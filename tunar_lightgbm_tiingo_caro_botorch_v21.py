"""CARO v2.1 com BoTorch/GPyTorch para retuning do LightGBM sobre Tiingo.

Versao: tiingo-lightgbm-caro-botorch-v2.1.0

Objetivo desta revisao:
- preservar integralmente a campanha BoTorch v2.0 como controle;
- impedir que incerteza alta domine a Expected Improvement quando o surrogate
  ja considera a regiao pouco promissora;
- privilegiar a regiao local do champion quando P(beat champion) esta baixa;
- reduzir a frequencia das recuperacoes globais por space filling.

A referencia historica de 43M continua fora da funcao objetivo, aquisicao,
promocao e criterio de parada.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch

import tunar_lightgbm_tiingo_caro as base
import tunar_lightgbm_tiingo_caro_botorch as v20

VERSION = "tiingo-lightgbm-caro-botorch-v2.1.0"
DIR_OUT = base.ROOT / "output" / "tiingo_lightgbm_caro_botorch_v21"
SEARCH_SEED = 20260919

# Na v2.0 a recuperacao global ocorria apos 4 trials sem melhoria. A auditoria
# mostrou 6 recuperacoes em 30 trials. Na v2.1 a exploracao global continua
# existindo, mas e deliberadamente mais rara.
STAGNATION_RECOVERY_TRIALS_V21 = 8

# Se nem o melhor ponto do pool tiver uma probabilidade razoavel de superar o
# champion, a v2.1 deixa de premiar regioes distantes apenas por incerteza e
# concentra a selecao na trust region (regiao de confianca) do champion.
LOW_CONFIDENCE_P_BEAT_ANCHOR = 0.10
LOCAL_RADIUS_MULTIPLIER = 1.50
LOCAL_RADIUS_MIN = 0.08
LOCAL_MIN_POOL = 64


def _space_filling_recovery(
    observations: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    known_hashes: set[str],
    seed: int,
    pool_size: int,
) -> dict[str, Any]:
    point = base.farthest_space_filling_point(
        observations,
        seed=seed + 31,
        pool_size=max(pool_size, 4096),
    )
    params = base.unit_to_settings(point)
    attempt = 0
    while base.settings_hash(params) in known_hashes and attempt < 20:
        attempt += 1
        point = base.farthest_space_filling_point(
            observations,
            seed=seed + 31 + attempt,
            pool_size=max(pool_size, 4096),
        )
        params = base.unit_to_settings(point)
    return {
        "params": params,
        "proposal_mode": "space_filling_recovery",
        "selection_reason": "stagnation_recovery_v21",
        "predicted": None,
        "pool": {
            "pool_size": max(pool_size, 4096),
            "trust_region_radius": float(state["trust_region_radius"]),
            "surrogate": "botorch_single_task_gp",
            "gp_backend": "gpytorch",
            "torch_device": str(v20.DEVICE),
            "v21_stagnation_threshold": STAGNATION_RECOVERY_TRIALS_V21,
        },
    }


def choose_proposal_botorch_v21(
    observations: list[dict[str, Any]],
    baseline: dict[str, Any],
    state: dict[str, Any],
    *,
    pool_size: int,
    exploration_weight: float,
    trial_index: int,
) -> dict[str, Any]:
    anchor = base.best_feasible(observations)
    known_hashes = {str(item["settings_hash"]) for item in observations}
    seed = SEARCH_SEED + trial_index * 1009
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if (
        int(state.get("no_improvement_streak", 0))
        >= STAGNATION_RECOVERY_TRIALS_V21
        and int(state.get("recovery_cooldown_remaining", 0)) <= 0
    ):
        return _space_filling_recovery(
            observations,
            state,
            known_hashes=known_hashes,
            seed=seed,
            pool_size=pool_size,
        )

    models = v20.fit_botorch_surrogates(observations)
    pool, pool_metadata = base.candidate_pool(
        observations,
        anchor,
        pool_size=pool_size,
        seed=seed,
        trust_region_radius=float(state["trust_region_radius"]),
        adaptive_trials_completed=int(state.get("adaptive_trials_completed", 0)),
    )

    params_rows: list[dict[str, Any]] = []
    unit_rows: list[np.ndarray] = []
    pool_hashes: set[str] = set()
    for point in pool:
        params = base.unit_to_settings(point)
        candidate_hash = base.settings_hash(params)
        if candidate_hash in known_hashes or candidate_hash in pool_hashes:
            continue
        pool_hashes.add(candidate_hash)
        params_rows.append(params)
        unit_rows.append(base.settings_to_unit(params))

    if not params_rows:
        raise RuntimeError("CARO-BoTorch v2.1 nao conseguiu gerar candidato inedito")

    x_pool = np.vstack(unit_rows)
    objective_mean, objective_std = v20.posterior_stats(models["objective"], x_pool)
    worst_mean, worst_std = v20.posterior_stats(models["log_worst_fold"], x_pool)
    dd_mean, dd_std = v20.posterior_stats(models["maximum_drawdown"], x_pool)

    best_score = float(anchor["objective"])
    control_score = base.robust_objective(
        baseline["metrics"], baseline["metrics"]
    )

    ei = v20.expected_improvement_torch(objective_mean, objective_std, best_score)
    p_positive_fold = v20.probability_above_torch(worst_mean, worst_std, 0.0)
    maxdd_floor = (
        float(baseline["metrics"]["strategy_maximum_drawdown"])
        - base.BASELINE_MAXDD_TOLERANCE
    )
    p_drawdown = v20.probability_above_torch(dd_mean, dd_std, maxdd_floor)
    p_feasible = p_positive_fold * p_drawdown
    p_beat_anchor_all = v20.probability_above_torch(
        objective_mean,
        objective_std,
        best_score,
    )
    p_beat_control_all = v20.probability_above_torch(
        objective_mean,
        objective_std,
        control_score,
    )

    std_scale = torch.quantile(objective_std, 0.90)
    if not torch.isfinite(std_scale) or float(std_scale.item()) <= 1e-12:
        std_scale = torch.tensor(1.0, dtype=v20.DTYPE, device=v20.DEVICE)
    exploration = objective_std / std_scale

    # v2.0 usava: EI * P(feasible) + exploration_weight * uncertainty.
    # Quando EI ficou muito pequena, a segunda parcela dominou e empurrou a
    # busca para regioes distantes. Na v2.1 a exploracao apenas MODULA uma
    # proposta que ja tem melhoria esperada e viabilidade.
    acquisition = (
        ei
        * p_feasible
        * (1.0 + float(exploration_weight) * exploration)
    )

    # Se a EI praticamente zerar em todo o pool, usamos P(beat anchor) como
    # fallback probabilistico, ainda condicionado a viabilidade. A incerteza
    # permanece multiplicativa: nunca ganha sozinha.
    acquisition_mode = "multiplicative_ei"
    if float(torch.max(acquisition).detach().cpu().item()) <= 1e-12:
        acquisition = (
            p_beat_anchor_all
            * p_feasible
            * (1.0 + 0.25 * float(exploration_weight) * exploration)
        )
        acquisition_mode = "probability_fallback"

    # Preferencia local quando o proprio GP esta pessimista em TODO o pool.
    # A distancia Chebyshev combina naturalmente com a trust region, pois o
    # candidate_pool gera vizinhos perturbando cada dimensao em +/- radius.
    anchor_vector = base.settings_to_unit(anchor["params"])
    distances = np.max(np.abs(x_pool - anchor_vector[None, :]), axis=1)
    max_p_beat_anchor = float(
        torch.max(p_beat_anchor_all).detach().cpu().item()
    )
    low_confidence_local_mode = max_p_beat_anchor < LOW_CONFIDENCE_P_BEAT_ANCHOR
    candidate_mask = np.ones(len(params_rows), dtype=bool)
    local_radius = max(
        LOCAL_RADIUS_MIN,
        float(state["trust_region_radius"]) * LOCAL_RADIUS_MULTIPLIER,
    )

    if low_confidence_local_mode:
        candidate_mask = distances <= local_radius
        if int(candidate_mask.sum()) < LOCAL_MIN_POOL:
            nearest = np.argsort(distances)[: min(LOCAL_MIN_POOL, len(distances))]
            candidate_mask = np.zeros(len(params_rows), dtype=bool)
            candidate_mask[nearest] = True

    eligible_indices = np.flatnonzero(candidate_mask)
    if len(eligible_indices) == 0:
        eligible_indices = np.arange(len(params_rows))

    eligible_tensor = torch.as_tensor(
        eligible_indices,
        dtype=torch.long,
        device=v20.DEVICE,
    )
    local_acquisition = acquisition.index_select(0, eligible_tensor)
    local_position = int(torch.argmax(local_acquisition).item())
    selected_index = int(eligible_indices[local_position])

    p_beat_control = p_beat_control_all[selected_index]
    p_beat_anchor = p_beat_anchor_all[selected_index]

    def scalar(tensor: torch.Tensor) -> float:
        return float(tensor.detach().cpu().item())

    selection_reason = "botorch_v21_constrained_ei"
    if low_confidence_local_mode:
        selection_reason += "_local_trust_region"
    if acquisition_mode == "probability_fallback":
        selection_reason += "_probability_fallback"

    return {
        "params": params_rows[selected_index],
        "proposal_mode": "adaptive_probability",
        "selection_reason": selection_reason,
        "predicted": {
            "objective_mean": scalar(objective_mean[selected_index]),
            "objective_std": scalar(objective_std[selected_index]),
            "expected_improvement": scalar(ei[selected_index]),
            "p_feasible": scalar(p_feasible[selected_index]),
            "p_positive_worst_fold": scalar(p_positive_fold[selected_index]),
            "p_drawdown_guardrail": scalar(p_drawdown[selected_index]),
            "p_beat_control": scalar(p_beat_control),
            "p_beat_anchor": scalar(p_beat_anchor),
            "acquisition": scalar(acquisition[selected_index]),
            "acquisition_mode": acquisition_mode,
            "low_confidence_local_mode": bool(low_confidence_local_mode),
            "max_pool_p_beat_anchor": max_p_beat_anchor,
            "distance_to_anchor_chebyshev": float(distances[selected_index]),
        },
        "pool": {
            "pool_size_requested": int(pool_size),
            "pool_size_unique": int(len(params_rows)),
            "pool_size_eligible": int(len(eligible_indices)),
            "surrogate": "botorch_single_task_gp",
            "gp_backend": "gpytorch",
            "torch_device": str(v20.DEVICE),
            "torch_dtype": str(v20.DTYPE),
            "local_radius": float(local_radius),
            "low_confidence_threshold": LOW_CONFIDENCE_P_BEAT_ANCHOR,
            "stagnation_recovery_trials": STAGNATION_RECOVERY_TRIALS_V21,
            **pool_metadata,
        },
    }


def main() -> int:
    # Isola completamente a v2.1: a campanha v2.0 permanece em
    # output/tiingo_lightgbm_caro_botorch e nunca e sobrescrita.
    v20.VERSION = VERSION
    v20.DIR_OUT = DIR_OUT
    v20.SEARCH_SEED = SEARCH_SEED

    base.VERSION = VERSION
    base.DIR_OUT = DIR_OUT
    base.SEARCH_SEED = SEARCH_SEED
    base.choose_proposal = choose_proposal_botorch_v21
    base.ensure_campaign = v20.ensure_campaign_botorch

    base.log(f"CARO-BoTorch v2.1 | device={v20.DEVICE}")
    base.log("v2.0 preservada como controle; resultados gravados em pasta separada")
    base.log(
        "Aquisicao v2.1: EI condicionada a viabilidade, exploracao multiplicativa, "
        "preferencia local sob baixa confianca e recovery global mais raro"
    )
    base.log("Referencia de 43M: contexto externo apenas; NAO e objetivo nem parada")
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
