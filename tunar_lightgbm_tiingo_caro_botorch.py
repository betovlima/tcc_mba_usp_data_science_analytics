"""CARO v2 com BoTorch/GPyTorch para retuning do LightGBM sobre Tiingo.

Versao: tiingo-lightgbm-caro-botorch-v2.0.0

Este experimento reutiliza exatamente o mesmo warm-up LHS e o mesmo motor de
backtest do CARO v1. A unica troca metodologica e o surrogate Bayesiano:
scikit-learn GaussianProcessRegressor -> BoTorch SingleTaskGP/GPyTorch.

A referencia historica de 43M nao participa da funcao objetivo, da aquisicao,
da promocao nem de qualquer criterio de parada.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import tunar_lightgbm_tiingo_caro as base

VERSION = "tiingo-lightgbm-caro-botorch-v2.0.0"
DIR_OUT = base.ROOT / "output" / "tiingo_lightgbm_caro_botorch"
SEARCH_SEED = 20260918


def _device() -> torch.device:
    requested = os.getenv("TCC_BOTORCH_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("TCC_BOTORCH_DEVICE=cuda, mas CUDA nao esta disponivel")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _device()
DTYPE = torch.double


def _tensor(values: np.ndarray | list[float]) -> torch.Tensor:
    return torch.as_tensor(values, dtype=DTYPE, device=DEVICE)


def _fit_single_task_gp(x: np.ndarray, y: np.ndarray) -> SingleTaskGP:
    train_x = _tensor(np.asarray(x, dtype=float))
    train_y = _tensor(np.asarray(y, dtype=float).reshape(-1, 1))
    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def fit_botorch_surrogates(
    observations: list[dict[str, Any]],
) -> dict[str, SingleTaskGP]:
    x = np.vstack([base.settings_to_unit(item["params"]) for item in observations])
    objective = np.asarray([float(item["objective"]) for item in observations], dtype=float)
    log_worst = np.log(
        np.asarray(
            [max(1e-12, float(item["metrics"]["worst_fold_multiple"])) for item in observations],
            dtype=float,
        )
    )
    maxdd = np.asarray(
        [float(item["metrics"]["strategy_maximum_drawdown"]) for item in observations],
        dtype=float,
    )
    return {
        "objective": _fit_single_task_gp(x, objective),
        "log_worst_fold": _fit_single_task_gp(x, log_worst),
        "maximum_drawdown": _fit_single_task_gp(x, maxdd),
    }


def posterior_stats(model: SingleTaskGP, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        posterior = model.posterior(_tensor(np.asarray(x, dtype=float)))
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1).clamp_min(1e-12)
        return mean, variance.sqrt()


def expected_improvement_torch(
    mean: torch.Tensor,
    std: torch.Tensor,
    best: float,
    xi: float = 0.01,
) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
        torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
    )
    safe_std = std.clamp_min(1e-12)
    improvement = mean - float(best) - float(xi)
    z = improvement / safe_std
    ei = improvement * normal.cdf(z) + safe_std * torch.exp(normal.log_prob(z))
    return ei.clamp_min(0.0)


def probability_above_torch(
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
        torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
    )
    safe_std = std.clamp_min(1e-12)
    return normal.cdf((mean - float(threshold)) / safe_std).clamp(0.0, 1.0)


def choose_proposal_botorch(
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
        int(state.get("no_improvement_streak", 0)) >= base.STAGNATION_RECOVERY_TRIALS
        and int(state.get("recovery_cooldown_remaining", 0)) <= 0
    ):
        point = base.farthest_space_filling_point(
            observations,
            seed=seed + 31,
            pool_size=max(pool_size, 2048),
        )
        params = base.unit_to_settings(point)
        attempt = 0
        while base.settings_hash(params) in known_hashes and attempt < 20:
            attempt += 1
            point = base.farthest_space_filling_point(
                observations,
                seed=seed + 31 + attempt,
                pool_size=max(pool_size, 2048),
            )
            params = base.unit_to_settings(point)
        return {
            "params": params,
            "proposal_mode": "space_filling_recovery",
            "selection_reason": "stagnation_recovery",
            "predicted": None,
            "pool": {
                "pool_size": max(pool_size, 2048),
                "trust_region_radius": float(state["trust_region_radius"]),
                "surrogate": "botorch_single_task_gp",
                "device": str(DEVICE),
            },
        }

    models = fit_botorch_surrogates(observations)
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
        raise RuntimeError("CARO-BoTorch nao conseguiu gerar candidato inedito")

    x_pool = np.vstack(unit_rows)
    objective_mean, objective_std = posterior_stats(models["objective"], x_pool)
    worst_mean, worst_std = posterior_stats(models["log_worst_fold"], x_pool)
    dd_mean, dd_std = posterior_stats(models["maximum_drawdown"], x_pool)

    best_score = float(anchor["objective"])
    ei = expected_improvement_torch(objective_mean, objective_std, best_score)
    p_positive_fold = probability_above_torch(worst_mean, worst_std, 0.0)
    maxdd_floor = (
        float(baseline["metrics"]["strategy_maximum_drawdown"])
        - base.BASELINE_MAXDD_TOLERANCE
    )
    p_drawdown = probability_above_torch(dd_mean, dd_std, maxdd_floor)
    p_feasible = p_positive_fold * p_drawdown

    std_scale = torch.quantile(objective_std, 0.90)
    if not torch.isfinite(std_scale) or float(std_scale.item()) <= 1e-12:
        std_scale = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)
    exploration = objective_std / std_scale
    acquisition = ei * p_feasible + float(exploration_weight) * exploration

    selected_index = int(torch.argmax(acquisition).item())
    control_score = base.robust_objective(
        baseline["metrics"], baseline["metrics"]
    )
    p_beat_control = probability_above_torch(
        objective_mean[selected_index:selected_index + 1],
        objective_std[selected_index:selected_index + 1],
        control_score,
    )[0]
    p_beat_anchor = probability_above_torch(
        objective_mean[selected_index:selected_index + 1],
        objective_std[selected_index:selected_index + 1],
        best_score,
    )[0]

    def scalar(tensor: torch.Tensor) -> float:
        return float(tensor.detach().cpu().item())

    return {
        "params": params_rows[selected_index],
        "proposal_mode": "adaptive_probability",
        "selection_reason": "botorch_constrained_expected_improvement",
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
        },
        "pool": {
            "pool_size_requested": int(pool_size),
            "pool_size_unique": int(len(params_rows)),
            "surrogate": "botorch_single_task_gp",
            "gp_backend": "gpytorch",
            "torch_device": str(DEVICE),
            "torch_dtype": str(DTYPE),
            **pool_metadata,
        },
    }


def ensure_campaign_botorch(
    lhs_campaign: dict[str, Any],
    current_signature: dict[str, Any],
    warmup_count: int,
) -> None:
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    source_signature = (
        (lhs_campaign.get("dataset_signature") or {}).get("combined_sha256")
    )
    if source_signature != current_signature.get("combined_sha256"):
        raise RuntimeError(
            "O dataset atual nao corresponde ao dataset congelado da campanha LHS"
        )

    path = DIR_OUT / "campaign.json"
    existing = base.read_json_optional(path)
    if existing is not None:
        if existing.get("script_version") != VERSION:
            raise RuntimeError("A pasta CARO-BoTorch existente pertence a outra versao")
        if (
            (existing.get("dataset_signature") or {}).get("combined_sha256")
            != current_signature.get("combined_sha256")
        ):
            raise RuntimeError("Dataset Tiingo mudou desde o inicio do CARO-BoTorch")
        return

    manifest = {
        "schema_version": 1,
        "script_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "tiingo_total_causal_lightgbm_caro_botorch",
        "warmup_source": base.SOURCE_LHS_DIR.relative_to(base.ROOT).as_posix(),
        "warmup_observation_count": int(warmup_count),
        "search_seed": SEARCH_SEED,
        "search_space": [dict(spec) for spec in base.SEARCH_SPACE],
        "objective_weights": base.OBJECTIVE_WEIGHTS,
        "historical_reference_capital_context_only": base.HISTORICAL_REFERENCE_CAPITAL,
        "selection_uses_43m_reference": False,
        "stopping_uses_43m_reference": False,
        "fixed_components": [
            "Tiingo RAW snapshot",
            "causal split normalization",
            "causal dividend normalization",
            "features",
            "targets",
            "walk-forward folds",
            "rotation policy",
            "fees and slippage",
        ],
        "tuned_component": "LightGBM hyperparameters only",
        "surrogate": "BoTorch SingleTaskGP + GPyTorch",
        "acquisition": "constrained_expected_improvement_plus_exploration",
        "torch_device": str(DEVICE),
        "torch_dtype": str(DTYPE),
        "dataset_signature": current_signature,
    }
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=base.json_default) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    # O restante da campanha (warm-up, retomada, backtest, folds, persistencia,
    # trust region e parada) e deliberadamente o mesmo do CARO v1.
    base.VERSION = VERSION
    base.DIR_OUT = DIR_OUT
    base.SEARCH_SEED = SEARCH_SEED
    base.choose_proposal = choose_proposal_botorch
    base.ensure_campaign = ensure_campaign_botorch

    base.log(f"Surrogate experimental: BoTorch/GPyTorch | device={DEVICE}")
    base.log("Comparacao controlada: somente o surrogate Bayesiano foi trocado")
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
