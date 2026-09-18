"""CARO adaptativo do LightGBM sobre o baseline oficial Tiingo 56 split-causal.

Versao: tiingo-56-lightgbm-caro-v1.0.0

Esta etapa deve ser executada APOS a campanha LHS de tunar_lightgbm_tiingo.py.
Ela reutiliza baseline + observacoes LHS concluidas como warm-up e passa a
propor candidatos sequencialmente com Gaussian Process (processo gaussiano),
Expected Improvement (melhoria esperada), probabilidade de viabilidade,
exploracao global, trust region (regiao de confianca) e recuperacao por
estagnacao.

A referencia historica de 43M da Alpaca NAO participa da funcao objetivo,
da aquisicao, da promocao ou de qualquer criterio de parada.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import CONFIG
from tcc_engine.execution import apply_slippage, calculate_reference_fees
from tunar_lightgbm_tiingo import (
    BASELINE_MAXDD_TOLERANCE,
    DIR_EVENTS,
    DIR_SPLITS,
    DIR_TIINGO,
    HISTORICAL_REFERENCE_CAPITAL,
    ROOT,
    dataset_signature,
    json_default,
    load_tiingo_split_causal,
    read_json_optional,
)

VERSION = "tiingo-56-lightgbm-caro-v1.0.0"
SOURCE_LHS_DIR = ROOT / "output" / "tiingo_56_lightgbm_lhs_v1"
DIR_OUT = ROOT / "output" / "tiingo_56_lightgbm_caro_v1"

SEARCH_SEED = 20260917
DEFAULT_TRIALS = 30
DEFAULT_POOL_SIZE = 2048
DEFAULT_EXPLORATION_WEIGHT = 0.15

TRUST_REGION_INITIAL = 0.20
TRUST_REGION_MIN = 0.04
TRUST_REGION_MAX = 0.40
TRUST_REGION_SUCCESS_EXPANSION = 1.25
TRUST_REGION_FAILURE_CONTRACTION = 0.70
TRUST_REGION_FAILURE_TOLERANCE = 3

GLOBAL_POOL_FRACTION_INITIAL = 0.30
GLOBAL_POOL_FRACTION_MIN = 0.15
ANCHOR_LOCAL_FRACTION = 0.50
STAGNATION_RECOVERY_TRIALS = 4
RECOVERY_COOLDOWN_TRIALS = 2

OBJECTIVE_WEIGHTS = {
    "log_capital_ratio": 1.00,
    "log_worst_fold_ratio": 1.00,
    "log_geometric_fold_ratio": 0.50,
    "sharpe_delta": 0.25,
    "maximum_drawdown_delta": 2.00,
    "fold_stability_delta": 0.25,
}

# O espaco e propositalmente o mesmo da campanha LHS.
SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {"name": "n_estimators", "kind": "int", "min": 180, "max": 700},
    {"name": "learning_rate", "kind": "log_float", "min": 0.008, "max": 0.080},
    {"name": "max_depth", "kind": "int", "min": 2, "max": 6},
    {"name": "num_leaves", "kind": "int", "min": 4, "max": 48},
    {"name": "min_child_samples", "kind": "int", "min": 10, "max": 80},
    {"name": "min_child_weight", "kind": "log_float", "min": 0.5, "max": 20.0},
    {"name": "subsample", "kind": "float", "min": 0.65, "max": 1.00},
    {"name": "subsample_freq", "kind": "categorical", "values": (0, 1)},
    {"name": "colsample_bytree", "kind": "float", "min": 0.60, "max": 1.00},
    {"name": "reg_alpha", "kind": "log_float", "min": 1e-4, "max": 1.5},
    {"name": "reg_lambda", "kind": "log_float", "min": 0.10, "max": 10.0},
    {"name": "max_bin", "kind": "categorical", "values": (63, 127, 255)},
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def settings_hash(settings: dict[str, Any]) -> str:
    encoded = json.dumps(
        settings, sort_keys=True, separators=(",", ":"), default=json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def normalized_value(value: Any, spec: dict[str, Any]) -> float:
    kind = str(spec["kind"])
    if kind == "categorical":
        values = list(spec["values"])
        try:
            index = values.index(value)
        except ValueError:
            index = int(np.argmin([abs(float(value) - float(item)) for item in values]))
        return 0.0 if len(values) <= 1 else float(index / (len(values) - 1))

    low = float(spec["min"])
    high = float(spec["max"])
    numeric = float(value)
    if kind == "log_float":
        numeric = math.log(max(numeric, low))
        low = math.log(low)
        high = math.log(high)
    return 0.0 if high <= low else clip01((numeric - low) / (high - low))


def settings_to_unit(settings: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [normalized_value(settings[spec["name"]], spec) for spec in SEARCH_SPACE],
        dtype=float,
    )


def int_from_unit(low: int, high: int, u: float) -> int:
    value = low + int(math.floor(clip01(u) * (high - low + 1)))
    return int(min(high, max(low, value)))


def unit_to_settings(point: np.ndarray) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for u, spec in zip(point, SEARCH_SPACE):
        kind = str(spec["kind"])
        name = str(spec["name"])
        u = clip01(float(u))
        if kind == "int":
            values[name] = int_from_unit(int(spec["min"]), int(spec["max"]), u)
        elif kind == "float":
            low = float(spec["min"])
            high = float(spec["max"])
            values[name] = float(low + u * (high - low))
        elif kind == "log_float":
            low = math.log(float(spec["min"]))
            high = math.log(float(spec["max"]))
            values[name] = float(math.exp(low + u * (high - low)))
        elif kind == "categorical":
            options = list(spec["values"])
            index = int_from_unit(0, len(options) - 1, u)
            values[name] = options[index]
        else:
            raise RuntimeError(f"Tipo de parametro desconhecido: {kind}")

    # Restricao estrutural do LightGBM: folhas coerentes com a profundidade.
    depth = int(values["max_depth"])
    max_leaves = max(2, min(48, 2 ** depth))
    min_leaves = 2 if max_leaves < 4 else 4
    values["num_leaves"] = int(
        min(max_leaves, max(min_leaves, int(values["num_leaves"])))
    )
    values["subsample_freq"] = int(values["subsample_freq"])
    values["max_bin"] = int(values["max_bin"])
    values["n_jobs"] = -1
    return values


def build_config(candidate_id: str, params: dict[str, Any]):
    settings = deepcopy(CONFIG.research_model_settings)
    settings["schema_version"] = 3
    settings["settings_revision"] = 3
    settings["profile_id"] = f"tcc-tiingo-caro-{candidate_id}"
    settings["lightgbm"] = deepcopy(params)
    return CONFIG.model_copy(
        update={
            "assets": tuple(CONFIG.assets),
            "calendar_anchor_assets": tuple(CONFIG.assets),
            "research_reference_assets": tuple(CONFIG.assets),
            "research_candidate_assets": (),
            "research_model_settings": settings,
            "research_model_family": "lightgbm_utility",
            "random_state": 42,
            "deterministic_execution": True,
            "numeric_thread_limit": 1,
        }
    )


def fold_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    folds = list(metrics.get("walk_forward_folds") or [])
    if not folds:
        raise RuntimeError("Resultado sem walk_forward_folds")
    multiples: list[float] = []
    log_growth: list[float] = []
    drawdowns: list[float] = []
    for fold in folds:
        start = float(fold["strategy_starting_capital"])
        end = float(fold["strategy_ending_capital"])
        multiple = end / start if start > 0 else float("nan")
        if not np.isfinite(multiple) or multiple <= 0:
            multiple = 1e-12
        multiples.append(float(multiple))
        log_growth.append(float(math.log(multiple)))
        drawdowns.append(abs(float(fold.get("maximum_drawdown") or 0.0)))
    return {
        "fold_count": len(folds),
        "fold_multiples": multiples,
        "worst_fold_multiple": float(min(multiples)),
        "median_fold_multiple": float(np.median(multiples)),
        "geometric_mean_fold_multiple": float(math.exp(float(np.mean(log_growth)))),
        "fold_log_growth_std": float(np.std(log_growth)),
        "all_folds_positive": bool(all(value > 1.0 for value in multiples)),
        "mean_fold_drawdown_abs": float(np.mean(drawdowns)),
        "worst_fold_drawdown_abs": float(max(drawdowns)),
    }


def compact_result(
    candidate_id: str,
    kind: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    fm = fold_metrics(metrics)
    return {
        "schema_version": 1,
        "script_version": VERSION,
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "status": "completed",
        "elapsed_seconds": float(elapsed),
        "params": deepcopy(params),
        "settings_hash": settings_hash(params),
        "metrics": {
            "strategy_ending_capital": float(metrics["strategy_ending_capital"]),
            "strategy_return": float(metrics["strategy_return"]),
            "strategy_cagr": float(metrics["strategy_cagr"]),
            "strategy_sharpe": float(metrics["strategy_sharpe"]),
            "strategy_maximum_drawdown": float(metrics["strategy_maximum_drawdown"]),
            "capital_rotations": int(metrics.get("capital_rotations") or 0),
            "effective_switch_margin": metrics.get("effective_switch_margin"),
            "effective_switch_margin_mean": metrics.get("effective_switch_margin_mean"),
            "walk_forward_folds": metrics.get("walk_forward_folds"),
            **fm,
        },
    }


def read_lhs_warmup() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    campaign = read_json_optional(SOURCE_LHS_DIR / "campaign.json")
    candidates = read_json_optional(SOURCE_LHS_DIR / "candidates.json")
    if not isinstance(campaign, dict) or not isinstance(candidates, list):
        raise RuntimeError(
            "Campanha LHS nao encontrada. Finalize primeiro: "
            "python tunar_lightgbm_tiingo.py"
        )

    observations: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        result = read_json_optional(SOURCE_LHS_DIR / candidate_id / "result.json")
        if result is None:
            missing.append(candidate_id)
            continue
        if result.get("status") != "completed":
            errors.append(candidate_id)
            continue
        observations.append(
            {
                "candidate_id": candidate_id,
                "kind": str(result.get("candidate_kind") or candidate.get("kind") or "lhs"),
                "params": deepcopy(result["params"]),
                "metrics": deepcopy(result["metrics"]),
                "source": "lhs_warmup",
            }
        )

    if missing or errors:
        parts = []
        if missing:
            parts.append("ausentes=" + ",".join(missing))
        if errors:
            parts.append("com_erro=" + ",".join(errors))
        raise RuntimeError(
            "A campanha LHS precisa estar 100% concluida antes do CARO: "
            + " | ".join(parts)
        )
    if len(observations) != len(candidates):
        raise RuntimeError("Warm-up LHS incompleto")
    if len(observations) < 4:
        raise RuntimeError("CARO exige pelo menos quatro observacoes concluidas")

    return observations, campaign


def baseline_observation(observations: list[dict[str, Any]]) -> dict[str, Any]:
    for item in observations:
        if item["candidate_id"] == "candidate_000":
            return item
    raise RuntimeError("candidate_000 baseline nao encontrado no warm-up")


def robust_objective(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> float:
    eps = 1e-12
    capital = max(eps, float(metrics["strategy_ending_capital"]))
    b_capital = max(eps, float(baseline_metrics["strategy_ending_capital"]))
    worst = max(eps, float(metrics["worst_fold_multiple"]))
    b_worst = max(eps, float(baseline_metrics["worst_fold_multiple"]))
    geom = max(eps, float(metrics["geometric_mean_fold_multiple"]))
    b_geom = max(eps, float(baseline_metrics["geometric_mean_fold_multiple"]))
    sharpe = float(metrics["strategy_sharpe"])
    b_sharpe = float(baseline_metrics["strategy_sharpe"])
    maxdd = float(metrics["strategy_maximum_drawdown"])
    b_maxdd = float(baseline_metrics["strategy_maximum_drawdown"])
    instability = float(metrics["fold_log_growth_std"])
    b_instability = float(baseline_metrics["fold_log_growth_std"])

    return float(
        OBJECTIVE_WEIGHTS["log_capital_ratio"] * math.log(capital / b_capital)
        + OBJECTIVE_WEIGHTS["log_worst_fold_ratio"] * math.log(worst / b_worst)
        + OBJECTIVE_WEIGHTS["log_geometric_fold_ratio"] * math.log(geom / b_geom)
        + OBJECTIVE_WEIGHTS["sharpe_delta"] * (sharpe - b_sharpe)
        + OBJECTIVE_WEIGHTS["maximum_drawdown_delta"] * (maxdd - b_maxdd)
        + OBJECTIVE_WEIGHTS["fold_stability_delta"] * (b_instability - instability)
    )


def is_feasible(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> bool:
    maxdd_floor = (
        float(baseline_metrics["strategy_maximum_drawdown"])
        - BASELINE_MAXDD_TOLERANCE
    )
    return bool(
        bool(metrics["all_folds_positive"])
        and float(metrics["worst_fold_multiple"]) > 1.0
        and float(metrics["strategy_maximum_drawdown"]) >= maxdd_floor
    )


def enrich_observations(
    observations: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in observations:
        row = deepcopy(item)
        row["objective"] = robust_objective(row["metrics"], baseline_metrics)
        row["feasible"] = is_feasible(row["metrics"], baseline_metrics)
        row["settings_hash"] = settings_hash(row["params"])
        enriched.append(row)
    return enriched


def best_feasible(observations: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in observations if bool(item.get("feasible"))]
    if not eligible:
        raise RuntimeError("Nenhuma observacao viavel para ancorar o CARO")
    return max(
        eligible,
        key=lambda item: (
            float(item["objective"]),
            float(item["metrics"]["strategy_ending_capital"]),
            float(item["metrics"]["worst_fold_multiple"]),
        ),
    )


def gaussian_process(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> GaussianProcessRegressor:
    dimensions = x.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.full(dimensions, 0.30, dtype=float),
            length_scale_bounds=(0.03, 3.0),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=2,
        alpha=1e-8,
    )
    model.fit(x, y)
    return model


def fit_surrogates(
    observations: list[dict[str, Any]],
    seed: int,
) -> dict[str, GaussianProcessRegressor]:
    x = np.vstack([settings_to_unit(item["params"]) for item in observations])
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
        "objective": gaussian_process(x, objective, seed),
        "log_worst_fold": gaussian_process(x, log_worst, seed + 1),
        "maximum_drawdown": gaussian_process(x, maxdd, seed + 2),
    }


def candidate_pool(
    observations: list[dict[str, Any]],
    anchor: dict[str, Any],
    *,
    pool_size: int,
    seed: int,
    trust_region_radius: float,
    adaptive_trials_completed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    dimensions = len(SEARCH_SPACE)
    rng = np.random.default_rng(seed)
    global_fraction = max(
        GLOBAL_POOL_FRACTION_MIN,
        GLOBAL_POOL_FRACTION_INITIAL - 0.01 * adaptive_trials_completed,
    )
    anchor_fraction = ANCHOR_LOCAL_FRACTION
    if global_fraction + anchor_fraction > 0.90:
        anchor_fraction = 0.90 - global_fraction
    top_fraction = 1.0 - global_fraction - anchor_fraction

    global_count = max(1, int(round(pool_size * global_fraction)))
    anchor_count = max(1, int(round(pool_size * anchor_fraction)))
    top_count = max(1, pool_size - global_count - anchor_count)

    global_points = qmc.LatinHypercube(
        d=dimensions, seed=seed + 17
    ).random(n=global_count)

    anchor_vector = settings_to_unit(anchor["params"])
    anchor_points = np.clip(
        anchor_vector
        + rng.uniform(
            -trust_region_radius,
            trust_region_radius,
            size=(anchor_count, dimensions),
        ),
        0.0,
        1.0,
    )

    ranked = sorted(
        observations,
        key=lambda item: (
            bool(item.get("feasible")),
            float(item["objective"]),
        ),
        reverse=True,
    )
    centers = np.vstack(
        [settings_to_unit(item["params"]) for item in ranked[: min(5, len(ranked))]]
    )
    center_indices = rng.integers(0, len(centers), size=top_count)
    top_radius = min(TRUST_REGION_MAX, trust_region_radius * 1.35)
    top_points = np.clip(
        centers[center_indices]
        + rng.uniform(-top_radius, top_radius, size=(top_count, dimensions)),
        0.0,
        1.0,
    )

    return np.vstack([global_points, anchor_points, top_points]), {
        "trust_region_radius": float(trust_region_radius),
        "global_fraction": float(global_fraction),
        "anchor_fraction": float(anchor_fraction),
        "top_regions_fraction": float(top_fraction),
    }


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    std = np.asarray(std, dtype=float)
    improvement = np.asarray(mean, dtype=float) - float(best) - float(xi)
    safe_std = np.maximum(std, 1e-12)
    z = improvement / safe_std
    ei = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
    ei = np.where(std <= 1e-12, 0.0, ei)
    return np.maximum(0.0, ei)


def probability_above(
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> np.ndarray:
    safe_std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    z = (np.asarray(mean, dtype=float) - float(threshold)) / safe_std
    probability = norm.cdf(z)
    deterministic = np.asarray(std, dtype=float) <= 1e-12
    probability = np.where(
        deterministic,
        (np.asarray(mean, dtype=float) >= threshold).astype(float),
        probability,
    )
    return np.clip(probability, 0.0, 1.0)


def farthest_space_filling_point(
    observations: list[dict[str, Any]],
    *,
    seed: int,
    pool_size: int,
) -> np.ndarray:
    dimensions = len(SEARCH_SPACE)
    pool = qmc.LatinHypercube(d=dimensions, seed=seed).random(n=pool_size)
    observed = np.vstack([settings_to_unit(item["params"]) for item in observations])
    distances = np.sqrt(
        ((pool[:, None, :] - observed[None, :, :]) ** 2).sum(axis=2)
    )
    min_distance = distances.min(axis=1)
    return pool[int(np.argmax(min_distance))]


def choose_proposal(
    observations: list[dict[str, Any]],
    baseline: dict[str, Any],
    state: dict[str, Any],
    *,
    pool_size: int,
    exploration_weight: float,
    trial_index: int,
) -> dict[str, Any]:
    anchor = best_feasible(observations)
    known_hashes = {str(item["settings_hash"]) for item in observations}
    seed = SEARCH_SEED + trial_index * 1009

    if (
        int(state.get("no_improvement_streak", 0)) >= STAGNATION_RECOVERY_TRIALS
        and int(state.get("recovery_cooldown_remaining", 0)) <= 0
    ):
        point = farthest_space_filling_point(
            observations,
            seed=seed + 31,
            pool_size=max(pool_size, 2048),
        )
        params = unit_to_settings(point)
        attempt = 0
        while settings_hash(params) in known_hashes and attempt < 20:
            attempt += 1
            point = farthest_space_filling_point(
                observations,
                seed=seed + 31 + attempt,
                pool_size=max(pool_size, 2048),
            )
            params = unit_to_settings(point)
        return {
            "params": params,
            "proposal_mode": "space_filling_recovery",
            "selection_reason": "stagnation_recovery",
            "predicted": None,
            "pool": {
                "pool_size": max(pool_size, 2048),
                "trust_region_radius": float(state["trust_region_radius"]),
            },
        }

    models = fit_surrogates(observations, seed)
    pool, pool_metadata = candidate_pool(
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
        params = unit_to_settings(point)
        candidate_hash = settings_hash(params)
        if candidate_hash in known_hashes or candidate_hash in pool_hashes:
            continue
        pool_hashes.add(candidate_hash)
        params_rows.append(params)
        unit_rows.append(settings_to_unit(params))

    if not params_rows:
        raise RuntimeError("CARO nao conseguiu gerar candidato inedito")

    x_pool = np.vstack(unit_rows)
    objective_mean, objective_std = models["objective"].predict(
        x_pool, return_std=True
    )
    worst_mean, worst_std = models["log_worst_fold"].predict(
        x_pool, return_std=True
    )
    dd_mean, dd_std = models["maximum_drawdown"].predict(
        x_pool, return_std=True
    )

    best_score = float(anchor["objective"])
    ei = expected_improvement(objective_mean, objective_std, best_score)
    p_positive_fold = probability_above(worst_mean, worst_std, 0.0)
    maxdd_floor = (
        float(baseline["metrics"]["strategy_maximum_drawdown"])
        - BASELINE_MAXDD_TOLERANCE
    )
    p_drawdown = probability_above(dd_mean, dd_std, maxdd_floor)
    p_feasible = p_positive_fold * p_drawdown

    std_scale = float(np.quantile(objective_std, 0.90)) if len(objective_std) else 1.0
    if not np.isfinite(std_scale) or std_scale <= 1e-12:
        std_scale = 1.0
    exploration = np.asarray(objective_std, dtype=float) / std_scale
    acquisition = ei * p_feasible + float(exploration_weight) * exploration

    selected_index = int(np.argmax(acquisition))
    control_score = robust_objective(
        baseline["metrics"], baseline["metrics"]
    )
    p_beat_control = probability_above(
        np.asarray([objective_mean[selected_index]]),
        np.asarray([objective_std[selected_index]]),
        control_score,
    )[0]
    p_beat_anchor = probability_above(
        np.asarray([objective_mean[selected_index]]),
        np.asarray([objective_std[selected_index]]),
        best_score,
    )[0]

    return {
        "params": params_rows[selected_index],
        "proposal_mode": "adaptive_probability",
        "selection_reason": "constrained_expected_improvement",
        "predicted": {
            "objective_mean": float(objective_mean[selected_index]),
            "objective_std": float(objective_std[selected_index]),
            "expected_improvement": float(ei[selected_index]),
            "p_feasible": float(p_feasible[selected_index]),
            "p_positive_worst_fold": float(p_positive_fold[selected_index]),
            "p_drawdown_guardrail": float(p_drawdown[selected_index]),
            "p_beat_control": float(p_beat_control),
            "p_beat_anchor": float(p_beat_anchor),
            "acquisition": float(acquisition[selected_index]),
        },
        "pool": {
            "pool_size_requested": int(pool_size),
            "pool_size_unique": int(len(params_rows)),
            **pool_metadata,
        },
    }


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": VERSION,
        "trust_region_radius": TRUST_REGION_INITIAL,
        "success_streak": 0,
        "failure_streak": 0,
        "no_improvement_streak": 0,
        "recovery_cooldown_remaining": 0,
        "stagnation_recoveries": 0,
        "adaptive_trials_completed": 0,
        "champion_revision": 0,
        "last_champion_candidate_id": "candidate_000",
    }


def evolve_state(
    state: dict[str, Any],
    *,
    promoted: bool,
    proposal_mode: str,
    candidate_id: str,
) -> dict[str, Any]:
    next_state = deepcopy(state)
    if proposal_mode == "space_filling_recovery":
        next_state["no_improvement_streak"] = 0
        next_state["failure_streak"] = 0
        next_state["success_streak"] = 0
        next_state["recovery_cooldown_remaining"] = RECOVERY_COOLDOWN_TRIALS
        next_state["stagnation_recoveries"] = (
            int(next_state.get("stagnation_recoveries", 0)) + 1
        )
        return next_state

    next_state["adaptive_trials_completed"] = (
        int(next_state.get("adaptive_trials_completed", 0)) + 1
    )
    if int(next_state.get("recovery_cooldown_remaining", 0)) > 0:
        next_state["recovery_cooldown_remaining"] = (
            int(next_state["recovery_cooldown_remaining"]) - 1
        )

    if promoted:
        next_state["success_streak"] = int(next_state.get("success_streak", 0)) + 1
        next_state["failure_streak"] = 0
        next_state["no_improvement_streak"] = 0
        next_state["champion_revision"] = int(
            next_state.get("champion_revision", 0)
        ) + 1
        next_state["last_champion_candidate_id"] = candidate_id
        next_state["trust_region_radius"] = min(
            TRUST_REGION_MAX,
            float(next_state["trust_region_radius"])
            * TRUST_REGION_SUCCESS_EXPANSION,
        )
    else:
        next_state["success_streak"] = 0
        next_state["failure_streak"] = int(next_state.get("failure_streak", 0)) + 1
        next_state["no_improvement_streak"] = (
            int(next_state.get("no_improvement_streak", 0)) + 1
        )
        if int(next_state["failure_streak"]) >= TRUST_REGION_FAILURE_TOLERANCE:
            next_state["trust_region_radius"] = max(
                TRUST_REGION_MIN,
                float(next_state["trust_region_radius"])
                * TRUST_REGION_FAILURE_CONTRACTION,
            )
            next_state["failure_streak"] = 0
    return next_state


def run_candidate(
    candidate_id: str,
    proposal: dict[str, Any],
    series: dict[str, pd.DataFrame],
    save_curves: bool,
) -> dict[str, Any]:
    candidate_dir = DIR_OUT / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result_path = candidate_dir / "result.json"
    proposal_path = candidate_dir / "proposal.json"

    existing = read_json_optional(result_path)
    if existing and existing.get("status") == "completed":
        log(f"[{candidate_id}] ja concluido; reutilizando resultado")
        return existing

    proposal_path.write_text(
        json.dumps(
            proposal, indent=2, ensure_ascii=False, default=json_default
        )
        + "\n",
        encoding="utf-8",
    )

    params = deepcopy(proposal["params"])
    config = build_config(candidate_id, params)
    started = time.perf_counter()
    log(
        f"[{candidate_id}] iniciando | mode={proposal['proposal_mode']} "
        f"| reason={proposal['selection_reason']}"
    )
    predicted = proposal.get("predicted")
    if isinstance(predicted, dict):
        log(
            f"[{candidate_id}] CARO | P(beat control)={predicted['p_beat_control']:.1%} "
            f"| P(feasible)={predicted['p_feasible']:.1%} "
            f"| EI={predicted['expected_improvement']:.5f}"
        )

    def progress(percent: float, stage: str, completed: int) -> None:
        log(f"[{candidate_id}] {percent:5.1f}% | {stage}")

    try:
        results = run_rotation_models(
            series,
            config,
            calculate_reference_fees,
            apply_slippage,
            progress_callback=progress,
            technical_log_callback=lambda message: None,
        )
        if len(results) != 1:
            raise RuntimeError(
                f"Esperava uma execucao; recebidas={len(results)}"
            )
        result = results[0]
        metrics = dict(result.metrics)
        payload = compact_result(
            candidate_id,
            "caro",
            params,
            metrics,
            time.perf_counter() - started,
        )
        payload["proposal"] = {
            "proposal_mode": proposal["proposal_mode"],
            "selection_reason": proposal["selection_reason"],
            "predicted": deepcopy(proposal.get("predicted")),
            "pool": deepcopy(proposal.get("pool")),
        }
        result_path.write_text(
            json.dumps(
                payload, indent=2, ensure_ascii=False, default=json_default
            )
            + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            list(metrics.get("walk_forward_folds") or [])
        ).to_csv(candidate_dir / "folds.csv", index=False)

        if save_curves:
            predictions = result.predictions.copy()
            cols = [
                c
                for c in (
                    "strategy_equity",
                    "selected_asset",
                    "selected_score",
                    "decision_score",
                    "trade_action",
                    "trade_reason",
                    "walk_forward_fold",
                )
                if c in predictions.columns
            ]
            curve = predictions[cols].copy() if cols else predictions
            if curve.index.name is not None or not isinstance(
                curve.index, pd.RangeIndex
            ):
                curve = curve.reset_index()
            curve.to_csv(candidate_dir / "equity_curve.csv", index=False)

        log(
            f"[{candidate_id}] concluido | "
            f"capital=US$ {payload['metrics']['strategy_ending_capital']:,.2f} "
            f"| worst_fold={payload['metrics']['worst_fold_multiple']:.3f}x "
            f"| maxDD={payload['metrics']['strategy_maximum_drawdown']:.2%}"
        )
        return payload
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "script_version": VERSION,
            "candidate_id": candidate_id,
            "candidate_kind": "caro",
            "status": "error",
            "elapsed_seconds": time.perf_counter() - started,
            "params": params,
            "settings_hash": settings_hash(params),
            "proposal": deepcopy(proposal),
            "error": f"{type(exc).__name__}: {exc}",
        }
        result_path.write_text(
            json.dumps(
                payload, indent=2, ensure_ascii=False, default=json_default
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"[{candidate_id}] ERRO | {payload['error']}")
        return payload


def load_completed_caro() -> list[dict[str, Any]]:
    if not DIR_OUT.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(DIR_OUT.glob("caro_*/result.json")):
        result = read_json_optional(path)
        if isinstance(result, dict) and result.get("status") == "completed":
            rows.append(
                {
                    "candidate_id": result["candidate_id"],
                    "kind": "caro",
                    "params": deepcopy(result["params"]),
                    "metrics": deepcopy(result["metrics"]),
                    "source": "caro",
                    "proposal": deepcopy(result.get("proposal")),
                }
            )
    return rows


def rebuild_state(
    warmup: list[dict[str, Any]],
    completed_caro: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    state = initial_state()
    observations = enrich_observations(warmup, baseline_metrics)
    anchor = best_feasible(observations)

    for item in completed_caro:
        enriched = enrich_observations([item], baseline_metrics)[0]
        previous_anchor = anchor
        promoted = bool(
            enriched["feasible"]
            and float(enriched["objective"]) > float(previous_anchor["objective"]) + 1e-9
        )
        observations.append(enriched)
        if promoted:
            anchor = enriched
        proposal_mode = str(
            ((item.get("proposal") or {}).get("proposal_mode"))
            or "adaptive_probability"
        )
        state = evolve_state(
            state,
            promoted=promoted,
            proposal_mode=proposal_mode,
            candidate_id=str(item["candidate_id"]),
        )
    return state


def write_outputs(
    observations: list[dict[str, Any]],
    baseline: dict[str, Any],
    state: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for item in observations:
        metrics = item["metrics"]
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "candidate_kind": item["kind"],
                "source": item["source"],
                "objective": item["objective"],
                "feasible": item["feasible"],
                "ending_capital": metrics["strategy_ending_capital"],
                "cagr": metrics["strategy_cagr"],
                "sharpe": metrics["strategy_sharpe"],
                "max_drawdown": metrics["strategy_maximum_drawdown"],
                "worst_fold_multiple": metrics["worst_fold_multiple"],
                "geometric_mean_fold_multiple": metrics[
                    "geometric_mean_fold_multiple"
                ],
                "fold_log_growth_std": metrics["fold_log_growth_std"],
                "all_folds_positive": metrics["all_folds_positive"],
                **item["params"],
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["feasible", "objective", "ending_capital"],
        ascending=[False, False, False],
    )
    df.to_csv(DIR_OUT / "leaderboard.csv", index=False)

    champion = best_feasible(observations)
    baseline_score = robust_objective(baseline["metrics"], baseline["metrics"])
    summary = {
        "schema_version": 1,
        "script_version": VERSION,
        "selection_uses_43m_reference": False,
        "historical_reference_capital_context_only": HISTORICAL_REFERENCE_CAPITAL,
        "baseline_candidate_id": baseline["candidate_id"],
        "baseline_objective": baseline_score,
        "completed_observations_total": len(observations),
        "completed_caro_trials": sum(
            1 for item in observations if item["source"] == "caro"
        ),
        "champion": {
            "candidate_id": champion["candidate_id"],
            "source": champion["source"],
            "objective": champion["objective"],
            "feasible": champion["feasible"],
            "params": champion["params"],
            "metrics": champion["metrics"],
        },
        "state": state,
    }
    (DIR_OUT / "campaign_summary.json").write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False, default=json_default
        )
        + "\n",
        encoding="utf-8",
    )
    (DIR_OUT / "best_caro.json").write_text(
        json.dumps(
            summary["champion"],
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    (DIR_OUT / "state.json").write_text(
        json.dumps(
            state, indent=2, ensure_ascii=False, default=json_default
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_campaign(
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
    existing = read_json_optional(path)
    if existing is not None:
        if existing.get("script_version") != VERSION:
            raise RuntimeError(
                "A pasta CARO existente pertence a outra versao do script"
            )
        if (
            (existing.get("dataset_signature") or {}).get("combined_sha256")
            != current_signature.get("combined_sha256")
        ):
            raise RuntimeError(
                "Dataset Tiingo mudou desde o inicio do CARO; campanha abortada"
            )
        return

    manifest = {
        "schema_version": 1,
        "script_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "tiingo_56_split_causal_lightgbm_caro",
        "warmup_source": SOURCE_LHS_DIR.relative_to(ROOT).as_posix(),
        "warmup_observation_count": int(warmup_count),
        "search_seed": SEARCH_SEED,
        "search_space": [dict(spec) for spec in SEARCH_SPACE],
        "objective_weights": OBJECTIVE_WEIGHTS,
        "historical_reference_capital_context_only": HISTORICAL_REFERENCE_CAPITAL,
        "selection_uses_43m_reference": False,
        "stopping_uses_43m_reference": False,
        "fixed_components": [
            "Tiingo RAW snapshot",
            "causal split normalization",
            "dividends audit-only (not applied)",
            "features",
            "targets",
            "walk-forward folds",
            "rotation policy",
            "fees and slippage",
        ],
        "tuned_component": "LightGBM hyperparameters only",
        "surrogate": "GaussianProcessRegressor + Matern 5/2",
        "acquisition": "constrained_expected_improvement_plus_exploration",
        "trust_region": {
            "initial": TRUST_REGION_INITIAL,
            "minimum": TRUST_REGION_MIN,
            "maximum": TRUST_REGION_MAX,
            "success_expansion": TRUST_REGION_SUCCESS_EXPANSION,
            "failure_contraction": TRUST_REGION_FAILURE_CONTRACTION,
            "failure_tolerance": TRUST_REGION_FAILURE_TOLERANCE,
        },
        "dataset_signature": current_signature,
    }
    path.write_text(
        json.dumps(
            manifest, indent=2, ensure_ascii=False, default=json_default
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CARO/Bayesiano para LightGBM sobre Tiingo 56 split-causal"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Orcamento total de candidatos CARO da campanha",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximo de novos candidatos nesta invocacao; retomada preservada",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Tamanho do pool de propostas avaliado pelo surrogate",
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=DEFAULT_EXPLORATION_WEIGHT,
        help="Peso da incerteza na funcao de aquisicao",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Parada opcional por N trials adaptativos sem novo champion; "
            "0 desabilita. Nunca usa 43M."
        ),
    )
    parser.add_argument(
        "--save-curves",
        action="store_true",
        help="Salvar equity_curve.csv de cada candidato CARO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials deve ser >= 1")
    if args.pool_size < 256:
        raise SystemExit("--pool-size deve ser >= 256")
    if args.exploration_weight < 0:
        raise SystemExit("--exploration-weight deve ser >= 0")

    for directory in (DIR_TIINGO, DIR_EVENTS, DIR_SPLITS):
        if not directory.exists():
            raise RuntimeError(f"Diretorio ausente: {directory}")

    log(f"Campanha: {VERSION}")
    log("CARO usa o LHS concluido como warm-up e propoe candidatos sequenciais")
    log("Somente hiperparametros LightGBM variam")
    log("Referencia de 43M: contexto externo apenas; NAO e objetivo nem parada")

    warmup, lhs_campaign = read_lhs_warmup()
    baseline = baseline_observation(warmup)
    baseline_metrics = baseline["metrics"]
    log(f"Warm-up LHS validado | observacoes={len(warmup)}")
    log(
        f"Control Tiingo | capital=US$ "
        f"{float(baseline_metrics['strategy_ending_capital']):,.2f} "
        f"| worst_fold={float(baseline_metrics['worst_fold_multiple']):.3f}x"
    )

    log("Validando assinatura SHA-256 do dataset congelado")
    signature = dataset_signature()
    ensure_campaign(lhs_campaign, signature, len(warmup))
    log(f"Dataset signature: {signature['combined_sha256']}")

    completed_caro = load_completed_caro()
    if len(completed_caro) >= args.trials:
        log(
            f"Orcamento ja atendido | CARO concluidos={len(completed_caro)}/"
            f"{args.trials}"
        )
        observations = enrich_observations(
            warmup + completed_caro, baseline_metrics
        )
        state = rebuild_state(warmup, completed_caro, baseline_metrics)
        write_outputs(observations, baseline, state)
        return 0

    state = rebuild_state(warmup, completed_caro, baseline_metrics)
    observations = enrich_observations(
        warmup + completed_caro, baseline_metrics
    )
    write_outputs(observations, baseline, state)

    remaining = int(args.trials) - len(completed_caro)
    to_run = remaining
    if args.limit is not None:
        to_run = min(to_run, max(0, int(args.limit)))
    if to_run <= 0:
        log("Nenhum novo candidato selecionado nesta invocacao")
        return 0

    log(
        f"CARO retomado | concluidos={len(completed_caro)}/{args.trials} "
        f"| novos nesta execucao={to_run}"
    )
    log("Carregando baseline Tiingo 56 split-causal uma unica vez")
    series = load_tiingo_split_causal()

    for local_position in range(to_run):
        trial_number = len(completed_caro) + 1
        candidate_id = f"caro_{trial_number:03d}"
        anchor_before = best_feasible(observations)
        log(
            f"Trial CARO {trial_number}/{args.trials} | "
            f"anchor={anchor_before['candidate_id']} "
            f"| score={float(anchor_before['objective']):.5f} "
            f"| trust_region={float(state['trust_region_radius']):.3f}"
        )

        proposal = choose_proposal(
            observations,
            baseline,
            state,
            pool_size=int(args.pool_size),
            exploration_weight=float(args.exploration_weight),
            trial_index=trial_number,
        )
        result = run_candidate(
            candidate_id,
            proposal,
            series,
            bool(args.save_curves),
        )
        if result.get("status") != "completed":
            log(
                "Candidato terminou com erro; campanha interrompida para "
                "preservar a sequencia adaptativa"
            )
            return 2

        new_item = {
            "candidate_id": result["candidate_id"],
            "kind": "caro",
            "params": deepcopy(result["params"]),
            "metrics": deepcopy(result["metrics"]),
            "source": "caro",
            "proposal": deepcopy(result.get("proposal")),
        }
        enriched = enrich_observations([new_item], baseline_metrics)[0]
        promoted = bool(
            enriched["feasible"]
            and float(enriched["objective"])
            > float(anchor_before["objective"]) + 1e-9
        )
        observations.append(enriched)
        completed_caro.append(new_item)
        state = evolve_state(
            state,
            promoted=promoted,
            proposal_mode=str(proposal["proposal_mode"]),
            candidate_id=candidate_id,
        )
        write_outputs(observations, baseline, state)

        champion = best_feasible(observations)
        log(
            f"[{candidate_id}] objective={float(enriched['objective']):.5f} "
            f"| feasible={bool(enriched['feasible'])} "
            f"| promoted={promoted}"
        )
        log(
            f"Champion CARO atual: {champion['candidate_id']} | "
            f"capital=US$ "
            f"{float(champion['metrics']['strategy_ending_capital']):,.2f} "
            f"| worst_fold="
            f"{float(champion['metrics']['worst_fold_multiple']):.3f}x "
            f"| score={float(champion['objective']):.5f}"
        )

        if (
            int(args.patience) > 0
            and int(state.get("adaptive_trials_completed", 0))
            >= int(args.patience)
            and int(state.get("no_improvement_streak", 0))
            >= int(args.patience)
        ):
            log(
                f"Parada por estagnacao configurada: {args.patience} "
                "trials sem novo champion"
            )
            break

    log(
        f"Campanha atualizada | CARO concluidos={len(completed_caro)}/"
        f"{args.trials}"
    )
    log(f"Resultados: {DIR_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
