"""Execucao comparativa Control vs Soft Horizon Consensus."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from engine.modelo_lightgbm import executar_lightgbm
from engine.rotacao import (
    _construir_folds_walk_forward,
    _desempenho_folds,
    preparar_painel_rotacao,
)
from engine.configuracao import (
    CONFIG,
    SOFT_HORIZON_CONSENSUS_PENALTY,
    StandaloneBacktestConfig,
    construir_configuracao_controle,
    construir_configuracao_soft,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia


def build_variant_configs(
    frames: dict[str, pd.DataFrame],
    base: StandaloneBacktestConfig = CONFIG,
) -> tuple[StandaloneBacktestConfig, StandaloneBacktestConfig]:
    eligible = tuple(frames)
    prepared = base.copiar_modelo(update={"assets": eligible})
    control = construir_configuracao_controle(prepared, assets=eligible)
    soft = construir_configuracao_soft(
        prepared,
        assets=eligible,
        penalty_strength=SOFT_HORIZON_CONSENSUS_PENALTY,
    )
    return control, soft


def build_folds(
    frames: dict[str, pd.DataFrame],
    config: StandaloneBacktestConfig,
) -> tuple[pd.DatetimeIndex, list[dict[str, Any]]]:
    _, common_dates, _ = preparar_painel_rotacao(frames, config)
    folds = _construir_folds_walk_forward(common_dates, config)
    return common_dates, folds


def summarize_metrics(
    result: Any,
    folds: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    fold_rows = deepcopy(
        result.metrics.get("walk_forward_folds") or []
    )
    if not fold_rows:
        fold_rows = _desempenho_folds(
            result.predictions,
            folds,
            initial_capital,
        )
    worst_fold = (
        min(float(row["strategy_return"]) for row in fold_rows)
        if fold_rows
        else None
    )
    simulation = deepcopy(result.metrics.get("simulation_profile") or {})
    output = {
        "ending_capital": float(result.metrics.get("strategy_ending_capital") or 0.0),
        "strategy_return": float(result.metrics.get("strategy_return") or 0.0),
        "cagr": float(result.metrics.get("strategy_cagr") or 0.0),
        "sharpe": float(result.metrics.get("strategy_sharpe") or 0.0),
        "maximum_drawdown": float(
            result.metrics.get("strategy_maximum_drawdown") or 0.0
        ),
        "worst_fold_return": worst_fold,
        "folds": fold_rows,
        "buy_hold_ending_capital": float(
            result.metrics.get("buy_hold_ending_capital") or 0.0
        ),
        "buy_hold_return": float(result.metrics.get("buy_hold_return") or 0.0),
        "buy_hold_cagr": float(result.metrics.get("buy_hold_cagr") or 0.0),
        "buy_hold_sharpe": float(result.metrics.get("buy_hold_sharpe") or 0.0),
        "buy_hold_maximum_drawdown": float(
            result.metrics.get("buy_hold_maximum_drawdown") or 0.0
        ),
        "benchmark_name": result.metrics.get("benchmark_name"),
        "calendar_source_asset": result.metrics.get("calendar_source_asset"),
        "requested_compute_device": result.metrics.get("requested_compute_device"),
        "effective_compute_device": result.metrics.get("effective_compute_device"),
        "predictive_diagnostics": deepcopy(
            result.metrics.get("lightgbm_predictive_diagnostics") or {}
        ),
        "simulation_profile": simulation,
        "simulation_session_count": simulation.get("session_count"),
        "switch_margin_calibration_details": deepcopy(
            result.metrics.get("switch_margin_calibration_details") or []
        ),
    }
    for key, value in result.metrics.items():
        if (
            str(key).startswith("soft_horizon_consensus_")
            or str(key).startswith("oos_inference_cache_")
        ):
            output[str(key)] = value
    return output


def run_variant(
    label: str,
    frames: dict[str, pd.DataFrame],
    config: StandaloneBacktestConfig,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(f"[final] starting {label}", flush=True)
    results = executar_lightgbm(
        frames,
        config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        progress_callback=lambda p, stage, completed: print(
            f"[final] {label} progress={p:.1f}% "
            f"completed={completed} stage={stage}",
            flush=True,
        ),
        technical_log_callback=lambda message: print(
            f"[technical] {label} {message}",
            flush=True,
        ),
    )
    if not results:
        raise RuntimeError(f"{label} returned no result.")
    result = results[0]
    metrics = summarize_metrics(result, folds, float(config.initial_capital))
    worst_fold = metrics["worst_fold_return"]
    worst_fold_text = (
        f"{float(worst_fold):.4%}" if worst_fold is not None else "n/a"
    )
    print(
        f"[final] completed {label} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"buy_hold={metrics['buy_hold_ending_capital']:,.2f} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={worst_fold_text}",
        flush=True,
    )
    return result, metrics


def compare_variants(
    control_metrics: dict[str, Any],
    soft_metrics: dict[str, Any],
) -> dict[str, Any]:
    control_capital = float(control_metrics["ending_capital"])
    soft_capital = float(soft_metrics["ending_capital"])
    delta = soft_capital - control_capital
    return {
        "control_ending_capital": control_capital,
        "soft_ending_capital": soft_capital,
        "soft_minus_control_capital": delta,
        "soft_vs_control_ratio": (
            soft_capital / control_capital - 1.0
            if control_capital > 0
            else None
        ),
        "control_cagr": float(control_metrics["cagr"]),
        "soft_cagr": float(soft_metrics["cagr"]),
        "control_sharpe": float(control_metrics["sharpe"]),
        "soft_sharpe": float(soft_metrics["sharpe"]),
        "control_maximum_drawdown": float(control_metrics["maximum_drawdown"]),
        "soft_maximum_drawdown": float(soft_metrics["maximum_drawdown"]),
        "control_worst_fold_return": control_metrics["worst_fold_return"],
        "soft_worst_fold_return": soft_metrics["worst_fold_return"],
        "soft_changed_base_actions": int(
            soft_metrics.get("soft_horizon_consensus_changed_base_actions") or 0
        ),
        "requested_compute_device": soft_metrics.get("requested_compute_device"),
        "effective_compute_device": soft_metrics.get("effective_compute_device"),
    }
