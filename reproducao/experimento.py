"""Apoio enxuto para a reproducao oficial do U67 Control."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from engine.configuracao import (
    CONFIG,
    StandaloneBacktestConfig,
    construir_configuracao_controle,
)
from engine.rotacao import _desempenho_folds


def build_control_config(
    frames: dict[str, pd.DataFrame],
    base: StandaloneBacktestConfig = CONFIG,
) -> StandaloneBacktestConfig:
    """Cria a configuracao Control somente com os ativos elegiveis."""
    eligible = tuple(frames)
    prepared = base.copiar_modelo(update={"assets": eligible})
    return construir_configuracao_controle(
        prepared,
        assets=eligible,
    )


def summarize_metrics(
    result: Any,
    folds: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    """Extrai as metricas usadas no artefato final da reproducao."""
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
    return {
        "ending_capital": float(
            result.metrics.get("strategy_ending_capital") or 0.0
        ),
        "strategy_return": float(
            result.metrics.get("strategy_return") or 0.0
        ),
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
        "buy_hold_return": float(
            result.metrics.get("buy_hold_return") or 0.0
        ),
        "buy_hold_cagr": float(
            result.metrics.get("buy_hold_cagr") or 0.0
        ),
        "buy_hold_sharpe": float(
            result.metrics.get("buy_hold_sharpe") or 0.0
        ),
        "buy_hold_maximum_drawdown": float(
            result.metrics.get("buy_hold_maximum_drawdown") or 0.0
        ),
        "benchmark_name": result.metrics.get("benchmark_name"),
        "benchmark_assets": list(result.metrics.get("benchmark_assets") or []),
        "benchmark_asset_count": result.metrics.get("benchmark_asset_count"),
        "benchmark_same_universe": result.metrics.get("benchmark_same_universe"),
        "calendar_source_asset": result.metrics.get(
            "calendar_source_asset"
        ),
        "requested_compute_device": result.metrics.get(
            "requested_compute_device"
        ),
        "effective_compute_device": result.metrics.get(
            "effective_compute_device"
        ),
        "predictive_diagnostics": deepcopy(
            result.metrics.get("lightgbm_predictive_diagnostics") or {}
        ),
        "simulation_profile": simulation,
        "simulation_session_count": simulation.get("session_count"),
    }

