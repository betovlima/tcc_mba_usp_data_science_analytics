from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .rotacao import (
    ROTATION_FEATURES,
    SUPPORTED_ROTATION_MODES,
    _construir_folds_walk_forward,
    _datas_decisao_analise,
    preparar_painel_rotacao,
)

def _configuracoes_pesquisa(config: Any) -> dict[str, Any]:
    raw = getattr(config, "research_model_settings", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}

def _configuracoes_lightgbm(config: Any) -> dict[str, Any]:
    settings = _configuracoes_pesquisa(config)
    lightgbm = settings.get("lightgbm")
    if not isinstance(lightgbm, dict):
        raise ValueError("The LightGBM execution snapshot is missing its protected model settings.")
    required = {
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "min_child_weight", "subsample", "subsample_freq",
        "colsample_bytree", "reg_alpha", "reg_lambda", "max_bin", "n_jobs",
    }
    missing = sorted(required.difference(lightgbm))
    if missing:
        raise ValueError("LightGBM research settings are incomplete: " + ", ".join(missing))
    resolved = dict(lightgbm)
    # v10.8.68: RMSE-based early stopping is deliberately disabled for the
    # economic ranking model. Predictive MAE/RMSE remain diagnostics only.
    resolved["early_stopping_enabled"] = False
    return resolved


def selecionar_switch_margin(
    candidate_scores: list[tuple[float, float]],
) -> dict[str, float]:
    """Seleciona a margem com maior score de calibracao."""
    if not candidate_scores:
        raise ValueError("Switch-margin calibration produced no candidates.")

    selected_margin = float(candidate_scores[0][0])
    selected_score = float(candidate_scores[0][1])
    for candidate, score in candidate_scores[1:]:
        candidate = float(candidate)
        score = float(score)
        if score > selected_score:
            selected_margin = candidate
            selected_score = score

    return {
        "selected_candidate_margin": selected_margin,
        "selected_calibration_score": selected_score,
    }


def _construir_contexto_execucao(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    calendar_override: pd.DatetimeIndex | None = None,
    calendar_source_label: str | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DatetimeIndex,
    str,
    list[str],
    list[dict[str, Any]],
    pd.DatetimeIndex,
    dict[pd.Timestamp, int],
    dict[pd.Timestamp, dict[str, Any]],
]:
    if config.strategy_mode not in SUPPORTED_ROTATION_MODES:
        raise ValueError(f"Unsupported research strategy mode: {config.strategy_mode}.")
    frames, common_dates, calendar_source_asset = preparar_painel_rotacao(
        bars_by_symbol,
        config,
        calendar_override=calendar_override,
        calendar_source_label=calendar_source_label,
    )
    symbols = sorted(frames)
    folds = _construir_folds_walk_forward(common_dates, config)
    all_decision_dates = _datas_decisao_analise(common_dates, folds, config)
    decision_to_fold: dict[pd.Timestamp, int] = {}
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] = {}
    for fold in folds:
        for timestamp in fold["decision_dates"][:-1]:
            key = pd.Timestamp(timestamp)
            decision_to_fold[key] = int(fold["fold_id"])
            decision_metadata[key] = {
                "fold_id": int(fold["fold_id"]),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
            }
    return (
        frames,
        common_dates,
        calendar_source_asset,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    )

def _diagnosticos_erro_regressao(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | None]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not bool(valid.any()):
        return {
            "rows": 0,
            "absolute_error_sum": 0.0,
            "squared_error_sum": 0.0,
            "mae": None,
            "rmse": None,
        }
    residual = predicted_values[valid] - actual_values[valid]
    abs_sum = float(np.abs(residual).sum())
    sq_sum = float(np.square(residual).sum())
    rows = int(valid.sum())
    return {
        "rows": rows,
        "absolute_error_sum": abs_sum,
        "squared_error_sum": sq_sum,
        "mae": abs_sum / rows,
        "rmse": math.sqrt(sq_sum / rows),
    }

def _diagnosticos_ajuste_modelo_lightgbm(
    model: Any,
    train_frame: pd.DataFrame,
    *,
    target_column: str,
    configured_estimators: int,
) -> dict[str, Any]:
    train_prediction = model.predict(train_frame[ROTATION_FEATURES])
    train_diag = _diagnosticos_erro_regressao(
        train_frame[target_column].to_numpy(dtype=np.float64),
        np.asarray(train_prediction, dtype=np.float64),
    )

    gain = np.asarray(
        model.booster_.feature_importance(importance_type="gain"),
        dtype=np.float64,
    )
    gain = np.where(np.isfinite(gain), gain, 0.0)
    gain_total = float(gain.sum())
    importance = {
        feature: (float(value / gain_total) if gain_total > 0 else 0.0)
        for feature, value in zip(ROTATION_FEATURES, gain)
    }

    return {
        "configured_estimators": int(configured_estimators),
        "best_iteration": int(configured_estimators),
        "early_stopping_used": False,
        "train": train_diag,
        "feature_importance_gain": importance,
    }

def _ajustar_modelos_lightgbm(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
    *,
    phase: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
    target_column: str = "forward_risk_adjusted_utility",
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM research requires lightgbm. Install requirements.txt."
        ) from exc

    minimum_rows = int(config.rotation_minimum_training_rows)
    settings = _configuracoes_lightgbm(config)
    active_device = "cpu"
    fitted: dict[str, Any] = {}
    started = time.perf_counter()

    def technical(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    technical(
        f"model=lightgbm phase={phase} event=fit_start device=cpu "
        f"models={len(symbols)} train_sessions={len(train_dates)} "
        f"estimators={int(settings['n_estimators'])} "
        f"seed={int(config.random_state)} early_stopping=false"
    )

    for position, symbol in enumerate(symbols, start=1):
        frame = frames[symbol].loc[train_dates].dropna(
            subset=[target_column, *ROTATION_FEATURES]
        )
        if len(frame) < minimum_rows:
            if progress_callback is not None:
                progress_callback(position, len(symbols), active_device)
            continue

        model = LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            num_leaves=int(settings["num_leaves"]),
            min_child_samples=int(settings["min_child_samples"]),
            min_child_weight=float(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            subsample_freq=int(settings["subsample_freq"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            max_bin=int(settings["max_bin"]),
            random_state=int(config.random_state),
            n_jobs=int(settings["n_jobs"]),
            device_type="cpu",
            deterministic=bool(config.deterministic_execution),
            force_col_wise=bool(config.deterministic_execution),
            verbosity=-1,
        )
        model.fit(
            frame[ROTATION_FEATURES],
            frame[target_column],
        )
        model._fit_diagnostics = _diagnosticos_ajuste_modelo_lightgbm(
            model,
            frame,
            target_column=target_column,
            configured_estimators=int(settings["n_estimators"]),
        )
        fitted[symbol] = model
        if progress_callback is not None:
            progress_callback(position, len(symbols), active_device)

    technical(
        f"model=lightgbm phase={phase} event=fit_complete device=cpu "
        f"models={len(fitted)} "
        f"duration_seconds={time.perf_counter() - started:.3f}"
    )
    return fitted

