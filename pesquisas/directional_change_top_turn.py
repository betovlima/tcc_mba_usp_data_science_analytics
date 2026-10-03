"""Directional Change Top-Turn + LightGBM challenger v1.3.0-dev.2.

A v1.3.0-dev.1 mostrou que um alvo generico de drawdown em cinco sessoes gera
muitos falsos positivos. Esta versao preserva as features Directional Change,
mas reformula o alvo para um primeiro evento de virada perto de uma maxima e
usa uma calibracao orientada a precisao, com confirmacao temporal antes da
saida.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _configuracoes_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    _crescimento_politica_simples,
    _desempenho_folds,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
)
from pesquisas.directional_change_lightgbm import (
    DIRECTIONAL_CHANGE_THRESHOLDS,
    MODEL_FEATURES,
    _directional_change_state,
    _tag_threshold,
)

RESEARCH_VERSION = "1.3.0-dev.2"
TOP_TURN_HORIZON_SESSIONS = 5
TOP_TURN_ATR_MULTIPLIER = 1.5
TOP_TURN_THRESHOLD_MIN = 0.02
TOP_TURN_THRESHOLD_MAX = 0.08
TOP_TURN_CONTINUATION_RATIO = 0.50
TOP_TURN_NEAR_HIGH_20 = 0.05
TOP_TURN_MIN_UP_REGIME_SHARE = 2.0 / 3.0
TOP_TURN_MIN_RETURN_20 = 0.0
TOP_TURN_CONFIRMATION_SESSIONS = 2
PROBABILITY_THRESHOLD_CANDIDATES = (
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)
MINIMUM_CLASS_ROWS = 20
MINIMUM_CALIBRATION_ALERTS = 5


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    fbeta_05: float | None
    balanced_accuracy: float | None
    precision: float | None
    recall: float | None
    predicted_positives: int
    positives: int
    negatives: int
    observations: int

    def as_dict(self, *, fold_id: int) -> dict[str, Any]:
        return {
            "fold_id": int(fold_id),
            "probability_threshold": float(self.threshold),
            "fbeta_05": self.fbeta_05,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "predicted_positives": int(self.predicted_positives),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "observations": int(self.observations),
        }


def _top_turn_targets(
    frame: pd.DataFrame,
    *,
    horizon: int = TOP_TURN_HORIZON_SESSIONS,
) -> pd.DataFrame:
    """Primeiro evento: queda relevante antes de uma continuacao da alta."""
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    atr_pct = pd.to_numeric(frame["atr_pct_14"], errors="coerce").to_numpy(
        dtype=float
    )
    target = np.full(len(frame), np.nan, dtype=float)
    threshold_values = np.full(len(frame), np.nan, dtype=float)
    continuation_values = np.full(len(frame), np.nan, dtype=float)
    event_step = np.full(len(frame), np.nan, dtype=float)

    for index in range(len(frame)):
        end = index + 1 + int(horizon)
        current = close[index]
        current_atr = atr_pct[index]
        if (
            end > len(frame)
            or not np.isfinite(current)
            or current <= 0
            or not np.isfinite(current_atr)
            or current_atr <= 0
        ):
            continue

        future = close[index + 1:end]
        if len(future) != int(horizon) or not np.isfinite(future).all():
            continue

        threshold = float(
            np.clip(
                float(TOP_TURN_ATR_MULTIPLIER) * float(current_atr),
                float(TOP_TURN_THRESHOLD_MIN),
                float(TOP_TURN_THRESHOLD_MAX),
            )
        )
        continuation = float(
            max(
                0.01,
                float(TOP_TURN_CONTINUATION_RATIO) * threshold,
            )
        )
        threshold_values[index] = threshold
        continuation_values[index] = continuation

        label = 0.0
        for step, future_close in enumerate(future, start=1):
            move = float(future_close / current - 1.0)
            if move <= -threshold:
                label = 1.0
                event_step[index] = float(step)
                break
            if move >= continuation:
                label = 0.0
                event_step[index] = float(step)
                break
        target[index] = label

    return pd.DataFrame(
        {
            "forward_down_reversal": target,
            "forward_top_turn_threshold": threshold_values,
            "forward_top_turn_continuation": continuation_values,
            "forward_top_turn_event_step": event_step,
        },
        index=frame.index,
    )


def adicionar_top_turn_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    close = pd.to_numeric(output["close"], errors="coerce")
    regime_columns: list[str] = []
    pressure_columns: list[str] = []

    for threshold in DIRECTIONAL_CHANGE_THRESHOLDS:
        dc = _directional_change_state(close, float(threshold))
        output = output.join(dc)
        tag = _tag_threshold(float(threshold))
        regime_columns.append(f"dc_regime_{tag}")
        pressure_columns.append(f"dc_reversal_pressure_{tag}")

    regimes = output[regime_columns].apply(pd.to_numeric, errors="coerce")
    output["dc_up_regime_share"] = (regimes > 0).mean(axis=1)
    output["dc_down_regime_share"] = (regimes < 0).mean(axis=1)
    output["dc_regime_agreement"] = regimes.mean(axis=1).abs()
    output["dc_reversal_pressure_max"] = output[pressure_columns].max(axis=1)

    distance_high = pd.to_numeric(
        output["distance_from_high_20"],
        errors="coerce",
    )
    return_20 = pd.to_numeric(output["return_20"], errors="coerce")
    output["top_turn_eligible"] = (
        (output["dc_up_regime_share"] >= TOP_TURN_MIN_UP_REGIME_SHARE)
        & (distance_high >= -TOP_TURN_NEAR_HIGH_20)
        & (return_20 > TOP_TURN_MIN_RETURN_20)
    )

    output = output.join(_top_turn_targets(output))
    return output.replace([np.inf, -np.inf], np.nan)


def _preparar_frames(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        symbol: adicionar_top_turn_features(frame)
        for symbol, frame in frames.items()
    }


def _ajustar_modelos_top_turn(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("Top-Turn research requires lightgbm.") from exc

    settings = _configuracoes_lightgbm(config)
    minimum_rows = int(config.rotation_minimum_training_rows)
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        frame = frames[symbol].reindex(train_dates)
        eligible = frame["top_turn_eligible"].fillna(False).astype(bool)
        train = frame.loc[eligible].dropna(
            subset=["forward_down_reversal", *MODEL_FEATURES]
        )
        target = train["forward_down_reversal"].astype(int)
        positives = int((target == 1).sum())
        negatives = int((target == 0).sum())
        diagnostic = {
            "asset": symbol,
            "eligible_training_rows": int(len(train)),
            "positive_rows": positives,
            "negative_rows": negatives,
            "fitted": False,
        }
        if (
            len(train) < max(100, minimum_rows // 4)
            or positives < MINIMUM_CLASS_ROWS
            or negatives < MINIMUM_CLASS_ROWS
        ):
            rows.append(diagnostic)
            continue

        model = LGBMClassifier(
            objective="binary",
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
            class_weight="balanced",
            verbosity=-1,
        )
        model.fit(train[list(MODEL_FEATURES)], target)
        models[symbol] = model
        diagnostic["fitted"] = True
        rows.append(diagnostic)

    return models, rows


def _probabilidades(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, dict[str, float]]:
    cache: dict[pd.Timestamp, dict[str, float]] = {
        pd.Timestamp(date): {} for date in dates
    }
    for symbol in symbols:
        model = models.get(symbol)
        if model is None:
            continue
        frame = frames[symbol].reindex(dates)
        eligible = frame["top_turn_eligible"].fillna(False).astype(bool)
        valid = eligible & frame[list(MODEL_FEATURES)].notna().all(axis=1)
        if not bool(valid.any()):
            continue
        rows = frame.loc[valid, list(MODEL_FEATURES)]
        predicted = model.predict_proba(rows)[:, 1]
        for timestamp, probability in zip(
            rows.index,
            predicted,
            strict=True,
        ):
            cache[pd.Timestamp(timestamp)][symbol] = float(probability)
    return cache


def _metricas_classificacao(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    predicted = probabilities >= float(threshold)
    positive = y_true == 1
    negative = y_true == 0
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum((~predicted) & positive))
    tn = int(np.sum((~predicted) & negative))
    fp = int(np.sum(predicted & negative))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else None
    beta2 = 0.25
    if precision is None or (precision == 0.0 and recall == 0.0):
        fbeta = 0.0
    else:
        fbeta = (
            (1.0 + beta2) * float(precision) * float(recall)
            / (beta2 * float(precision) + float(recall))
        )
    return {
        "fbeta_05": float(fbeta),
        "balanced_accuracy": 0.5 * (float(recall) + float(specificity)),
        "precision": precision,
        "recall": float(recall),
        "predicted_positives": int(np.sum(predicted)),
    }


def calibrar_limiar(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    *,
    candidates: Iterable[float] = PROBABILITY_THRESHOLD_CANDIDATES,
) -> CalibrationResult:
    cache = _probabilidades(models, frames, symbols, calibration_dates)
    targets: list[int] = []
    probabilities: list[float] = []

    for timestamp in calibration_dates:
        key = pd.Timestamp(timestamp)
        for symbol in symbols:
            probability = cache.get(key, {}).get(symbol)
            if probability is None:
                continue
            frame = frames[symbol]
            if key not in frame.index:
                continue
            value = frame.at[key, "forward_down_reversal"]
            if pd.isna(value):
                continue
            targets.append(int(value))
            probabilities.append(float(probability))

    if not targets:
        return CalibrationResult(
            threshold=0.75,
            fbeta_05=None,
            balanced_accuracy=None,
            precision=None,
            recall=None,
            predicted_positives=0,
            positives=0,
            negatives=0,
            observations=0,
        )

    y_true = np.asarray(targets, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    best: tuple[float, float, dict[str, float | int | None]] | None = None

    for candidate in candidates:
        metrics = _metricas_classificacao(
            y_true,
            probs,
            float(candidate),
        )
        if int(metrics["predicted_positives"] or 0) < MINIMUM_CALIBRATION_ALERTS:
            continue
        score = float(metrics["fbeta_05"] or 0.0)
        row = (score, float(candidate), metrics)
        if best is None or (row[0], row[1]) > (best[0], best[1]):
            best = row

    if best is None:
        threshold = 0.75
        metrics = _metricas_classificacao(y_true, probs, threshold)
    else:
        threshold = float(best[1])
        metrics = best[2]

    return CalibrationResult(
        threshold=threshold,
        fbeta_05=float(metrics["fbeta_05"] or 0.0),
        balanced_accuracy=float(metrics["balanced_accuracy"] or 0.0),
        precision=(
            float(metrics["precision"])
            if metrics["precision"] is not None
            else None
        ),
        recall=float(metrics["recall"] or 0.0),
        predicted_positives=int(metrics["predicted_positives"] or 0),
        positives=int((y_true == 1).sum()),
        negatives=int((y_true == 0).sum()),
        observations=int(len(y_true)),
    )


def _envolver_politica_top_turn(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    probabilities: dict[pd.Timestamp, dict[str, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    probability_threshold: float,
    config: Any,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]],
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    streak_by_symbol: dict[str, int] = {}

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        base_target, base_score = base_policy(
            timestamp,
            current_position,
            holding_days,
        )
        key = pd.Timestamp(timestamp)
        diagnostic = decision_diagnostics.setdefault(key, {})
        symbol = symbols[current_position - 1] if current_position > 0 else None

        probability = None
        eligible = False
        if symbol is not None:
            probability = probabilities.get(key, {}).get(symbol)
            if key in frames[symbol].index:
                eligible = bool(
                    frames[symbol].at[key, "top_turn_eligible"]
                )

        control_holds = bool(
            current_position > 0
            and int(base_target) == int(current_position)
            and holding_days >= int(config.rotation_min_holding_days)
        )
        above_threshold = bool(
            control_holds
            and eligible
            and probability is not None
            and np.isfinite(float(probability))
            and float(probability) >= float(probability_threshold)
        )

        if symbol is not None:
            if above_threshold:
                streak_by_symbol[symbol] = (
                    int(streak_by_symbol.get(symbol, 0)) + 1
                )
            else:
                streak_by_symbol[symbol] = 0
        streak = int(streak_by_symbol.get(symbol, 0)) if symbol else 0
        triggered = bool(
            above_threshold
            and streak >= int(TOP_TURN_CONFIRMATION_SESSIONS)
        )

        diagnostic.update(
            {
                "directional_change_schema_version": 2,
                "directional_change_research_version": RESEARCH_VERSION,
                "directional_change_probability": (
                    float(probability) if probability is not None else None
                ),
                "directional_change_probability_threshold": float(
                    probability_threshold
                ),
                "directional_change_top_turn_eligible": bool(eligible),
                "directional_change_confirmation_streak": int(streak),
                "directional_change_confirmation_required": int(
                    TOP_TURN_CONFIRMATION_SESSIONS
                ),
                "directional_change_base_target_asset": (
                    "CASH"
                    if int(base_target) <= 0
                    else symbols[int(base_target) - 1]
                ),
                "directional_change_exit_triggered": triggered,
            }
        )

        if not triggered:
            if not control_holds and symbol is not None:
                streak_by_symbol[symbol] = 0
            return int(base_target), float(base_score)

        streak_by_symbol[symbol] = 0
        diagnostic["directional_change_base_reason"] = diagnostic.get(
            "decision_reason"
        )
        diagnostic.update(
            {
                "decision_reason": "DIRECTIONAL_CHANGE_TOP_TURN_EXIT",
                "final_action_asset": "CASH",
                "final_action_score": 0.0,
                "decision_is_rotation": False,
                "decision_is_entry": False,
                "decision_is_exit_to_cash": True,
            }
        )
        return 0, 0.0

    return policy


def executar_directional_change_top_turn(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> Any:
    if int(config.rotation_model_repetitions) != 1:
        raise ValueError("Top-Turn v1.3.0-dev.2 requires one repetition.")

    (
        frames,
        common_dates,
        calendar_source_asset,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _construir_contexto_execucao(bars_by_symbol, config)
    frames = _preparar_frames(frames)

    policies: dict[int, Callable] = {}
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
    calibration_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []

    total_folds = len(folds)
    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        if progress_callback is not None:
            progress_callback(
                5.0 + 75.0 * ((fold_position - 1) / max(1, total_folds)),
                f"Top-Turn fold {fold_position}/{total_folds} training",
                fold_position - 1,
            )

        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):
            int(fold["calibration_end_index"])
        ]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]

        calibration_utility_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config,
            phase=f"top_turn_fold_{fold_id}_utility_calibration",
        )
        candidate_margins = tuple(
            float(value)
            for value in config.rotation_switch_margin_candidates
        )
        best_candidate = candidate_margins[0]
        best_score = float("-inf")
        for candidate in candidate_margins:
            calibration_policy = _politica_utilidade(
                calibration_utility_models,
                frames,
                symbols,
                config,
                candidate,
            )
            score = _crescimento_politica_simples(
                calibration_policy,
                frames,
                symbols,
                calibration_dates,
                config,
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        reversal_calibration_models, calibration_fit = (
            _ajustar_modelos_top_turn(
                frames,
                symbols,
                train_dates,
                config,
            )
        )
        calibration = calibrar_limiar(
            reversal_calibration_models,
            frames,
            symbols,
            calibration_dates,
        )
        calibration_rows.append(calibration.as_dict(fold_id=fold_id))
        for row in calibration_fit:
            fit_rows.append(
                {"fold_id": fold_id, "phase": "calibration", **row}
            )

        final_utility_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"top_turn_fold_{fold_id}_utility_final",
        )
        final_reversal_models, final_fit = _ajustar_modelos_top_turn(
            frames,
            symbols,
            final_fit_dates,
            config,
        )
        for row in final_fit:
            fit_rows.append({"fold_id": fold_id, "phase": "final", **row})

        decision_dates = pd.DatetimeIndex(fold["decision_dates"])
        utility_cache, _ = _precalcular_utilidades_modelo(
            final_utility_models,
            frames,
            symbols,
            decision_dates,
            config,
        )
        probability_cache = _probabilidades(
            final_reversal_models,
            frames,
            symbols,
            decision_dates,
        )

        effective_margin = max(
            float(config.rotation_switch_margin),
            float(best_candidate),
        )
        base_policy = _politica_utilidade(
            final_utility_models,
            frames,
            symbols,
            config,
            effective_margin,
            decision_diagnostics=diagnostics,
            fold_id=fold_id,
            calibrated_switch_margin=float(best_candidate),
            utility_cache=utility_cache,
        )
        policies[fold_id] = _envolver_politica_top_turn(
            base_policy,
            probabilities=probability_cache,
            frames=frames,
            symbols=symbols,
            probability_threshold=float(calibration.threshold),
            config=config,
            decision_diagnostics=diagnostics,
        )
        margin_rows.append(
            {
                "fold_id": fold_id,
                "calibrated_candidate_margin": float(best_candidate),
                "effective_switch_margin": float(effective_margin),
                "calibration_risk_adjusted_score": float(best_score),
            }
        )

    if progress_callback is not None:
        progress_callback(
            85.0,
            "Directional Change Top-Turn OOS replay",
            total_folds,
        )

    scheduled = _politica_agendada(policies, decision_to_fold)
    result = _simular_exato(
        "directional_change_top_turn_lightgbm",
        scheduled,
        frames,
        symbols,
        all_decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Directional Change Top-Turn + LightGBM",
        method_line=(
            "- Directional Change Top-Turn + LightGBM estimates a "
            "precision-oriented first-passage probability of a downward "
            "turn while the held asset is in an up regime near a recent high; "
            "two consecutive confirmations are required before exiting to "
            "CASH at the next open."
        ),
    )
    result.backend = "directional_change_top_turn_lightgbm"

    probabilities = pd.to_numeric(
        result.predictions.get(
            "directional_change_probability",
            pd.Series(index=result.predictions.index, dtype=float),
        ),
        errors="coerce",
    )
    triggers = result.predictions.get(
        "directional_change_exit_triggered",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)

    result.metrics.update(
        {
            "backend": "directional_change_top_turn_lightgbm",
            "model_family": "directional_change_top_turn_lightgbm",
            "strategy_label": "Directional Change Top-Turn + LightGBM",
            "directional_change_research_version": RESEARCH_VERSION,
            "directional_change_thresholds": list(
                DIRECTIONAL_CHANGE_THRESHOLDS
            ),
            "directional_change_top_turn_horizon_sessions": int(
                TOP_TURN_HORIZON_SESSIONS
            ),
            "directional_change_top_turn_near_high_20": float(
                TOP_TURN_NEAR_HIGH_20
            ),
            "directional_change_confirmation_sessions": int(
                TOP_TURN_CONFIRMATION_SESSIONS
            ),
            "directional_change_probability_threshold_mean": float(
                np.mean(
                    [
                        row["probability_threshold"]
                        for row in calibration_rows
                    ]
                )
            ),
            "directional_change_exit_triggers": int(triggers.sum()),
            "directional_change_probability_observations": int(
                probabilities.notna().sum()
            ),
            "directional_change_calibration": calibration_rows,
            "directional_change_fit": fit_rows,
            "walk_forward_fold_count": len(folds),
            "walk_forward_folds": _desempenho_folds(
                result.predictions,
                folds,
                float(config.initial_capital),
            ),
            "calendar_source_asset": calendar_source_asset,
            "requested_compute_device": "cpu",
            "effective_compute_device": "cpu",
        }
    )

    margin_by_fold = {int(row["fold_id"]): row for row in margin_rows}
    for row in result.metrics["walk_forward_folds"]:
        row.update(margin_by_fold.get(int(row["fold_id"]), {}))

    if progress_callback is not None:
        progress_callback(
            100.0,
            "Directional Change Top-Turn + LightGBM completed",
            total_folds,
        )
    return result


def comparar_control_top_turn(
    control_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    control_peak: dict[str, Any],
    challenger_peak: dict[str, Any],
) -> dict[str, Any]:
    control_capital = float(control_metrics["ending_capital"])
    challenger_capital = float(challenger_metrics["ending_capital"])
    control_distance = control_peak.get(
        "median_exit_distance_from_peak_pct"
    )
    challenger_distance = challenger_peak.get(
        "median_exit_distance_from_peak_pct"
    )
    return {
        "research_version": RESEARCH_VERSION,
        "control_ending_capital": control_capital,
        "top_turn_ending_capital": challenger_capital,
        "top_turn_minus_control_capital": challenger_capital - control_capital,
        "top_turn_vs_control_ratio": (
            challenger_capital / control_capital - 1.0
            if control_capital > 0
            else None
        ),
        "control_cagr": control_metrics.get("cagr"),
        "top_turn_cagr": challenger_metrics.get("cagr"),
        "control_sharpe": control_metrics.get("sharpe"),
        "top_turn_sharpe": challenger_metrics.get("sharpe"),
        "control_maximum_drawdown": control_metrics.get("maximum_drawdown"),
        "top_turn_maximum_drawdown": challenger_metrics.get(
            "maximum_drawdown"
        ),
        "control_worst_fold_return": control_metrics.get("worst_fold_return"),
        "top_turn_worst_fold_return": challenger_metrics.get(
            "worst_fold_return"
        ),
        "control_median_exit_distance_from_peak_pct": control_distance,
        "top_turn_median_exit_distance_from_peak_pct": challenger_distance,
        "median_exit_distance_improvement_pct_points": (
            float(control_distance) - float(challenger_distance)
            if control_distance is not None
            and challenger_distance is not None
            else None
        ),
        "control_median_peak_capture_pct": control_peak.get(
            "median_peak_capture_pct"
        ),
        "top_turn_median_peak_capture_pct": challenger_peak.get(
            "median_peak_capture_pct"
        ),
        "control_median_days_from_peak_to_exit": control_peak.get(
            "median_days_from_peak_to_exit"
        ),
        "top_turn_median_days_from_peak_to_exit": challenger_peak.get(
            "median_days_from_peak_to_exit"
        ),
        "top_turn_exit_triggers": int(
            challenger_metrics.get("directional_change_exit_triggers") or 0
        ),
    }
