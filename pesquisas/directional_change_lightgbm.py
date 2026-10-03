"""Pesquisa Directional Change + LightGBM.

Este modulo e o unico ponto de evolucao desta linha de pesquisa. O historico
fica no Git; novas tentativas substituem a implementacao corrente em vez de
criar novos arquivos versionados.

Versao atual: Top-Turn. O classificador estima uma virada de alta para baixa
perto de maxima recente, com calibracao orientada a precision e duas
confirmacoes antes de antecipar a saida do Control.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable
import zipfile

import numpy as np
import pandas as pd

from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _configuracoes_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    ROTATION_FEATURES,
    _crescimento_politica_simples,
    _desempenho_folds,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
)

RESEARCH_VERSION = "1.4.0-dev.1"
DIRECTIONAL_CHANGE_THRESHOLDS = (0.02, 0.04, 0.08)
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

BOCPD_HAZARD_LAMBDAS = (20, 60, 120)
BOCPD_SCORE_THRESHOLDS = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.12,
    0.18,
    0.25,
)
BOCPD_MAX_RUN_LENGTH = 252
BOCPD_SHORT_RUN_MAX = 3
BOCPD_PRIOR_MEAN = 0.0
BOCPD_PRIOR_KAPPA = 1.0
BOCPD_NEAR_HIGH_20 = 0.05
BOCPD_MIN_RETURN_20 = 0.0
BOCPD_CONFIRMATION_SESSIONS = 2

def _tag_threshold(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}pct"

def directional_change_feature_names(
    thresholds: Iterable[float] = DIRECTIONAL_CHANGE_THRESHOLDS,
) -> tuple[str, ...]:
    names: list[str] = []
    for threshold in thresholds:
        tag = _tag_threshold(threshold)
        names.extend(
            [
                f"dc_regime_{tag}",
                f"dc_distance_from_extreme_{tag}",
                f"dc_overshoot_{tag}",
                f"dc_sessions_since_event_{tag}",
                f"dc_reversal_pressure_{tag}",
            ]
        )
    names.extend(
        [
            "dc_up_regime_share",
            "dc_down_regime_share",
            "dc_regime_agreement",
            "dc_reversal_pressure_max",
        ]
    )
    return tuple(names)

DC_FEATURES = directional_change_feature_names()
MODEL_FEATURES = tuple(ROTATION_FEATURES) + DC_FEATURES

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

@dataclass(frozen=True)
class BOCPDCalibrationResult:
    hazard_lambda: int
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
            "hazard_lambda": int(self.hazard_lambda),
            "score_threshold": float(self.threshold),
            "fbeta_05": self.fbeta_05,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "predicted_positives": int(self.predicted_positives),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "observations": int(self.observations),
        }


def _directional_change_state(
    close: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    regime = np.full(len(values), np.nan, dtype=float)
    distance = np.full(len(values), np.nan, dtype=float)
    overshoot = np.full(len(values), np.nan, dtype=float)
    age = np.full(len(values), np.nan, dtype=float)
    pressure = np.full(len(values), np.nan, dtype=float)

    mode = 0
    running_high = np.nan
    running_low = np.nan
    confirmation_price = np.nan
    sessions_since_event = 0

    for index, price in enumerate(values):
        if not np.isfinite(price) or price <= 0:
            continue

        if not np.isfinite(running_high):
            running_high = price
            running_low = price
            confirmation_price = price
            sessions_since_event = 0
        else:
            if mode >= 0:
                running_high = max(float(running_high), price)
            if mode <= 0:
                running_low = min(float(running_low), price)

            changed = False
            if mode in {0, 1} and price <= float(running_high) * (1.0 - threshold):
                mode = -1
                running_low = price
                confirmation_price = price
                changed = True
            elif mode in {0, -1} and price >= float(running_low) * (1.0 + threshold):
                mode = 1
                running_high = price
                confirmation_price = price
                changed = True

            sessions_since_event = 0 if changed else sessions_since_event + 1

        regime[index] = float(mode)
        age[index] = float(sessions_since_event)
        overshoot[index] = (
            float(price / confirmation_price - 1.0)
            if np.isfinite(confirmation_price) and confirmation_price > 0
            else np.nan
        )

        if mode == 1 and np.isfinite(running_high) and running_high > 0:
            distance[index] = float(price / running_high - 1.0)
            pressure[index] = max(
                0.0,
                float((running_high - price) / running_high / threshold),
            )
        elif mode == -1 and np.isfinite(running_low) and running_low > 0:
            distance[index] = float(price / running_low - 1.0)
            pressure[index] = max(
                0.0,
                float((price - running_low) / running_low / threshold),
            )
        else:
            distance[index] = 0.0
            pressure[index] = 0.0

    tag = _tag_threshold(threshold)
    return pd.DataFrame(
        {
            f"dc_regime_{tag}": regime,
            f"dc_distance_from_extreme_{tag}": distance,
            f"dc_overshoot_{tag}": overshoot,
            f"dc_sessions_since_event_{tag}": age,
            f"dc_reversal_pressure_{tag}": pressure,
        },
        index=close.index,
    )

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
    output["bocpd_eligible"] = (
        (distance_high >= -BOCPD_NEAR_HIGH_20)
        & (return_20 > BOCPD_MIN_RETURN_20)
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

def _normal_pdf(
    value: float,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    variance = np.maximum(
        np.asarray(variance, dtype=float),
        1e-9,
    )
    delta = float(value) - np.asarray(mean, dtype=float)
    return np.exp(-0.5 * delta * delta / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def _bocpd_downward_scores(
    frame: pd.DataFrame,
    *,
    hazard_lambda: int,
    max_run_length: int = BOCPD_MAX_RUN_LENGTH,
    short_run_max: int = BOCPD_SHORT_RUN_MAX,
) -> pd.Series:
    """Causal BOCPD score for a recent downward regime change."""
    returns = pd.to_numeric(frame["return_1"], errors="coerce")
    previous_vol = pd.to_numeric(
        frame["vol_20"],
        errors="coerce",
    ).shift(1)
    standardized = (returns / previous_vol.replace(0, np.nan)).clip(-8.0, 8.0)

    run_prob = np.asarray([1.0], dtype=float)
    means = np.asarray([float(BOCPD_PRIOR_MEAN)], dtype=float)
    kappas = np.asarray([float(BOCPD_PRIOR_KAPPA)], dtype=float)
    hazard = 1.0 / max(2.0, float(hazard_lambda))
    scores = np.full(len(frame), np.nan, dtype=float)

    for index, value in enumerate(standardized.to_numpy(dtype=float)):
        if not np.isfinite(value):
            continue

        predictive_variance = 1.0 + 1.0 / np.maximum(kappas, 1e-9)
        predictive = _normal_pdf(
            float(value),
            means,
            predictive_variance,
        )
        joint = run_prob * predictive

        changepoint = float(np.sum(joint * hazard))
        growth = joint * (1.0 - hazard)
        next_prob = np.concatenate(
            ([changepoint], growth)
        )

        if len(next_prob) > int(max_run_length) + 1:
            next_prob = next_prob[: int(max_run_length) + 1]

        total = float(np.sum(next_prob))
        if not np.isfinite(total) or total <= 0:
            next_prob = np.asarray([1.0], dtype=float)
        else:
            next_prob = next_prob / total

        updated_kappas = kappas + 1.0
        updated_means = (
            kappas * means + float(value)
        ) / updated_kappas
        prior_kappa = float(BOCPD_PRIOR_KAPPA)
        prior_mean = float(BOCPD_PRIOR_MEAN)
        cp_mean = (
            prior_kappa * prior_mean + float(value)
        ) / (prior_kappa + 1.0)

        next_means = np.concatenate(
            ([cp_mean], updated_means)
        )
        next_kappas = np.concatenate(
            ([prior_kappa + 1.0], updated_kappas)
        )
        keep = len(next_prob)
        means = next_means[:keep]
        kappas = next_kappas[:keep]
        run_prob = next_prob

        short_mass = float(
            np.sum(run_prob[: int(short_run_max) + 1])
        )
        downside = max(0.0, -float(value))
        downside_weight = downside / (1.0 + downside)
        scores[index] = short_mass * downside_weight

    return pd.Series(scores, index=frame.index, dtype=float)


def _precalcular_bocpd_scores(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    hazard_lambdas: Iterable[int] = BOCPD_HAZARD_LAMBDAS,
) -> dict[int, dict[str, pd.Series]]:
    return {
        int(hazard_lambda): {
            symbol: _bocpd_downward_scores(
                frames[symbol],
                hazard_lambda=int(hazard_lambda),
            )
            for symbol in symbols
        }
        for hazard_lambda in hazard_lambdas
    }


def calibrar_bocpd(
    score_cache: dict[int, dict[str, pd.Series]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
) -> BOCPDCalibrationResult:
    best: tuple[
        float,
        float,
        float,
        int,
        dict[str, float | int | None],
        int,
        int,
        int,
    ] | None = None

    for hazard_lambda in BOCPD_HAZARD_LAMBDAS:
        targets: list[int] = []
        scores: list[float] = []

        for symbol in symbols:
            frame = frames[symbol].reindex(calibration_dates)
            eligible = frame["bocpd_eligible"].fillna(False).astype(bool)
            target = pd.to_numeric(
                frame["forward_down_reversal"],
                errors="coerce",
            )
            signal = score_cache[int(hazard_lambda)][symbol].reindex(
                calibration_dates
            )
            valid = eligible & target.notna() & signal.notna()
            if not bool(valid.any()):
                continue
            targets.extend(target.loc[valid].astype(int).tolist())
            scores.extend(signal.loc[valid].astype(float).tolist())

        if not targets:
            continue

        y_true = np.asarray(targets, dtype=int)
        values = np.asarray(scores, dtype=float)
        positives = int((y_true == 1).sum())
        negatives = int((y_true == 0).sum())

        for threshold in BOCPD_SCORE_THRESHOLDS:
            metrics = _metricas_classificacao(
                y_true,
                values,
                float(threshold),
            )
            predicted = int(metrics["predicted_positives"] or 0)
            if predicted < MINIMUM_CALIBRATION_ALERTS:
                continue

            fbeta = float(metrics["fbeta_05"] or 0.0)
            precision = float(metrics["precision"] or 0.0)
            candidate = (
                fbeta,
                precision,
                float(threshold),
                int(hazard_lambda),
                metrics,
                positives,
                negatives,
                int(len(y_true)),
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate

    if best is None:
        return BOCPDCalibrationResult(
            hazard_lambda=60,
            threshold=0.05,
            fbeta_05=None,
            balanced_accuracy=None,
            precision=None,
            recall=None,
            predicted_positives=0,
            positives=0,
            negatives=0,
            observations=0,
        )

    (
        _fbeta,
        _precision,
        threshold,
        hazard_lambda,
        metrics,
        positives,
        negatives,
        observations,
    ) = best
    return BOCPDCalibrationResult(
        hazard_lambda=int(hazard_lambda),
        threshold=float(threshold),
        fbeta_05=float(metrics["fbeta_05"] or 0.0),
        balanced_accuracy=float(
            metrics["balanced_accuracy"] or 0.0
        ),
        precision=(
            float(metrics["precision"])
            if metrics["precision"] is not None
            else None
        ),
        recall=float(metrics["recall"] or 0.0),
        predicted_positives=int(
            metrics["predicted_positives"] or 0
        ),
        positives=int(positives),
        negatives=int(negatives),
        observations=int(observations),
    )


def _envolver_politica_bocpd(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    score_cache: dict[str, pd.Series],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    score_threshold: float,
    hazard_lambda: int,
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

        score = None
        eligible = False
        if symbol is not None and key in frames[symbol].index:
            eligible = bool(frames[symbol].at[key, "bocpd_eligible"])
            series = score_cache.get(symbol)
            if series is not None and key in series.index:
                value = series.loc[key]
                if pd.notna(value):
                    score = float(value)

        control_holds = bool(
            current_position > 0
            and int(base_target) == int(current_position)
            and holding_days >= int(config.rotation_min_holding_days)
        )
        above_threshold = bool(
            control_holds
            and eligible
            and score is not None
            and np.isfinite(float(score))
            and float(score) >= float(score_threshold)
        )

        if symbol is not None:
            if above_threshold:
                streak_by_symbol[symbol] = int(
                    streak_by_symbol.get(symbol, 0)
                ) + 1
            else:
                streak_by_symbol[symbol] = 0
        streak = int(streak_by_symbol.get(symbol, 0)) if symbol else 0
        triggered = bool(
            above_threshold
            and streak >= int(BOCPD_CONFIRMATION_SESSIONS)
        )

        diagnostic.update(
            {
                "bocpd_schema_version": 1,
                "bocpd_research_version": RESEARCH_VERSION,
                "bocpd_score": score,
                "bocpd_score_threshold": float(score_threshold),
                "bocpd_hazard_lambda": int(hazard_lambda),
                "bocpd_eligible": bool(eligible),
                "bocpd_confirmation_streak": int(streak),
                "bocpd_confirmation_required": int(
                    BOCPD_CONFIRMATION_SESSIONS
                ),
                "bocpd_base_target_asset": (
                    "CASH"
                    if int(base_target) <= 0
                    else symbols[int(base_target) - 1]
                ),
                "bocpd_exit_triggered": triggered,
            }
        )

        if not triggered:
            if not control_holds and symbol is not None:
                streak_by_symbol[symbol] = 0
            return int(base_target), float(base_score)

        streak_by_symbol[symbol] = 0
        diagnostic["bocpd_base_reason"] = diagnostic.get(
            "decision_reason"
        )
        diagnostic.update(
            {
                "decision_reason": "BOCPD_DOWNWARD_CHANGE_EXIT",
                "final_action_asset": "CASH",
                "final_action_score": 0.0,
                "decision_is_rotation": False,
                "decision_is_entry": False,
                "decision_is_exit_to_cash": True,
            }
        )
        return 0, 0.0

    return policy


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
    suppressed_triggers: set[tuple[pd.Timestamp, str]] | None = None,
    suppressed_assets: set[str] | None = None,
    disable_all_triggers: bool = False,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    streak_by_symbol: dict[str, int] = {}
    suppressed = suppressed_triggers or set()
    suppressed_asset_set = {str(item) for item in (suppressed_assets or set())}

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
        candidate_trigger = bool(
            above_threshold
            and streak >= int(TOP_TURN_CONFIRMATION_SESSIONS)
        )
        trigger_key = (
            (key, str(symbol))
            if symbol is not None
            else None
        )
        suppressed_by_key = bool(
            candidate_trigger
            and trigger_key is not None
            and trigger_key in suppressed
        )
        suppressed_by_asset = bool(
            candidate_trigger
            and symbol is not None
            and str(symbol) in suppressed_asset_set
        )
        suppressed_here = bool(
            suppressed_by_key or suppressed_by_asset
        )
        triggered = bool(
            candidate_trigger
            and not suppressed_here
            and not disable_all_triggers
        )
        if candidate_trigger and not triggered and symbol is not None:
            # A suppressed intervention must require a fresh confirmation
            # sequence before it can fire again on a later date.
            streak_by_symbol[symbol] = 0
            streak = 0

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
                "directional_change_ablation_candidate_trigger": candidate_trigger,
                "directional_change_ablation_suppressed": suppressed_here,
                "directional_change_ablation_suppressed_by_key": (
                    suppressed_by_key
                ),
                "directional_change_ablation_suppressed_by_asset": (
                    suppressed_by_asset
                ),
                "directional_change_ablation_disable_all": bool(
                    disable_all_triggers
                ),
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

def _agrupar_gatilhos_ablation(
    trigger_rows: pd.DataFrame,
) -> dict[tuple[str, str], set[tuple[pd.Timestamp, str]]]:
    """Agrupa gatilhos originais por ativo e por fold para replays OOS."""
    groups: dict[
        tuple[str, str],
        set[tuple[pd.Timestamp, str]],
    ] = {}

    for _, trigger_row in trigger_rows.iterrows():
        decision_timestamp = pd.Timestamp(trigger_row["decision_date"])
        asset = str(
            trigger_row.get("current_asset")
            or trigger_row.get("previous_asset")
            or ""
        )
        key = (decision_timestamp, asset)

        groups.setdefault(("asset", asset), set()).add(key)

        fold_value = trigger_row.get("walk_forward_fold")
        if fold_value is not None and not pd.isna(fold_value):
            fold_id = str(int(fold_value))
            groups.setdefault(("fold", fold_id), set()).add(key)

    return groups


def executar_directional_change_lightgbm(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
    run_ablation: bool = True,
) -> Any:
    if int(config.rotation_model_repetitions) != 1:
        raise ValueError("Directional Change research requires one repetition.")

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
    replay_specs: dict[int, dict[str, Any]] = {}

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
        replay_specs[fold_id] = {
            "utility_models": final_utility_models,
            "utility_cache": utility_cache,
            "effective_margin": float(effective_margin),
            "calibrated_margin": float(best_candidate),
            "probability_cache": probability_cache,
            "probability_threshold": float(calibration.threshold),
        }
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
        "directional_change_lightgbm",
        scheduled,
        frames,
        symbols,
        all_decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Directional Change + LightGBM",
        method_line=(
            "- Directional Change + LightGBM estimates a "
            "precision-oriented first-passage probability of a downward "
            "turn while the held asset is in an up regime near a recent high; "
            "two consecutive confirmations are required before exiting to "
            "CASH at the next open."
        ),
    )
    result.backend = "directional_change_lightgbm"

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
            "backend": "directional_change_lightgbm",
            "model_family": "directional_change_lightgbm",
            "strategy_label": "Directional Change + LightGBM",
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

    if not run_ablation:
        if progress_callback is not None:
            progress_callback(
                100.0,
                "Directional Change + LightGBM completed",
                total_folds,
            )
        return result

    def replay_with_ablation(
        suppressed_triggers: set[tuple[pd.Timestamp, str]] | None = None,
        *,
        suppressed_assets: set[str] | None = None,
        suppressed_folds: set[int] | None = None,
        disable_all_triggers: bool = False,
    ) -> Any:
        replay_diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        replay_policies: dict[int, Callable] = {}
        for replay_fold_id, spec in replay_specs.items():
            replay_base_policy = _politica_utilidade(
                spec["utility_models"],
                frames,
                symbols,
                config,
                float(spec["effective_margin"]),
                decision_diagnostics=replay_diagnostics,
                fold_id=int(replay_fold_id),
                calibrated_switch_margin=float(spec["calibrated_margin"]),
                utility_cache=spec["utility_cache"],
            )
            replay_policies[int(replay_fold_id)] = _envolver_politica_top_turn(
                replay_base_policy,
                probabilities=spec["probability_cache"],
                frames=frames,
                symbols=symbols,
                probability_threshold=float(spec["probability_threshold"]),
                config=config,
                decision_diagnostics=replay_diagnostics,
                suppressed_triggers=suppressed_triggers,
                suppressed_assets=suppressed_assets,
                disable_all_triggers=bool(
                    disable_all_triggers
                    or int(replay_fold_id) in (suppressed_folds or set())
                ),
            )

        return _simular_exato(
            "directional_change_lightgbm_ablation",
            _politica_agendada(replay_policies, decision_to_fold),
            frames,
            symbols,
            all_decision_dates,
            config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=replay_diagnostics,
            model_label="Directional Change + LightGBM ablation",
            method_line=(
                "- Ablation replay reuses the exact fitted OOS models and "
                "suppresses selected Top-Turn interventions without retraining."
            ),
        )

    trigger_rows = result.predictions.loc[triggers].copy()
    ablation_rows: list[dict[str, Any]] = []
    full_capital = float(result.metrics["strategy_ending_capital"])

    for position, (execution_timestamp, trigger_row) in enumerate(
        trigger_rows.iterrows(),
        start=1,
    ):
        decision_timestamp = pd.Timestamp(trigger_row["decision_date"])
        trigger_asset = str(
            trigger_row.get("current_asset")
            or trigger_row.get("previous_asset")
            or ""
        )
        key = (decision_timestamp, trigger_asset)

        if progress_callback is not None:
            progress_callback(
                90.0 + 8.0 * position / max(1, len(trigger_rows)),
                (
                    "Directional Change ablation "
                    f"{position}/{len(trigger_rows)} "
                    f"{decision_timestamp.date()} {trigger_asset}"
                ),
                total_folds,
            )

        ablated = replay_with_ablation({key})
        ablated_capital = float(
            ablated.metrics["strategy_ending_capital"]
        )
        ablation_rows.append(
            {
                "trigger_number": int(position),
                "decision_timestamp": decision_timestamp,
                "execution_timestamp": pd.Timestamp(execution_timestamp),
                "asset": trigger_asset,
                "walk_forward_fold": trigger_row.get("walk_forward_fold"),
                "probability": trigger_row.get(
                    "directional_change_probability"
                ),
                "probability_threshold": trigger_row.get(
                    "directional_change_probability_threshold"
                ),
                "full_top_turn_ending_capital": full_capital,
                "without_trigger_ending_capital": ablated_capital,
                "trigger_contribution_to_ending_capital": (
                    full_capital - ablated_capital
                ),
                "trigger_contribution_ratio": (
                    full_capital / ablated_capital - 1.0
                    if ablated_capital > 0
                    else None
                ),
            }
        )

    group_rows: list[dict[str, Any]] = []
    groups = _agrupar_gatilhos_ablation(trigger_rows)
    ordered_groups = sorted(groups.items(), key=lambda item: item[0])

    for group_position, (
        (group_type, group_value),
        suppressed_keys,
    ) in enumerate(ordered_groups, start=1):
        if progress_callback is not None:
            progress_callback(
                98.0
                + 1.5
                * group_position
                / max(1, len(ordered_groups)),
                (
                    "Directional Change grouped ablation "
                    f"{group_position}/{len(ordered_groups)} "
                    f"{group_type}={group_value}"
                ),
                total_folds,
            )

        if group_type == "asset":
            grouped = replay_with_ablation(
                suppressed_assets={str(group_value)}
            )
            suppression_scope = "all_oos_triggers_for_asset"
        elif group_type == "fold":
            grouped = replay_with_ablation(
                suppressed_folds={int(group_value)}
            )
            suppression_scope = "all_oos_triggers_for_fold"
        else:
            raise ValueError(
                f"Unsupported ablation group type: {group_type}"
            )

        grouped_capital = float(
            grouped.metrics["strategy_ending_capital"]
        )
        group_rows.append(
            {
                "group_type": str(group_type),
                "group_value": str(group_value),
                "original_trigger_count": int(len(suppressed_keys)),
                "suppression_scope": suppression_scope,
                "full_top_turn_ending_capital": full_capital,
                "without_group_ending_capital": grouped_capital,
                "group_contribution_to_ending_capital": (
                    full_capital - grouped_capital
                ),
                "group_contribution_ratio": (
                    full_capital / grouped_capital - 1.0
                    if grouped_capital > 0
                    else None
                ),
            }
        )

    if progress_callback is not None:
        progress_callback(
            99.7,
            "Directional Change ablation without all triggers",
            total_folds,
        )

    no_overlay = replay_with_ablation(disable_all_triggers=True)
    no_overlay_capital = float(
        no_overlay.metrics["strategy_ending_capital"]
    )
    control_capital_reference = None
    result.metrics["directional_change_ablation"] = {
        "schema_version": 3,
        "method": (
            "leave_one_trigger_out_plus_full_scope_group_suppression_"
            "without_retraining"
        ),
        "trigger_count": int(len(trigger_rows)),
        "full_top_turn_ending_capital": full_capital,
        "without_all_triggers_ending_capital": no_overlay_capital,
        "control_capital_reference": control_capital_reference,
        "rows": ablation_rows,
        "group_rows": group_rows,
    }

    if progress_callback is not None:
        progress_callback(
            100.0,
            "Directional Change + LightGBM completed with ablation",
            total_folds,
        )
    return result

def executar_bocpd_overlay(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> Any:
    if int(config.rotation_model_repetitions) != 1:
        raise ValueError("BOCPD comparison requires one repetition.")

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
    score_cache = _precalcular_bocpd_scores(
        frames,
        symbols,
    )

    policies: dict[int, Callable] = {}
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
    calibration_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []

    total_folds = len(folds)
    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        if progress_callback is not None:
            progress_callback(
                5.0 + 75.0 * ((fold_position - 1) / max(1, total_folds)),
                f"BOCPD fold {fold_position}/{total_folds} training",
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
            phase=f"bocpd_fold_{fold_id}_utility_calibration",
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
            candidate_score = _crescimento_politica_simples(
                calibration_policy,
                frames,
                symbols,
                calibration_dates,
                config,
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate

        calibration = calibrar_bocpd(
            score_cache,
            frames,
            symbols,
            calibration_dates,
        )
        calibration_rows.append(
            calibration.as_dict(fold_id=fold_id)
        )

        final_utility_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"bocpd_fold_{fold_id}_utility_final",
        )
        decision_dates = pd.DatetimeIndex(fold["decision_dates"])
        utility_cache, _ = _precalcular_utilidades_modelo(
            final_utility_models,
            frames,
            symbols,
            decision_dates,
            config,
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
        selected_scores = {
            symbol: score_cache[int(calibration.hazard_lambda)][symbol]
            for symbol in symbols
        }
        policies[fold_id] = _envolver_politica_bocpd(
            base_policy,
            score_cache=selected_scores,
            frames=frames,
            symbols=symbols,
            score_threshold=float(calibration.threshold),
            hazard_lambda=int(calibration.hazard_lambda),
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
            "BOCPD OOS replay",
            total_folds,
        )

    result = _simular_exato(
        "bocpd_overlay",
        _politica_agendada(policies, decision_to_fold),
        frames,
        symbols,
        all_decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Bayesian Online Change Point Detection",
        method_line=(
            "- BOCPD tracks causal posterior run-length mass on "
            "volatility-standardized daily returns. A downward short-run "
            "change signal near a recent high must exceed a calibration-only "
            "threshold on two consecutive sessions before an exit to CASH."
        ),
    )
    result.backend = "bocpd_overlay"

    scores = pd.to_numeric(
        result.predictions.get(
            "bocpd_score",
            pd.Series(index=result.predictions.index, dtype=float),
        ),
        errors="coerce",
    )
    triggers = result.predictions.get(
        "bocpd_exit_triggered",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)

    result.metrics.update(
        {
            "backend": "bocpd_overlay",
            "model_family": "bayesian_online_changepoint_detection",
            "strategy_label": "BOCPD Overlay",
            "bocpd_research_version": RESEARCH_VERSION,
            "bocpd_hazard_candidates": list(BOCPD_HAZARD_LAMBDAS),
            "bocpd_score_threshold_candidates": list(
                BOCPD_SCORE_THRESHOLDS
            ),
            "bocpd_max_run_length": int(BOCPD_MAX_RUN_LENGTH),
            "bocpd_short_run_max": int(BOCPD_SHORT_RUN_MAX),
            "bocpd_confirmation_sessions": int(
                BOCPD_CONFIRMATION_SESSIONS
            ),
            "bocpd_exit_triggers": int(triggers.sum()),
            "bocpd_score_observations": int(scores.notna().sum()),
            "bocpd_calibration": calibration_rows,
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
            "BOCPD completed",
            total_folds,
        )
    return result


def comparar_control_directional_change(
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
        "directional_change_ending_capital": challenger_capital,
        "directional_change_minus_control_capital": challenger_capital - control_capital,
        "directional_change_vs_control_ratio": (
            challenger_capital / control_capital - 1.0
            if control_capital > 0
            else None
        ),
        "control_cagr": control_metrics.get("cagr"),
        "directional_change_cagr": challenger_metrics.get("cagr"),
        "control_sharpe": control_metrics.get("sharpe"),
        "directional_change_sharpe": challenger_metrics.get("sharpe"),
        "control_maximum_drawdown": control_metrics.get("maximum_drawdown"),
        "directional_change_maximum_drawdown": challenger_metrics.get(
            "maximum_drawdown"
        ),
        "control_worst_fold_return": control_metrics.get("worst_fold_return"),
        "directional_change_worst_fold_return": challenger_metrics.get(
            "worst_fold_return"
        ),
        "control_median_exit_distance_from_peak_pct": control_distance,
        "directional_change_median_exit_distance_from_peak_pct": challenger_distance,
        "median_exit_distance_improvement_pct_points": (
            float(control_distance) - float(challenger_distance)
            if control_distance is not None
            and challenger_distance is not None
            else None
        ),
        "control_median_peak_capture_pct": control_peak.get(
            "median_peak_capture_pct"
        ),
        "directional_change_median_peak_capture_pct": challenger_peak.get(
            "median_peak_capture_pct"
        ),
        "control_median_days_from_peak_to_exit": control_peak.get(
            "median_days_from_peak_to_exit"
        ),
        "directional_change_median_days_from_peak_to_exit": challenger_peak.get(
            "median_days_from_peak_to_exit"
        ),
        "directional_change_exit_triggers": int(
            challenger_metrics.get("directional_change_exit_triggers") or 0
        ),
    }

def calcular_peak_exit(
    trades: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calcula proximidade do topo usando os mesmos OHLC do backtest."""
    rows: list[dict[str, Any]] = []
    if trades is None or trades.empty:
        return {
            "closed_positions": 0,
            "median_exit_distance_from_peak_pct": None,
            "median_peak_capture_pct": None,
            "median_max_runup_pct": None,
            "median_days_from_peak_to_exit": None,
        }, pd.DataFrame()

    for trade in trades.to_dict(orient="records"):
        action = str(trade.get("action") or "").upper()
        if action not in {"SELL", "FINAL_SELL"}:
            continue
        symbol = str(trade.get("asset") or "")
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        entry_at = pd.Timestamp(trade.get("entry_timestamp"))
        exit_at = pd.Timestamp(trade.get("timestamp"))
        if entry_at.tzinfo is None:
            entry_at = entry_at.tz_localize("UTC")
        if exit_at.tzinfo is None:
            exit_at = exit_at.tz_localize("UTC")
        entry_price = float(trade.get("entry_price") or 0.0)
        exit_price = float(trade.get("execution_price") or 0.0)
        if entry_price <= 0 or exit_price <= 0:
            continue

        include_exit = action == "FINAL_SELL"
        holding = frame.loc[
            (frame.index >= entry_at)
            & (
                (frame.index <= exit_at)
                if include_exit
                else (frame.index < exit_at)
            )
        ]
        highs = pd.to_numeric(
            holding.get("high"),
            errors="coerce",
        ).dropna()
        peak_price = float(entry_price)
        peak_at = entry_at
        if not highs.empty:
            candidate_at = highs.idxmax()
            candidate_price = float(highs.loc[candidate_at])
            if candidate_price >= peak_price:
                peak_price = candidate_price
                peak_at = pd.Timestamp(candidate_at)

        peak_gain = max(0.0, peak_price / entry_price - 1.0)
        exit_gain = exit_price / entry_price - 1.0
        distance = (
            max(0.0, (peak_price - exit_price) / peak_price) * 100.0
            if peak_price > 0
            else None
        )
        capture = (
            min(1.0, max(0.0, exit_gain) / peak_gain) * 100.0
            if peak_gain > 0
            else None
        )
        days_after_peak = int(
            ((frame.index > peak_at) & (frame.index <= exit_at)).sum()
        )

        post = frame.loc[
            frame.index > exit_at
            if action == "FINAL_SELL"
            else frame.index >= exit_at
        ]
        post_10 = None
        if len(post) >= 10:
            post_highs = pd.to_numeric(
                post.head(10).get("high"),
                errors="coerce",
            ).dropna()
            if not post_highs.empty:
                post_10 = max(
                    0.0,
                    float(post_highs.max()) / exit_price - 1.0,
                ) * 100.0

        rows.append(
            {
                "asset": symbol,
                "action": action,
                "entry_timestamp": entry_at,
                "exit_timestamp": exit_at,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "peak_price_while_held": peak_price,
                "peak_timestamp_while_held": peak_at,
                "exit_distance_from_peak_pct": distance,
                "peak_capture_pct": capture,
                "max_runup_pct": peak_gain * 100.0,
                "days_from_peak_to_exit": days_after_peak,
                "post_exit_peak_10d_pct": post_10,
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "closed_positions": 0,
            "median_exit_distance_from_peak_pct": None,
            "median_peak_capture_pct": None,
            "median_max_runup_pct": None,
            "median_days_from_peak_to_exit": None,
        }, detail

    def median(column: str) -> float | None:
        values = pd.to_numeric(detail[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else None

    return {
        "closed_positions": int(len(detail)),
        "median_exit_distance_from_peak_pct": median(
            "exit_distance_from_peak_pct"
        ),
        "median_peak_capture_pct": median("peak_capture_pct"),
        "median_max_runup_pct": median("max_runup_pct"),
        "median_days_from_peak_to_exit": median(
            "days_from_peak_to_exit"
        ),
        "median_post_exit_peak_10d_pct": median(
            "post_exit_peak_10d_pct"
        ),
    }, detail


def criar_pacote_analise(diretorio_resultados: Path) -> Path:
    """Gera um unico ZIP estavel com os artefatos da execucao corrente."""
    diretorio = Path(diretorio_resultados)
    if not diretorio.exists():
        raise FileNotFoundError(diretorio)

    destino = diretorio / "pacote_analise.zip"
    if destino.exists():
        destino.unlink()

    with zipfile.ZipFile(
        destino,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as arquivo:
        for path in sorted(diretorio.rglob("*")):
            if not path.is_file() or path.resolve() == destino.resolve():
                continue
            arquivo.write(path, arcname=path.relative_to(diretorio))

    return destino


def sinal_sonoro_conclusao() -> None:
    """Emite dois tons no Windows; usa bell do terminal como fallback."""
    try:
        import winsound

        winsound.Beep(880, 220)
        time.sleep(0.08)
        winsound.Beep(1175, 420)
        return
    except (ImportError, RuntimeError, OSError):
        pass

    try:
        sys.stdout.write("\\a")
        sys.stdout.flush()
    except Exception:
        return
