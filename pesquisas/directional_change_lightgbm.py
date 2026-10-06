"""Pesquisa de reversao: Top-Turn, BOCPD e HSMM.

Este modulo e o unico ponto de evolucao desta linha de pesquisa. O historico
fica no Git; novas tecnicas sao comparadas no mesmo protocolo OOS sem criar
arquivos de codigo paralelos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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
    _selecionar_switch_margin_fold,
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

RESEARCH_VERSION = "1.17.0-dev.1"
TOP_GAP_RESEARCH_VERSION = "1.0.0-dev.1"
EXPECTED_EXECUTION_SCHEMA = "intelligent-asset-search-u59-v1"
EXPECTED_COMPARISON_FILE = "asset_search.json"
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

BOTTOM_TURN_HORIZON_SESSIONS = TOP_TURN_HORIZON_SESSIONS
BOTTOM_TURN_ATR_MULTIPLIER = TOP_TURN_ATR_MULTIPLIER
BOTTOM_TURN_THRESHOLD_MIN = TOP_TURN_THRESHOLD_MIN
BOTTOM_TURN_THRESHOLD_MAX = TOP_TURN_THRESHOLD_MAX
BOTTOM_TURN_CONTINUATION_RATIO = TOP_TURN_CONTINUATION_RATIO
BOTTOM_TURN_NEAR_LOW_20 = 0.05
BOTTOM_TURN_MIN_DOWN_REGIME_SHARE = 2.0 / 3.0
BOTTOM_TURN_MAX_RETURN_20 = 0.0
BOTTOM_TURN_CONFIRMATION_SESSIONS = 2
BOTTOM_TURN_MAX_WAIT_SESSIONS = BOTTOM_TURN_HORIZON_SESSIONS
FIXED_COOLDOWN_SESSIONS = BOTTOM_TURN_HORIZON_SESSIONS
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

HSMM_STATE_COUNT = 3
HSMM_MAX_DURATION = 60
HSMM_MIN_TRAINING_ROWS = 160
HSMM_SCORE_THRESHOLDS = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.35,
    0.50,
    0.65,
)
HSMM_NEAR_HIGH_20 = 0.05
HSMM_MIN_RETURN_20 = 0.0
HSMM_CONFIRMATION_SESSIONS = 2
HSMM_DURATION_SMOOTHING = 0.5
HSMM_TRANSITION_SMOOTHING = 0.5

HAZARD_HORIZON_SESSIONS = TOP_TURN_HORIZON_SESSIONS
HAZARD_NEAR_HIGH_20 = 0.05
HAZARD_MIN_RETURN_20 = 0.0
HAZARD_CONFIRMATION_SESSIONS = 2
HAZARD_MIN_TRAINING_ROWS = 500
HAZARD_SCORE_THRESHOLDS = (
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.35,
    0.50,
    0.65,
)
HAZARD_FEATURES = (
    "return_1",
    "return_5",
    "return_20",
    "vol_20",
    "atr_pct_14",
    "distance_from_high_20",
    "ema_distance_20",
    "ema_slope_20_5",
    "rsi_14",
    "trend_efficiency_20",
    "dc_up_regime_share",
    "dc_reversal_pressure_max",
)

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


@dataclass(frozen=True)
class HSMMCalibrationResult:
    threshold: float
    fbeta_05: float | None
    balanced_accuracy: float | None
    precision: float | None
    recall: float | None
    predicted_positives: int
    positives: int
    negatives: int
    observations: int
    fitted_assets: int

    def as_dict(self, *, fold_id: int) -> dict[str, Any]:
        return {
            "fold_id": int(fold_id),
            "score_threshold": float(self.threshold),
            "fbeta_05": self.fbeta_05,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "predicted_positives": int(self.predicted_positives),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "observations": int(self.observations),
            "fitted_assets": int(self.fitted_assets),
        }


@dataclass(frozen=True)
class HazardCalibrationResult:
    threshold: float
    fbeta_05: float | None
    balanced_accuracy: float | None
    precision: float | None
    recall: float | None
    predicted_positives: int
    positives: int
    negatives: int
    observations: int
    training_rows: int
    training_events: int

    def as_dict(self, *, fold_id: int) -> dict[str, Any]:
        return {
            "fold_id": int(fold_id),
            "score_threshold": float(self.threshold),
            "fbeta_05": self.fbeta_05,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "predicted_positives": int(self.predicted_positives),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "observations": int(self.observations),
            "training_rows": int(self.training_rows),
            "training_events": int(self.training_events),
        }


@dataclass(frozen=True)
class HSMMAssetModel:
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    duration_pmf: np.ndarray
    initial_probabilities: np.ndarray


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
    down_event_step = np.full(len(frame), np.nan, dtype=float)
    competing_event_step = np.full(len(frame), np.nan, dtype=float)
    observed_down_event = np.full(len(frame), np.nan, dtype=float)
    censored = np.full(len(frame), np.nan, dtype=float)

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
        censored[index] = 1.0
        for step, future_close in enumerate(future, start=1):
            move = float(future_close / current - 1.0)
            if move <= -threshold:
                label = 1.0
                event_step[index] = float(step)
                down_event_step[index] = float(step)
                observed_down_event[index] = 1.0
                censored[index] = 0.0
                break
            if move >= continuation:
                label = 0.0
                event_step[index] = float(step)
                competing_event_step[index] = float(step)
                observed_down_event[index] = 0.0
                censored[index] = 0.0
                break
        target[index] = label

    return pd.DataFrame(
        {
            "forward_down_reversal": target,
            "forward_top_turn_threshold": threshold_values,
            "forward_top_turn_continuation": continuation_values,
            "forward_top_turn_event_step": event_step,
            "hazard_down_event_step": down_event_step,
            "hazard_competing_event_step": competing_event_step,
            "hazard_observed_down_event": observed_down_event,
            "hazard_right_censored": censored,
        },
        index=frame.index,
    )

def _bottom_turn_targets(
    frame: pd.DataFrame,
    *,
    horizon: int = BOTTOM_TURN_HORIZON_SESSIONS,
) -> pd.DataFrame:
    """Primeiro evento: recuperacao relevante antes de nova continuacao da queda."""
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
                float(BOTTOM_TURN_ATR_MULTIPLIER) * float(current_atr),
                float(BOTTOM_TURN_THRESHOLD_MIN),
                float(BOTTOM_TURN_THRESHOLD_MAX),
            )
        )
        continuation = float(
            max(
                0.01,
                float(BOTTOM_TURN_CONTINUATION_RATIO) * threshold,
            )
        )
        threshold_values[index] = threshold
        continuation_values[index] = continuation

        label = 0.0
        for step, future_close in enumerate(future, start=1):
            move = float(future_close / current - 1.0)
            if move >= threshold:
                label = 1.0
                event_step[index] = float(step)
                break
            if move <= -continuation:
                label = 0.0
                event_step[index] = float(step)
                break
        target[index] = label

    return pd.DataFrame(
        {
            "forward_up_reversal": target,
            "forward_bottom_turn_threshold": threshold_values,
            "forward_bottom_turn_continuation": continuation_values,
            "forward_bottom_turn_event_step": event_step,
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
    distance_low = pd.to_numeric(
        output["distance_from_low_20"],
        errors="coerce",
    )
    return_20 = pd.to_numeric(output["return_20"], errors="coerce")
    output["top_turn_eligible"] = (
        (output["dc_up_regime_share"] >= TOP_TURN_MIN_UP_REGIME_SHARE)
        & (distance_high >= -TOP_TURN_NEAR_HIGH_20)
        & (return_20 > TOP_TURN_MIN_RETURN_20)
    )
    output["bottom_turn_eligible"] = (
        (output["dc_down_regime_share"] >= BOTTOM_TURN_MIN_DOWN_REGIME_SHARE)
        & (distance_low <= BOTTOM_TURN_NEAR_LOW_20)
        & (return_20 < BOTTOM_TURN_MAX_RETURN_20)
    )
    output["bocpd_eligible"] = (
        (distance_high >= -BOCPD_NEAR_HIGH_20)
        & (return_20 > BOCPD_MIN_RETURN_20)
    )
    output["hsmm_eligible"] = (
        (distance_high >= -HSMM_NEAR_HIGH_20)
        & (return_20 > HSMM_MIN_RETURN_20)
    )
    output["hazard_eligible"] = (
        (distance_high >= -HAZARD_NEAR_HIGH_20)
        & (return_20 > HAZARD_MIN_RETURN_20)
    )

    output = output.join(_top_turn_targets(output))
    output = output.join(_bottom_turn_targets(output))
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


def _hazard_person_period_dataset(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []

    for symbol in symbols:
        frame = frames[symbol].reindex(dates)
        feature_frame = frame.loc[:, list(HAZARD_FEATURES)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        eligible = frame["hazard_eligible"].fillna(False).astype(bool)

        for position, _timestamp in enumerate(frame.index):
            if not bool(eligible.iloc[position]):
                continue
            features = feature_frame.iloc[position].to_numpy(dtype=float)
            if not np.isfinite(features).all():
                continue

            down_step_value = frame.iloc[position].get(
                "hazard_down_event_step"
            )
            competing_step_value = frame.iloc[position].get(
                "hazard_competing_event_step"
            )
            censored_value = frame.iloc[position].get(
                "hazard_right_censored"
            )
            if pd.isna(censored_value):
                continue

            down_step = (
                int(down_step_value)
                if pd.notna(down_step_value)
                else None
            )
            competing_step = (
                int(competing_step_value)
                if pd.notna(competing_step_value)
                else None
            )

            if down_step is not None:
                risk_end = min(
                    int(HAZARD_HORIZON_SESSIONS),
                    int(down_step),
                )
            elif competing_step is not None:
                # Competing continuation censors the downside process
                # at the start of that interval.
                risk_end = min(
                    int(HAZARD_HORIZON_SESSIONS),
                    max(0, int(competing_step) - 1),
                )
            else:
                # No event observed through the horizon: right-censored
                # after contributing survival information for all intervals.
                risk_end = int(HAZARD_HORIZON_SESSIONS)

            for step in range(1, risk_end + 1):
                rows.append(
                    [
                        *features.tolist(),
                        float(step) / float(HAZARD_HORIZON_SESSIONS),
                    ]
                )
                labels.append(
                    int(
                        down_step is not None
                        and int(step) == int(down_step)
                    )
                )

    if not rows:
        return (
            np.empty((0, len(HAZARD_FEATURES) + 1), dtype=float),
            np.empty((0,), dtype=int),
        )
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def _fit_hazard_model(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    *,
    random_state: int,
) -> tuple[Any | None, int, int]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = _hazard_person_period_dataset(
        frames,
        symbols,
        train_dates,
    )
    training_rows = int(len(y_train))
    training_events = int(np.sum(y_train == 1))
    if (
        training_rows < int(HAZARD_MIN_TRAINING_ROWS)
        or training_events < int(MINIMUM_CLASS_ROWS)
        or int(np.sum(y_train == 0)) < int(MINIMUM_CLASS_ROWS)
    ):
        return None, training_rows, training_events

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "hazard",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model, training_rows, training_events


def _predict_hazard_scores(
    model: Any | None,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}

    for symbol in symbols:
        frame = frames[symbol]
        output = pd.Series(np.nan, index=frame.index, dtype=float)
        if model is None:
            result[symbol] = output
            continue

        feature_frame = frame.loc[:, list(HAZARD_FEATURES)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        eligible = frame["hazard_eligible"].fillna(False).astype(bool)
        valid = eligible & feature_frame.notna().all(axis=1)
        valid_positions = np.flatnonzero(valid.to_numpy(dtype=bool))
        if len(valid_positions) == 0:
            result[symbol] = output
            continue

        feature_values = feature_frame.iloc[
            valid_positions
        ].to_numpy(dtype=float)
        design_rows: list[list[float]] = []
        for features in feature_values:
            for step in range(1, int(HAZARD_HORIZON_SESSIONS) + 1):
                design_rows.append(
                    [
                        *features.tolist(),
                        float(step) / float(HAZARD_HORIZON_SESSIONS),
                    ]
                )

        probabilities = model.predict_proba(
            np.asarray(design_rows, dtype=float)
        )[:, 1]
        probabilities = probabilities.reshape(
            len(valid_positions),
            int(HAZARD_HORIZON_SESSIONS),
        )
        cumulative = 1.0 - np.prod(
            1.0 - np.clip(probabilities, 0.0, 1.0),
            axis=1,
        )
        output.iloc[valid_positions] = cumulative
        result[symbol] = output

    return result


def calibrar_hazard(
    score_cache: dict[str, pd.Series],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    *,
    training_rows: int,
    training_events: int,
) -> HazardCalibrationResult:
    targets: list[int] = []
    scores: list[float] = []

    for symbol in symbols:
        frame = frames[symbol].reindex(calibration_dates)
        signal = score_cache[symbol].reindex(calibration_dates)
        eligible = frame["hazard_eligible"].fillna(False).astype(bool)
        observed = pd.to_numeric(
            frame["hazard_observed_down_event"],
            errors="coerce",
        )
        valid = eligible & observed.notna() & signal.notna()
        if not bool(valid.any()):
            continue
        targets.extend(observed.loc[valid].astype(int).tolist())
        scores.extend(signal.loc[valid].astype(float).tolist())

    if not targets:
        return HazardCalibrationResult(
            threshold=0.25,
            fbeta_05=None,
            balanced_accuracy=None,
            precision=None,
            recall=None,
            predicted_positives=0,
            positives=0,
            negatives=0,
            observations=0,
            training_rows=int(training_rows),
            training_events=int(training_events),
        )

    y_true = np.asarray(targets, dtype=int)
    values = np.asarray(scores, dtype=float)
    best: tuple[
        float,
        float,
        float,
        dict[str, float | int | None],
    ] | None = None

    for threshold in HAZARD_SCORE_THRESHOLDS:
        metrics = _metricas_classificacao(
            y_true,
            values,
            float(threshold),
        )
        if int(metrics["predicted_positives"] or 0) < MINIMUM_CALIBRATION_ALERTS:
            continue
        candidate = (
            float(metrics["fbeta_05"] or 0.0),
            float(metrics["precision"] or 0.0),
            float(threshold),
            metrics,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        threshold = 0.25
        metrics = _metricas_classificacao(
            y_true,
            values,
            threshold,
        )
    else:
        threshold = float(best[2])
        metrics = best[3]

    return HazardCalibrationResult(
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
        positives=int((y_true == 1).sum()),
        negatives=int((y_true == 0).sum()),
        observations=int(len(y_true)),
        training_rows=int(training_rows),
        training_events=int(training_events),
    )


def _envolver_politica_hazard(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    score_cache: dict[str, pd.Series],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    score_threshold: float,
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
            eligible = bool(frames[symbol].at[key, "hazard_eligible"])
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
            and streak >= int(HAZARD_CONFIRMATION_SESSIONS)
        )

        diagnostic.update(
            {
                "hazard_schema_version": 1,
                "hazard_research_version": RESEARCH_VERSION,
                "hazard_cumulative_downside_probability": score,
                "hazard_score_threshold": float(score_threshold),
                "hazard_eligible": bool(eligible),
                "hazard_confirmation_streak": int(streak),
                "hazard_confirmation_required": int(
                    HAZARD_CONFIRMATION_SESSIONS
                ),
                "hazard_base_target_asset": (
                    "CASH"
                    if int(base_target) <= 0
                    else symbols[int(base_target) - 1]
                ),
                "hazard_exit_triggered": triggered,
            }
        )

        if not triggered:
            if not control_holds and symbol is not None:
                streak_by_symbol[symbol] = 0
            return int(base_target), float(base_score)

        streak_by_symbol[symbol] = 0
        diagnostic["hazard_base_reason"] = diagnostic.get(
            "decision_reason"
        )
        diagnostic.update(
            {
                "decision_reason": "HAZARD_SURVIVAL_EXIT",
                "final_action_asset": "CASH",
                "final_action_score": 0.0,
                "decision_is_rotation": False,
                "decision_is_entry": False,
                "decision_is_exit_to_cash": True,
            }
        )
        return 0, 0.0

    return policy


def _hsmm_observations(frame: pd.DataFrame) -> pd.DataFrame:
    previous_vol = pd.to_numeric(
        frame["vol_20"],
        errors="coerce",
    ).shift(1)
    daily = pd.to_numeric(frame["return_1"], errors="coerce")
    slope = pd.to_numeric(frame["ema_slope_20_5"], errors="coerce")
    scale_daily = previous_vol.replace(0, np.nan)
    scale_slope = (
        previous_vol.replace(0, np.nan) * np.sqrt(5.0)
    )
    return pd.DataFrame(
        {
            "hsmm_standardized_return": (
                daily / scale_daily
            ).clip(-8.0, 8.0),
            "hsmm_standardized_trend": (
                slope / scale_slope
            ).clip(-8.0, 8.0),
        },
        index=frame.index,
    )


def _hsmm_gaussian_logpdf(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    variances = np.maximum(
        np.asarray(variances, dtype=float),
        1e-6,
    )
    delta = (
        np.asarray(values, dtype=float)[None, :]
        - np.asarray(means, dtype=float)
    )
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variances)
        + delta * delta / variances,
        axis=1,
    )


def _fit_hsmm_asset(
    frame: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    *,
    random_state: int,
) -> HSMMAssetModel | None:
    from sklearn.mixture import GaussianMixture

    observations = _hsmm_observations(frame).reindex(train_dates)
    clean = observations.dropna()
    if len(clean) < int(HSMM_MIN_TRAINING_ROWS):
        return None

    values = clean.to_numpy(dtype=float)
    mixture = GaussianMixture(
        n_components=int(HSMM_STATE_COUNT),
        covariance_type="diag",
        reg_covar=1e-5,
        random_state=int(random_state),
        n_init=5,
    )
    mixture.fit(values)
    labels_raw = mixture.predict(values)

    raw_means = np.asarray(mixture.means_, dtype=float)
    order = np.argsort(
        raw_means[:, 0] + 0.35 * raw_means[:, 1]
    )
    raw_to_state = {
        int(raw_index): int(state_index)
        for state_index, raw_index in enumerate(order)
    }
    labels = np.asarray(
        [raw_to_state[int(value)] for value in labels_raw],
        dtype=int,
    )
    means = raw_means[order]
    variances = np.asarray(
        mixture.covariances_,
        dtype=float,
    )[order]

    state_count = int(HSMM_STATE_COUNT)
    max_duration = int(HSMM_MAX_DURATION)
    transition_counts = np.full(
        (state_count, state_count),
        float(HSMM_TRANSITION_SMOOTHING),
        dtype=float,
    )
    duration_counts = np.full(
        (state_count, max_duration),
        float(HSMM_DURATION_SMOOTHING),
        dtype=float,
    )
    initial_counts = np.full(
        state_count,
        1.0,
        dtype=float,
    )

    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            state = int(labels[start])
            duration = int(index - start)
            segments.append((state, duration))
            duration_counts[
                state,
                min(duration, max_duration) - 1,
            ] += 1.0
            start = index

    if segments:
        initial_counts[int(segments[0][0])] += 1.0
    for (left_state, _), (right_state, _) in zip(
        segments[:-1],
        segments[1:],
        strict=True,
    ):
        if int(left_state) != int(right_state):
            transition_counts[
                int(left_state),
                int(right_state),
            ] += 1.0

    for state in range(state_count):
        transition_counts[state, state] = 0.0
        total = float(transition_counts[state].sum())
        if total <= 0:
            transition_counts[state] = 1.0
            transition_counts[state, state] = 0.0

    transition = transition_counts / transition_counts.sum(
        axis=1,
        keepdims=True,
    )
    duration_pmf = duration_counts / duration_counts.sum(
        axis=1,
        keepdims=True,
    )
    initial_probabilities = initial_counts / initial_counts.sum()

    return HSMMAssetModel(
        means=means,
        variances=variances,
        transition=transition,
        duration_pmf=duration_pmf,
        initial_probabilities=initial_probabilities,
    )


def _filter_hsmm_asset(
    frame: pd.DataFrame,
    model: HSMMAssetModel,
) -> pd.DataFrame:
    observations = _hsmm_observations(frame)
    state_count = int(HSMM_STATE_COUNT)
    max_duration = int(HSMM_MAX_DURATION)
    posterior = np.full(
        (len(frame), state_count),
        np.nan,
        dtype=float,
    )
    score = np.full(len(frame), np.nan, dtype=float)

    alpha: np.ndarray | None = None
    previous_state = np.asarray(
        model.initial_probabilities,
        dtype=float,
    )

    for index, values in enumerate(
        observations.to_numpy(dtype=float)
    ):
        if not np.isfinite(values).all():
            continue

        log_likelihood = _hsmm_gaussian_logpdf(
            values,
            model.means,
            model.variances,
        )
        likelihood = np.exp(
            log_likelihood - float(np.max(log_likelihood))
        )

        if alpha is None:
            alpha = np.zeros(
                (state_count, max_duration),
                dtype=float,
            )
            for state in range(state_count):
                alpha[state, :] = (
                    float(model.initial_probabilities[state])
                    * model.duration_pmf[state]
                    * float(likelihood[state])
                )
        else:
            next_alpha = np.zeros_like(alpha)

            # Remaining duration > 1: deterministic persistence.
            next_alpha[:, :-1] += alpha[:, 1:]

            # Remaining duration == 1: transition to another state
            # and draw an explicit new duration.
            ending_mass = alpha[:, 0]
            for source in range(state_count):
                source_mass = float(ending_mass[source])
                if source_mass <= 0:
                    continue
                for target in range(state_count):
                    transition_probability = float(
                        model.transition[source, target]
                    )
                    if transition_probability <= 0:
                        continue
                    next_alpha[target, :] += (
                        source_mass
                        * transition_probability
                        * model.duration_pmf[target]
                    )

            next_alpha *= likelihood[:, None]
            alpha = next_alpha

        total = float(alpha.sum())
        if not np.isfinite(total) or total <= 0:
            alpha = np.zeros(
                (state_count, max_duration),
                dtype=float,
            )
            for state in range(state_count):
                alpha[state, :] = (
                    float(model.initial_probabilities[state])
                    * model.duration_pmf[state]
                    * float(likelihood[state])
                )
            total = float(alpha.sum())

        if total <= 0:
            continue

        alpha = alpha / total
        state_probability = alpha.sum(axis=1)
        posterior[index, :] = state_probability

        down_probability = float(state_probability[0])
        neutral_probability = float(state_probability[1])
        previous_up = float(previous_state[2])
        score[index] = previous_up * (
            down_probability + 0.5 * neutral_probability
        )
        previous_state = state_probability

    return pd.DataFrame(
        {
            "hsmm_down_probability": posterior[:, 0],
            "hsmm_neutral_probability": posterior[:, 1],
            "hsmm_up_probability": posterior[:, 2],
            "hsmm_reversal_score": score,
        },
        index=frame.index,
    )


def _fit_hsmm_models(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    *,
    random_state: int,
) -> dict[str, HSMMAssetModel]:
    models: dict[str, HSMMAssetModel] = {}
    for symbol in symbols:
        model = _fit_hsmm_asset(
            frames[symbol],
            train_dates,
            random_state=int(random_state),
        )
        if model is not None:
            models[symbol] = model
    return models


def _precalcular_hsmm_scores(
    models: dict[str, HSMMAssetModel],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        symbol: _filter_hsmm_asset(frames[symbol], models[symbol])
        for symbol in symbols
        if symbol in models
    }


def calibrar_hsmm(
    filtered: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
) -> HSMMCalibrationResult:
    targets: list[int] = []
    scores: list[float] = []

    for symbol in symbols:
        signal_frame = filtered.get(symbol)
        if signal_frame is None:
            continue
        frame = frames[symbol].reindex(calibration_dates)
        signals = signal_frame.reindex(calibration_dates)
        eligible = frame["hsmm_eligible"].fillna(False).astype(bool)
        target = pd.to_numeric(
            frame["forward_down_reversal"],
            errors="coerce",
        )
        score = pd.to_numeric(
            signals["hsmm_reversal_score"],
            errors="coerce",
        )
        valid = eligible & target.notna() & score.notna()
        if not bool(valid.any()):
            continue
        targets.extend(target.loc[valid].astype(int).tolist())
        scores.extend(score.loc[valid].astype(float).tolist())

    if not targets:
        return HSMMCalibrationResult(
            threshold=0.25,
            fbeta_05=None,
            balanced_accuracy=None,
            precision=None,
            recall=None,
            predicted_positives=0,
            positives=0,
            negatives=0,
            observations=0,
            fitted_assets=int(len(filtered)),
        )

    y_true = np.asarray(targets, dtype=int)
    values = np.asarray(scores, dtype=float)
    best: tuple[
        float,
        float,
        float,
        dict[str, float | int | None],
    ] | None = None

    for threshold in HSMM_SCORE_THRESHOLDS:
        metrics = _metricas_classificacao(
            y_true,
            values,
            float(threshold),
        )
        if int(metrics["predicted_positives"] or 0) < MINIMUM_CALIBRATION_ALERTS:
            continue
        candidate = (
            float(metrics["fbeta_05"] or 0.0),
            float(metrics["precision"] or 0.0),
            float(threshold),
            metrics,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        threshold = 0.25
        metrics = _metricas_classificacao(
            y_true,
            values,
            threshold,
        )
    else:
        threshold = float(best[2])
        metrics = best[3]

    return HSMMCalibrationResult(
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
        positives=int((y_true == 1).sum()),
        negatives=int((y_true == 0).sum()),
        observations=int(len(y_true)),
        fitted_assets=int(len(filtered)),
    )


def _envolver_politica_hsmm(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    filtered: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    score_threshold: float,
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
        down_probability = None
        neutral_probability = None
        up_probability = None
        eligible = False

        if symbol is not None and key in frames[symbol].index:
            eligible = bool(frames[symbol].at[key, "hsmm_eligible"])
            signal_frame = filtered.get(symbol)
            if signal_frame is not None and key in signal_frame.index:
                row = signal_frame.loc[key]
                for name, target in (
                    ("hsmm_reversal_score", "score"),
                    ("hsmm_down_probability", "down"),
                    ("hsmm_neutral_probability", "neutral"),
                    ("hsmm_up_probability", "up"),
                ):
                    value = row.get(name)
                    if pd.notna(value):
                        if target == "score":
                            score = float(value)
                        elif target == "down":
                            down_probability = float(value)
                        elif target == "neutral":
                            neutral_probability = float(value)
                        else:
                            up_probability = float(value)

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
            and streak >= int(HSMM_CONFIRMATION_SESSIONS)
        )

        diagnostic.update(
            {
                "hsmm_schema_version": 1,
                "hsmm_research_version": RESEARCH_VERSION,
                "hsmm_reversal_score": score,
                "hsmm_score_threshold": float(score_threshold),
                "hsmm_down_probability": down_probability,
                "hsmm_neutral_probability": neutral_probability,
                "hsmm_up_probability": up_probability,
                "hsmm_eligible": bool(eligible),
                "hsmm_confirmation_streak": int(streak),
                "hsmm_confirmation_required": int(
                    HSMM_CONFIRMATION_SESSIONS
                ),
                "hsmm_base_target_asset": (
                    "CASH"
                    if int(base_target) <= 0
                    else symbols[int(base_target) - 1]
                ),
                "hsmm_exit_triggered": triggered,
            }
        )

        if not triggered:
            if not control_holds and symbol is not None:
                streak_by_symbol[symbol] = 0
            return int(base_target), float(base_score)

        streak_by_symbol[symbol] = 0
        diagnostic["hsmm_base_reason"] = diagnostic.get(
            "decision_reason"
        )
        diagnostic.update(
            {
                "decision_reason": "HSMM_REGIME_TRANSITION_EXIT",
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

def _ajustar_modelos_bottom_turn(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("Bottom-Turn research requires lightgbm.") from exc

    settings = _configuracoes_lightgbm(config)
    minimum_rows = int(config.rotation_minimum_training_rows)
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        frame = frames[symbol].reindex(train_dates)
        eligible = frame["bottom_turn_eligible"].fillna(False).astype(bool)
        train = frame.loc[eligible].dropna(
            subset=["forward_up_reversal", *MODEL_FEATURES]
        )
        target = train["forward_up_reversal"].astype(int)
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


def _probabilidades_bottom_turn(
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
        eligible = frame["bottom_turn_eligible"].fillna(False).astype(bool)
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


def calibrar_limiar_bottom_turn(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    *,
    candidates: Iterable[float] = PROBABILITY_THRESHOLD_CANDIDATES,
) -> CalibrationResult:
    cache = _probabilidades_bottom_turn(
        models,
        frames,
        symbols,
        calibration_dates,
    )
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
            value = frame.at[key, "forward_up_reversal"]
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


def _envolver_politica_bottom_turn(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    probabilities: dict[pd.Timestamp, dict[str, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    probability_threshold: float,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]],
    activate_only_after_top_turn: bool = False,
    max_wait_sessions: int | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Confirma entradas perto de fundo sem substituir a escolha do Control.

    No modo v2, o gate so e armado por uma saida Top-Turn e pode bloquear no
    maximo max_wait_sessions sessoes de decisao seguintes.
    """
    streak_by_symbol: dict[str, int] = {}
    gate_armed = False
    gate_wait_sessions = 0

    if activate_only_after_top_turn:
        if max_wait_sessions is None:
            max_wait_sessions = int(BOTTOM_TURN_MAX_WAIT_SESSIONS)
        if int(max_wait_sessions) <= 0:
            raise ValueError("Bottom-Turn max_wait_sessions must be positive.")

    def reset_streaks() -> None:
        for known_symbol in list(streak_by_symbol):
            streak_by_symbol[known_symbol] = 0

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        nonlocal gate_armed, gate_wait_sessions

        base_target, base_score = base_policy(
            timestamp,
            current_position,
            holding_days,
        )
        key = pd.Timestamp(timestamp)
        diagnostic = decision_diagnostics.setdefault(key, {})

        top_exit_triggered = bool(
            diagnostic.get("directional_change_exit_triggered", False)
        )

        if (
            activate_only_after_top_turn
            and current_position > 0
            and int(base_target) <= 0
            and top_exit_triggered
        ):
            gate_armed = True
            gate_wait_sessions = 0
            reset_streaks()
            diagnostic.update(
                {
                    "bottom_turn_schema_version": 2,
                    "bottom_turn_research_version": RESEARCH_VERSION,
                    "bottom_turn_gate_armed_event": True,
                    "bottom_turn_gate_active": True,
                    "bottom_turn_gate_wait_session": 0,
                    "bottom_turn_gate_max_wait_sessions": int(
                        max_wait_sessions
                    ),
                    "bottom_turn_gate_expired": False,
                    "bottom_turn_entry_candidate": False,
                    "bottom_turn_entry_triggered": False,
                    "bottom_turn_entry_blocked": False,
                }
            )
            return int(base_target), float(base_score)

        if current_position > 0:
            if activate_only_after_top_turn:
                gate_armed = False
                gate_wait_sessions = 0
            reset_streaks()
            diagnostic.update(
                {
                    "bottom_turn_gate_armed_event": False,
                    "bottom_turn_gate_active": False,
                    "bottom_turn_gate_wait_session": 0,
                    "bottom_turn_gate_expired": False,
                    "bottom_turn_entry_candidate": False,
                    "bottom_turn_entry_triggered": False,
                    "bottom_turn_entry_blocked": False,
                }
            )
            return int(base_target), float(base_score)

        if activate_only_after_top_turn and not gate_armed:
            reset_streaks()
            diagnostic.update(
                {
                    "bottom_turn_gate_armed_event": False,
                    "bottom_turn_gate_active": False,
                    "bottom_turn_gate_wait_session": 0,
                    "bottom_turn_gate_expired": False,
                    "bottom_turn_entry_candidate": False,
                    "bottom_turn_entry_triggered": False,
                    "bottom_turn_entry_blocked": False,
                }
            )
            return int(base_target), float(base_score)

        if activate_only_after_top_turn:
            gate_wait_sessions += 1
            wait_session = int(gate_wait_sessions)
            max_wait = int(max_wait_sessions)
        else:
            wait_session = 0
            max_wait = 0

        if int(base_target) <= 0:
            expired = bool(
                activate_only_after_top_turn
                and wait_session >= max_wait
            )
            diagnostic.update(
                {
                    "bottom_turn_gate_armed_event": False,
                    "bottom_turn_gate_active": bool(
                        gate_armed if activate_only_after_top_turn else True
                    ),
                    "bottom_turn_gate_wait_session": wait_session,
                    "bottom_turn_gate_max_wait_sessions": (
                        max_wait if activate_only_after_top_turn else None
                    ),
                    "bottom_turn_gate_expired": expired,
                    "bottom_turn_entry_candidate": False,
                    "bottom_turn_entry_triggered": False,
                    "bottom_turn_entry_blocked": False,
                }
            )
            if expired:
                gate_armed = False
                gate_wait_sessions = 0
                reset_streaks()
            return int(base_target), float(base_score)

        symbol = symbols[int(base_target) - 1]
        for known_symbol in list(streak_by_symbol):
            if known_symbol != symbol:
                streak_by_symbol[known_symbol] = 0

        probability = probabilities.get(key, {}).get(symbol)
        eligible = bool(
            key in frames[symbol].index
            and frames[symbol].at[key, "bottom_turn_eligible"]
        )
        above_threshold = bool(
            eligible
            and probability is not None
            and np.isfinite(float(probability))
            and float(probability) >= float(probability_threshold)
        )
        if above_threshold:
            streak_by_symbol[symbol] = int(
                streak_by_symbol.get(symbol, 0)
            ) + 1
        else:
            streak_by_symbol[symbol] = 0

        streak = int(streak_by_symbol.get(symbol, 0))
        triggered = bool(
            above_threshold
            and streak >= int(BOTTOM_TURN_CONFIRMATION_SESSIONS)
        )
        expires_after_decision = bool(
            activate_only_after_top_turn
            and wait_session >= max_wait
            and not triggered
        )

        diagnostic.update(
            {
                "bottom_turn_schema_version": (
                    2 if activate_only_after_top_turn else 1
                ),
                "bottom_turn_research_version": RESEARCH_VERSION,
                "bottom_turn_probability": (
                    float(probability) if probability is not None else None
                ),
                "bottom_turn_probability_threshold": float(
                    probability_threshold
                ),
                "bottom_turn_eligible": bool(eligible),
                "bottom_turn_confirmation_streak": int(streak),
                "bottom_turn_confirmation_required": int(
                    BOTTOM_TURN_CONFIRMATION_SESSIONS
                ),
                "bottom_turn_base_target_asset": symbol,
                "bottom_turn_gate_armed_event": False,
                "bottom_turn_gate_active": bool(
                    gate_armed if activate_only_after_top_turn else True
                ),
                "bottom_turn_gate_wait_session": wait_session,
                "bottom_turn_gate_max_wait_sessions": (
                    max_wait if activate_only_after_top_turn else None
                ),
                "bottom_turn_gate_expired": expires_after_decision,
                "bottom_turn_entry_candidate": True,
                "bottom_turn_entry_triggered": triggered,
                "bottom_turn_entry_blocked": not triggered,
            }
        )

        if triggered:
            reset_streaks()
            if activate_only_after_top_turn:
                gate_armed = False
                gate_wait_sessions = 0
            diagnostic["decision_reason"] = "BOTTOM_TURN_CONFIRMED_ENTRY"
            diagnostic["final_action_asset"] = symbol
            diagnostic["final_action_score"] = float(base_score)
            diagnostic["decision_is_entry"] = True
            diagnostic["decision_is_exit_to_cash"] = False
            return int(base_target), float(base_score)

        diagnostic["bottom_turn_base_reason"] = diagnostic.get(
            "decision_reason"
        )
        diagnostic.update(
            {
                "decision_reason": "BOTTOM_TURN_WAIT_IN_CASH",
                "final_action_asset": "CASH",
                "final_action_score": 0.0,
                "decision_is_rotation": False,
                "decision_is_entry": False,
                "decision_is_exit_to_cash": False,
            }
        )

        if expires_after_decision:
            gate_armed = False
            gate_wait_sessions = 0
            reset_streaks()
        return 0, 0.0

    return policy



def _envolver_politica_cooldown_pos_top_turn(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]],
    cooldown_sessions: int = FIXED_COOLDOWN_SESSIONS,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Ablacao sem ML: espera fixa apos uma saida Top-Turn.

    O cooldown e armado somente por uma saida Top-Turn real. As proximas
    cooldown_sessions sessoes de decisao permanecem em CASH; na sessao
    seguinte a politica Top-Turn volta a operar normalmente.
    """
    if int(cooldown_sessions) <= 0:
        raise ValueError("cooldown_sessions must be positive.")

    gate_armed = False
    gate_wait_sessions = 0

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        nonlocal gate_armed, gate_wait_sessions

        base_target, base_score = base_policy(
            timestamp,
            current_position,
            holding_days,
        )
        key = pd.Timestamp(timestamp)
        diagnostic = decision_diagnostics.setdefault(key, {})
        top_exit_triggered = bool(
            diagnostic.get("directional_change_exit_triggered", False)
        )

        if (
            current_position > 0
            and int(base_target) <= 0
            and top_exit_triggered
        ):
            gate_armed = True
            gate_wait_sessions = 0
            diagnostic.update(
                {
                    "cooldown_schema_version": 1,
                    "cooldown_research_version": RESEARCH_VERSION,
                    "cooldown_gate_armed_event": True,
                    "cooldown_gate_active": True,
                    "cooldown_wait_session": 0,
                    "cooldown_max_wait_sessions": int(cooldown_sessions),
                    "cooldown_gate_expired": False,
                    "cooldown_entry_blocked": False,
                }
            )
            return int(base_target), float(base_score)

        if current_position > 0:
            gate_armed = False
            gate_wait_sessions = 0
            diagnostic.update(
                {
                    "cooldown_gate_armed_event": False,
                    "cooldown_gate_active": False,
                    "cooldown_wait_session": 0,
                    "cooldown_max_wait_sessions": int(cooldown_sessions),
                    "cooldown_gate_expired": False,
                    "cooldown_entry_blocked": False,
                }
            )
            return int(base_target), float(base_score)

        if not gate_armed:
            diagnostic.update(
                {
                    "cooldown_gate_armed_event": False,
                    "cooldown_gate_active": False,
                    "cooldown_wait_session": 0,
                    "cooldown_max_wait_sessions": int(cooldown_sessions),
                    "cooldown_gate_expired": False,
                    "cooldown_entry_blocked": False,
                }
            )
            return int(base_target), float(base_score)

        gate_wait_sessions += 1
        wait_session = int(gate_wait_sessions)
        expired = bool(wait_session >= int(cooldown_sessions))
        blocked = bool(int(base_target) > 0)

        diagnostic.update(
            {
                "cooldown_schema_version": 1,
                "cooldown_research_version": RESEARCH_VERSION,
                "cooldown_gate_armed_event": False,
                "cooldown_gate_active": True,
                "cooldown_wait_session": wait_session,
                "cooldown_max_wait_sessions": int(cooldown_sessions),
                "cooldown_gate_expired": expired,
                "cooldown_entry_blocked": blocked,
            }
        )

        if blocked:
            diagnostic["cooldown_base_reason"] = diagnostic.get(
                "decision_reason"
            )
            diagnostic.update(
                {
                    "decision_reason": "TOP_TURN_FIXED_COOLDOWN_WAIT",
                    "final_action_asset": "CASH",
                    "final_action_score": 0.0,
                    "decision_is_rotation": False,
                    "decision_is_entry": False,
                    "decision_is_exit_to_cash": False,
                }
            )

        if expired:
            gate_armed = False
            gate_wait_sessions = 0

        if blocked:
            return 0, 0.0
        return int(base_target), float(base_score)

    return policy


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


def executar_bottom_turn_lightgbm(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    include_top_turn_exit: bool = False,
    post_top_turn_only: bool = False,
    max_wait_sessions: int | None = None,
    include_fixed_cooldown_ablation: bool = False,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> Any:
    """Executa Bottom-Turn puro ou ciclo combinado Top-Turn + Bottom-Turn."""
    if int(config.rotation_model_repetitions) != 1:
        raise ValueError("Bottom-Turn research requires one repetition.")
    if include_fixed_cooldown_ablation and not include_top_turn_exit:
        raise ValueError(
            "Fixed cooldown ablation requires include_top_turn_exit=True."
        )

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
    cooldown_policies: dict[int, Callable] = {}
    cooldown_diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
    bottom_calibration_rows: list[dict[str, Any]] = []
    bottom_fit_rows: list[dict[str, Any]] = []
    top_calibration_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []

    total_folds = len(folds)
    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        label = "Top+Bottom" if include_top_turn_exit else "Bottom-Turn"
        if progress_callback is not None:
            progress_callback(
                5.0 + 75.0 * ((fold_position - 1) / max(1, total_folds)),
                f"{label} fold {fold_position}/{total_folds} training",
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
            phase=f"bottom_turn_fold_{fold_id}_utility_calibration",
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

        bottom_calibration_models, calibration_fit = (
            _ajustar_modelos_bottom_turn(
                frames,
                symbols,
                train_dates,
                config,
            )
        )
        bottom_calibration = calibrar_limiar_bottom_turn(
            bottom_calibration_models,
            frames,
            symbols,
            calibration_dates,
        )
        bottom_calibration_rows.append(
            bottom_calibration.as_dict(fold_id=fold_id)
        )
        for row in calibration_fit:
            bottom_fit_rows.append(
                {"fold_id": fold_id, "phase": "calibration", **row}
            )

        top_calibration = None
        if include_top_turn_exit:
            top_calibration_models, _ = _ajustar_modelos_top_turn(
                frames,
                symbols,
                train_dates,
                config,
            )
            top_calibration = calibrar_limiar(
                top_calibration_models,
                frames,
                symbols,
                calibration_dates,
            )
            top_calibration_rows.append(
                top_calibration.as_dict(fold_id=fold_id)
            )

        final_utility_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"bottom_turn_fold_{fold_id}_utility_final",
        )
        final_bottom_models, final_fit = _ajustar_modelos_bottom_turn(
            frames,
            symbols,
            final_fit_dates,
            config,
        )
        for row in final_fit:
            bottom_fit_rows.append(
                {"fold_id": fold_id, "phase": "final", **row}
            )

        final_top_models: dict[str, Any] = {}
        if include_top_turn_exit:
            final_top_models, _ = _ajustar_modelos_top_turn(
                frames,
                symbols,
                final_fit_dates,
                config,
            )

        decision_dates = pd.DatetimeIndex(fold["decision_dates"])
        utility_cache, _ = _precalcular_utilidades_modelo(
            final_utility_models,
            frames,
            symbols,
            decision_dates,
            config,
        )
        bottom_probability_cache = _probabilidades_bottom_turn(
            final_bottom_models,
            frames,
            symbols,
            decision_dates,
        )
        top_probability_cache = (
            _probabilidades(
                final_top_models,
                frames,
                symbols,
                decision_dates,
            )
            if include_top_turn_exit
            else {}
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

        entry_base_policy = base_policy
        if include_top_turn_exit:
            if top_calibration is None:
                raise RuntimeError("Top-Turn calibration missing.")
            entry_base_policy = _envolver_politica_top_turn(
                base_policy,
                probabilities=top_probability_cache,
                frames=frames,
                symbols=symbols,
                probability_threshold=float(top_calibration.threshold),
                config=config,
                decision_diagnostics=diagnostics,
            )

        policies[fold_id] = _envolver_politica_bottom_turn(
            entry_base_policy,
            probabilities=bottom_probability_cache,
            frames=frames,
            symbols=symbols,
            probability_threshold=float(bottom_calibration.threshold),
            decision_diagnostics=diagnostics,
            activate_only_after_top_turn=bool(post_top_turn_only),
            max_wait_sessions=max_wait_sessions,
        )

        if include_fixed_cooldown_ablation:
            if top_calibration is None:
                raise RuntimeError("Top-Turn calibration missing for cooldown.")
            cooldown_base_policy = _politica_utilidade(
                final_utility_models,
                frames,
                symbols,
                config,
                effective_margin,
                decision_diagnostics=cooldown_diagnostics,
                fold_id=fold_id,
                calibrated_switch_margin=float(best_candidate),
                utility_cache=utility_cache,
            )
            cooldown_top_policy = _envolver_politica_top_turn(
                cooldown_base_policy,
                probabilities=top_probability_cache,
                frames=frames,
                symbols=symbols,
                probability_threshold=float(top_calibration.threshold),
                config=config,
                decision_diagnostics=cooldown_diagnostics,
            )
            cooldown_policies[fold_id] = (
                _envolver_politica_cooldown_pos_top_turn(
                    cooldown_top_policy,
                    decision_diagnostics=cooldown_diagnostics,
                    cooldown_sessions=int(FIXED_COOLDOWN_SESSIONS),
                )
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
            (
                "Top+Bottom OOS replay"
                if include_top_turn_exit
                else "Bottom-Turn OOS replay"
            ),
            total_folds,
        )

    backend = (
        "top_bottom_turn_lightgbm"
        if include_top_turn_exit
        else "bottom_turn_lightgbm"
    )
    result = _simular_exato(
        backend,
        _politica_agendada(policies, decision_to_fold),
        frames,
        symbols,
        all_decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label=(
            "Directional Change + LightGBM Top+Bottom"
            if include_top_turn_exit
            else "Directional Change + LightGBM Bottom-Turn"
        ),
        method_line=(
            "- Bottom-Turn uses Directional Change + LightGBM to gate only "
            "CASH-to-asset entries already selected by the Control policy. "
            "It estimates whether a relevant upward first-passage move occurs "
            "before renewed downside continuation near a recent low; two "
            "consecutive confirmations are required."
        ),
    )
    result.backend = backend

    bottom_probabilities = pd.to_numeric(
        result.predictions.get(
            "bottom_turn_probability",
            pd.Series(index=result.predictions.index, dtype=float),
        ),
        errors="coerce",
    )
    bottom_triggers = result.predictions.get(
        "bottom_turn_entry_triggered",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)
    bottom_blocks = result.predictions.get(
        "bottom_turn_entry_blocked",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)
    top_triggers = result.predictions.get(
        "directional_change_exit_triggered",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)

    result.metrics.update(
        {
            "backend": backend,
            "model_family": "directional_change_lightgbm_bottom_turn",
            "strategy_label": (
                "Top-Turn + Bottom-Turn"
                if include_top_turn_exit
                else "Bottom-Turn"
            ),
            "bottom_turn_research_version": RESEARCH_VERSION,
            "bottom_turn_horizon_sessions": int(
                BOTTOM_TURN_HORIZON_SESSIONS
            ),
            "bottom_turn_near_low_20": float(BOTTOM_TURN_NEAR_LOW_20),
            "bottom_turn_confirmation_sessions": int(
                BOTTOM_TURN_CONFIRMATION_SESSIONS
            ),
            "bottom_turn_probability_threshold_mean": float(
                np.mean(
                    [
                        row["probability_threshold"]
                        for row in bottom_calibration_rows
                    ]
                )
            ),
            "bottom_turn_entry_triggers": int(bottom_triggers.sum()),
            "bottom_turn_entry_blocks": int(bottom_blocks.sum()),
            "bottom_turn_probability_observations": int(
                bottom_probabilities.notna().sum()
            ),
            "bottom_turn_calibration": bottom_calibration_rows,
            "bottom_turn_fit": bottom_fit_rows,
            "combined_top_turn_enabled": bool(include_top_turn_exit),
            "combined_top_turn_exit_triggers": int(top_triggers.sum()),
            "combined_top_turn_calibration": top_calibration_rows,
            "bottom_turn_post_top_only": bool(post_top_turn_only),
            "bottom_turn_max_wait_sessions": (
                int(max_wait_sessions)
                if max_wait_sessions is not None
                else (
                    int(BOTTOM_TURN_MAX_WAIT_SESSIONS)
                    if post_top_turn_only
                    else None
                )
            ),
            "bottom_turn_gate_expirations": int(
                result.predictions.get(
                    "bottom_turn_gate_expired",
                    pd.Series(
                        False,
                        index=result.predictions.index,
                        dtype=bool,
                    ),
                ).fillna(False).astype(bool).sum()
            ),
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

    cooldown_result = None
    if include_fixed_cooldown_ablation:
        cooldown_result = _simular_exato(
            "top_turn_fixed_cooldown_5",
            _politica_agendada(cooldown_policies, decision_to_fold),
            frames,
            symbols,
            all_decision_dates,
            config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=cooldown_diagnostics,
            model_label="Top-Turn + Fixed Cooldown 5",
            method_line=(
                "- Ablation baseline: after each real Top-Turn exit, remain "
                "in CASH for exactly five decision sessions. No Bottom-Turn "
                "probability, threshold or confirmation is consulted."
            ),
        )
        cooldown_result.backend = "top_turn_fixed_cooldown_5"
        cooldown_blocks = cooldown_result.predictions.get(
            "cooldown_entry_blocked",
            pd.Series(
                False,
                index=cooldown_result.predictions.index,
                dtype=bool,
            ),
        ).fillna(False).astype(bool)
        cooldown_arms = cooldown_result.predictions.get(
            "cooldown_gate_armed_event",
            pd.Series(
                False,
                index=cooldown_result.predictions.index,
                dtype=bool,
            ),
        ).fillna(False).astype(bool)
        cooldown_expirations = cooldown_result.predictions.get(
            "cooldown_gate_expired",
            pd.Series(
                False,
                index=cooldown_result.predictions.index,
                dtype=bool,
            ),
        ).fillna(False).astype(bool)
        cooldown_result.metrics.update(
            {
                "backend": "top_turn_fixed_cooldown_5",
                "model_family": "top_turn_fixed_cooldown_ablation",
                "strategy_label": "Top-Turn + Fixed Cooldown 5",
                "cooldown_research_version": RESEARCH_VERSION,
                "cooldown_sessions": int(FIXED_COOLDOWN_SESSIONS),
                "cooldown_uses_bottom_turn_ml": False,
                "cooldown_gate_armed_events": int(cooldown_arms.sum()),
                "cooldown_entry_blocks": int(cooldown_blocks.sum()),
                "cooldown_gate_expirations": int(
                    cooldown_expirations.sum()
                ),
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": _desempenho_folds(
                    cooldown_result.predictions,
                    folds,
                    float(config.initial_capital),
                ),
                "calendar_source_asset": calendar_source_asset,
                "requested_compute_device": "cpu",
                "effective_compute_device": "cpu",
            }
        )
        for row in cooldown_result.metrics["walk_forward_folds"]:
            row.update(margin_by_fold.get(int(row["fold_id"]), {}))

    if progress_callback is not None:
        progress_callback(
            100.0,
            (
                "Top+Bottom + cooldown ablation completed"
                if include_fixed_cooldown_ablation
                else (
                    "Top+Bottom completed"
                    if include_top_turn_exit
                    else "Bottom-Turn completed"
                )
            ),
            total_folds,
        )
    if include_fixed_cooldown_ablation:
        return result, cooldown_result
    return result


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
        candidate_scores: list[tuple[float, float]] = []
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
            candidate_scores.append((float(candidate), float(score)))

        margin_selection = _selecionar_switch_margin_fold(
            config,
            fold_id,
            candidate_scores,
        )
        best_candidate = float(
            margin_selection["selected_candidate_margin"]
        )
        best_score = float(
            margin_selection["selected_calibration_score"]
        )
        auto_best_candidate = float(
            margin_selection["auto_candidate_margin"]
        )
        auto_best_score = float(
            margin_selection["auto_calibration_score"]
        )

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
                "auto_calibrated_candidate_margin": float(
                    auto_best_candidate
                ),
                "auto_calibration_risk_adjusted_score": float(
                    auto_best_score
                ),
                "margin_selection_source": str(
                    margin_selection["selection_source"]
                ),
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

def executar_hazard_survival_overlay(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> Any:
    if int(config.rotation_model_repetitions) != 1:
        raise ValueError("Hazard/Survival comparison requires one repetition.")

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
    margin_rows: list[dict[str, Any]] = []

    total_folds = len(folds)
    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        if progress_callback is not None:
            progress_callback(
                5.0 + 75.0 * ((fold_position - 1) / max(1, total_folds)),
                f"Hazard fold {fold_position}/{total_folds} training",
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
            phase=f"hazard_fold_{fold_id}_utility_calibration",
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

        calibration_model, training_rows, training_events = _fit_hazard_model(
            frames,
            symbols,
            train_dates,
            random_state=int(config.random_state) + fold_id,
        )
        calibration_scores = _predict_hazard_scores(
            calibration_model,
            frames,
            symbols,
        )
        calibration = calibrar_hazard(
            calibration_scores,
            frames,
            symbols,
            calibration_dates,
            training_rows=training_rows,
            training_events=training_events,
        )
        calibration_rows.append(
            calibration.as_dict(fold_id=fold_id)
        )

        final_utility_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"hazard_fold_{fold_id}_utility_final",
        )
        final_hazard_model, final_training_rows, final_training_events = (
            _fit_hazard_model(
                frames,
                symbols,
                final_fit_dates,
                random_state=int(config.random_state) + 1000 + fold_id,
            )
        )
        final_scores = _predict_hazard_scores(
            final_hazard_model,
            frames,
            symbols,
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
        policies[fold_id] = _envolver_politica_hazard(
            base_policy,
            score_cache=final_scores,
            frames=frames,
            symbols=symbols,
            score_threshold=float(calibration.threshold),
            config=config,
            decision_diagnostics=diagnostics,
        )
        margin_rows.append(
            {
                "fold_id": fold_id,
                "calibrated_candidate_margin": float(best_candidate),
                "effective_switch_margin": float(effective_margin),
                "calibration_risk_adjusted_score": float(best_score),
                "hazard_training_rows": int(final_training_rows),
                "hazard_training_events": int(final_training_events),
            }
        )

    if progress_callback is not None:
        progress_callback(
            85.0,
            "Hazard/Survival OOS replay",
            total_folds,
        )

    result = _simular_exato(
        "hazard_survival_overlay",
        _politica_agendada(policies, decision_to_fold),
        frames,
        symbols,
        all_decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Discrete-Time Hazard / Survival",
        method_line=(
            "- Cause-specific discrete-time logistic hazard models downside "
            "events over five sessions. Continuation is treated as a "
            "competing censoring event and no-event paths are right-censored "
            "at the horizon. Threshold selection uses calibration-only data."
        ),
    )
    result.backend = "hazard_survival_overlay"

    scores = pd.to_numeric(
        result.predictions.get(
            "hazard_cumulative_downside_probability",
            pd.Series(index=result.predictions.index, dtype=float),
        ),
        errors="coerce",
    )
    triggers = result.predictions.get(
        "hazard_exit_triggered",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)

    result.metrics.update(
        {
            "backend": "hazard_survival_overlay",
            "model_family": "discrete_time_cause_specific_hazard",
            "strategy_label": "Hazard/Survival",
            "hazard_research_version": RESEARCH_VERSION,
            "hazard_horizon_sessions": int(HAZARD_HORIZON_SESSIONS),
            "hazard_score_threshold_candidates": list(
                HAZARD_SCORE_THRESHOLDS
            ),
            "hazard_confirmation_sessions": int(
                HAZARD_CONFIRMATION_SESSIONS
            ),
            "hazard_exit_triggers": int(triggers.sum()),
            "hazard_score_observations": int(scores.notna().sum()),
            "hazard_calibration": calibration_rows,
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
            "Hazard/Survival completed",
            total_folds,
        )
    return result


def executar_hsmm_overlay(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> Any:
    if int(config.rotation_model_repetitions) != 1:
        raise ValueError("HSMM comparison requires one repetition.")

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
    margin_rows: list[dict[str, Any]] = []

    total_folds = len(folds)
    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        if progress_callback is not None:
            progress_callback(
                5.0 + 75.0 * ((fold_position - 1) / max(1, total_folds)),
                f"HSMM fold {fold_position}/{total_folds} training",
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
            phase=f"hsmm_fold_{fold_id}_utility_calibration",
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

        calibration_models = _fit_hsmm_models(
            frames,
            symbols,
            train_dates,
            random_state=int(config.random_state) + fold_id,
        )
        calibration_filtered = _precalcular_hsmm_scores(
            calibration_models,
            frames,
            symbols,
        )
        calibration = calibrar_hsmm(
            calibration_filtered,
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
            phase=f"hsmm_fold_{fold_id}_utility_final",
        )
        final_hsmm_models = _fit_hsmm_models(
            frames,
            symbols,
            final_fit_dates,
            random_state=int(config.random_state) + 1000 + fold_id,
        )
        final_filtered = _precalcular_hsmm_scores(
            final_hsmm_models,
            frames,
            symbols,
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
        policies[fold_id] = _envolver_politica_hsmm(
            base_policy,
            filtered=final_filtered,
            frames=frames,
            symbols=symbols,
            score_threshold=float(calibration.threshold),
            config=config,
            decision_diagnostics=diagnostics,
        )
        margin_rows.append(
            {
                "fold_id": fold_id,
                "calibrated_candidate_margin": float(best_candidate),
                "effective_switch_margin": float(effective_margin),
                "calibration_risk_adjusted_score": float(best_score),
                "hsmm_fitted_assets": int(len(final_hsmm_models)),
            }
        )

    if progress_callback is not None:
        progress_callback(
            85.0,
            "HSMM OOS replay",
            total_folds,
        )

    result = _simular_exato(
        "hsmm_overlay",
        _politica_agendada(policies, decision_to_fold),
        frames,
        symbols,
        all_decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Explicit-Duration HSMM",
        method_line=(
            "- HSMM fits three Gaussian latent regimes on training-only "
            "standardized return and trend observations, estimates explicit "
            "state-duration distributions, and filters regime probabilities "
            "causally. A transition away from the prior up regime near a "
            "recent high requires two confirmations before exit to CASH."
        ),
    )
    result.backend = "hsmm_overlay"

    scores = pd.to_numeric(
        result.predictions.get(
            "hsmm_reversal_score",
            pd.Series(index=result.predictions.index, dtype=float),
        ),
        errors="coerce",
    )
    triggers = result.predictions.get(
        "hsmm_exit_triggered",
        pd.Series(False, index=result.predictions.index, dtype=bool),
    ).fillna(False).astype(bool)

    result.metrics.update(
        {
            "backend": "hsmm_overlay",
            "model_family": "explicit_duration_hsmm",
            "strategy_label": "Explicit-Duration HSMM",
            "hsmm_research_version": RESEARCH_VERSION,
            "hsmm_state_count": int(HSMM_STATE_COUNT),
            "hsmm_max_duration": int(HSMM_MAX_DURATION),
            "hsmm_score_threshold_candidates": list(
                HSMM_SCORE_THRESHOLDS
            ),
            "hsmm_confirmation_sessions": int(
                HSMM_CONFIRMATION_SESSIONS
            ),
            "hsmm_exit_triggers": int(triggers.sum()),
            "hsmm_score_observations": int(scores.notna().sum()),
            "hsmm_calibration": calibration_rows,
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
            "HSMM completed",
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

def calcular_bottom_entry(
    trades: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    *,
    oos_start: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Mede a qualidade de entradas CASH -> ativo em relacao ao fundo recente."""
    if trades is None or trades.empty:
        return {"cash_entries": 0}, pd.DataFrame()

    ordered = trades.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered = ordered.sort_values("timestamp", ignore_index=True)
    if oos_start is not None:
        start = pd.Timestamp(oos_start)
        start = (
            start.tz_localize("UTC")
            if start.tzinfo is None
            else start.tz_convert("UTC")
        )
    else:
        start = pd.Timestamp(ordered["timestamp"].min())

    rows: list[dict[str, Any]] = []
    cash_since = start

    for _, trade in ordered.iterrows():
        action = str(trade.get("action") or "").upper()
        from_asset = str(trade.get("rotation_from_asset") or "").upper()
        to_asset = str(trade.get("rotation_to_asset") or "").upper()
        timestamp = pd.Timestamp(trade["timestamp"])

        if action == "SELL" and to_asset == "CASH":
            cash_since = timestamp
            continue

        if action != "BUY" or from_asset != "CASH":
            continue

        asset = str(trade.get("asset") or "")
        frame = frames.get(asset)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local.index = pd.to_datetime(local.index, utc=True)
        local = local.sort_index()

        decision_value = trade.get("decision_timestamp")
        decision_at = (
            pd.Timestamp(decision_value)
            if decision_value is not None and not pd.isna(decision_value)
            else timestamp
        )
        decision_at = (
            decision_at.tz_localize("UTC")
            if decision_at.tzinfo is None
            else decision_at.tz_convert("UTC")
        )

        window = local.loc[
            (local.index >= cash_since)
            & (local.index <= decision_at)
        ]
        lows = pd.to_numeric(window.get("low"), errors="coerce").dropna()
        if lows.empty:
            cash_since = timestamp
            continue

        bottom_at = pd.Timestamp(lows.idxmin())
        bottom_price = float(lows.loc[bottom_at])
        entry_price = float(trade.get("execution_price"))
        raw_distance = (
            float(entry_price / bottom_price - 1.0)
            if bottom_price > 0
            else np.nan
        )
        entry_distance = max(0.0, raw_distance) if np.isfinite(raw_distance) else np.nan

        prior = window.loc[window.index <= bottom_at]
        highs = pd.to_numeric(prior.get("high"), errors="coerce").dropna()
        prior_high = float(highs.max()) if not highs.empty else np.nan
        if (
            np.isfinite(prior_high)
            and prior_high > bottom_price
            and np.isfinite(entry_price)
        ):
            capture = 100.0 * (
                1.0
                - max(0.0, entry_price - bottom_price)
                / (prior_high - bottom_price)
            )
            capture = float(np.clip(capture, 0.0, 100.0))
        else:
            capture = None

        positions = local.index.get_indexer([bottom_at, timestamp])
        if (positions >= 0).all():
            days_from_bottom = int(max(0, positions[1] - positions[0]))
        else:
            days_from_bottom = None

        row: dict[str, Any] = {
            "asset": asset,
            "entry_timestamp": timestamp,
            "decision_timestamp": decision_at,
            "cash_since": cash_since,
            "bottom_price_before_entry": bottom_price,
            "bottom_timestamp_before_entry": bottom_at,
            "entry_price": entry_price,
            "entry_distance_from_bottom_pct": (
                float(entry_distance * 100.0)
                if np.isfinite(entry_distance)
                else None
            ),
            "bottom_capture_pct": capture,
            "days_from_bottom_to_entry": days_from_bottom,
            "walk_forward_fold": trade.get("walk_forward_fold"),
            "decision_reason": trade.get("decision_reason"),
        }

        if timestamp in local.index:
            entry_position = int(local.index.get_loc(timestamp))
            for horizon in (5, 10, 20):
                future_position = entry_position + int(horizon)
                if future_position < len(local):
                    future_close = float(
                        pd.to_numeric(
                            local["close"],
                            errors="coerce",
                        ).iloc[future_position]
                    )
                    row[f"post_entry_return_{horizon}d_pct"] = float(
                        (future_close / entry_price - 1.0) * 100.0
                    )
                    forward = pd.to_numeric(
                        local["low"],
                        errors="coerce",
                    ).iloc[
                        entry_position:
                        future_position + 1
                    ].dropna()
                    row[f"continued_drawdown_{horizon}d_pct"] = (
                        float(
                            (float(forward.min()) / entry_price - 1.0)
                            * 100.0
                        )
                        if not forward.empty
                        else None
                    )
                else:
                    row[f"post_entry_return_{horizon}d_pct"] = None
                    row[f"continued_drawdown_{horizon}d_pct"] = None

        rows.append(row)
        cash_since = timestamp

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"cash_entries": 0}, detail

    def median(column: str) -> float | None:
        values = pd.to_numeric(detail[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else None

    summary: dict[str, Any] = {
        "cash_entries": int(len(detail)),
        "median_entry_distance_from_bottom_pct": median(
            "entry_distance_from_bottom_pct"
        ),
        "median_bottom_capture_pct": median("bottom_capture_pct"),
        "median_days_from_bottom_to_entry": median(
            "days_from_bottom_to_entry"
        ),
    }
    for horizon in (5, 10, 20):
        summary[f"median_post_entry_return_{horizon}d_pct"] = median(
            f"post_entry_return_{horizon}d_pct"
        )
        summary[f"median_continued_drawdown_{horizon}d_pct"] = median(
            f"continued_drawdown_{horizon}d_pct"
        )
    return summary, detail


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



def decompor_distancia_topo_operacoes(
    trades: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    *,
    entry_lookback_sessions: int = 20,
    post_exit_sessions: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Diagnostica onde cada operacao deixou retorno potencial.

    Esta analise e estritamente ex post e nao altera a politica U67. Os quatro
    gaps sao contrafactuais e podem se sobrepor, portanto nao devem ser somados
    como se fossem uma decomposicao contabil fechada.

    Gaps medidos:
    - entrada_tardia: distancia da entrada ao menor low das sessoes anteriores;
    - escolha_ativo: melhor ativo U67 no mesmo intervalo de holding;
    - permanencia_excessiva: pico do ativo durante o holding menos a saida;
    - saida_precoce: melhor high nas sessoes posteriores a saida.

    O objetivo e localizar a principal fonte de oportunidade para a proxima
    pesquisa, e nao criar uma regra usando informacao futura.
    """
    columns = (
        "operation_id",
        "asset",
        "entry_timestamp",
        "exit_timestamp",
        "holding_bars",
        "walk_forward_fold",
        "entry_price",
        "exit_price",
        "quantity",
        "actual_position_return_pct",
        "entry_reference_low",
        "entry_reference_low_timestamp",
        "entry_late_gap_pct",
        "entry_late_gap_proxy_usd",
        "best_same_holding_asset",
        "selected_same_holding_market_return_pct",
        "best_same_holding_market_return_pct",
        "asset_selection_gap_pct",
        "asset_selection_gap_proxy_usd",
        "peak_price_while_held",
        "peak_timestamp_while_held",
        "days_from_peak_to_exit",
        "excess_holding_gap_pct",
        "excess_holding_gap_proxy_usd",
        "post_exit_best_high",
        "post_exit_best_timestamp",
        "early_exit_gap_pct",
        "early_exit_gap_proxy_usd",
        "dominant_gap",
        "dominant_gap_pct",
        "dominant_gap_proxy_usd",
    )
    empty = pd.DataFrame(columns=columns)
    if trades is None or trades.empty:
        return {
            "research_version": TOP_GAP_RESEARCH_VERSION,
            "closed_positions": 0,
            "entry_lookback_sessions": int(entry_lookback_sessions),
            "post_exit_sessions": int(post_exit_sessions),
            "warning": "ex_post_non_additive_counterfactuals",
        }, empty, pd.DataFrame()

    normalized_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local.index = pd.to_datetime(local.index, utc=True)
        normalized_frames[str(symbol)] = local.sort_index()

    rows: list[dict[str, Any]] = []
    closed = trades.loc[
        trades["action"].astype(str).str.upper().isin({"SELL", "FINAL_SELL"})
    ].copy()
    closed = closed.sort_values("timestamp", ignore_index=True)

    for operation_number, trade in enumerate(
        closed.to_dict(orient="records"),
        start=1,
    ):
        asset = str(trade.get("asset") or "")
        frame = normalized_frames.get(asset)
        if frame is None or frame.empty:
            continue

        entry_value = trade.get("entry_timestamp")
        exit_value = trade.get("timestamp")
        if entry_value is None or exit_value is None:
            continue
        entry_at = pd.Timestamp(entry_value)
        exit_at = pd.Timestamp(exit_value)
        entry_at = (
            entry_at.tz_localize("UTC")
            if entry_at.tzinfo is None
            else entry_at.tz_convert("UTC")
        )
        exit_at = (
            exit_at.tz_localize("UTC")
            if exit_at.tzinfo is None
            else exit_at.tz_convert("UTC")
        )
        if entry_at not in frame.index or exit_at not in frame.index:
            continue

        entry_price = float(trade.get("entry_price") or 0.0)
        exit_price = float(trade.get("execution_price") or 0.0)
        quantity = float(trade.get("quantity") or 0.0)
        if entry_price <= 0 or exit_price <= 0 or quantity <= 0:
            continue
        exposure = quantity * entry_price

        action = str(trade.get("action") or "").upper()
        selected_entry_market = float(frame.loc[entry_at, "open"])
        selected_exit_market = float(
            frame.loc[exit_at, "close" if action == "FINAL_SELL" else "open"]
        )
        selected_market_return = (
            selected_exit_market / selected_entry_market - 1.0
            if selected_entry_market > 0
            else np.nan
        )

        # Entrada tardia: somente informacao anterior a entrada, mas usada aqui
        # como diagnostico ex post, nao como gatilho operacional.
        entry_position = int(frame.index.get_loc(entry_at))
        lookback_start = max(
            0,
            entry_position - max(1, int(entry_lookback_sessions)) + 1,
        )
        entry_window = frame.iloc[lookback_start : entry_position + 1]
        entry_lows = pd.to_numeric(
            entry_window.get("low"),
            errors="coerce",
        ).dropna()
        if entry_lows.empty:
            entry_reference_low = np.nan
            entry_reference_at = pd.NaT
            entry_gap = 0.0
        else:
            entry_reference_at = pd.Timestamp(entry_lows.idxmin())
            entry_reference_low = float(entry_lows.loc[entry_reference_at])
            entry_gap = max(
                0.0,
                (entry_price - entry_reference_low) / entry_price,
            )

        # Escolha de ativo: mantem exatamente as datas da operacao observada e
        # pergunta qual ativo U67 teria tido maior retorno bruto no intervalo.
        best_asset = asset
        best_return = selected_market_return
        for candidate, candidate_frame in normalized_frames.items():
            if (
                entry_at not in candidate_frame.index
                or exit_at not in candidate_frame.index
            ):
                continue
            candidate_entry = float(candidate_frame.loc[entry_at, "open"])
            candidate_exit = float(
                candidate_frame.loc[
                    exit_at,
                    "close" if action == "FINAL_SELL" else "open",
                ]
            )
            if candidate_entry <= 0 or not np.isfinite(candidate_exit):
                continue
            candidate_return = candidate_exit / candidate_entry - 1.0
            if (
                not np.isfinite(best_return)
                or candidate_return > best_return
            ):
                best_return = float(candidate_return)
                best_asset = candidate

        selection_gap = (
            max(0.0, float(best_return - selected_market_return))
            if np.isfinite(best_return)
            and np.isfinite(selected_market_return)
            else 0.0
        )

        # Permanencia excessiva: quanto do retorno potencial do proprio ativo
        # foi devolvido entre o pico observado durante o holding e a saida.
        holding = frame.loc[
            (frame.index >= entry_at)
            & (
                (frame.index <= exit_at)
                if action == "FINAL_SELL"
                else (frame.index < exit_at)
            )
        ]
        highs = pd.to_numeric(holding.get("high"), errors="coerce").dropna()
        peak_price = entry_price
        peak_at = entry_at
        if not highs.empty:
            candidate_peak_at = pd.Timestamp(highs.idxmax())
            candidate_peak = float(highs.loc[candidate_peak_at])
            if candidate_peak >= peak_price:
                peak_price = candidate_peak
                peak_at = candidate_peak_at
        excess_holding_gap = max(
            0.0,
            (peak_price - exit_price) / entry_price,
        )
        days_from_peak_to_exit = int(
            ((frame.index > peak_at) & (frame.index <= exit_at)).sum()
        )

        # Saida precoce: upside que apareceu logo depois da venda. O horizonte
        # e deliberadamente curto e fica registrado no resumo.
        exit_position = int(frame.index.get_loc(exit_at))
        post = frame.iloc[
            exit_position + 1 :
            exit_position + 1 + max(1, int(post_exit_sessions))
        ]
        post_highs = pd.to_numeric(post.get("high"), errors="coerce").dropna()
        if post_highs.empty:
            post_high = exit_price
            post_high_at = pd.NaT
            early_exit_gap = 0.0
        else:
            post_high_at = pd.Timestamp(post_highs.idxmax())
            post_high = float(post_highs.loc[post_high_at])
            early_exit_gap = max(
                0.0,
                (post_high - exit_price) / entry_price,
            )

        gap_map = {
            "entrada_tardia": float(entry_gap),
            "escolha_ativo": float(selection_gap),
            "permanencia_excessiva": float(excess_holding_gap),
            "saida_precoce": float(early_exit_gap),
        }
        dominant_gap = max(gap_map, key=gap_map.get)
        dominant_value = float(gap_map[dominant_gap])

        rows.append(
            {
                "operation_id": int(operation_number),
                "asset": asset,
                "entry_timestamp": entry_at,
                "exit_timestamp": exit_at,
                "holding_bars": trade.get("holding_bars"),
                "walk_forward_fold": trade.get("walk_forward_fold"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "actual_position_return_pct": float(
                    (exit_price / entry_price - 1.0) * 100.0
                ),
                "entry_reference_low": (
                    float(entry_reference_low)
                    if np.isfinite(entry_reference_low)
                    else None
                ),
                "entry_reference_low_timestamp": entry_reference_at,
                "entry_late_gap_pct": float(entry_gap * 100.0),
                "entry_late_gap_proxy_usd": float(exposure * entry_gap),
                "best_same_holding_asset": best_asset,
                "selected_same_holding_market_return_pct": (
                    float(selected_market_return * 100.0)
                    if np.isfinite(selected_market_return)
                    else None
                ),
                "best_same_holding_market_return_pct": (
                    float(best_return * 100.0)
                    if np.isfinite(best_return)
                    else None
                ),
                "asset_selection_gap_pct": float(selection_gap * 100.0),
                "asset_selection_gap_proxy_usd": float(
                    exposure * selection_gap
                ),
                "peak_price_while_held": float(peak_price),
                "peak_timestamp_while_held": peak_at,
                "days_from_peak_to_exit": days_from_peak_to_exit,
                "excess_holding_gap_pct": float(
                    excess_holding_gap * 100.0
                ),
                "excess_holding_gap_proxy_usd": float(
                    exposure * excess_holding_gap
                ),
                "post_exit_best_high": float(post_high),
                "post_exit_best_timestamp": post_high_at,
                "early_exit_gap_pct": float(early_exit_gap * 100.0),
                "early_exit_gap_proxy_usd": float(
                    exposure * early_exit_gap
                ),
                "dominant_gap": dominant_gap,
                "dominant_gap_pct": dominant_value * 100.0,
                "dominant_gap_proxy_usd": float(
                    exposure * dominant_value
                ),
            }
        )

    detail = pd.DataFrame(rows, columns=columns)
    if detail.empty:
        return {
            "research_version": TOP_GAP_RESEARCH_VERSION,
            "closed_positions": 0,
            "entry_lookback_sessions": int(entry_lookback_sessions),
            "post_exit_sessions": int(post_exit_sessions),
            "warning": "ex_post_non_additive_counterfactuals",
        }, detail, pd.DataFrame()

    gap_columns = {
        "entrada_tardia": (
            "entry_late_gap_pct",
            "entry_late_gap_proxy_usd",
        ),
        "escolha_ativo": (
            "asset_selection_gap_pct",
            "asset_selection_gap_proxy_usd",
        ),
        "permanencia_excessiva": (
            "excess_holding_gap_pct",
            "excess_holding_gap_proxy_usd",
        ),
        "saida_precoce": (
            "early_exit_gap_pct",
            "early_exit_gap_proxy_usd",
        ),
    }
    summary_rows: list[dict[str, Any]] = []
    for cause, (pct_column, usd_column) in gap_columns.items():
        pct_values = pd.to_numeric(
            detail[pct_column],
            errors="coerce",
        ).dropna()
        usd_values = pd.to_numeric(
            detail[usd_column],
            errors="coerce",
        ).dropna()
        dominant_count = int((detail["dominant_gap"] == cause).sum())
        summary_rows.append(
            {
                "cause": cause,
                "operations": int(len(detail)),
                "dominant_operations": dominant_count,
                "dominant_share": float(
                    dominant_count / len(detail)
                ),
                "median_gap_pct": (
                    float(pct_values.median())
                    if not pct_values.empty
                    else None
                ),
                "mean_gap_pct": (
                    float(pct_values.mean())
                    if not pct_values.empty
                    else None
                ),
                "p90_gap_pct": (
                    float(pct_values.quantile(0.90))
                    if not pct_values.empty
                    else None
                ),
                "proxy_usd_sum_non_additive": (
                    float(usd_values.sum())
                    if not usd_values.empty
                    else 0.0
                ),
                "proxy_usd_median": (
                    float(usd_values.median())
                    if not usd_values.empty
                    else None
                ),
            }
        )
    by_cause = pd.DataFrame(summary_rows).sort_values(
        ["dominant_share", "proxy_usd_sum_non_additive"],
        ascending=False,
        ignore_index=True,
    )

    summary = {
        "research_version": TOP_GAP_RESEARCH_VERSION,
        "closed_positions": int(len(detail)),
        "entry_lookback_sessions": int(entry_lookback_sessions),
        "post_exit_sessions": int(post_exit_sessions),
        "warning": "ex_post_non_additive_counterfactuals",
        "dominant_cause": (
            str(by_cause.iloc[0]["cause"])
            if not by_cause.empty
            else None
        ),
        "dominant_cause_share": (
            float(by_cause.iloc[0]["dominant_share"])
            if not by_cause.empty
            else None
        ),
        "median_actual_position_return_pct": float(
            pd.to_numeric(
                detail["actual_position_return_pct"],
                errors="coerce",
            ).median()
        ),
    }
    return summary, detail, by_cause


def calcular_metricas_peak_gatilhos(
    peak_trades: pd.DataFrame,
    predictions: pd.DataFrame,
    trigger_column: str,
) -> dict[str, float | None]:
    """Resume Peak Exit apenas das saídas realmente causadas pelo overlay."""
    if (
        peak_trades is None
        or peak_trades.empty
        or predictions is None
        or predictions.empty
        or trigger_column not in predictions.columns
    ):
        return {
            "median_exit_distance_from_peak_pct": None,
            "median_peak_capture_pct": None,
        }

    prediction_frame = predictions.copy()
    mask = prediction_frame[trigger_column].fillna(False).astype(bool)
    trigger_rows = prediction_frame.loc[mask].copy()
    if trigger_rows.empty:
        return {
            "median_exit_distance_from_peak_pct": None,
            "median_peak_capture_pct": None,
        }

    detail = peak_trades.copy()
    detail["exit_timestamp"] = pd.to_datetime(
        detail["exit_timestamp"],
        utc=True,
    )
    selected: list[pd.Series] = []

    for timestamp, row in trigger_rows.iterrows():
        execution_at = pd.Timestamp(timestamp)
        if execution_at.tzinfo is None:
            execution_at = execution_at.tz_localize("UTC")
        else:
            execution_at = execution_at.tz_convert("UTC")

        asset = str(
            row.get("current_asset")
            or row.get("previous_asset")
            or row.get("selected_asset")
            or ""
        )
        if not asset or asset == "nan":
            continue

        exact = detail.loc[
            (detail["asset"].astype(str) == asset)
            & (detail["exit_timestamp"] == execution_at)
        ]
        if exact.empty:
            continue
        selected.append(exact.iloc[0])

    if not selected:
        return {
            "median_exit_distance_from_peak_pct": None,
            "median_peak_capture_pct": None,
        }

    selected_frame = pd.DataFrame(selected)

    def median(column: str) -> float | None:
        values = pd.to_numeric(
            selected_frame[column],
            errors="coerce",
        ).dropna()
        return float(values.median()) if not values.empty else None

    return {
        "median_exit_distance_from_peak_pct": median(
            "exit_distance_from_peak_pct"
        ),
        "median_peak_capture_pct": median("peak_capture_pct"),
    }


def criar_pacote_analise(
    diretorio_resultados: Path,
    *,
    comparison_file: str | None = None,
    execution_schema: str | None = None,
    archive_name: str = "pacote_analise.zip",
) -> Path:
    """Gera ZIP estavel para runners independentes.

    Cada runner pode informar seu proprio arquivo principal e schema. Isso
    permite separar busca de ativos de avaliacao financeira sem misturar
    artefatos. Os defaults preservam compatibilidade com o runner de busca.
    """
    diretorio = Path(diretorio_resultados)
    if not diretorio.exists():
        raise FileNotFoundError(diretorio)

    expected_file = str(comparison_file or EXPECTED_COMPARISON_FILE)
    expected_schema = str(execution_schema or EXPECTED_EXECUTION_SCHEMA)
    comparison_path = diretorio / expected_file
    if not comparison_path.exists():
        raise RuntimeError(
            "Pacote recusado: o runner espera "
            f"{expected_file} com schema {expected_schema}, "
            "mas o arquivo nao existe. Atualize a branch, reinicie o kernel "
            "do Spyder e execute o runner correto."
        )

    try:
        comparison_payload = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Pacote recusado: nao foi possivel validar {comparison_path.name}."
        ) from exc

    observed_schema = comparison_payload.get("execution_schema")
    if observed_schema != expected_schema:
        raise RuntimeError(
            "Pacote recusado por execution_schema incompatível: "
            f"esperado={expected_schema!r} observado={observed_schema!r}. "
            "Execute o arquivo Spyder correspondente a este pacote."
        )

    destino = diretorio / str(archive_name)
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
    """Sinal audivel de conclusao que nunca pode derrubar a pesquisa."""
    mechanisms: list[str] = []

    try:
        import winsound

        # PlaySound e sincrono por padrao quando SND_ASYNC nao e usado.
        # Algumas versoes de Python/Windows nao expoem SND_SYNC.
        try:
            alias_flag = getattr(winsound, "SND_ALIAS", None)
            if alias_flag is not None:
                winsound.PlaySound(
                    "SystemExclamation",
                    int(alias_flag),
                )
                mechanisms.append("PlaySound:SystemExclamation")
        except Exception:
            pass

        try:
            message_type = getattr(winsound, "MB_ICONASTERISK", -1)
            winsound.MessageBeep(int(message_type))
            mechanisms.append("MessageBeep")
        except Exception:
            pass

        try:
            winsound.Beep(880, 220)
            time.sleep(0.08)
            winsound.Beep(1175, 420)
            mechanisms.append("Beep")
        except Exception:
            pass
    except Exception:
        pass

    if not mechanisms:
        try:
            sys.stdout.write("\\a")
            sys.stdout.flush()
            mechanisms.append("terminal-bell")
        except Exception:
            mechanisms.append("none")

    try:
        print(
            "[sound] completion mechanisms=" + ",".join(mechanisms),
            flush=True,
        )
    except Exception:
        pass
