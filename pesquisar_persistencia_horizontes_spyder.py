"""Pesquisa U67+10 v1.0.11: ablacoes causais de permanencia e margem.

Control e unico backtest; selecao de candidatos e diagnostica.
Reproduz o checkpoint, corrige cobertura de labels H01..H60, treina
modelo direto de diferencas e preserva auditoria independente.
Sem download, sem alteracao da reproducao oficial ou de sistemas externos.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from engine.configuracao import (
    CONFIG,
    U67_REQUESTED_ASSETS,
    U67_EXPECTED_EXCLUSIONS,
    construir_configuracao_controle,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _configuracoes_lightgbm,
    _construir_contexto_execucao,
    selecionar_switch_margin,
)
from engine.rotacao import (
    ROTATION_FEATURES,
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import prepare_model_frames
from pesquisar_expansao_u67_spyder import (
    ANALYSIS_END_DATE,
    DISCOVERY_SELECTED_CANDIDATES,
    EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL,
    FULL_DATA,
    _merge_utility_caches,
    _prepare_candidate_frames,
    _train_baseline_states,
    _train_candidate_caches,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "persistencia_horizontes_u77_v111"
VERSION = "horizon-persistence-oos-v1.0.11"
CONFIDENCE_MULTIPLIERS = (0.0, 0.5, 1.0)  # historico v1.0.3
ANCHOR_BLEND = 0.5  # historico v1.0.3
HOLDING_CANDIDATES = (1, 2, 3, 4)  # historico v1.0.4
RELATIVE_HORIZONS = (5, 10, 20, 40)
SHORT_HORIZONS = (5, 10)
MEDIUM_HORIZONS = (20, 40)
MIN_GROUP_ADVANTAGE_PCT = 0.005
CANDIDATE_REPORT_TOP_K = 10
CHECKPOINT_TOLERANCE_DOLLARS = 0.01
CAPTURE_HORIZONS = (2, 3, 5, 10)
EXTENDED_AUDIT_HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60)
AUDIT_ONLY_HORIZONS = (1, 15, 20, 30, 40, 60)
ALL_RESEARCH_HORIZONS = EXTENDED_AUDIT_HORIZONS
AUDIT_TOP_K = 5
# Ablacao pre-registrada: mesmo sinal de utilidade, uma alteracao por vez.
UTILITY_ABLATION_HOLD = (0, 2)
UTILITY_ABLATION_MARGINS = ("zero", "base", "calibrated")
UTILITY_ABLATION_PRIMARY_HORIZON = 5
UTILITY_ABLATION_PRIMARY_METRIC = "H05_switch_precision_vs_hold"
AUTONOMOUS_POLICY_METHODS = ("net_gain", "borda_pareto", "topsis", "consensus")
AUTONOMOUS_MIN_HOLD = 2
AUTONOMOUS_SWITCH_MARGIN = 0.0  # nenhum ajuste no OOS
AUTONOMOUS_CASH_EXIT_MARGIN = 0.0
AUTONOMOUS_MIN_MATURE_SWITCHES_PER_FOLD = 10
RANKING_METHODS = (
    "predicted_return", "net_advantage", "borda",
    "topsis", "pareto", "probabilistic", "robust",
    "learning_to_rank",
)
RANKING_EVALUATION_HORIZONS = (2, 3, 5, 10)
RANKING_PRIMARY_METRIC = "ndcg_at_5"
RANKING_CALIBRATION_MIN_DAYS = 15
RANKING_PROBABILITY_VARIANCE_FLOOR = 1e-6
RANKING_LOG_EVERY_DAYS = 25

PAIR_REFERENCE_COUNT = 2
PAIR_SELECTED_TOP_K = 5
PAIR_TRAINING_FEATURES = (
    "return_2", "return_5", "return_10", "return_20",
    "vol_5", "vol_20", "ema_distance_20", "ema_distance_50",
    "trend_efficiency_20", "distance_from_high_20",
    "rsi_14", "atr_pct_14", "volume_zscore_20",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return safe_json(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(safe_json(obj), ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def select_purged_horizon_training_dates(
    *,
    calendar: pd.DatetimeIndex,
    training_end_index: int,
    evaluation_start_index: int,
    horizon: int,
    min_training_rows: int,
) -> pd.DatetimeIndex:
    """Keep all training labels that mature strictly before evaluation starts.

    The official fold already has a 60-session purge. Removing another h
    sessions from the training interval double-purges the first fold and
    reduces its 700 available rows below the LightGBM minimum.
    """
    h = int(horizon)
    end = int(training_end_index)
    evaluation = int(evaluation_start_index)
    minimum = int(min_training_rows)
    if h <= 0 or end <= 0 or evaluation > len(calendar) or end > len(calendar):
        raise ValueError("Invalid horizon or training/evaluation calendar indices.")
    if end - 1 + h >= evaluation:
        raise RuntimeError(
            "CausalHorizonPurgeInsufficient: "
            f"last training target index={end - 1 + h} must be strictly "
            f"before evaluation_start_index={evaluation} for H{h}. "
            "Never train using a label that matures inside evaluation."
        )
    if end < minimum:
        raise RuntimeError(
            "HorizonTrainingRowsInsufficient: "
            f"available={end} minimum={minimum} H{h}. "
            "Do not silently train with fewer rows or skip models."
        )
    return pd.DatetimeIndex(calendar[:end])


def uncertainty_from_matured_calibration(
    *,
    h: int,
    timestamps: pd.DatetimeIndex,
    full_dates: pd.DatetimeIndex,
    latest_label_date_index: int,
    symbols: list[str],
    frames: dict[str, pd.DataFrame],
    predictions: dict[pd.Timestamp, np.ndarray],
) -> tuple[float, int]:
    """Erro absoluto p75 somente de alvos totalmente maturados antes do OOS."""
    lookup = {pd.Timestamp(date): i for i, date in enumerate(full_dates)}
    errors: list[float] = []
    matured_dates_with_predictions = 0
    for day in timestamps:
        timestamp = pd.Timestamp(day)
        index = lookup.get(timestamp)
        if index is None or index + h >= latest_label_date_index:
            continue
        predicted = predictions.get(timestamp)
        if predicted is None:
            continue
        observed_this_day = False
        for column, symbol in enumerate(symbols, start=1):
            actual = frames[symbol].loc[timestamp, f"forward_horizon_utility_{h}"]
            forecast = float(predicted[column])
            if np.isfinite(actual) and np.isfinite(forecast):
                errors.append(abs(float(actual) - forecast))
                observed_this_day = True
        if observed_this_day:
            matured_dates_with_predictions += 1
    if len(errors) < 30 or matured_dates_with_predictions < 15:
        raise RuntimeError(
            f"H{h}: insuficientes residuos maturados para medir a incerteza "
            f"antes do OOS: {len(errors)}. Nao usar rotulos futuros."
        )
    return float(np.quantile(np.asarray(errors), 0.75)), len(errors)


class HorizonPersistencePolicy:
    """Politica cronologica com memoria da previsao na entrada.

    A incerteza e um proxy empirico, NAO probabilidade calibrada.
    A regra de margem financeira original continua ativa; a confianca
    acrescenta evidencia por horizonte, sem intervalo de holding fixo.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        aggregate: dict[pd.Timestamp, np.ndarray],
        per_horizon: dict[int, dict[pd.Timestamp, np.ndarray]],
        uncertainties: dict[int, float],
        weights: dict[int, float],
        margin: float,
        cash_threshold: float,
        entry_threshold: float,
        confidence_multiplier: float,
        audit: list[dict[str, Any]] | None = None,
        fold_id: int | None = None,
    ) -> None:
        self.symbols = list(symbols)
        self.date_to_index = {pd.Timestamp(day): i for i, day in enumerate(dates)}
        self.aggregate = aggregate
        self.per_horizon = per_horizon
        self.uncertainties = uncertainties
        self.weights = weights
        self.margin = float(margin)
        self.cash_threshold = float(cash_threshold)
        self.entry_threshold = float(entry_threshold)
        self.confidence_multiplier = float(confidence_multiplier)
        self.audit = audit
        self.fold_id = fold_id
        self.anchor: dict[str, Any] | None = None

    def __call__(self, day: pd.Timestamp, current: int, holding: int) -> tuple[int, float]:
        timestamp = pd.Timestamp(day)
        index = self.date_to_index[timestamp]
        values = self.aggregate[timestamp]
        if not np.isfinite(values[1:]).any():
            return 0, 0.0
        best = int(np.nanargmax(values))
        q_best = float(values[best])
        q_current = float(values[current])
        scores = {
            h: self.per_horizon[h].get(timestamp)
            for h in self.weights
        }
        if current > 0 and (
            self.anchor is None or self.anchor["position"] != current
        ):
            # Crossing a fold can inherit a live position whose original
            # entry forecast is not carried by this fold-local policy.
            self.anchor = {
                "position": current,
                "date_index": index,
                "scores": {
                    h: float(row[current]) if row is not None else float("nan")
                    for h, row in scores.items()
                },
                "observed_at_entry": False,
            }

        target = current
        reason = "HOLD_CURRENT"
        horizon_advantage: float | None = None
        uncertainty: float | None = None
        age: int | None = None
        if best == 0 or q_best <= self.cash_threshold:
            target = 0
            reason = "CASH_THRESHOLD"
        elif current == 0:
            target = best if q_best >= self.entry_threshold else 0
            reason = "ENTER_BEST" if target else "MIN_EXPECTED_EDGE"
        elif best == current:
            reason = "CURRENT_BEST"
        elif q_best < q_current + self.margin:
            reason = "SWITCH_MARGIN"
        else:
            assert self.anchor is not None
            age = max(0, index - int(self.anchor["date_index"]))
            available = []
            for h, weight in self.weights.items():
                row = scores[h]
                old = float(self.anchor["scores"].get(h, float("nan")))
                if row is None or not (np.isfinite(row[current]) and np.isfinite(row[best])):
                    continue
                remaining = max(0.0, 1.0 - age / float(h))
                if remaining > 0 and np.isfinite(old):
                    available.append((h, weight * remaining, old))
            if not available:
                # No non-expired entry horizon (or inherited fold boundary):
                # compare fresh horizon scores rather than manufacture an old one.
                available = [
                    (h, weight, float(scores[h][current]))
                    for h, weight in self.weights.items()
                    if scores[h] is not None
                    and np.isfinite(scores[h][current])
                    and np.isfinite(scores[h][best])
                ]
            total_weight = sum(w for _, w, _ in available)
            if total_weight <= 0:
                reason = "INSUFFICIENT_HORIZON_EVIDENCE"
            else:
                new_current = sum(
                    w * float(scores[h][current]) for h, w, _ in available
                ) / total_weight
                old_current = sum(w * old for _, w, old in available) / total_weight
                new_best = sum(
                    w * float(scores[h][best]) for h, w, _ in available
                ) / total_weight
                horizon_advantage = new_best - (
                    ANCHOR_BLEND * old_current
                    + (1.0 - ANCHOR_BLEND) * new_current
                )
                uncertainty = sum(
                    w * float(self.uncertainties[h]) for h, w, _ in available
                ) / total_weight
                if horizon_advantage > self.confidence_multiplier * uncertainty:
                    target = best
                    reason = "ROTATE_HORIZON_CONFIDENT"
                else:
                    reason = "HORIZON_CONFIDENCE_NOT_MET"

        if self.audit is not None:
            row: dict[str, Any] = {
                "fold_id": self.fold_id,
                "decision_date": timestamp.isoformat(),
                "current_asset": "CASH" if current == 0 else self.symbols[current - 1],
                "raw_best_asset": "CASH" if best == 0 else self.symbols[best - 1],
                "target_asset": "CASH" if target == 0 else self.symbols[target - 1],
                "reason": reason,
                "holding_days_simulator": holding,
                "forecast_age_sessions": age,
                "entry_forecast_origin": (
                    "observed_entry"
                    if self.anchor is not None and self.anchor["observed_at_entry"]
                    else "fold_boundary_proxy" if self.anchor is not None else None
                ),
                "q_best": q_best,
                "q_current": q_current,
                "q_margin": self.margin,
                "horizon_advantage": horizon_advantage,
                "horizon_uncertainty_p75": uncertainty,
                "confidence_multiplier": self.confidence_multiplier,
            }
            for h in self.weights:
                pred = scores[h]
                row[f"H{h}_current_today"] = (
                    float(pred[current]) if pred is not None and current > 0 else None
                )
                row[f"H{h}_best_today"] = (
                    float(pred[best]) if pred is not None and best > 0 else None
                )
                row[f"H{h}_entry_forecast"] = (
                    self.anchor["scores"].get(h)
                    if self.anchor is not None and current > 0
                    else None
                )
            self.audit.append(row)

        if target == 0:
            self.anchor = None
        elif target != current:
            self.anchor = {
                "position": target,
                "date_index": index,
                "scores": {
                    h: float(row[target]) if row is not None else float("nan")
                    for h, row in scores.items()
                },
                "observed_at_entry": True,
            }
        return target, float(values[target])


def make_policy(
    *,
    symbols: list[str],
    dates: pd.DatetimeIndex,
    aggregate: dict[pd.Timestamp, np.ndarray],
    per_horizon: dict[int, dict[pd.Timestamp, np.ndarray]],
    uncertainties: dict[int, float],
    config: Any,
    margin: float,
    multiplier: float,
    audit: list[dict[str, Any]] | None = None,
    fold_id: int | None = None,
) -> HorizonPersistencePolicy:
    weights = np.asarray(config.rotation_target_horizon_weights, dtype=float)
    weights /= weights.sum()
    return HorizonPersistencePolicy(
        symbols=symbols,
        dates=dates,
        aggregate=aggregate,
        per_horizon=per_horizon,
        uncertainties=uncertainties,
        weights=dict(zip(config.rotation_target_horizons, weights)),
        margin=margin,
        cash_threshold=float(config.rotation_cash_threshold),
        entry_threshold=(
            float(config.rotation_cash_threshold)
            + float(config.rotation_min_expected_edge)
        ),
        confidence_multiplier=multiplier,
        audit=audit,
        fold_id=fold_id,
    )




def calibrate_pair_gap_error(
    *,
    horizon: int,
    days: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    last_matured_exclusive: int,
    symbols: list[str],
    frames: dict[str, pd.DataFrame],
    aggregate: dict[pd.Timestamp, np.ndarray],
    predictions: dict[pd.Timestamp, np.ndarray],
) -> tuple[float, int, int]:
    """Quantile across days of within-day pairwise prediction errors.

    At each training-calibration day, compare the aggregate leader with
    every other asset, but use only H-specific labels completely observed
    BEFORE the threshold-selection interval. Each day's 75th percentile
    is one statistical observation, to avoid counting 74 correlated
    leader/challenger pairs as independent days.
    """
    lookups = {pd.Timestamp(date): n for n, date in enumerate(calendar)}
    daily_pair_p75: list[float] = []
    pairs_used = 0
    h = int(horizon)
    for date in days:
        stamp = pd.Timestamp(date)
        idx = lookups.get(stamp)
        if idx is None or idx + h >= int(last_matured_exclusive):
            continue
        aggregate_row = aggregate.get(stamp)
        horizon_row = predictions.get(stamp)
        if aggregate_row is None or horizon_row is None:
            continue
        valid = np.flatnonzero(
            np.isfinite(aggregate_row[1:]) & np.isfinite(horizon_row[1:])
        ) + 1
        if len(valid) < 2:
            continue
        leader = int(valid[np.argmax(aggregate_row[valid])])
        leader_label = float(frames[symbols[leader - 1]].at[
            stamp, f"forward_horizon_utility_{h}"
        ])
        if not np.isfinite(leader_label):
            continue
        errors: list[float] = []
        for pos in valid:
            pos = int(pos)
            if pos == leader:
                continue
            counterpart_label = float(frames[symbols[pos - 1]].at[
                stamp, f"forward_horizon_utility_{h}"
            ])
            if not np.isfinite(counterpart_label):
                continue
            predicted_gap = float(horizon_row[leader] - horizon_row[pos])
            realized_gap = leader_label - counterpart_label
            errors.append(abs(predicted_gap - realized_gap))
        if errors:
            daily_pair_p75.append(float(np.quantile(errors, 0.75)))
            pairs_used += len(errors)
    if len(daily_pair_p75) < 15 or pairs_used < 30:
        raise RuntimeError(
            f"H{h}: evidência histórica insuficiente para calibração "
            f"do erro de diferença entre ativos: "
            f"days={len(daily_pair_p75)}, pairs={pairs_used}. "
            "Não utilizar dados posteriores ao corte."
        )
    return float(np.quantile(daily_pair_p75, 0.75)), len(daily_pair_p75), pairs_used


class EarlyHorizonOverridePolicy:
    """Preserve Control decisions except a horizon-supported early rotation.

    The fold boundary can inherit an asset without its original entry
    forecast. In that case the early override is disabled until a new
    observed entry; the ordinary Control decision is still executable.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        aggregate: dict[pd.Timestamp, np.ndarray],
        per_horizon: dict[int, dict[pd.Timestamp, np.ndarray]],
        pair_error_p75: dict[int, float],
        weights: dict[int, float],
        control_policy: Any,
        margin: float,
        minimum_holding: int,
        confidence_multiplier: float,
        audit: list[dict[str, Any]] | None = None,
        fold_id: int | None = None,
    ) -> None:
        self.symbols = symbols
        self.indices = {pd.Timestamp(d): n for n, d in enumerate(dates)}
        self.aggregate = aggregate
        self.per_horizon = per_horizon
        self.pair_error_p75 = pair_error_p75
        self.weights = weights
        self.control_policy = control_policy
        self.margin = float(margin)
        self.minimum_holding = int(minimum_holding)
        self.confidence_multiplier = float(confidence_multiplier)
        self.audit = audit
        self.fold_id = fold_id
        self.entry: dict[str, Any] | None = None

    def __call__(
        self, timestamp: pd.Timestamp, current: int, holding: int
    ) -> tuple[int, float]:
        stamp = pd.Timestamp(timestamp)
        normal_action, normal_score = self.control_policy(
            stamp, current, holding
        )
        row = self.aggregate[stamp]
        date_index = self.indices[stamp]
        horizon_rows = {
            int(h): self.per_horizon[int(h)].get(stamp)
            for h in self.weights
        }
        if current > 0 and (
            self.entry is None or self.entry["position"] != current
        ):
            # Never fabricate a historical entry prediction at a fold boundary.
            self.entry = {
                "position": current,
                "observed": False,
                "date_index": date_index,
                "horizon": {},
            }

        action, score = int(normal_action), float(normal_score)
        candidate = None
        fresh_advantage = None
        original_advantage = None
        uncertainty = None
        reason = "CONTROL_UNCHANGED"
        eligible = (
            0 < current <= len(self.symbols)
            and 0 < holding < self.minimum_holding
            and action == current
            and bool(self.entry and self.entry["observed"])
            and np.isfinite(row[current])
            and np.isfinite(row[1:]).any()
        )

        if eligible:
            candidate = int(np.nanargmax(row))
            if candidate <= 0 or candidate == current:
                reason = "NO_ALTERNATIVE"
            elif not (
                np.isfinite(row[candidate])
                and row[candidate] > 0.0
                and row[candidate] >= row[current] + self.margin
            ):
                reason = "AGGREGATE_MARGIN_NOT_MET"
            else:
                age = max(0, date_index - int(self.entry["date_index"]))
                differences = []
                for h, weight in self.weights.items():
                    forecast = horizon_rows[h]
                    old = self.entry["horizon"].get(h)
                    if (
                        forecast is None
                        or old is None
                        or not np.isfinite(old)
                        or not (
                            np.isfinite(forecast[current])
                            and np.isfinite(forecast[candidate])
                        )
                    ):
                        continue
                    remaining = max(0.0, 1.0 - age / float(h))
                    effective_weight = float(weight) * remaining
                    if effective_weight <= 0.0:
                        continue
                    differences.append(
                        (
                            effective_weight,
                            float(forecast[candidate] - forecast[current]),
                            float(forecast[candidate] - old),
                            float(self.pair_error_p75[h]),
                        )
                    )
                total_weight = sum(w for w, _, _, _ in differences)
                if total_weight <= 0:
                    reason = "MISSING_ACTIVE_HORIZON"
                else:
                    fresh_advantage = sum(
                        w * gap for w, gap, _, _ in differences
                    ) / total_weight
                    original_advantage = sum(
                        w * gap for w, _, gap, _ in differences
                    ) / total_weight
                    uncertainty = sum(
                        w * sigma for w, _, _, sigma in differences
                    ) / total_weight
                    threshold = self.confidence_multiplier * uncertainty
                    if (
                        fresh_advantage > threshold
                        and original_advantage > threshold
                    ):
                        action, score = candidate, float(row[candidate])
                        reason = "EARLY_OVERRIDE_HORIZON_PAIRS"
                    else:
                        reason = "PAIR_EVIDENCE_INSUFFICIENT"
        elif 0 < current and holding < self.minimum_holding:
            reason = "FOLD_INHERITED_OR_NO_ANCHOR"
        elif holding >= self.minimum_holding:
            reason = "CONTROL_HOLDING_ALREADY_ELIGIBLE"

        if self.audit is not None:
            self.audit.append({
                "fold_id": self.fold_id,
                "decision_date": stamp.isoformat(),
                "current_asset": (
                    self.symbols[current - 1] if current > 0 else "CASH"
                ),
                "control_target": (
                    self.symbols[normal_action - 1]
                    if normal_action > 0 else "CASH"
                ),
                "best_candidate": (
                    self.symbols[candidate - 1]
                    if candidate is not None and candidate > 0 else None
                ),
                "final_target": (
                    self.symbols[action - 1] if action > 0 else "CASH"
                ),
                "holding_sessions": holding,
                "entry_forecast_observed": bool(
                    self.entry and self.entry["observed"]
                ),
                "reason": reason,
                "early_override": action != normal_action,
                "fresh_pair_advantage": fresh_advantage,
                "entry_pair_advantage": original_advantage,
                "pair_error_p75": uncertainty,
                "confidence_multiplier": self.confidence_multiplier,
            })

        if action != current:
            if action > 0:
                self.entry = {
                    "position": action,
                    "observed": True,
                    "date_index": date_index,
                    "horizon": {
                        int(h): (
                            float(forecast[action])
                            if forecast is not None
                            and np.isfinite(forecast[action])
                            else float("nan")
                        )
                        for h, forecast in horizon_rows.items()
                    },
                }
            else:
                self.entry = None
        return int(action), float(score)


def make_early_override(
    *,
    symbols: list[str],
    dates: pd.DatetimeIndex,
    aggregate: dict[pd.Timestamp, np.ndarray],
    per_horizon: dict[int, dict[pd.Timestamp, np.ndarray]],
    pair_error_p75: dict[int, float],
    config: Any,
    margin: float,
    multiplier: float,
    audit: list[dict[str, Any]] | None = None,
    fold_id: int | None = None,
) -> EarlyHorizonOverridePolicy:
    w = np.asarray(config.rotation_target_horizon_weights, dtype=float)
    w /= w.sum()
    control = _politica_utilidade(
        {}, {}, symbols, config, margin, utility_cache=aggregate
    )
    return EarlyHorizonOverridePolicy(
        symbols=symbols,
        dates=dates,
        aggregate=aggregate,
        per_horizon=per_horizon,
        pair_error_p75=pair_error_p75,
        weights=dict(zip((int(h) for h in config.rotation_target_horizons), w)),
        control_policy=control,
        margin=margin,
        minimum_holding=int(config.rotation_min_holding_days),
        confidence_multiplier=multiplier,
        audit=audit,
        fold_id=fold_id,
    )

def scenario_metrics(result: Any, folds: list[dict[str, Any]], capital: float) -> dict[str, Any]:
    m = summarize_metrics(result, folds, capital)
    m.update({
        "capital_rotations": int(result.metrics.get("capital_rotations") or 0),
        "total_transaction_fees": float(result.metrics.get("total_transaction_fees") or 0.0),
        "simulated_buys": int(result.metrics.get("simulated_buys") or 0),
        "simulated_sells": int(result.metrics.get("simulated_sells") or 0),
    })
    return m



def select_calibrated_holding(scores: dict[int, float]) -> int:
    """Seleciona apenas na calibracao. Desempate prefere o Control (2)."""
    expected = set(HOLDING_CANDIDATES)
    if set(scores) != expected:
        raise ValueError("A selecao exige todos os holdings pre-registrados.")
    if not all(np.isfinite(float(v)) for v in scores.values()):
        raise ValueError("Scores de calibracao nao finitos. Nao usar OOS.")
    return min(
        HOLDING_CANDIDATES,
        key=lambda h: (-float(scores[h]), abs(h - 2), h),
    )


def causal_hold_guard_features(
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Somente variaveis calculadas na data da decisao; nao usa forward labels."""
    rows: list[dict[str, Any]] = []
    features = (
        "vol_20", "vol_60", "return_5",
        "trend_efficiency_20", "distance_from_high_20",
    )
    for day, record in sorted(diagnostics.items()):
        if not record.get("min_hold_guard_applied"):
            continue
        symbol = str(record.get("current_asset") or "")
        if symbol == "CASH" or symbol not in frames:
            continue
        stamp = pd.Timestamp(day)
        frame = frames[symbol]
        if stamp not in frame.index:
            raise RuntimeError(
                f"Diagnostico temporal sem frame: {symbol} {stamp.isoformat()}"
            )
        values: dict[str, Any] = {}
        for feature in features:
            if feature not in frame.columns:
                raise RuntimeError(
                    f"Caracteristica historica ausente: {symbol}.{feature}"
                )
            value = float(frame.at[stamp, feature])
            values[feature] = value if np.isfinite(value) else None
        rows.append({
            "decision_date": stamp.isoformat(),
            "fold_id": record.get("decision_fold_id"),
            "asset": symbol,
            "holding_sessions": record.get("holding_days_at_decision"),
            "raw_best_asset": record.get("raw_best_asset"),
            "best_vs_current_gap": record.get("best_vs_current_gap"),
            **values,
        })
    return pd.DataFrame(rows, columns=[
        "decision_date", "fold_id", "asset", "holding_sessions",
        "raw_best_asset", "best_vs_current_gap", *features,
    ])



def add_horizon_percentage_training_targets(
    frames: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = RELATIVE_HORIZONS,
) -> None:
    """Rotulos FUTUROS apenas de treino; jamais valores usados nas decisoes."""
    for symbol, frame in frames.items():
        for h in horizons:
            source = f"forward_horizon_net_log_return_{h}"
            target = f"forward_horizon_net_return_pct_{h}"
            if source not in frame.columns:
                raise RuntimeError(f"{symbol}: missing frozen training target {source}")
            raw = pd.to_numeric(frame[source], errors="coerce")
            frame[target] = np.expm1(raw)
            frame.loc[~np.isfinite(frame[target]), target] = np.nan


def relative_horizon_candidates(
    *,
    date: pd.Timestamp,
    incumbent: int,
    control_target: int,
    symbols: list[str],
    aggregate: np.ndarray,
    forecasts: dict[int, dict[pd.Timestamp, np.ndarray]],
    min_group_advantage: float,
    require_each_horizon: bool,
    effective_margin: float,
    top_k: int = CANDIDATE_REPORT_TOP_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compara somente previsoes contemporaneas. Sem leitura de labels."""
    if incumbent <= 0 or incumbent > len(symbols):
        return [], []
    timestamp = pd.Timestamp(date)
    expected_length = len(symbols) + 1
    arrays: dict[int, np.ndarray] = {}
    for h in RELATIVE_HORIZONS:
        cache = forecasts.get(h, {})
        value = cache.get(timestamp)
        if value is None:
            return [], []
        values = np.asarray(value, dtype=np.float64)
        if len(values) != expected_length:
            raise ValueError(f"H{h}: expected {expected_length} model scores")
        arrays[h] = values
    if not all(np.isfinite(arrays[h][incumbent]) for h in RELATIVE_HORIZONS):
        return [], []

    reference = control_target if control_target > 0 else incumbent
    if not all(np.isfinite(arrays[h][reference]) for h in RELATIVE_HORIZONS):
        return [], []
    current_short = float(np.mean([arrays[h][incumbent] for h in SHORT_HORIZONS]))
    current_medium = float(np.mean([arrays[h][incumbent] for h in MEDIUM_HORIZONS]))
    baseline_short = float(np.mean([arrays[h][reference] for h in SHORT_HORIZONS]))
    baseline_medium = float(np.mean([arrays[h][reference] for h in MEDIUM_HORIZONS]))

    ranking: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for pos, symbol in enumerate(symbols, start=1):
        if pos == incumbent:
            continue
        if any(not np.isfinite(arrays[h][pos]) for h in RELATIVE_HORIZONS):
            continue
        preds = {h: float(arrays[h][pos]) for h in RELATIVE_HORIZONS}
        short = float(np.mean([preds[h] for h in SHORT_HORIZONS]))
        medium = float(np.mean([preds[h] for h in MEDIUM_HORIZONS]))
        short_advantage = short - current_short
        medium_advantage = medium - current_medium
        short_vs_control = short - baseline_short
        medium_vs_control = medium - baseline_medium
        individual = {h: preds[h] - float(arrays[h][incumbent]) for h in RELATIVE_HORIZONS}
        aggregate_score = float(aggregate[pos])
        utility_valid = bool(
            np.isfinite(aggregate_score)
            and aggregate_score > 0.0
        )
        passes_each = all(individual[h] > 0.0 for h in RELATIVE_HORIZONS)
        qualifies = bool(
            utility_valid
            and short > 0.0
            and medium > 0.0
            and short_advantage >= min_group_advantage
            and medium_advantage >= min_group_advantage
            and (not require_each_horizon or passes_each)
            and (
                control_target in (0, incumbent, pos)
                or (
                    short_vs_control > 0.0
                    and medium_vs_control > 0.0
                )
            )
            and (
                not require_each_horizon
                or aggregate_score >= float(aggregate[incumbent]) + effective_margin
            )
        )
        row = {
            "decision_date": timestamp.isoformat(),
            "current_asset": symbols[incumbent - 1],
            "control_target": (
                symbols[control_target - 1] if control_target > 0 else "CASH"
            ),
            "candidate": symbol,
            "candidate_position": pos,
            "net_expected_short_pct": short,
            "net_expected_medium_pct": medium,
            "current_expected_short_pct": current_short,
            "current_expected_medium_pct": current_medium,
            "delta_short_pct_points": short_advantage,
            "delta_medium_pct_points": medium_advantage,
            "delta_vs_control_short_pct_points": short_vs_control,
            "delta_vs_control_medium_pct_points": medium_vs_control,
            "candidate_aggregate_utility": aggregate_score,
            "qualifies": qualifies,
            **{f"candidate_expected_H{h}_pct": preds[h] for h in RELATIVE_HORIZONS},
            **{f"current_expected_H{h}_pct": float(arrays[h][incumbent]) for h in RELATIVE_HORIZONS},
            **{f"delta_H{h}_pct_points": individual[h] for h in RELATIVE_HORIZONS},
        }
        ranking.append(row)
        if qualifies:
            eligible.append(row)
    key = lambda row: (
        -0.5 * (
            row["delta_short_pct_points"] + row["delta_medium_pct_points"]
        ),
        -min(row["delta_short_pct_points"], row["delta_medium_pct_points"]),
        row["candidate"],
    )
    ranking.sort(key=key)
    eligible.sort(key=key)
    return eligible, ranking[:top_k]


class RelativeHorizonGainPolicy:
    """Control como fallback; tentativa de escolher lider relativo H5/H10/H20/H40."""

    def __init__(
        self,
        *,
        symbols: list[str],
        aggregate: dict[pd.Timestamp, np.ndarray],
        forecasts: dict[int, dict[pd.Timestamp, np.ndarray]],
        control_policy: Any,
        minimum_holding: int,
        margin: float,
        require_each_horizon: bool,
        min_group_advantage: float = MIN_GROUP_ADVANTAGE_PCT,
        fold_id: int,
        audit: list[dict[str, Any]] | None = None,
        ranking_audit: list[dict[str, Any]] | None = None,
    ) -> None:
        self.symbols = symbols
        self.aggregate = aggregate
        self.forecasts = forecasts
        self.control_policy = control_policy
        self.minimum_holding = int(minimum_holding)
        self.margin = float(margin)
        self.require_each_horizon = bool(require_each_horizon)
        self.min_group_advantage = float(min_group_advantage)
        self.fold_id = int(fold_id)
        self.audit = audit
        self.ranking_audit = ranking_audit

    def __call__(self, timestamp: pd.Timestamp, current: int, holding: int) -> tuple[int, float]:
        day = pd.Timestamp(timestamp)
        normal_target, normal_score = self.control_policy(day, current, holding)
        action = int(normal_target)
        score = float(normal_score)
        reason = "CONTROL_FALLBACK"
        candidate_count = 0
        chosen: dict[str, Any] | None = None
        if current <= 0:
            reason = "NO_INCUMBENT_CASH"
        elif holding < self.minimum_holding:
            reason = "MIN_HOLD_PRESERVED"
        elif normal_target == 0:
            reason = "CONTROL_CASH_EXIT_PRESERVED"
        else:
            eligible, ranked = relative_horizon_candidates(
                date=day,
                incumbent=current,
                control_target=int(normal_target),
                symbols=self.symbols,
                aggregate=self.aggregate[day],
                forecasts=self.forecasts,
                min_group_advantage=self.min_group_advantage,
                require_each_horizon=self.require_each_horizon,
                effective_margin=self.margin,
            )
            candidate_count = len(eligible)
            if self.ranking_audit is not None:
                self.ranking_audit.extend([
                    {
                        "fold_id": self.fold_id,
                        "holding_sessions": holding,
                        "rule": "each_horizon" if self.require_each_horizon else "group_average",
                        **item,
                    } for item in ranked
                ])
            if eligible:
                chosen = eligible[0]
                proposal = int(chosen["candidate_position"])
                if proposal != normal_target:
                    action = proposal
                    score = float(self.aggregate[day][proposal])
                    reason = "RELATIVE_HORIZON_SWITCH"
                else:
                    reason = "CONTROL_TARGET_ALREADY_BEST"
            else:
                reason = "NO_SUPERIOR_HORIZON_CANDIDATE"
        if self.audit is not None:
            self.audit.append({
                "decision_date": day.isoformat(),
                "fold_id": self.fold_id,
                "rule": "each_horizon" if self.require_each_horizon else "group_average",
                "holding_sessions": holding,
                "current_asset": self.symbols[current - 1] if current > 0 else "CASH",
                "control_target": self.symbols[normal_target - 1] if normal_target > 0 else "CASH",
                "final_target": self.symbols[action - 1] if action > 0 else "CASH",
                "reason": reason,
                "changes_control_decision": action != normal_target,
                "eligible_candidate_count": candidate_count,
                "chosen_delta_short_pct_points": (
                    chosen["delta_short_pct_points"] if chosen else None
                ),
                "chosen_delta_medium_pct_points": (
                    chosen["delta_medium_pct_points"] if chosen else None
                ),
            })
        return action, score



def build_horizon_return_labels(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    config: Any,
    horizons: tuple[int, ...] = ALL_RESEARCH_HORIZONS,
) -> None:
    """Recalcula TODOS os H independentemente do horizonte maximo.

    Precos e features de t conhecidos; rotulos t+H servem APENAS
    ao treinamento pre-OOS e avaliacao posterior. Nunca a selecao.
    Reindexar primeiro evita encolher H quando ativo perde dias.
    """
    cost = min(
        0.25, 2.0 * (
            max(0.0, float(config.slippage_bps)) / 10000.0
            + max(0.0, float(config.commission_rate))
        )
    )
    calendar = pd.DatetimeIndex(common_dates)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Fixed calendar must be ordered and unique")
    for symbol, frame in frames.items():
        if not frame.index.is_unique:
            raise RuntimeError(f"{symbol}: duplicated price index")
        open_prices = pd.to_numeric(
            frame["open"].reindex(calendar), errors="coerce"
        )
        closes = pd.to_numeric(
            frame["close"].reindex(calendar), errors="coerce"
        )
        entry = open_prices.shift(-1)
        for h in horizons:
            exit_price = closes.shift(-int(h))
            valid = (
                np.isfinite(entry) & (entry > 0)
                & np.isfinite(exit_price) & (exit_price > 0)
            )
            net_log = np.log((exit_price / entry).where(valid)) + math.log1p(-cost)
            target = f"forward_horizon_net_return_pct_{h}"
            frame[target] = np.expm1(net_log.reindex(frame.index))
            frame.loc[~np.isfinite(frame[target]), target] = np.nan

def fixed_horizon_value(
    *,
    day: pd.Timestamp,
    horizon: int,
    symbol: str,
    frames: dict[str, pd.DataFrame],
    maturity: dict[pd.Timestamp, int],
) -> float | None:
    """Outcome so depois da selecao, com maturidade dentro do MESMO fold."""
    key = pd.Timestamp(day)
    if maturity.get(key, 0) < horizon:
        return None
    frame = frames.get(symbol)
    target = f"forward_horizon_net_return_pct_{horizon}"
    if frame is None or key not in frame.index or target not in frame:
        return None
    actual = float(frame.at[key, target])
    return actual if np.isfinite(actual) else None


def select_horizon_candidates(
    *,
    day: pd.Timestamp,
    incumbent: str,
    symbols: list[str],
    forecast: dict[int, dict[pd.Timestamp, np.ndarray]],
    horizons: tuple[int, ...],
    extra_switch_cost: float,
    top_k: int = AUDIT_TOP_K,
    capture_require_short_medium_advantage: bool = False,
    candidate_cost_fn: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ranking somente por previsoes da data t, nunca por label futuro."""
    if incumbent not in symbols:
        return [], {"decision_date": pd.Timestamp(day).isoformat(),
                    "incumbent": incumbent, "best_candidate": None}
    i = symbols.index(incumbent) + 1
    expected_len = len(symbols) + 1
    arrays: dict[int, np.ndarray] = {}
    timestamp = pd.Timestamp(day)
    for h in horizons:
        item = forecast.get(h, {}).get(timestamp)
        if item is None:
            raise RuntimeError(f"No point-in-time forecasts: {timestamp} H{h}")
        arr = np.asarray(item, dtype=np.float64)
        if len(arr) != expected_len or not np.isfinite(arr[i]):
            raise RuntimeError(
                f"Invalid incumbent forecast {timestamp} {incumbent} H{h}"
            )
        arrays[h] = arr
    records: list[dict[str, Any]] = []
    for pos, symbol in enumerate(symbols, start=1):
        if pos == i or not all(np.isfinite(arrays[h][pos]) for h in horizons):
            continue
        row: dict[str, Any] = {
            "decision_date": timestamp.isoformat(),
            "incumbent": incumbent,
            "candidate": symbol,
            "extra_switch_cost_fraction": extra_switch_cost,
        }
        per_session_advantages = []
        positive = 0
        actual_cost = (
            float(candidate_cost_fn(symbol))
            if candidate_cost_fn is not None else float(extra_switch_cost)
        )
        row["extra_switch_cost_fraction"] = actual_cost
        for h in horizons:
            candidate = float(arrays[h][pos])
            current = float(arrays[h][i])
            gap = candidate - current - actual_cost
            per_session_advantages.append(gap / float(h))
            positive += gap > 0.0
            row[f"H{h}_candidate_pred_pct"] = candidate
            row[f"H{h}_incumbent_pred_pct"] = current
            row[f"H{h}_pred_increment_pct_points"] = gap * 100.0
        row["positive_horizon_count"] = int(positive)
        row["mean_daily_pred_increment_pct_points"] = (
            float(np.mean(per_session_advantages)) * 100.0
        )
        records.append(row)
    records.sort(key=lambda r: (
        -float(r["mean_daily_pred_increment_pct_points"]),
        -int(r["positive_horizon_count"]), r["candidate"],
    ))
    def qualifies(row: dict[str, Any]) -> bool:
        if row["mean_daily_pred_increment_pct_points"] <= 0.0:
            return False
        if not capture_require_short_medium_advantage:
            return True
        short_mean = float(np.mean([
            row[f"H{h}_pred_increment_pct_points"] for h in (2, 3)
        ]))
        medium_mean = float(np.mean([
            row[f"H{h}_pred_increment_pct_points"] for h in (5, 10)
        ]))
        return bool(
            row["positive_horizon_count"] >= 3
            and short_mean > 0.0 and medium_mean > 0.0
        )

    selected = next((r for r in records if qualifies(r)), None)
    summary = {
        "decision_date": timestamp.isoformat(),
        "incumbent": incumbent,
        "best_candidate": selected["candidate"] if selected else None,
        "best_pred_mean_daily_gap_pp": (
            selected["mean_daily_pred_increment_pct_points"]
            if selected else None
        ),
        "positive_horizon_count": (
            selected["positive_horizon_count"] if selected else 0
        ),
        "eligible_candidate_count": sum(qualifies(r) for r in records),
        "ranked_candidate_count": len(records),
    }
    if selected is not None:
        summary["extra_switch_cost_fraction"] = selected[
            "extra_switch_cost_fraction"
        ]
        summary.update({
            f"H{h}_pred_increment_pct_points":
                selected[f"H{h}_pred_increment_pct_points"]
            for h in horizons
        })
    return records[:top_k], summary


def attach_fixed_horizon_outcomes(
    ranked_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...],
    folds_maturity: dict[int, dict[pd.Timestamp, int]],
    frames: dict[str, pd.DataFrame],
    extra_switch_cost: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Somente pos-hoc: labels jamais entram na escolha da carteira."""
    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        day = pd.Timestamp(row["decision_date"])
        available = folds_maturity[int(row["fold_id"])]
        current = str(row["incumbent"])
        candidate = row.get("candidate", row.get("best_candidate"))
        for h in horizons:
            key = f"H{h}_"
            now = fixed_horizon_value(
                day=day, horizon=h, symbol=current,
                frames=frames, maturity=available,
            )
            new = (
                fixed_horizon_value(
                    day=day, horizon=h, symbol=str(candidate),
                    frames=frames, maturity=available,
                )
                if candidate else None
            )
            enriched[key + "realized_incumbent_pct"] = now
            enriched[key + "realized_candidate_pct"] = new
            row_cost = float(
                row.get("extra_switch_cost_fraction", extra_switch_cost)
            )
            enriched[key + "realized_increment_pp"] = (
                (new - now - row_cost) * 100.0
                if new is not None and now is not None else None
            )
            predicted = row.get(key + "pred_increment_pct_points")
            enriched[key + "pred_vs_realized_gap_error_pp"] = (
                float(predicted) - enriched[key + "realized_increment_pp"]
                if predicted is not None
                and enriched[key + "realized_increment_pp"] is not None
                else None
            )
        return enriched
    return [evaluate(r) for r in ranked_rows], [
        evaluate(r) for r in selections
    ]


def summarize_selection_quality(
    evaluated: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Taxa de vantagem realizada e erro em pontos percentuais por fold/H."""
    output = []
    fold_ids = sorted({int(r["fold_id"]) for r in evaluated})
    for fold_id in [*fold_ids, None]:
        fold_rows = [
            r for r in evaluated
            if r.get("best_candidate") is not None
            and (fold_id is None or int(r["fold_id"]) == fold_id)
        ]
        for h in horizons:
            deltas = np.asarray([
                float(row[f"H{h}_realized_increment_pp"])
                for row in fold_rows
                if row.get(f"H{h}_realized_increment_pp") is not None
            ], dtype=float)
            pred = np.asarray([
                float(row[f"H{h}_pred_increment_pct_points"])
                for row in fold_rows
                if row.get(f"H{h}_realized_increment_pp") is not None
            ], dtype=float)
            output.append({
                "fold_id": fold_id if fold_id is not None else "ALL",
                "horizon": h,
                "selected": len(fold_rows),
                "matured_evaluable": len(deltas),
                "wins_vs_incumbent": int((deltas > 0).sum()),
                "hit_rate": float((deltas > 0).mean()) if len(deltas) else None,
                "mean_realized_increment_pp": (
                    float(deltas.mean()) if len(deltas) else None
                ),
                "mean_predicted_increment_pp": (
                    float(pred.mean()) if len(pred) else None
                ),
                "prediction_mae_pp": (
                    float(np.abs(pred - deltas).mean()) if len(pred) else None
                ),
                "prediction_rmse_pp": (
                    float(np.sqrt(np.mean((pred - deltas) ** 2)))
                    if len(pred) else None
                ),
                "sign_accuracy": (
                    float(np.mean((pred > 0) == (deltas > 0)))
                    if len(pred) else None
                ),
            })
    return output



PAIR_MIN_TRAIN_DAYS = 15
PAIR_CALIBRATION_FRACTION = 2.0 / 3.0
PAIR_ERROR_QUANTILE = 0.75


def estimated_incremental_pair_cost(
    *,
    day: pd.Timestamp,
    incumbent: str,
    control_target: str,
    candidate: str,
    frames: dict[str, pd.DataFrame],
    config: Any,
) -> float:
    """Friccao proporcional conservadora, calculada com fechamentos em t.

    Precos futuros, quantidade futura e capital final nao sao consultados.
    TAF/CAT por acao estimadas com notional de US$ 1 e sem cap,
    como limite conservador; nao substituir por taxas reais do backtest.
    """
    def one_side(symbol: str, side: str) -> float:
        frame = frames[symbol]
        price = float(frame.at[pd.Timestamp(day), "close"])
        if not np.isfinite(price) or price <= 0.0:
            raise RuntimeError(f"Invalid same-day quote {symbol} {day}")
        base = (
            max(0.0, float(config.slippage_bps)) / 10000.0
            + max(0.0, float(config.commission_rate))
            + max(0.0, float(config.cat_fee_per_share)) / price
        )
        if side == "SELL":
            base += (
                max(0.0, float(config.sec_fee_rate))
                + max(0.0, float(config.taf_fee_per_share)) / price
            )
        return float(base)

    if control_target == "CASH":
        return float("inf")
    # Se Control ja gira, compara apenas custo de compra do novo
    # candidato contra custo de compra do alvo congelado. O custo
    # de venda da posicao anterior se cancela nos dois cenarios.
    if control_target != incumbent:
        return max(
            0.0,
            one_side(candidate, "BUY") - one_side(control_target, "BUY"),
        )
    return (
        one_side(incumbent, "SELL") + one_side(candidate, "BUY")
    )


def historical_control_pair_context(
    *,
    days: pd.DatetimeIndex,
    policy: Any,
) -> list[dict[str, Any]]:
    """Estado historico do Control na calibracao, apenas decisoes conhecidas."""
    current, holding = 0, 0
    records: list[dict[str, Any]] = []
    for day in days:
        t = pd.Timestamp(day)
        action, _ = policy(t, current, holding)
        records.append({
            "decision_date": t,
            "incumbent_position": int(current),
            "control_position": int(action),
            "holding_days": int(holding),
        })
        if int(action) == current:
            holding = holding + 1 if current > 0 else 0
        else:
            current = int(action)
            holding = 1 if current > 0 else 0
    return records


def rank_direct_pair_gap(
    *,
    day: pd.Timestamp,
    incumbent: str,
    control_target: str,
    symbols: list[str],
    forecasts: dict[int, dict[pd.Timestamp, np.ndarray]],
    frames: dict[str, pd.DataFrame],
    config: Any,
    horizons: tuple[int, ...] = CAPTURE_HORIZONS,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Um par (candidato, benchmark Control) por linha, sem labels futuros."""
    if incumbent == "CASH" or control_target == "CASH":
        return []
    if incumbent not in symbols or control_target not in symbols:
        raise ValueError("Historical control asset outside locked universe")
    t = pd.Timestamp(day)
    pos_current = symbols.index(incumbent) + 1
    pos_control = symbols.index(control_target) + 1
    arrays: dict[int, np.ndarray] = {}
    for h in horizons:
        arr = forecasts.get(h, {}).get(t)
        if arr is None:
            raise RuntimeError(f"H{h} prediction missing on {t}")
        array = np.asarray(arr, dtype=float)
        if len(array) != len(symbols) + 1:
            raise RuntimeError("Horizon forecast universe mismatch")
        arrays[h] = array
    if any(
        not np.isfinite(arrays[h][pos_current])
        or not np.isfinite(arrays[h][pos_control])
        for h in horizons
    ):
        return []
    rows: list[dict[str, Any]] = []
    for pos, candidate in enumerate(symbols, start=1):
        if pos in (pos_current, pos_control):
            continue
        if any(not np.isfinite(arrays[h][pos]) for h in horizons):
            continue
        friction = estimated_incremental_pair_cost(
            day=t, incumbent=incumbent, control_target=control_target,
            candidate=candidate, frames=frames, config=config,
        )
        if not np.isfinite(friction):
            continue
        raw: dict[int, float] = {}
        current: dict[int, float] = {}
        for h in horizons:
            raw[h] = float(
                arrays[h][pos] - arrays[h][pos_control] - friction
            )
            current[h] = float(
                arrays[h][pos] - arrays[h][pos_current] - friction
            )
        short = float(np.mean([raw[h] for h in (2, 3)]))
        medium = float(np.mean([raw[h] for h in (5, 10)]))
        current_short = float(np.mean([current[h] for h in (2, 3)]))
        current_medium = float(np.mean([current[h] for h in (5, 10)]))
        score = float(np.mean([raw[h] / h for h in horizons]))
        qualifies = (
            short > 0.0 and medium > 0.0
            and current_short > 0.0 and current_medium > 0.0
            and sum(raw[h] > 0 for h in horizons) >= 3
        )
        rows.append({
            "decision_date": t.isoformat(),
            "incumbent": incumbent,
            "control_target": control_target,
            "candidate": candidate,
            "incremental_cost_fraction": friction,
            "raw_pair_mean_daily_gap": score,
            "raw_pair_short_gap": short,
            "raw_pair_medium_gap": medium,
            "raw_incumbent_short_gap": current_short,
            "raw_incumbent_medium_gap": current_medium,
            "raw_qualifies": qualifies,
            **{f"H{h}_raw_pair_gap_fraction": raw[h] for h in horizons},
            **{f"H{h}_raw_incumbent_gap_fraction": current[h]
               for h in horizons},
        })
    rows.sort(key=lambda r: (-r["raw_pair_mean_daily_gap"], r["candidate"]))
    return rows[:limit] if limit > 0 else rows


def train_pairwise_winner_calibration(
    *,
    selected: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    split_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    horizons: tuple[int, ...] = CAPTURE_HORIZONS,
    minimum_days: int = PAIR_MIN_TRAIN_DAYS,
) -> dict[str, Any]:
    """Slope/uncertainty do par vencedor, estimados SO com labels maturados.

    O embargo ate split_date impede que um rotulo de treino atravesse
    para a validacao historica; OOS jamais participa.
    """
    dates = pd.DatetimeIndex(calendar)
    lookup = {pd.Timestamp(t): i for i, t in enumerate(dates)}
    split = lookup[pd.Timestamp(split_date)]
    cutoff = lookup[pd.Timestamp(cutoff_date)]
    evidence: dict[int, list[tuple[float, float, str]]] = {
        h: [] for h in horizons
    }
    for row in selected:
        t = pd.Timestamp(row["decision_date"])
        index = lookup.get(t)
        if index is None:
            raise RuntimeError("Pair training date not in calendar")
        if t >= split_date:
            continue
        candidate, reference = row["candidate"], row["control_target"]
        friction = float(row["incremental_cost_fraction"])
        for h in horizons:
            if index + h >= split or index + h >= cutoff:
                continue
            source = f"forward_horizon_net_return_pct_{h}"
            cand = float(frames[candidate].at[t, source])
            base = float(frames[reference].at[t, source])
            if not (np.isfinite(cand) and np.isfinite(base)):
                continue
            predicted = float(row[f"H{h}_raw_pair_gap_fraction"])
            realized = cand - base - friction
            evidence[h].append((predicted, realized, t.isoformat()))
    calib: dict[str, Any] = {
        "split_date": pd.Timestamp(split_date).isoformat(),
        "cutoff_date": pd.Timestamp(cutoff_date).isoformat(),
        "quantile": PAIR_ERROR_QUANTILE,
        "by_horizon": {},
    }
    for h in horizons:
        values = evidence[h]
        distinct = len({x[2] for x in values})
        if distinct < minimum_days:
            raise RuntimeError(
                f"Insufficient distinct causal pair-days H{h}: {distinct}"
            )
        x = np.asarray([v[0] for v in values], dtype=float)
        y = np.asarray([v[1] for v in values], dtype=float)
        slope = float(np.clip(
            float(np.dot(x, y)) / max(float(np.dot(x, x)), 1e-12),
            0.0, 1.0,
        ))
        residual = y - slope * x
        q75 = float(np.quantile(np.abs(residual), PAIR_ERROR_QUANTILE))
        calib["by_horizon"][str(h)] = {
            "slope": slope,
            "abs_residual_p75": q75,
            "training_unique_days": distinct,
            "training_pairs": len(values),
            "mean_raw_predicted_gap": float(x.mean()),
            "mean_realized_pair_gap": float(y.mean()),
        }
    return calib


def calibrated_pair_choice(
    ranking: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    require_confidence: bool = True,
    horizons: tuple[int, ...] = CAPTURE_HORIZONS,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Ajusta diferenca entre pares; nao seleciona parametros por OOS."""
    enriched = []
    for base in ranking:
        row = dict(base)
        gaps = {}
        current = {}
        for h in horizons:
            params = calibration["by_horizon"][str(h)]
            slope = float(params["slope"])
            gaps[h] = slope * float(row[f"H{h}_raw_pair_gap_fraction"])
            current[h] = slope * float(row[f"H{h}_raw_incumbent_gap_fraction"])
            row[f"H{h}_calibrated_pair_gap_fraction"] = gaps[h]
            row[f"H{h}_historical_error_p75"] = float(params["abs_residual_p75"])
        row["calibrated_mean_daily_gap"] = float(
            np.mean([gaps[h] / h for h in horizons])
        )
        row["calibrated_short_gap"] = float(np.mean(
            [gaps[2], gaps[3]]
        ))
        row["calibrated_medium_gap"] = float(np.mean(
            [gaps[5], gaps[10]]
        ))
        row["calibrated_incumbent_short_gap"] = float(np.mean(
            [current[2], current[3]]
        ))
        row["calibrated_incumbent_medium_gap"] = float(np.mean(
            [current[5], current[10]]
        ))
        row["calibrated_min_confidence_daily"] = float(np.mean([
            float(calibration["by_horizon"][str(h)]["abs_residual_p75"])
            / h for h in horizons
        ]))
        row["calibrated_qualifies"] = bool(
            base["raw_qualifies"]
            and row["calibrated_short_gap"] > 0.0
            and row["calibrated_medium_gap"] > 0.0
            and row["calibrated_incumbent_short_gap"] > 0.0
            and row["calibrated_incumbent_medium_gap"] > 0.0
            and (not require_confidence or
                 row["calibrated_mean_daily_gap"] >
                 row["calibrated_min_confidence_daily"])
        )
        enriched.append(row)
    enriched.sort(key=lambda r: (
        -r["calibrated_mean_daily_gap"], r["candidate"]
    ))
    return next((r for r in enriched if r["calibrated_qualifies"]), None), enriched


def assess_pairwise_fixed_outcomes(
    records: list[dict[str, Any]],
    *,
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    horizon_end_exclusive: pd.Timestamp,
    horizons: tuple[int, ...] = CAPTURE_HORIZONS,
) -> list[dict[str, Any]]:
    """Liga observacoes futuras SOMENTE apos escolha do par.

    Dentro do proprio fold, incluindo a primeira decisao pre-abertura;
    nenhuma observacao cruza limite do fold.
    """
    days = pd.DatetimeIndex(calendar)
    index = {pd.Timestamp(t): i for i, t in enumerate(days)}
    cutoff = int(days.searchsorted(pd.Timestamp(horizon_end_exclusive), side="left"))
    outcome = []
    for record in records:
        row = dict(record)
        day = pd.Timestamp(row["decision_date"])
        offset = index[day]
        for h in horizons:
            key = f"H{h}"
            reference = row["control_target"]
            candidate = row.get("candidate")
            if candidate is None or offset + h >= cutoff:
                row[key + "_realized_gap_fraction"] = None
                row[key + "_hit"] = None
                continue
            source = f"forward_horizon_net_return_pct_{h}"
            cand = float(frames[candidate].at[day, source])
            base = float(frames[reference].at[day, source])
            if not (np.isfinite(cand) and np.isfinite(base)):
                row[key + "_realized_gap_fraction"] = None
                row[key + "_hit"] = None
                continue
            cost = float(row["incremental_cost_fraction"])
            gap = cand - base - cost
            row[key + "_realized_gap_fraction"] = gap
            row[key + "_hit"] = bool(gap > 0.0)
            current_asset = row.get("incumbent")
            current_realized = (
                float(frames[current_asset].at[day, source])
                if current_asset in frames and day in frames[current_asset].index
                else np.nan
            )
            row[key + "_realized_vs_incumbent_fraction"] = (
                cand - current_realized - cost
                if np.isfinite(current_realized) else None
            )
            prediction_key = (
                f"{key}_calibrated_pair_gap_fraction"
                if f"{key}_calibrated_pair_gap_fraction" in row
                else f"{key}_raw_pair_gap_fraction"
            )
            row[key + "_prediction_error_fraction"] = (
                float(row[prediction_key]) - gap
                if prediction_key in row else None
            )
        outcome.append(row)
    return outcome


def aggregate_pairwise_quality(
    evaluated: list[dict[str, Any]],
    horizons: tuple[int, ...] = CAPTURE_HORIZONS,
) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted({int(r["fold_id"]) for r in evaluated}):
        selected = [r for r in evaluated if int(r["fold_id"]) == fold]
        for h in horizons:
            available = [
                r for r in selected
                if r.get(f"H{h}_realized_gap_fraction") is not None
            ]
            realized = np.asarray([
                float(r[f"H{h}_realized_gap_fraction"])
                for r in available
            ], dtype=float)
            error = np.asarray([
                float(r[f"H{h}_prediction_error_fraction"])
                for r in available if r.get(
                    f"H{h}_prediction_error_fraction"
                ) is not None
            ], dtype=float)
            incumbent_realized = np.asarray([
                float(r[f"H{h}_realized_vs_incumbent_fraction"])
                for r in available
                if r.get(f"H{h}_realized_vs_incumbent_fraction") is not None
            ], dtype=float)
            rows.append({
                "fold_id": fold,
                "horizon": h,
                "selected": len(selected),
                "matured_evaluable": len(available),
                "realized_vs_incumbent_count": len(incumbent_realized),
                "mean_realized_vs_incumbent": (
                    float(incumbent_realized.mean())
                    if len(incumbent_realized) else None
                ),
                "incumbent_outperformance_rate": (
                    float(np.mean(incumbent_realized > 0.0))
                    if len(incumbent_realized) else None
                ),
                "realized_win_rate": (
                    float(np.mean(realized > 0)) if len(realized) else None
                ),
                "mean_realized_pair_gap_fraction": (
                    float(realized.mean()) if len(realized) else None
                ),
                "prediction_mae": (
                    float(np.abs(error).mean()) if len(error) else None
                ),
                "prediction_rmse": (
                    float(np.sqrt(np.mean(error**2))) if len(error) else None
                ),
            })
    return rows



def pair_feature_columns() -> list[str]:
    """Contrato de features direto, com orientacao candidato/referencia."""
    return [
        f"{prefix}_{feature}"
        for prefix in ("candidate", "reference", "difference")
        for feature in PAIR_TRAINING_FEATURES
    ]


def build_pair_feature_cube(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calendar: pd.DatetimeIndex,
) -> np.ndarray:
    """Feature(t) compartilhada, sem labels futuros ou preco de t+1."""
    return np.stack([
        frame.reindex(calendar)[list(PAIR_TRAINING_FEATURES)].to_numpy(
            dtype=np.float32
        )
        for symbol in symbols
        for frame in [frames[symbol]]
    ], axis=1)


def direct_pair_feature_block(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """A ordem importa: (A,B,A-B) difere de (B,A,B-A)."""
    a, b = np.asarray(candidate, dtype=np.float32), np.asarray(
        reference, dtype=np.float32
    )
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Candidate/reference feature matrices must match")
    return np.concatenate((a, b, a - b), axis=1)


def train_direct_pair_dataset(
    *,
    feature_cube: np.ndarray,
    target_cube: np.ndarray,
    training_date_indices: np.ndarray,
    horizon: int,
    evaluation_start_index: int,
    reference_count: int = PAIR_REFERENCE_COUNT,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Pares elegiveis por features e labels maturos antes da avaliacao.

    References sao rotativas e deterministicas, sem ranking por retorno
    realizado; produz A-B e B-A para simetria dos exemplos.
    """
    if reference_count < 1:
        raise ValueError("Need at least one historical reference")
    if feature_cube.ndim != 3 or target_cube.ndim != 2:
        raise ValueError("Invalid market arrays for pair training")
    n_dates, n_symbols, n_features = feature_cube.shape
    if target_cube.shape != (n_dates, n_symbols):
        raise ValueError("Target calendar or universe mismatch")
    if n_features != len(PAIR_TRAINING_FEATURES):
        raise ValueError("Feature count mismatch")
    if reference_count >= n_symbols:
        raise ValueError("Reference count must be smaller than universe")
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    unique_days = 0
    for day_index in training_date_indices:
        k = int(day_index)
        if not (0 <= k < n_dates) or k + int(horizon) >= evaluation_start_index:
            raise RuntimeError("Pair training target crosses evaluation boundary")
        row_features = feature_cube[k]
        label = target_cube[k]
        valid = np.isfinite(label) & np.isfinite(row_features).all(axis=1)
        references = [
            (k * 7 + j * max(1, n_symbols // reference_count)) % n_symbols
            for j in range(reference_count)
        ]
        if len(set(references)) != len(references):
            raise RuntimeError("Duplicate pair training references")
        added = 0
        for ref in references:
            if not valid[ref]:
                continue
            candidates = np.flatnonzero(valid)
            candidates = candidates[candidates != ref]
            if not len(candidates):
                continue
            cand = row_features[candidates]
            base = np.broadcast_to(
                row_features[ref], cand.shape
            )
            x = direct_pair_feature_block(cand, base)
            y = label[candidates] - label[ref]
            train_x.extend((x, direct_pair_feature_block(base, cand)))
            train_y.extend((y, -y))
            added += len(candidates)
        unique_days += int(added > 0)
    if unique_days < PAIR_MIN_TRAIN_DAYS or not train_x:
        raise RuntimeError(
            f"Pair training insufficient matured days={unique_days}"
        )
    values = np.concatenate(train_x, axis=0)
    labels = np.concatenate(train_y, axis=0)
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise RuntimeError("Invalid direct-pair training numeric values")
    return (
        pd.DataFrame(values, columns=pair_feature_columns()),
        labels.astype(np.float32),
        {
            "training_unique_days": unique_days,
            "examples": len(labels),
            "reference_count": reference_count,
            "horizon": horizon,
            "training_calendar_first_index": int(training_date_indices[0]),
            "training_calendar_last_index": int(training_date_indices[-1]),
            "evaluation_start_index": int(evaluation_start_index),
        },
    )


def fit_direct_pair_model(
    features: pd.DataFrame,
    labels: np.ndarray,
    config: Any,
    *,
    horizon: int,
    fold_id: int,
    phase: str,
) -> Any:
    """LightGBM de delta relativo, sem usar regressao individual."""
    from lightgbm import LGBMRegressor

    setting = _configuracoes_lightgbm(config)
    if len(features) != len(labels):
        raise ValueError("Direct-pair label/features length mismatch")
    if len(features) < int(config.rotation_minimum_training_rows):
        raise RuntimeError("Pair learning insufficient training examples")
    model = LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=int(setting["n_estimators"]),
        learning_rate=float(setting["learning_rate"]),
        max_depth=int(setting["max_depth"]),
        num_leaves=int(setting["num_leaves"]),
        min_child_samples=int(setting["min_child_samples"]),
        min_child_weight=float(setting["min_child_weight"]),
        subsample=float(setting["subsample"]),
        subsample_freq=int(setting["subsample_freq"]),
        colsample_bytree=float(setting["colsample_bytree"]),
        reg_alpha=float(setting["reg_alpha"]),
        reg_lambda=float(setting["reg_lambda"]),
        max_bin=int(setting["max_bin"]),
        random_state=int(config.random_state),
        n_jobs=1,
        device_type="cpu",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(features, labels)
    print(
        f"[pair-direct] fold={fold_id} H{horizon} phase={phase} "
        f"train_examples={len(labels)}",
        flush=True,
    )
    return model


def predict_direct_pair_gaps(
    models: dict[int, Any],
    row_features: np.ndarray,
    *,
    symbols: list[str],
    incumbent: str,
    control_target: str,
    day: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    config: Any,
    top_k: int = 0,
) -> list[dict[str, Any]]:
    """Prevê A-Control e A-incumbent para TODOS os candidatos elegiveis."""
    if (incumbent == "CASH" or control_target == "CASH"
        or incumbent not in symbols or control_target not in symbols):
        return []
    n = len(symbols)
    if row_features.shape != (n, len(PAIR_TRAINING_FEATURES)):
        raise ValueError("Pair daily features shape mismatch")
    positions = [
        pos for pos in range(n)
        if symbols[pos] not in (incumbent, control_target)
        and np.isfinite(row_features[pos]).all()
    ]
    if not positions:
        return []
    reference_index = symbols.index(control_target)
    current_index = symbols.index(incumbent)
    if not (
        np.isfinite(row_features[reference_index]).all()
        and np.isfinite(row_features[current_index]).all()
    ):
        return []
    candidate_features = row_features[positions]
    control_features = np.broadcast_to(
        row_features[reference_index], candidate_features.shape
    )
    incumbent_features = np.broadcast_to(
        row_features[current_index], candidate_features.shape
    )
    block_control = pd.DataFrame(
        direct_pair_feature_block(candidate_features, control_features),
        columns=pair_feature_columns(),
    )
    block_incumbent = pd.DataFrame(
        direct_pair_feature_block(candidate_features, incumbent_features),
        columns=pair_feature_columns(),
    )
    predicted = {
        h: np.asarray(model.predict(block_control), dtype=float)
        for h, model in models.items()
    }
    predicted_current = {
        h: np.asarray(model.predict(block_incumbent), dtype=float)
        for h, model in models.items()
    }
    rows = []
    for i, pos in enumerate(positions):
        symbol = symbols[pos]
        cost = estimated_incremental_pair_cost(
            day=day, incumbent=incumbent, control_target=control_target,
            candidate=symbol, frames=frames, config=config,
        )
        if not np.isfinite(cost):
            continue
        by_h = {h: float(predicted[h][i]) - cost for h in CAPTURE_HORIZONS}
        by_current = {
            h: float(predicted_current[h][i]) - cost for h in CAPTURE_HORIZONS
        }
        if any(not np.isfinite(v) for v in [*by_h.values(), *by_current.values()]):
            continue
        score = float(np.mean([by_h[h] / h for h in CAPTURE_HORIZONS]))
        qualifies = (
            float(np.mean([by_h[h] for h in (2, 3)])) > 0
            and float(np.mean([by_h[h] for h in (5, 10)])) > 0
            and float(np.mean([by_current[h] for h in (2, 3)])) > 0
            and float(np.mean([by_current[h] for h in (5, 10)])) > 0
            and sum(by_h[h] > 0 for h in CAPTURE_HORIZONS) >= 3
        )
        rows.append({
            "decision_date": pd.Timestamp(day).isoformat(),
            "incumbent": incumbent,
            "control_target": control_target,
            "candidate": symbol,
            "incremental_cost_fraction": cost,
            "raw_pair_mean_daily_gap": score,
            "raw_qualifies": bool(qualifies),
            **{f"H{h}_raw_pair_gap_fraction": by_h[h]
               for h in CAPTURE_HORIZONS},
            **{f"H{h}_raw_incumbent_gap_fraction": by_current[h]
               for h in CAPTURE_HORIZONS},
        })
    rows.sort(key=lambda r: (-r["raw_pair_mean_daily_gap"], r["candidate"]))
    return rows[:top_k] if top_k > 0 else rows


def calibrate_direct_pair_confidence(
    training_rows: list[dict[str, Any]],
    *,
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    split_date: pd.Timestamp,
    min_distinct_days: int = PAIR_MIN_TRAIN_DAYS,
) -> dict[str, Any]:
    """Estima vies e erro das previsoes do VENCEDOR, apenas dados maduros."""
    calendar_index = {pd.Timestamp(t): i for i, t in enumerate(calendar)}
    split_index = calendar_index[pd.Timestamp(split_date)]
    evidence: dict[int, list[tuple[float, float, str]]] = {
        h: [] for h in CAPTURE_HORIZONS
    }
    for row in training_rows:
        day = pd.Timestamp(row["decision_date"])
        index = calendar_index[day]
        if index >= split_index:
            continue
        for h in CAPTURE_HORIZONS:
            if index + h >= split_index:
                continue
            label = f"forward_horizon_net_return_pct_{h}"
            candidate = float(frames[row["candidate"]].at[day, label])
            baseline = float(frames[row["control_target"]].at[day, label])
            if not (np.isfinite(candidate) and np.isfinite(baseline)):
                continue
            actual = candidate - baseline - float(
                row["incremental_cost_fraction"]
            )
            pred = float(row[f"H{h}_raw_pair_gap_fraction"])
            evidence[h].append((pred, actual, day.isoformat()))
    calibration: dict[str, Any] = {
        "split_date": pd.Timestamp(split_date).isoformat(),
        "by_horizon": {},
        "method": "winner_pair_median_bias_and_p75_absolute_residual",
    }
    for h, sample in evidence.items():
        days_count = len({v[2] for v in sample})
        if days_count < min_distinct_days:
            raise RuntimeError(
                f"Insufficient mature winner-pair days H{h}: {days_count}"
            )
        errors = np.asarray([v[1] - v[0] for v in sample], dtype=float)
        bias = float(np.median(errors))
        dispersion = float(np.quantile(
            np.abs(errors - bias), PAIR_ERROR_QUANTILE
        ))
        calibration["by_horizon"][str(h)] = {
            "bias_correction": bias,
            "abs_residual_p75": dispersion,
            "training_unique_days": days_count,
            "mean_raw_gap": float(np.mean([v[0] for v in sample])),
            "mean_actual_gap": float(np.mean([v[1] for v in sample])),
        }
    return calibration


def choose_confident_direct_pair(
    ranking: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    require_confidence: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Mesmo filtro previamente registrado para todos os folds."""
    enriched: list[dict[str, Any]] = []
    for base in ranking:
        row = dict(base)
        corrected, current, lower = {}, {}, {}
        for h in CAPTURE_HORIZONS:
            params = calibration["by_horizon"][str(h)]
            correction = float(params["bias_correction"])
            error = float(params["abs_residual_p75"])
            corrected[h] = float(row[f"H{h}_raw_pair_gap_fraction"]) + correction
            current[h] = float(row[f"H{h}_raw_incumbent_gap_fraction"]) + correction
            lower[h] = corrected[h] - error
            row[f"H{h}_corrected_pair_gap_fraction"] = corrected[h]
            row[f"H{h}_calibrated_pair_gap_fraction"] = corrected[h]
            row[f"H{h}_pair_lower_bound_fraction"] = lower[h]
        row["corrected_mean_daily_gap"] = float(np.mean([
            corrected[h] / h for h in CAPTURE_HORIZONS
        ]))
        row["mean_lower_daily_gap"] = float(np.mean([
            lower[h] / h for h in CAPTURE_HORIZONS
        ]))
        row["calibrated_qualifies"] = bool(
            row["raw_qualifies"]
            and float(np.mean([corrected[h] for h in (2, 3)])) > 0
            and float(np.mean([corrected[h] for h in (5, 10)])) > 0
            and float(np.mean([current[h] for h in (2, 3)])) > 0
            and float(np.mean([current[h] for h in (5, 10)])) > 0
            and (
                not require_confidence
                or (
                    float(np.mean([lower[h] for h in (2, 3)])) > 0
                    and float(np.mean([lower[h] for h in (5, 10)])) > 0
                )
            )
        )
        enriched.append(row)
    enriched.sort(key=lambda r: (-r["mean_lower_daily_gap"], r["candidate"]))
    winner = next(
        (row for row in enriched if row["calibrated_qualifies"]), None
    )
    return winner, enriched


def fit_ranker_h05(
    *,
    feature_cube: np.ndarray,
    targets: np.ndarray,
    training_indices: np.ndarray,
    evaluation_start_index: int,
    config: Any,
    fold_id: int,
) -> tuple[Any, dict[str, Any]]:
    """LambdaMART opcional como competidor supervisionado, nao pos-processo.

    A relevancia vem SOMENTE de labels historicos plenamente maduros.
    O objetivo e comparar com os rankings sem treinamento adicional.
    """
    from lightgbm import LGBMRanker

    training_blocks: list[np.ndarray] = []
    training_labels: list[np.ndarray] = []
    groups: list[int] = []
    for index in training_indices:
        k = int(index)
        if k + 5 >= int(evaluation_start_index):
            raise RuntimeError("LTR label crosses chronological cutoff")
        obs = feature_cube[k]
        truth = targets[k]
        valid = np.isfinite(obs).all(axis=1) & np.isfinite(truth)
        selected = np.flatnonzero(valid)
        if len(selected) < 5:
            continue
        expected = truth[selected]
        order = np.argsort(expected, kind="stable")
        relevance = np.empty(len(selected), dtype=np.int32)
        relevance[order] = np.minimum(
            4, np.floor(
                np.arange(len(selected), dtype=float) * 5.0 / len(selected)
            ).astype(np.int32)
        )
        training_blocks.append(obs[selected])
        training_labels.append(relevance)
        groups.append(len(selected))
    if len(groups) < int(config.rotation_minimum_training_rows):
        raise RuntimeError(
            f"Too few historical LTR query days fold={fold_id}: {len(groups)}"
        )
    x = np.concatenate(training_blocks, axis=0)
    y = np.concatenate(training_labels, axis=0)
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=240, learning_rate=0.035, num_leaves=31,
        min_child_samples=30, reg_alpha=0.05, reg_lambda=0.20,
        random_state=int(config.random_state), n_jobs=1,
        verbosity=-1, deterministic=True, force_col_wise=True,
    )
    model.fit(
        pd.DataFrame(x, columns=list(PAIR_TRAINING_FEATURES)),
        y, group=groups,
    )
    return model, {
        "model": "LGBMRanker_LambdaMART",
        "target_horizon": 5,
        "training_query_days": len(groups),
        "training_asset_rows": int(len(x)),
        "last_training_date_index": int(training_indices[-1]),
        "evaluation_start_index": int(evaluation_start_index),
        "label_maturity_guard": True,
    }


def calibrate_asset_horizon_errors(
    *,
    forecasts: dict[int, dict[pd.Timestamp, np.ndarray]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_days: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    split_date: pd.Timestamp,
    horizons: tuple[int, ...] = RANKING_EVALUATION_HORIZONS,
) -> dict[int, dict[str, np.ndarray | int]]:
    """Media e desvio dos erros OOF anteriores a validacao historica.

    A estimativa probabilistica usa aproximacao gaussiana marginal.
    A correlacao de erro entre ativos nao foi estimada.
    """
    dates = pd.DatetimeIndex(calendar)
    lookup = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    end = lookup[pd.Timestamp(split_date)]
    n = len(symbols)
    summary: dict[int, dict[str, np.ndarray | int]] = {}
    for h in horizons:
        errors: list[list[float]] = [[] for _ in symbols]
        effective_days = 0
        for day in calibration_days:
            t = pd.Timestamp(day)
            k = lookup[t]
            if k + h >= end:
                continue
            predicted = forecasts[h].get(t)
            if predicted is None:
                raise RuntimeError(f"Missing historical forecast H{h} {t}")
            predicted = np.asarray(predicted, dtype=float)
            if len(predicted) != n + 1:
                raise RuntimeError("Historical calibration universe drift")
            daily = 0
            for i, sym in enumerate(symbols):
                if t not in frames[sym].index:
                    continue
                observed = float(
                    frames[sym].at[t, f"forward_horizon_net_return_pct_{h}"]
                )
                estimate = float(predicted[i + 1])
                if np.isfinite(observed) and np.isfinite(estimate):
                    errors[i].append(observed - estimate)
                    daily += 1
            effective_days += int(daily > 0)
        if effective_days < RANKING_CALIBRATION_MIN_DAYS:
            raise RuntimeError(
                f"Not enough matured calibration days for H{h}: {effective_days}"
            )
        pooled = np.asarray(
            [e for asset in errors for e in asset], dtype=float
        )
        if len(pooled) < 20:
            raise RuntimeError("Insufficient calibration observations")
        pooled_std = max(
            RANKING_PROBABILITY_VARIANCE_FLOOR,
            float(np.std(pooled, ddof=1)),
        )
        center = np.empty(n, dtype=float)
        scale = np.empty(n, dtype=float)
        for i, items in enumerate(errors):
            if len(items) < RANKING_CALIBRATION_MIN_DAYS:
                center[i] = float(np.mean(pooled))
                scale[i] = pooled_std
            else:
                vec = np.asarray(items, dtype=float)
                center[i] = float(np.mean(vec))
                scale[i] = max(
                    RANKING_PROBABILITY_VARIANCE_FLOOR,
                    float(np.std(vec, ddof=1)),
                )
        summary[h] = {
            "center": center,
            "scale": scale,
            "effective_days": effective_days,
            "min_asset_samples": int(min(map(len, errors))),
            "pooled_error_std": pooled_std,
        }
    return summary


def nondominated_layers(values: np.ndarray) -> np.ndarray:
    """Pareto fronts, O(n^2) por sessao, sem usar retornos observados."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Pareto input must be a finite 2D matrix")
    count = len(values)
    domination_count = np.zeros(count, dtype=np.int32)
    dominates: list[list[int]] = [[] for _ in range(count)]
    for i in range(count):
        better_equal = np.all(values[i] >= values, axis=1)
        better_strict = np.any(values[i] > values, axis=1)
        inferior = np.flatnonzero(better_equal & better_strict)
        dominates[i] = inferior.tolist()
        domination_count[inferior] += 1
    fronts = np.full(count, -1, dtype=np.int32)
    active = np.flatnonzero(domination_count == 0).tolist()
    level = 0
    while active:
        remaining = []
        for i in active:
            fronts[i] = level
            for j in dominates[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    remaining.append(j)
        active = sorted(set(remaining))
        level += 1
    if np.any(fronts < 0):
        raise RuntimeError("Pareto ranking not fully assigned")
    return fronts


def rank_with_all_methods(
    *,
    forecasts: dict[int, np.ndarray],
    symbols: list[str],
    reference_symbol: str,
    incremental_cost: np.ndarray,
    volatility: np.ndarray,
    historical_error: dict[int, dict[str, np.ndarray | int]],
    ltr_scores: np.ndarray,
    horizons: tuple[int, ...] = RANKING_EVALUATION_HORIZONS,
) -> dict[str, list[tuple[int, float]]]:
    """Oito rankings independentes sobre o MESMO universo da sessao.

    Somente previsoes, atributos presentes, custos estimados e
    calibracoes historicas; nunca inclui labels OOS no score.
    """
    n = len(symbols)
    ref = symbols.index(reference_symbol)
    predicted = np.column_stack([
        np.asarray(forecasts[h], dtype=float) for h in horizons
    ])
    costs = np.asarray(incremental_cost, dtype=float)
    risk = np.asarray(volatility, dtype=float)
    learned = np.asarray(ltr_scores, dtype=float)
    if (
        predicted.shape != (n, len(horizons))
        or costs.shape != (n,) or risk.shape != (n,)
        or learned.shape != (n,)
    ):
        raise ValueError("Rank methods universe mismatch")
    eligible = (
        np.isfinite(predicted).all(axis=1)
        & np.isfinite(costs) & np.isfinite(risk)
        & np.isfinite(learned)
    )
    if not eligible[ref]:
        return {}
    ids = np.flatnonzero(eligible)
    if len(ids) < 3:
        return {}
    horizons_arr = np.asarray(horizons, dtype=float)
    forecasts_by_h = predicted[ids]
    advantage = forecasts_by_h - predicted[ref] - costs[ids, None]
    daily = advantage / horizons_arr
    normalized_gain = daily.mean(axis=1)

    scores: dict[str, np.ndarray] = {
        "predicted_return": (forecasts_by_h / horizons_arr).mean(axis=1),
        "net_advantage": normalized_gain,
        "learning_to_rank": learned[ids],
    }
    # Borda: cada horizonte vota por posicao, e nao por magnitude extrema.
    ranks = np.empty_like(forecasts_by_h)
    for column in range(len(horizons)):
        order = sorted(
            range(len(ids)),
            key=lambda j: (-advantage[j, column], symbols[int(ids[j])]),
        )
        for rank, row_id in enumerate(order):
            ranks[row_id, column] = rank
    scores["borda"] = 1.0 - ranks.mean(axis=1) / max(1, len(ids) - 1)

    # TOPSIS: 4 retornos diarios, volatilidade, friccao, divergencia.
    criterion = np.column_stack([
        daily, -risk[ids], -costs[ids],
        -np.std(daily, axis=1),
    ])
    weights = np.asarray([0.15]*len(horizons)+[0.15, 0.15, 0.10])
    if len(weights) != criterion.shape[1]:
        raise RuntimeError("TOPSIS locked criteria dimension changed")
    den = np.maximum(
        np.linalg.norm(criterion, axis=0), 1e-12
    )
    scaled = criterion / den * weights
    ideal, anti = np.max(scaled, axis=0), np.min(scaled, axis=0)
    dplus = np.linalg.norm(scaled - ideal, axis=1)
    dminus = np.linalg.norm(scaled - anti, axis=1)
    scores["topsis"] = np.divide(
        dminus, dplus + dminus,
        out=np.full(len(ids), 0.5), where=(dplus + dminus) > 0,
    )

    # Pareto nao dominado: ganhos em 4 H e volatilidade menor.
    fronts = nondominated_layers(
        np.column_stack([daily, -risk[ids]])
    )
    scores["pareto"] = -fronts.astype(float) + (
        (ranks.shape[0] - ranks.mean(axis=1)) / max(2, len(ids))
    ) * 1e-2

    # Probabilistico e robusto: apenas erros maturados pre-OOS.
    relative_mean = np.column_stack([
        np.asarray(historical_error[h]["center"], dtype=float)[ids]
        - float(np.asarray(historical_error[h]["center"])[ref])
        for h in horizons
    ])
    relative_sigma = np.column_stack([
        np.sqrt(
            np.asarray(historical_error[h]["scale"], dtype=float)[ids] ** 2
            + float(np.asarray(historical_error[h]["scale"])[ref]) ** 2
        )
        for h in horizons
    ])
    relative_sigma = np.maximum(
        relative_sigma, RANKING_PROBABILITY_VARIANCE_FLOOR
    )
    from scipy.special import ndtr
    win_probability = ndtr(
        (advantage + relative_mean) / relative_sigma
    )
    scores["probabilistic"] = win_probability.mean(axis=1)
    scores["robust"] = np.min(
        (advantage + relative_mean - relative_sigma) / horizons_arr,
        axis=1,
    )

    return {
        method: [
            (int(ids[index]), float(value[index]))
            for index in sorted(
                range(len(ids)),
                key=lambda j: (-float(value[j]), symbols[int(ids[j])]),
            )
        ]
        for method, value in scores.items()
    }


def rank_quality_metrics(
    *,
    ranking: list[tuple[int, float]],
    realized: np.ndarray,
    reference_index: int,
    costs: np.ndarray,
    top_k: int = 5,
) -> dict[str, Any]:
    """Metricas post-hoc, nunca usadas para formar rankings em t."""
    from scipy.stats import kendalltau, spearmanr

    observed = np.asarray(realized, dtype=float)
    # Nunca substituir o Top-1 ausente pelo proximo ativo com label:
    # isso usaria a disponibilidade do futuro para mudar o vencedor.
    actual_first = int(ranking[0][0]) if ranking else -1
    ranked = [
        index for index, _ in ranking if np.isfinite(observed[index])
    ]
    if (
        actual_first < 0
        or not np.isfinite(observed[actual_first])
        or len(ranked) < 3
        or not np.isfinite(observed[reference_index])
    ):
        return {
            "evaluable_assets": len(ranked),
            "spearman": None, "kendall": None,
            "ndcg_at_5": None, "top1_gap": None,
            "top5_gap": None, "top1_win": None,
            "top1_regret": None,
        }
    forecast_order = np.asarray([
        score for index, score in ranking
        if np.isfinite(observed[index])
    ], dtype=float)
    actual = np.asarray([
        observed[i] - observed[reference_index] - costs[i]
        for i in ranked
    ], dtype=float)
    spear = float(spearmanr(forecast_order, actual).statistic)
    kend = float(kendalltau(forecast_order, actual).statistic)
    spear = spear if np.isfinite(spear) else None
    kend = kend if np.isfinite(kend) else None
    # Usar DCG sobre relevancia nao negativa mantendo a ordenacao real.
    relevance = np.maximum(
        0.0, actual - np.min(actual) + 1e-9
    )
    weights = 1.0 / np.log2(np.arange(2, len(actual) + 2))
    k = min(top_k, len(actual))
    ideal = np.sort(relevance)[::-1]
    ideal_dcg = float(np.dot(ideal[:k], weights[:k]))
    current_dcg = float(np.dot(relevance[:k], weights[:k]))
    ndcg = (
        current_dcg / ideal_dcg if ideal_dcg > 1e-12 else None
    )
    top1 = float(actual[0])
    # Top-5 original: nao substituir um membro sem label pelo sexto
    # colocado, pois isso altera a escolha depois de ver o futuro.
    first_five = [int(i) for i, _ in ranking[:top_k]]
    top5_complete = all(np.isfinite(observed[i]) for i in first_five)
    top5 = (
        float(np.mean([
            observed[i] - observed[reference_index] - costs[i]
            for i in first_five
        ])) if top5_complete else None
    )
    return {
        "evaluable_assets": len(ranked),
        "spearman": spear,
        "kendall": kend,
        "ndcg_at_5": ndcg,
        "top1_gap": top1,
        "top5_gap": top5,
        "top5_original_complete": top5_complete,
        "top1_win": int(top1 > 0.0),
        "top1_regret": float(np.max(actual) - top1),
    }


def summarise_ranking_methods(
    quality_rows: list[dict[str, Any]],
    methods: tuple[str, ...] = RANKING_METHODS,
) -> list[dict[str, Any]]:
    """Macro-folds descritivos: nenhum criterio reotimizado no OOS."""
    rows = []
    for method in methods:
        eligible = [
            r for r in quality_rows
            if r["method"] == method and r["ndcg_at_5"] is not None
        ]
        per_fold = []
        for fid in sorted({int(r["fold_id"]) for r in eligible}):
            grp = [r for r in eligible if int(r["fold_id"]) == fid]
            if not grp:
                continue
            per_fold.append({
                "fold_id": fid,
                "sessions": len({r["decision_date"] for r in grp}),
                "rows": len(grp),
                "mean_ndcg_at_5": float(np.mean([
                    r["ndcg_at_5"] for r in grp
                ])),
                "mean_spearman": float(np.mean([
                    r["spearman"] for r in grp
                    if r["spearman"] is not None
                ])),
                "mean_kendall": float(np.mean([
                    r["kendall"] for r in grp
                    if r["kendall"] is not None
                ])),
                "mean_top1_gap": float(np.mean([
                    r["top1_gap"] for r in grp
                ])),
                "top1_win_rate": float(np.mean([
                    r["top1_win"] for r in grp
                ])),
                "mean_top5_gap": (
                    float(np.mean([
                        r["top5_gap"] for r in grp
                        if r["top5_gap"] is not None
                    ]))
                    if any(r["top5_gap"] is not None for r in grp)
                    else None
                ),
                "mean_top1_regret": float(np.mean([
                    r["top1_regret"] for r in grp
                ])),
            })
        if not per_fold:
            rows.append({
                "method": method, "primary_metric": RANKING_PRIMARY_METRIC,
                "mean_ndcg_at_5": None, "folds": 0,
            })
            continue
        fields = (
            "mean_ndcg_at_5", "mean_spearman", "mean_kendall",
            "mean_top1_gap", "top1_win_rate", "mean_top5_gap",
            "mean_top1_regret",
        )
        rows.append({
            "method": method,
            "primary_metric": RANKING_PRIMARY_METRIC,
            "folds": len(per_fold),
            "evaluated_sessions": sum(r["sessions"] for r in per_fold),
            "evaluated_horizon_rows": sum(r["rows"] for r in per_fold),
            **{
                key: (
                    float(np.mean([
                        f[key] for f in per_fold if f[key] is not None
                    ]))
                    if any(f[key] is not None for f in per_fold)
                    else None
                )
                for key in fields
            },
            "fold_details": per_fold,
        })
    rows.sort(key=lambda r: (
        -(r.get("mean_ndcg_at_5") or -1.0), r["method"]
    ))
    for rank, row in enumerate(rows, 1):
        row["descriptive_rank"] = rank
    return rows



def point_in_time_switch_cost(
    *, day: pd.Timestamp, current_asset: str, candidate: str,
    frames: dict[str, pd.DataFrame], config: Any,
) -> float:
    """Friccao pre-negociacao: apenas close em t, nunca t+1."""
    if current_asset == candidate:
        return 0.0

    def side_fee(symbol: str, side: str) -> float:
        if symbol == "CASH":
            return 0.0
        price = float(frames[symbol].at[day, "close"])
        if not np.isfinite(price) or price <= 0:
            return float("inf")
        result = (
            max(0.0, float(config.slippage_bps)) / 10000.0
            + max(0.0, float(config.commission_rate))
            + max(0.0, float(config.cat_fee_per_share)) / price
        )
        if side == "SELL":
            result += (
                max(0.0, float(config.sec_fee_rate))
                + max(0.0, float(config.taf_fee_per_share)) / price
            )
        return float(result)

    return (
        side_fee(current_asset, "SELL")
        + side_fee(candidate, "BUY")
    )


def mathematical_rotation_choice(
    *,
    method: str,
    symbols: list[str],
    predicted_returns: np.ndarray,
    volatility: np.ndarray,
    switch_costs: np.ndarray,
    current_position: int,
    holding_sessions: int,
    minimum_holding: int = AUTONOMOUS_MIN_HOLD,
    switch_margin: float = AUTONOMOUS_SWITCH_MARGIN,
) -> tuple[int, float, dict[str, Any]]:
    """Politica sem Control e sem ML de ordenacao.

    Input: somente quatro retornos previstos, risco conhecido, custo
    estimado em t e estado da propria carteira. Atribui um score
    deterministico por metodo e aplica um filtro de troca versus manter.
    """
    if method not in AUTONOMOUS_POLICY_METHODS:
        raise ValueError(f"Unknown autonomous method: {method}")
    n = len(symbols)
    prediction = np.asarray(predicted_returns, dtype=float)
    risks = np.asarray(volatility, dtype=float)
    costs = np.asarray(switch_costs, dtype=float)
    if (
        prediction.shape != (n, len(CAPTURE_HORIZONS))
        or risks.shape != (n,) or costs.shape != (n,)
        or not 0 <= current_position <= n
    ):
        raise ValueError("Autonomous rotation input shape/index drift")
    current_id = current_position - 1
    current_valid = (
        current_position == 0 or (
            np.isfinite(prediction[current_id]).all()
            and np.isfinite(risks[current_id])
        )
    )
    base = {
        "method": method,
        "incumbent": (
            "CASH" if current_position == 0
            else symbols[current_id]
        ),
        "holding_sessions": int(holding_sessions),
        "uses_control_action_as_feature": False,
    }
    if not current_valid:
        return current_position, 0.0, {
            **base, "reason": "UNKNOWN_INCUMBENT_FAIL_CLOSED"
        }
    if (
        current_position > 0
        and holding_sessions < minimum_holding
    ):
        return current_position, 0.0, {
            **base, "reason": "MINIMUM_HOLD"
        }
    valid = (
        np.isfinite(prediction).all(axis=1)
        & np.isfinite(risks) & np.isfinite(costs)
    )
    ids = np.flatnonzero(valid)
    if not len(ids):
        return current_position, 0.0, {
            **base, "reason": "NO_ELIGIBLE_ASSET"
        }
    horizon = np.asarray(CAPTURE_HORIZONS, dtype=float)
    daily = (prediction[ids] - costs[ids, None]) / horizon
    forecast_score = np.mean(daily, axis=1)
    # Borda nao depende da escala do retorno previsto.
    horizon_rank = np.empty_like(daily)
    for j in range(len(horizon)):
        ordering = sorted(
            range(len(ids)),
            key=lambda k: (-daily[k, j], symbols[int(ids[k])]),
        )
        for rank, local_id in enumerate(ordering):
            horizon_rank[local_id, j] = rank
    borda = 1.0 - horizon_rank.mean(axis=1) / max(1, len(ids) - 1)
    front = nondominated_layers(
        np.column_stack([daily, -risks[ids]])
    )
    # TOPSIS normalizado com pesos PREDEFINIDOS.
    criteria = np.column_stack([
        daily, -risks[ids], -costs[ids],
        -np.std(daily, axis=1),
    ])
    weights = np.asarray([0.15]*len(horizon)+[0.15, 0.15, 0.10])
    scaled = criteria / np.maximum(
        np.linalg.norm(criteria, axis=0), 1e-12
    ) * weights
    ideal, anti = np.max(scaled, axis=0), np.min(scaled, axis=0)
    positive_dist = np.linalg.norm(scaled - ideal, axis=1)
    negative_dist = np.linalg.norm(scaled - anti, axis=1)
    topsis = np.divide(
        negative_dist, positive_dist + negative_dist,
        out=np.full(len(ids), 0.5),
        where=(positive_dist + negative_dist) > 0,
    )
    net_ranks = sorted(
        range(len(ids)),
        key=lambda j: (-forecast_score[j], symbols[int(ids[j])]),
    )
    borda_ranks = sorted(
        range(len(ids)),
        key=lambda j: (-borda[j], symbols[int(ids[j])]),
    )
    topsis_ranks = sorted(
        range(len(ids)),
        key=lambda j: (-topsis[j], symbols[int(ids[j])]),
    )
    if method == "net_gain":
        choices = net_ranks
    elif method == "borda_pareto":
        choices = sorted(
            range(len(ids)),
            key=lambda j: (
                int(front[j]), -float(borda[j]), symbols[int(ids[j])]
            ),
        )
    elif method == "topsis":
        choices = topsis_ranks
    else:
        # Consenso estrito: Borda, TOPSIS e score liquido devem
        # concordar no conjunto Top 5, e ativo deve ser nao dominado.
        group_size = min(5, len(ids))
        intersect = (
            set(borda_ranks[:group_size])
            & set(net_ranks[:group_size])
            & set(topsis_ranks[:group_size])
        )
        choices = [
            j for j in borda_ranks
            if j in intersect and front[j] == 0
        ]
        if not choices:
            return current_position, 0.0, {
                **base, "reason": "NO_MATH_CONSENSUS",
                "eligible_count": int(len(ids)),
            }
    leader = int(ids[choices[0]])
    raw_lead = float(forecast_score[choices[0]])
    current_forecast = (
        np.asarray(prediction[current_id], dtype=float)
        if current_position > 0 else
        np.zeros(len(horizon), dtype=float)
    )
    total_net_relative = (
        prediction[leader] - current_forecast - costs[leader]
    )
    daily_net_relative = total_net_relative / horizon
    net_advantage = float(np.mean(daily_net_relative))
    group_short = float(np.mean(daily_net_relative[:2]))
    group_medium = float(np.mean(daily_net_relative[2:]))
    capture_consistent = int((total_net_relative > 0).sum())
    score = raw_lead
    if leader == current_id:
        action, reason = current_position, "RANKED_INCUMBENT"
    elif not (
        net_advantage > switch_margin
        and group_short > 0.0 and group_medium > 0.0
    ):
        action, reason = current_position, "INSUFFICIENT_NET_EDGE"
    elif method == "consensus" and capture_consistent < 3:
        action, reason = current_position, "INSUFFICIENT_CONSENSUS_HORIZONS"
    elif current_position == 0 and raw_lead <= 0:
        action, reason = 0, "CASH_ENTRY_NOT_ATTRACTIVE"
    else:
        action, reason = leader + 1, "MATH_ROTATION"
    # A saida para caixa depende SOMENTE da propria previsao atual;
    # nao usa destino Control ou retorno realizado depois de t.
    current_daily = (
        float(np.mean(current_forecast / horizon))
        if current_position > 0 else 0.0
    )
    if (
        action == current_position and current_position > 0
        and current_daily < -AUTONOMOUS_CASH_EXIT_MARGIN
    ):
        action, reason = 0, "NEGATIVE_POSITION_EXPECTATION"
    return action, score, {
        **base,
        "reason": reason,
        "top_candidate": symbols[leader],
        "eligible_count": int(len(ids)),
        "predicted_net_advantage_per_day": net_advantage,
        "short_edge": group_short,
        "medium_edge": group_medium,
        "positive_horizons": capture_consistent,
        "ranking_score": raw_lead,
        "action": "CASH" if action == 0 else symbols[action - 1],
    }


class AutonomousMathematicalPolicy:
    """O simulador injeta posicao/progressao, nao a acao Control."""

    def __init__(
        self, *, method: str, symbols: list[str],
        forecast_by_fold: dict[int, dict[int, dict[pd.Timestamp, np.ndarray]]],
        decision_to_fold: dict[pd.Timestamp, int],
        frames: dict[str, pd.DataFrame],
        config: Any,
        audit: list[dict[str, Any]],
        diagnostics: dict[pd.Timestamp, dict[str, Any]],
    ) -> None:
        if method not in AUTONOMOUS_POLICY_METHODS:
            raise ValueError("Unregistered autonomous policy")
        self.method = method
        self.symbols = symbols
        self.forecast_by_fold = forecast_by_fold
        self.decision_to_fold = decision_to_fold
        self.frames = frames
        self.config = config
        self.audit = audit
        self.diagnostics = diagnostics

    def __call__(
        self, timestamp: pd.Timestamp, current: int, holding: int,
    ) -> tuple[int, float]:
        day = pd.Timestamp(timestamp)
        fid = int(self.decision_to_fold[day])
        forecast = np.column_stack([
            np.asarray(self.forecast_by_fold[fid][h][day], dtype=float)[1:]
            for h in CAPTURE_HORIZONS
        ])
        n = len(self.symbols)
        if forecast.shape != (n, len(CAPTURE_HORIZONS)):
            raise RuntimeError("Autonomous daily forecast shape drift")
        risks = np.asarray([
            float(self.frames[s].at[day, "vol_20"])
            if day in self.frames[s].index else np.nan
            for s in self.symbols
        ])
        current_asset = (
            self.symbols[current - 1] if current > 0 else "CASH"
        )
        costs = np.asarray([
            point_in_time_switch_cost(
                day=day, current_asset=current_asset,
                candidate=s, frames=self.frames, config=self.config,
            ) if day in self.frames[s].index else np.inf
            for s in self.symbols
        ], dtype=float)
        target, score, evidence = mathematical_rotation_choice(
            method=self.method, symbols=self.symbols,
            predicted_returns=forecast, volatility=risks,
            switch_costs=costs, current_position=current,
            holding_sessions=holding,
        )
        log = {
            "decision_date": day.isoformat(), "fold_id": fid,
            "current_asset": current_asset,
            "action_asset": (
                "CASH" if target == 0 else self.symbols[target - 1]
            ),
            "switch_cost_estimated_fraction": (
                float(costs[target-1]) if target > 0
                else point_in_time_switch_cost(
                    day=day, current_asset=current_asset,
                    candidate="CASH", frames=self.frames,
                    config=self.config,
                )
            ),
            **evidence,
        }
        self.audit.append(log)
        self.diagnostics[day] = {
            "decision_diagnostics_schema_version": 1,
            "current_asset": current_asset,
            "holding_days_at_decision": holding,
            "raw_best_asset": evidence.get("top_candidate"),
            "best_asset": evidence.get("top_candidate"),
            "best_score": evidence.get("ranking_score"),
            "final_action_asset": log["action_asset"],
            "final_action_score": score,
            "decision_reason": evidence["reason"],
            "min_hold_guard_applied": evidence["reason"] == "MINIMUM_HOLD",
            "effective_switch_margin": AUTONOMOUS_SWITCH_MARGIN,
        }
        return target, float(score)


def autonomous_decision_quality(
    *,
    audit: list[dict[str, Any]], frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex, folds: list[dict[str, Any]],
    config: Any,
    methods: tuple[str, ...] = AUTONOMOUS_POLICY_METHODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Atribui labels APOS simular; nenhuma pre-visao usa esses labels."""
    loc = {pd.Timestamp(t): i for i, t in enumerate(calendar)}
    boundaries = {
        int(f["fold_id"]): int(f["test_end_index"])
        for f in folds
    }
    quality = []
    for decision in audit:
        day = pd.Timestamp(decision["decision_date"])
        fid = int(decision["fold_id"])
        current = str(decision["current_asset"])
        action = str(decision["action_asset"])
        if day not in loc:
            raise RuntimeError("Decision calendar absent in post-hoc scoring")
        for h in CAPTURE_HORIZONS:
            if loc[day] + h >= boundaries[fid]:
                continue
            label = f"forward_horizon_net_return_pct_{h}"

            def label_at(asset: str) -> float:
                if asset == "CASH":
                    return 0.0
                frame = frames[asset]
                if day not in frame.index:
                    return float("nan")
                return float(frame.at[day, label])

            base = label_at(current)
            candidate = label_at(action)
            if not (np.isfinite(base) and np.isfinite(candidate)):
                continue
            cost = point_in_time_switch_cost(
                day=day, current_asset=current, candidate=action,
                frames=frames, config=config,
            )
            if not np.isfinite(cost):
                continue
            delta = float(candidate - base - cost)
            quality.append({
                "fold_id": fid,
                "decision_date": day.isoformat(),
                "method": decision["method"],
                "horizon": h,
                "current_asset": current,
                "target_asset": action,
                "executed_switch": action != current,
                "realized_switch_vs_hold": delta,
                "switch_positive": (
                    bool(delta > 0) if action != current else None
                ),
                "switch_fee_proxy": float(cost),
                "reason": decision["reason"],
            })
    summary: list[dict[str, Any]] = []
    for method in methods:
        for fid in sorted(boundaries):
            subset = [
                d for d in audit
                if d["method"] == method and int(d["fold_id"]) == fid
            ]
            eligible = [
                r for r in quality if r["method"] == method
                and r["fold_id"] == fid
            ]
            for h in CAPTURE_HORIZONS:
                mature = [r for r in eligible if r["horizon"] == h]
                swaps = [
                    r for r in mature if r["executed_switch"]
                ]
                gains = [
                    r["realized_switch_vs_hold"] for r in swaps
                ]
                summary.append({
                    "method": method, "fold_id": fid,
                    "horizon": h, "decisions": len(subset),
                    "rotations_proposed": sum(
                        d["action_asset"] != d["current_asset"]
                        for d in subset
                    ),
                    "matured_switches": len(swaps),
                    "switch_precision": (
                        float(np.mean(np.asarray(gains) > 0))
                        if gains else None
                    ),
                    "mean_switch_gain_vs_hold": (
                        float(np.mean(gains)) if gains else None
                    ),
                    "cash_decisions": sum(
                        d["action_asset"] == "CASH" for d in subset
                    ),
                })
    return quality, summary


def utility_ablation_variant_config(
    config: Any, *, minimum_hold: int, margin_mode: str,
    calibrated_margin: float,
) -> tuple[Any, float]:
    """Altera SOMENTE hold e margem mantendo features e modelos congelados."""
    if minimum_hold not in UTILITY_ABLATION_HOLD:
        raise ValueError("Unsupported ablation holding")
    if margin_mode not in UTILITY_ABLATION_MARGINS:
        raise ValueError("Unsupported ablation margin mode")
    base = float(config.rotation_switch_margin)
    if margin_mode == "zero":
        setting = config.copiar_modelo(update={
            "rotation_min_holding_days": minimum_hold,
            "rotation_switch_margin": 0.0,
        })
        margin = 0.0
    else:
        setting = config.copiar_modelo(update={
            "rotation_min_holding_days": minimum_hold,
        })
        margin = (
            base if margin_mode == "base"
            else max(base, float(calibrated_margin))
        )
    return setting, float(margin)


def prepare_utility_ablation_policies(
    *,
    config: Any,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    fold_caches: dict[int, dict[str, Any]],
    hold: int,
    margin_mode: str,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None,
) -> dict[int, Any]:
    """Mesma utilidade treinada do Control, decisoes geradas em separado."""
    policies: dict[int, Any] = {}
    for fid, cache in sorted(fold_caches.items()):
        setting, margin = utility_ablation_variant_config(
            config,
            minimum_hold=hold, margin_mode=margin_mode,
            calibrated_margin=float(cache["margin"]),
        )
        policies[fid] = _politica_utilidade(
            {}, frames, symbols, setting, margin,
            utility_cache=cache["oos"],
            decision_diagnostics=decision_diagnostics,
            fold_id=fid,
            calibrated_switch_margin=margin,
        )
    return policies


def summarize_ablation_decision_outcomes(
    *,
    audit: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    folds: list[dict[str, Any]],
    config: Any,
    methods: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Avalia uma unica definicao de precisao pos-hoc para TODAS variantes."""
    return autonomous_decision_quality(
        audit=audit, frames=frames, calendar=calendar,
        folds=folds, config=config, methods=methods,
    )


def one_step_utility_ablation(
    *,
    baseline_decisions: dict[pd.Timestamp, dict[str, Any]],
    policies: dict[str, dict[int, Any]],
    symbols: list[str],
    date_to_fold: dict[pd.Timestamp, int],
) -> list[dict[str, Any]]:
    """Confronto pareado SEM alterar o estado real do Control.

    Cada variante ve a MESMA posicao, data e holding do Control.
    Diferencas sao atribuiveis somente as regras hold/margem;
    replay financeiro separado mede efeitos dinamicos posteriores.
    """
    rows = []
    for date, reference in sorted(baseline_decisions.items()):
        day = pd.Timestamp(date)
        if day not in date_to_fold:
            continue
        fid = int(date_to_fold[day])
        incumbent = str(reference["current_asset"])
        current_pos = (
            0 if incumbent == "CASH" else symbols.index(incumbent) + 1
        )
        holding = int(reference["holding_days_at_decision"])
        official = str(reference["final_action_asset"])
        for method, per_fold in policies.items():
            action, score = per_fold[fid](day, current_pos, holding)
            predicted = "CASH" if action == 0 else symbols[action - 1]
            rows.append({
                "decision_date": day.isoformat(),
                "fold_id": fid, "method": method,
                "current_asset": incumbent,
                "holding_sessions": holding,
                "official_control_action": official,
                "counterfactual_action": predicted,
                "differs_from_control": predicted != official,
                "counterfactual_score": float(score),
                "same_state_as_official_control": True,
            })
    return rows


def main() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    data = SnapshotPaths.u67(ROOT)
    snapshot = validate_snapshot(data)
    if tuple(snapshot.get("assets") or ()) != tuple(U67_REQUESTED_ASSETS):
        raise RuntimeError("Snapshot congelado U67 divergente; pesquisa interrompida.")
    needed = tuple(DISCOVERY_SELECTED_CANDIDATES)
    absent = [
        symbol for symbol in needed
        if not (FULL_DATA.raw_bars / f"{symbol}.csv").exists()
        or not (FULL_DATA.corporate_actions / f"{symbol}.csv").exists()
    ]
    if absent:
        raise RuntimeError(
            "Histórico U67+10 nao esta congelado no cache local: "
            + ", ".join(absent)
            + ". Reutilize o cache da pesquisa anterior. Nenhum download foi tentado."
        )

    base_raw, exclusions, _, _ = prepare_model_frames(
        data, assets=U67_REQUESTED_ASSETS, csv_float_precision="round_trip"
    )
    if frozenset(str(e.get("symbol") or "") for e in exclusions) != U67_EXPECTED_EXCLUSIONS:
        raise RuntimeError("Exclusões estruturais inesperadas na reproducao U67.")
    baseline = [s for s in U67_REQUESTED_ASSETS if s in base_raw]
    config = build_control_config({s: base_raw[s] for s in baseline}, CONFIG)
    config = config.copiar_modelo(update={"analysis_end_date": ANALYSIS_END_DATE})
    (
        baseline_frames, common_dates, _, baseline_symbols, folds,
        all_decision_dates, decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao({s: base_raw[s] for s in baseline}, config)

    candidate_frames, continuity, hashes = _prepare_candidate_frames(
        list(needed), common_dates, all_decision_dates, config
    )
    missing = [s for s in needed if s not in candidate_frames]
    if missing:
        continuity.to_csv(OUT / "continuity_audit.csv", index=False)
        raise RuntimeError(
            "Ativos externos sem continuidade/identidade de snapshot: "
            + ", ".join(missing)
        )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    print("[training] Baseline U67 agregado e 10 ativos adicionais", flush=True)
    states, _, _ = _train_baseline_states(
        baseline_frames, common_dates, baseline_symbols, folds, config
    )
    candidates = {
        s: _train_candidate_caches(s, candidate_frames[s], states, config)
        for s in needed
    }
    symbols = sorted([*baseline_symbols, *needed])
    frames = {**baseline_frames, **candidate_frames}
    configuration = construir_configuracao_controle(config, assets=symbols)
    configuration_without_hold = configuration.copiar_modelo(
        update={"rotation_min_holding_days": 0}
    )
    policies_2: dict[int, Any] = {}
    policies_0: dict[int, Any] = {}
    control_guard_audit: dict[pd.Timestamp, dict[str, Any]] = {}
    no_hold_guard_audit: dict[pd.Timestamp, dict[str, Any]] = {}
    fold_caches: dict[int, dict[str, Any]] = {}
    for state in states:
        cal = _merge_utility_caches(
            state.calibration_cache,
            baseline_symbols,
            {s: candidates[s][state.fold_id].calibration_cache for s in needed},
            symbols,
        )
        oos = _merge_utility_caches(
            state.decision_cache,
            baseline_symbols,
            {s: candidates[s][state.fold_id].decision_cache for s in needed},
            symbols,
        )
        scores = []
        for candidate_margin in configuration.rotation_switch_margin_candidates:
            policy = _politica_utilidade(
                {}, frames, symbols, configuration,
                float(candidate_margin), utility_cache=cal
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, state.calibration_dates, configuration
            )
            scores.append((float(candidate_margin), float(score)))
        selection = selecionar_switch_margin(scores)
        margin = max(
            float(configuration.rotation_switch_margin),
            float(selection["selected_candidate_margin"]),
        )
        policies_2[state.fold_id] = _politica_utilidade(
            {}, frames, symbols, configuration, margin,
            utility_cache=oos, decision_diagnostics=control_guard_audit,
            fold_id=state.fold_id,
        )
        policies_0[state.fold_id] = _politica_utilidade(
            {}, frames, symbols, configuration_without_hold, margin,
            utility_cache=oos, decision_diagnostics=no_hold_guard_audit,
            fold_id=state.fold_id,
        )
        fold_caches[state.fold_id] = {
            "calibration": cal,
            "oos": oos,
            "margin": margin,
            "calibration_margin_scores": scores,
        }

    result_2 = _simular_exato(
        "horizon_research_control_holding_2",
        _politica_agendada(policies_2, decision_to_fold),
        frames, symbols, all_decision_dates, configuration,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="U67+10 hold=2 (frozen control)",
    )
    metrics_2 = scenario_metrics(result_2, folds, float(configuration.initial_capital))
    actual = metrics_2["ending_capital"]
    expected = EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL
    if abs(actual - expected) > CHECKPOINT_TOLERANCE_DOLLARS:
        raise RuntimeError(
            f"Controle U67+10 nao reproduz o checkpoint: {actual:.8f} vs {expected:.8f}"
        )
    print(f"[control] US$ {actual:,.2f} reproduzido ao centavo", flush=True)





    # Labels independentes H01..H60, reconstruidos sem corte global H60.
    build_horizon_return_labels(frames, common_dates, configuration)

    # v1.0.11: fatorial 2 x 3 sobre o MESMO cache de utilidade Control.
    # Nao altera a politica oficial, nem refaz ajustes nos modelos.
    ablation_variants: list[tuple[str, int, str]] = [
        (f"utility_hold{hold}_margin_{mode}", hold, mode)
        for hold in UTILITY_ABLATION_HOLD
        for mode in UTILITY_ABLATION_MARGINS
    ]
    utility_ablation_financial: list[dict[str, Any]] = []
    utility_ablation_decisions: list[dict[str, Any]] = []
    utility_ablation_logs: list[str] = []
    one_step_policies: dict[str, dict[int, Any]] = {}
    ablation_decision_to_fold = {
        pd.Timestamp(day): int(fid)
        for day, fid in decision_to_fold.items()
    }

    def ablation_log(message: str) -> None:
        line = "[utility-ablation-v111] " + message
        print(line, flush=True)
        utility_ablation_logs.append(line)

    ablation_log(
        "start same_pretrained_utility=TRUE "
        "factorial_hold=0,2 factor_margin=zero,base,calibrated "
        "primary_metric=H05_switch_precision_vs_hold"
    )
    for variant, hold, mode in ablation_variants:
        if hold == 2 and mode == "calibrated":
            result = result_2
            diagnostics = control_guard_audit
            ablation_log(
                f"variant={variant} REUSE frozen verified Control run"
            )
        else:
            diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
            replay_policies = prepare_utility_ablation_policies(
                config=configuration, frames=frames, symbols=symbols,
                fold_caches=fold_caches, hold=hold, margin_mode=mode,
                decision_diagnostics=diagnostics,
            )
            ablation_log(f"variant={variant} start replay")
            result = _simular_exato(
                f"utility_ablation_{variant}",
                _politica_agendada(
                    replay_policies, decision_to_fold,
                ),
                frames, symbols, all_decision_dates, configuration,
                calcular_taxas_referencia, aplicar_deslizamento,
                decision_metadata=decision_metadata,
                model_label=f"Utility factor ablation {variant}",
                method_line=(
                    "Identical fold-specific Control utility forecasts "
                    "with ONLY holding and switch margin factors changed."
                ),
            )
        score = scenario_metrics(
            result, folds, float(configuration.initial_capital)
        )
        audit_rows = [
            {
                "method": variant,
                "decision_date": pd.Timestamp(day).isoformat(),
                "fold_id": int(ablation_decision_to_fold[pd.Timestamp(day)]),
                "current_asset": str(diag["current_asset"]),
                "action_asset": str(diag["final_action_asset"]),
                "reason": str(diag["decision_reason"]),
                "holding_sessions": int(diag["holding_days_at_decision"]),
                "best_asset": str(diag["raw_best_asset"]),
                "best_vs_current_gap": diag.get("best_vs_current_gap"),
                "effective_switch_margin": diag.get("effective_switch_margin"),
                "minimum_hold_guard": bool(diag.get("min_hold_guard_applied")),
            }
            for day, diag in sorted(diagnostics.items())
            if pd.Timestamp(day) in ablation_decision_to_fold
        ]
        expected_days = len(all_decision_dates) - 1
        if len(audit_rows) != expected_days:
            raise RuntimeError(
                f"Incomplete ablation {variant}: "
                f"{len(audit_rows)} != {expected_days}"
            )
        utility_ablation_decisions.extend(audit_rows)
        utility_ablation_financial.append({
            "method": variant,
            "hold_sessions": hold,
            "margin_mode": mode,
            "same_control_utility_predictions": True,
            "control_action_used_as_input": False,
            "reused_official_control": bool(
                hold == 2 and mode == "calibrated"
            ),
            "ending_capital": score["ending_capital"],
            "sharpe": score["sharpe"],
            "maximum_drawdown": score["maximum_drawdown"],
            "worst_fold_return": score["worst_fold_return"],
            "capital_rotations": score["capital_rotations"],
            "fees": score["total_transaction_fees"],
            "market_exposure": result.metrics["market_exposure"],
            "relative_capital_vs_control": (
                score["ending_capital"] / actual
            ),
            "folds": score["folds"],
        })
        result.predictions.to_csv(
            OUT / f"{variant}_equity.csv", index_label="timestamp"
        )
        result.trades.to_csv(
            OUT / f"{variant}_trades.csv", index=False
        )
        ablation_log(
            f"variant={variant} done capital="
            f"{score['ending_capital']:,.2f} "
            f"sharpe={score['sharpe']:.4f} "
            f"maxdd={score['maximum_drawdown']:.3%} "
            f"rotations={score['capital_rotations']}"
        )
        if result is not result_2:
            del result

        one_step_policies[variant] = prepare_utility_ablation_policies(
            config=configuration, frames=frames, symbols=symbols,
            fold_caches=fold_caches, hold=hold, margin_mode=mode,
            decision_diagnostics=None,
        )

    one_step_rows = one_step_utility_ablation(
        baseline_decisions=control_guard_audit,
        policies=one_step_policies,
        symbols=symbols,
        date_to_fold=ablation_decision_to_fold,
    )
    # Contrafactual isolado de UMA decisao, avaliado so depois da
    # maturacao H05 e mantendo a posicao oficial como estado fixo.
    # Nao e um backtest de trajetoria alternativa.
    ablation_calendar_index = {
        pd.Timestamp(d): i for i, d in enumerate(common_dates)
    }
    ablation_folds = {int(f["fold_id"]): f for f in folds}
    for paired in one_step_rows:
        t = pd.Timestamp(paired["decision_date"])
        fid = int(paired["fold_id"])
        i = ablation_calendar_index[t]
        paired["H05_observable"] = False
        paired["H05_alternative_vs_control_fraction"] = None
        if i + UTILITY_ABLATION_PRIMARY_HORIZON >= int(
            ablation_folds[fid]["test_end_index"]
        ):
            continue
        incumbent = str(paired["current_asset"])
        alternative = str(paired["counterfactual_action"])
        reference = str(paired["official_control_action"])
        def observed(asset: str) -> float:
            if asset == "CASH":
                return 0.0
            if t not in frames[asset].index:
                return float("nan")
            return float(frames[asset].at[
                t, "forward_horizon_net_return_pct_5"
            ])
        real_alt = observed(alternative)
        real_ref = observed(reference)
        if not (np.isfinite(real_alt) and np.isfinite(real_ref)):
            continue
        cost_alt = point_in_time_switch_cost(
            day=t, current_asset=incumbent, candidate=alternative,
            frames=frames, config=configuration,
        )
        cost_ref = point_in_time_switch_cost(
            day=t, current_asset=incumbent, candidate=reference,
            frames=frames, config=configuration,
        )
        if not (np.isfinite(cost_alt) and np.isfinite(cost_ref)):
            continue
        paired["H05_observable"] = True
        paired["H05_alternative_vs_control_fraction"] = float(
            (real_alt - cost_alt) - (real_ref - cost_ref)
        )
    official_variant = "utility_hold2_margin_calibrated"
    if any(
        r["differs_from_control"]
        for r in one_step_rows if r["method"] == official_variant
    ):
        raise RuntimeError(
            "Official same-state Control must have identical decisions"
        )
    ablation_quality, ablation_quality_fold = (
        summarize_ablation_decision_outcomes(
            audit=utility_ablation_decisions,
            frames=frames, calendar=common_dates,
            folds=folds, config=configuration,
            methods=tuple(v[0] for v in ablation_variants),
        )
    )
    for row in utility_ablation_financial:
        method = row["method"]
        group = [
            q for q in ablation_quality_fold
            if q["method"] == method
            and q["horizon"] == UTILITY_ABLATION_PRIMARY_HORIZON
        ]
        mature = [q for q in group if q["switch_precision"] is not None]
        row["H05_precision_macro"] = (
            float(np.mean([q["switch_precision"] for q in mature]))
            if mature else None
        )
        row["H05_matured_rotations"] = sum(
            int(q["matured_switches"]) for q in group
        )
        paired = [
            r for r in one_step_rows if r["method"] == method
        ]
        row["one_step_action_disagreements_vs_control"] = sum(
            bool(r["differs_from_control"]) for r in paired
        )
        row["one_step_dates"] = len(paired)
        ablation_log(
            f"variant={method} H05_precision={row['H05_precision_macro']} "
            f"n_H05={row['H05_matured_rotations']} "
            f"same_state_diff={row['one_step_action_disagreements_vs_control']}"
        )
    (OUT / "utility_ablation.log").write_text(
        "\n".join(utility_ablation_logs) + "\n",
        encoding="utf-8",
    )
    if not set(PAIR_TRAINING_FEATURES).issubset(ROTATION_FEATURES):
        raise RuntimeError("Direct pair feature contract exceeds point-in-time features")
    feature_cube = build_pair_feature_cube(frames, symbols, common_dates)
    date_to_index = {
        pd.Timestamp(day): idx for idx, day in enumerate(common_dates)
    }
    target_cubes: dict[int, np.ndarray] = {
        h: np.stack([
            frames[symbol].reindex(common_dates)[
                f"forward_horizon_net_return_pct_{h}"
            ].to_numpy(dtype=np.float32)
            for symbol in symbols
        ], axis=1)
        for h in CAPTURE_HORIZONS
    }
    folds_by_id = {int(f["fold_id"]): f for f in folds}
    model_audit: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    pair_validation: list[dict[str, Any]] = []
    pair_raw: list[dict[str, Any]] = []
    pair_confident: list[dict[str, Any]] = []
    pair_top5: list[dict[str, Any]] = []
    audit_ranked: list[dict[str, Any]] = []
    audit_selected: list[dict[str, Any]] = []
    audit_per_horizon: list[dict[str, Any]] = []
    seen_oos_days: set[pd.Timestamp] = set()
    autonomous_forecast_by_fold: dict[
        int, dict[int, dict[pd.Timestamp, np.ndarray]]
    ] = {}
    ranking_quality: list[dict[str, Any]] = []
    ranking_calibrations: list[dict[str, Any]] = []
    ranking_output = OUT / "ranking_all_assets.csv"
    ranking_log = OUT / "ranking_methods.log"
    output_stream = ranking_output.open("w", newline="", encoding="utf-8")
    log_stream = ranking_log.open("w", encoding="utf-8")
    detailed = csv.writer(output_stream)
    detailed.writerow([
        "decision_date", "fold_id", "method", "rank", "asset",
        "ranking_score", "current_asset", "control_target",
        "incremental_cost_fraction",
        *[f"H{h}_predicted_net_fraction" for h in RANKING_EVALUATION_HORIZONS],
        *[f"H{h}_realized_net_fraction" for h in RANKING_EVALUATION_HORIZONS],
    ])

    def rank_log(message: str) -> None:
        text = f"[ranking-v109] {message}"
        print(text, flush=True)
        log_stream.write(text + "\n")
        log_stream.flush()

    rank_log(
        "start methods=" + ",".join(RANKING_METHODS)
        + " primary_metric=" + RANKING_PRIMARY_METRIC
        + " - diagnostic comparison, NOT OOS promotion"
    )

    for state in states:
        fid = int(state.fold_id)
        fold = folds_by_id[fid]
        calibration_days = pd.DatetimeIndex(state.calibration_dates)
        decision_days = pd.DatetimeIndex(state.decision_dates)
        split_index = int(len(calibration_days) * PAIR_CALIBRATION_FRACTION)
        split_date = pd.Timestamp(calibration_days[split_index])
        initial_indices = np.asarray([
            date_to_index[pd.Timestamp(t)] for t in state.train_dates
        ], dtype=np.int64)
        final_indices = np.asarray([
            date_to_index[pd.Timestamp(t)] for t in state.final_fit_dates
        ], dtype=np.int64)
        fold_models_cal: dict[int, Any] = {}
        fold_models_oos: dict[int, Any] = {}
        for h in CAPTURE_HORIZONS:
            initial_x, initial_y, init_info = train_direct_pair_dataset(
                feature_cube=feature_cube,
                target_cube=target_cubes[h],
                training_date_indices=initial_indices,
                horizon=h,
                evaluation_start_index=int(fold["calibration_start_index"]),
            )
            fold_models_cal[h] = fit_direct_pair_model(
                initial_x, initial_y, configuration,
                horizon=h, fold_id=fid, phase="historical_calibration",
            )
            del initial_x, initial_y
            final_x, final_y, final_info = train_direct_pair_dataset(
                feature_cube=feature_cube,
                target_cube=target_cubes[h],
                training_date_indices=final_indices,
                horizon=h,
                evaluation_start_index=int(fold["test_start_index"]),
            )
            fold_models_oos[h] = fit_direct_pair_model(
                final_x, final_y, configuration,
                horizon=h, fold_id=fid, phase="final_oos",
            )
            del final_x, final_y
            model_audit.append({
                "fold_id": fid,
                "model_family": "direct_pairwise_lightgbm",
                "horizon": h,
                "calibration_training": init_info,
                "final_training": final_info,
                "features": pair_feature_columns(),
                "target": "observed_percent_return_A_minus_B",
            })

        # Referencia historica: mesmo Control, mas sem dados de teste.
        # Na calibracao historica, nao usar margem cuja selecao
        # dependeu de observacoes posteriores nessa mesma janela.
        # A referencia proxy aplica somente a margem base ja fixada.
        historical_policy = _politica_utilidade(
            {}, frames, symbols, configuration,
            float(configuration.rotation_switch_margin),
            utility_cache=fold_caches[fid]["calibration"],
        )
        historical_positions = historical_control_pair_context(
            days=calibration_days, policy=historical_policy,
        )
        previous_pairs: list[dict[str, Any]] = []
        later_pairs: list[dict[str, Any]] = []
        for state_row in historical_positions:
            t = pd.Timestamp(state_row["decision_date"])
            current, target = (
                int(state_row["incumbent_position"]),
                int(state_row["control_position"]),
            )
            if (
                current <= 0 or target <= 0
                or int(state_row["holding_days"])
                < int(configuration.rotation_min_holding_days)
            ):
                continue
            ranked = predict_direct_pair_gaps(
                fold_models_cal, feature_cube[date_to_index[t]],
                symbols=symbols,
                incumbent=symbols[current - 1],
                control_target=symbols[target - 1],
                day=t, frames=frames, config=configuration,
            )
            if ranked:
                best = {"fold_id": fid, **ranked[0]}
                if t < split_date:
                    previous_pairs.append(best)
                else:
                    later_pairs.append(best)

        confidence = calibrate_direct_pair_confidence(
            previous_pairs, frames=frames, calendar=common_dates,
            split_date=split_date,
        )
        calibrations.append({
            "fold_id": fid,
            "historical_train_pairs": len(previous_pairs),
            "historical_validation_pairs": len(later_pairs),
            **confidence,
        })
        validation_candidates: list[dict[str, Any]] = []
        for raw in later_pairs:
            choice, enriched = choose_confident_direct_pair([raw], confidence)
            if enriched:
                validation_candidates.append({
                    **enriched[0],
                    "fold_id": fid,
                    "passes_confidence": choice is not None,
                })
        pair_validation.extend(assess_pairwise_fixed_outcomes(
            validation_candidates, frames=frames, calendar=common_dates,
            horizon_end_exclusive=pd.Timestamp(fold["test_start"]),
        ))

        # Calibracao probabilistica: forecasts OOF exclusivamente na
        # janela historica anterior a OOS, sem inspecao de labels futuros.
        historical_asset_forecasts = {}
        for h in RANKING_EVALUATION_HORIZONS:
            safe_hist = select_purged_horizon_training_dates(
                calendar=common_dates,
                training_end_index=int(fold["train_end_index"]),
                evaluation_start_index=int(fold["calibration_start_index"]),
                horizon=h,
                min_training_rows=int(
                    configuration.rotation_minimum_training_rows
                ),
            )
            hist_models = _ajustar_modelos_lightgbm(
                frames, symbols, safe_hist, configuration,
                phase=f"v109_historical_ranking_h{h}_fold{fid}",
                target_column=f"forward_horizon_net_return_pct_{h}",
            )
            if len(hist_models) != len(symbols):
                raise RuntimeError(
                    f"Ranking historical models incomplete fold={fid} H{h}"
                )
            predictions, _ = _precalcular_utilidades_modelo(
                hist_models, frames, symbols, calibration_days, configuration
            )
            historical_asset_forecasts[h] = predictions
            del hist_models
        historical_errors = calibrate_asset_horizon_errors(
            forecasts=historical_asset_forecasts,
            frames=frames, symbols=symbols,
            calibration_days=calibration_days,
            calendar=common_dates, split_date=split_date,
        )
        for h, summary in historical_errors.items():
            ranking_calibrations.append({
                "fold_id": fid, "horizon": h,
                "calibration_split": split_date.isoformat(),
                "calibration_matured_days": summary["effective_days"],
                "asset_min_observations": summary["min_asset_samples"],
                "pooled_residual_std": summary["pooled_error_std"],
                "approximation": "Gaussian marginal residual, "
                                 "zero cross-asset residual covariance",
            })
        del historical_asset_forecasts
        ltr_model, ltr_fit = fit_ranker_h05(
            feature_cube=feature_cube,
            targets=target_cubes[5],
            training_indices=final_indices,
            evaluation_start_index=int(fold["test_start_index"]),
            config=configuration, fold_id=fid,
        )
        model_audit.append({
            "fold_id": fid,
            "model_family": "ranking_supervised_LambdaMART",
            **ltr_fit,
        })
        rank_log(
            f"fold={fid} hist_calib="
            f"{len(calibration_days)}days "
            f"confidence_window_end={split_date.date()} "
            f"ltr_queries={ltr_fit['training_query_days']}"
        )

        # Auditoria ampliada independente do treino e da escolha dos pares.
        individual_forecasts: dict[int, dict[pd.Timestamp, np.ndarray]] = {}
        for h in EXTENDED_AUDIT_HORIZONS:
            safe_dates = select_purged_horizon_training_dates(
                calendar=common_dates,
                training_end_index=int(fold["final_fit_end_index"]),
                evaluation_start_index=int(fold["test_start_index"]),
                horizon=h,
                min_training_rows=int(configuration.rotation_minimum_training_rows),
            )
            audit_models = _ajustar_modelos_lightgbm(
                frames, symbols, safe_dates, configuration,
                phase=f"v108_independent_audit_h{h}_fold{fid}",
                target_column=f"forward_horizon_net_return_pct_{h}",
            )
            if len(audit_models) != len(symbols):
                raise RuntimeError(f"Audit H{h} fold={fid}: missing models")
            individual_forecasts[h], diag = _precalcular_utilidades_modelo(
                audit_models, frames, symbols, decision_days, configuration
            )
            model_audit.append({
                "fold_id": fid,
                "model_family": "independent_asset_audit",
                "horizon": h,
                "training_last": safe_dates[-1].isoformat(),
                "model_count": len(audit_models),
                "prediction_diagnostics": diag,
            })
            del audit_models
            print(f"[audit] fold={fid} H{h} ready", flush=True)

        # Cache estritamente fora da amostra para as quatro
        # politicas matematicas independentes do Control.
        autonomous_forecast_by_fold[fid] = {
            h: individual_forecasts[h] for h in CAPTURE_HORIZONS
        }
        for day in decision_days:
            t = pd.Timestamp(day)
            if decision_to_fold.get(t) != fid:
                continue
            if t in seen_oos_days:
                raise RuntimeError(f"Duplicated OOS decision: {t}")
            seen_oos_days.add(t)
            control = control_guard_audit.get(t)
            if control is None:
                raise RuntimeError(f"Control decision not recorded: {t}")
            incumbent = str(control["current_asset"])
            if incumbent == "CASH":
                continue
            reference = str(control["final_action_asset"])
            metadata = {
                "fold_id": fid,
                "holding_days": int(control["holding_days_at_decision"]),
                "control_decision_reason": control["decision_reason"],
            }
            audit_ranks, audit_best = select_horizon_candidates(
                day=t, incumbent=incumbent, symbols=symbols,
                forecast=individual_forecasts,
                horizons=EXTENDED_AUDIT_HORIZONS,
                extra_switch_cost=0.0,
                capture_require_short_medium_advantage=False,
                candidate_cost_fn=lambda symbol: estimated_incremental_pair_cost(
                    day=t, incumbent=incumbent, control_target=incumbent,
                    candidate=symbol, frames=frames, config=configuration,
                ),
            )
            audit_ranked.extend([
                {**metadata, "rank": rank, **row}
                for rank, row in enumerate(audit_ranks, start=1)
            ])
            audit_selected.append({**metadata, **audit_best})
            incumbent_index = symbols.index(incumbent) + 1
            for h in EXTENDED_AUDIT_HORIZONS:
                current_pred = individual_forecasts[h][t]
                options = sorted([
                    (
                        sym, float(current_pred[index] - current_pred[incumbent_index])
                        - estimated_incremental_pair_cost(
                            day=t, incumbent=incumbent, control_target=incumbent,
                            candidate=sym, frames=frames, config=configuration,
                        ),
                    )
                    for index, sym in enumerate(symbols, start=1)
                    if sym != incumbent and np.isfinite(current_pred[index])
                ], key=lambda row: (-row[1], row[0]))
                if options:
                    chosen = options[0]
                    audit_per_horizon.append({
                        **metadata,
                        "decision_date": t.isoformat(),
                        "incumbent": incumbent,
                        "horizon": h,
                        "top_candidate": chosen[0],
                        "pred_gap_fraction": chosen[1],
                        "estimated_incremental_cost_fraction":
                            estimated_incremental_pair_cost(
                                day=t, incumbent=incumbent,
                                control_target=incumbent,
                                candidate=chosen[0], frames=frames,
                                config=configuration,
                            ),
                        "audit_only": True,
                    })

            # Comparacao de TODOS os metodos sobre os mesmos arrays
            # congelados e universo elegivel. Todos os scores sao
            # construidos ANTES de consultar os retornos t+H.
            if reference != "CASH":
                present = feature_cube[date_to_index[t]]
                eligible = np.isfinite(present).all(axis=1)
                daily_vol = np.asarray([
                    float(frames[sym].at[t, "vol_20"])
                    if t in frames[sym].index else np.nan
                    for sym in symbols
                ], dtype=float)
                ranking_cost = np.full(
                    len(symbols), np.nan, dtype=float
                )
                for pos, sym in enumerate(symbols):
                    if not eligible[pos] or not np.isfinite(daily_vol[pos]):
                        continue
                    if sym in (reference, incumbent):
                        ranking_cost[pos] = 0.0
                    else:
                        ranking_cost[pos] = estimated_incremental_pair_cost(
                            day=t, incumbent=incumbent,
                            control_target=reference, candidate=sym,
                            frames=frames, config=configuration,
                        )
                ltr_predictions = np.full(
                    len(symbols), np.nan, dtype=float
                )
                valid = np.flatnonzero(eligible)
                if len(valid):
                    ltr_predictions[valid] = np.asarray(
                        ltr_model.predict(pd.DataFrame(
                            present[valid], columns=list(PAIR_TRAINING_FEATURES)
                        )), dtype=float
                    )
                daily_forecasts = {
                    h: np.asarray(
                        individual_forecasts[h][t], dtype=float
                    )[1:]
                    for h in RANKING_EVALUATION_HORIZONS
                }
                selections = rank_with_all_methods(
                    forecasts=daily_forecasts,
                    symbols=symbols,
                    reference_symbol=reference,
                    incremental_cost=ranking_cost,
                    volatility=daily_vol,
                    historical_error=historical_errors,
                    ltr_scores=ltr_predictions,
                )
                if len(selections) != len(RANKING_METHODS):
                    raise RuntimeError(
                        f"Incomplete ranking methods at fold={fid} date={t}: "
                        f"{list(selections)}"
                    )
                observed = {}
                for h in RANKING_EVALUATION_HORIZONS:
                    matured = (
                        date_to_index[t] + h
                        < int(fold["test_end_index"])
                    )
                    observed[h] = (
                        np.asarray(target_cubes[h][date_to_index[t]], dtype=float)
                        if matured else
                        np.full(len(symbols), np.nan, dtype=float)
                    )
                reference_id = symbols.index(reference)
                # Mesmo conjunto de datas comparaveis em CADA H para
                # todos os oito metodos. Nunca filtrar um metodo com
                # o resultado futuro sem aplicar regra identica aos outros.
                comparable_horizon = {
                    h: bool(
                        np.isfinite(observed[h][reference_id])
                        and all(
                            np.isfinite(observed[h][order[0][0]])
                            for order in selections.values()
                        )
                    )
                    for h in RANKING_EVALUATION_HORIZONS
                }
                for method, sorted_assets in selections.items():
                    for rank, (asset_idx, score) in enumerate(sorted_assets, 1):
                        detailed.writerow([
                            t.isoformat(), fid, method, rank,
                            symbols[asset_idx], score,
                            incumbent, reference, ranking_cost[asset_idx],
                            *[daily_forecasts[h][asset_idx]
                              for h in RANKING_EVALUATION_HORIZONS],
                            *[observed[h][asset_idx]
                              if np.isfinite(observed[h][asset_idx])
                              else ""
                              for h in RANKING_EVALUATION_HORIZONS],
                        ])
                    for h in RANKING_EVALUATION_HORIZONS:
                        quality = (
                            rank_quality_metrics(
                                ranking=sorted_assets,
                                realized=observed[h],
                                reference_index=reference_id,
                                costs=ranking_cost,
                            )
                            if comparable_horizon[h] else {
                                "evaluable_assets": int(
                                    np.isfinite(observed[h]).sum()
                                ),
                                "spearman": None, "kendall": None,
                                "ndcg_at_5": None, "top1_gap": None,
                                "top5_gap": None, "top1_win": None,
                                "top1_regret": None,
                            }
                        )
                        ranking_quality.append({
                            "common_top1_evaluable": comparable_horizon[h],
                            "decision_date": t.isoformat(),
                            "fold_id": fid, "method": method,
                            "horizon": h,
                            "asset_universe": len(sorted_assets),
                            "reference": reference,
                            "top1_asset": symbols[sorted_assets[0][0]],
                            **quality,
                        })
                if len(seen_oos_days) % RANKING_LOG_EVERY_DAYS == 0:
                    leaders = " ".join(
                        f"{method}={symbols[values[0][0]]}"
                        for method, values in selections.items()
                    )
                    rank_log(f"fold={fid} day={t.date()} {leaders}")

            if (
                reference == "CASH" or
                int(control["holding_days_at_decision"])
                < int(configuration.rotation_min_holding_days)
            ):
                continue
            full_ranking = predict_direct_pair_gaps(
                fold_models_oos, feature_cube[date_to_index[t]],
                symbols=symbols, incumbent=incumbent,
                control_target=reference, day=t,
                frames=frames, config=configuration,
            )
            selected, confidence_ranked = choose_confident_direct_pair(
                full_ranking, confidence
            )
            pair_top5.extend([
                {**metadata, "rank": rank, **row}
                for rank, row in enumerate(confidence_ranked[:5], start=1)
            ])
            best_raw = next(
                (row for row in full_ranking if row["raw_qualifies"]), None
            )
            if best_raw is not None:
                pair_raw.append({
                    **metadata, "selection_kind": "raw",
                    **best_raw,
                })
            if selected is not None:
                pair_confident.append({
                    **metadata, "selection_kind": "confident",
                    **selected,
                })
        del fold_models_cal, fold_models_oos, ltr_model
        fold_metrics = summarise_ranking_methods([
            row for row in ranking_quality if row["fold_id"] == fid
        ])
        if fold_metrics:
            rank_log(
                f"fold={fid} descriptive_best_ndcg5="
                f"{fold_metrics[0]['method']} "
                f"ndcg5={fold_metrics[0].get('mean_ndcg_at_5')}"
            )
        print(
            f"[fold] {fid} historical_winners={len(previous_pairs)} "
            f"validated={len(later_pairs)} "
            f"OOS_confident={len([r for r in pair_confident if r['fold_id']==fid])}",
            flush=True,
        )

    output_stream.flush()
    output_stream.close()
    rank_log("all candidate rows written, computing macro-fold leaderboard")
    ranking_leaderboard = summarise_ranking_methods(ranking_quality)
    for record in ranking_leaderboard:
        rank_log(
            "leaderboard "
            f"rank={record['descriptive_rank']} "
            f"method={record['method']} "
            f"ndcg5={record.get('mean_ndcg_at_5')} "
            f"spearman={record.get('mean_spearman')} "
            f"kendall={record.get('mean_kendall')} "
            f"top1_gap={record.get('mean_top1_gap')}"
        )
    ranking_by_horizon: list[dict[str, Any]] = []
    for h in RANKING_EVALUATION_HORIZONS:
        for item in summarise_ranking_methods([
            row for row in ranking_quality if row["horizon"] == h
        ]):
            ranking_by_horizon.append({
                "horizon": h,
                **{key: value for key, value in item.items()
                   if key != "fold_details"},
            })
        top = next(
            (row for row in ranking_by_horizon
             if row["horizon"] == h and row["descriptive_rank"] == 1),
            None,
        )
        if top:
            rank_log(
                f"horizon=H{h:02d} best_ordering={top['method']} "
                f"ndcg_at_5={top.get('mean_ndcg_at_5')}"
            )
    log_stream.flush()
    log_stream.close()

    # SIMULACOES AUTONOMAS: cada politica utiliza estado/proprio
    # portfolio, previsoes OOS e custos em t. A acao Control nao
    # entra nas regras matematicas.
    autonomous_audit: list[dict[str, Any]] = []
    autonomous_financial: list[dict[str, Any]] = []
    rotation_logs: list[str] = []

    def rotation_log(message: str) -> None:
        line = f"[autonomous-v110] {message}"
        print(line, flush=True)
        rotation_logs.append(line)

    rotation_log(
        "begin independent policies=" + ",".join(AUTONOMOUS_POLICY_METHODS)
    )
    for method in AUTONOMOUS_POLICY_METHODS:
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        policy_rows: list[dict[str, Any]] = []
        policy = AutonomousMathematicalPolicy(
            method=method, symbols=symbols,
            forecast_by_fold=autonomous_forecast_by_fold,
            decision_to_fold=decision_to_fold,
            frames=frames, config=configuration,
            audit=policy_rows, diagnostics=diagnostics,
        )
        rotation_log(f"start policy={method}")
        def progress(fraction: float, stage: str) -> None:
            if fraction >= 0.999 or abs(fraction - 0.12) < 1e-4:
                rotation_log(
                    f"policy={method} progress={fraction:.1%} stage={stage}"
                )

        result = _simular_exato(
            f"autonomous_math_{method}",
            policy,
            frames, symbols, all_decision_dates, configuration,
            calcular_taxas_referencia, aplicar_deslizamento,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=diagnostics,
            model_label=f"Autonomous mathematical rotation: {method}",
            method_line=(
                "Deterministic rotation from separately trained "
                "horizon forecasts; no Control decisions as policy inputs"
            ),
            simulation_progress_callback=progress,
        )
        figures = scenario_metrics(
            result, folds, float(configuration.initial_capital)
        )
        independent_score = {
            "method": method, "variant": f"math_{method}",
            "ending_capital": figures["ending_capital"],
            "sharpe": figures["sharpe"],
            "maximum_drawdown": figures["maximum_drawdown"],
            "worst_fold_return": figures["worst_fold_return"],
            "capital_rotations": figures["capital_rotations"],
            "fees": figures["total_transaction_fees"],
            "market_exposure": result.metrics["market_exposure"],
            "avg_holding_sessions": result.metrics["average_holding_days"],
            "folds": figures["folds"],
            "relative_to_control_capital_ratio": (
                figures["ending_capital"] / actual
            ),
            "uses_control_action_as_policy_input": False,
            "is_exploratory_repeated_oos": True,
        }
        autonomous_financial.append(independent_score)
        autonomous_audit.extend(policy_rows)
        result.predictions.to_csv(
            OUT / f"autonomous_{method}_equity.csv",
            index_label="timestamp",
        )
        result.trades.to_csv(
            OUT / f"autonomous_{method}_trades.csv", index=False
        )
        rotation_log(
            f"completed policy={method} capital="
            f"{figures['ending_capital']:,.2f} "
            f"sharpe={figures['sharpe']:.4f} "
            f"max_dd={figures['maximum_drawdown']:.3%} "
            f"rotations={figures['capital_rotations']} "
            f"fees={figures['total_transaction_fees']:,.2f}"
        )
        del result, policy

    # O Control e avaliado pela mesma definicao de melhoria/erro
    # de trocar vs manter, sem interferir nas politicas alternativas.
    control_audit = [{
        "method": "control", "decision_date": t.isoformat(),
        "fold_id": int(decision_to_fold[t]),
        "current_asset": str(record["current_asset"]),
        "action_asset": str(record["final_action_asset"]),
        "reason": str(record["decision_reason"]),
    } for t, record in sorted(control_guard_audit.items())
        if t in decision_to_fold and t in seen_oos_days
    ]
    rotation_quality, rotation_quality_summary = autonomous_decision_quality(
        audit=autonomous_audit + control_audit,
        frames=frames, calendar=common_dates,
        folds=folds, config=configuration,
        methods=("control", *AUTONOMOUS_POLICY_METHODS),
    )
    autonomous_financial.insert(0, {
        "method": "control", "variant": "control_holding_2",
        "ending_capital": actual, "sharpe": metrics_2["sharpe"],
        "maximum_drawdown": metrics_2["maximum_drawdown"],
        "worst_fold_return": metrics_2["worst_fold_return"],
        "capital_rotations": metrics_2["capital_rotations"],
        "fees": metrics_2["total_transaction_fees"],
        "market_exposure": result_2.metrics["market_exposure"],
        "avg_holding_sessions": result_2.metrics["average_holding_days"],
        "folds": metrics_2["folds"],
        "relative_to_control_capital_ratio": 1.0,
        "uses_control_action_as_policy_input": False,
        "is_exploratory_repeated_oos": True,
    })
    for row in autonomous_financial:
        rotation_log(
            f"comparison method={row['method']} "
            f"ending_capital={row['ending_capital']:,.2f} "
            f"rotations={row['capital_rotations']}"
        )
    for method in ("control", *AUTONOMOUS_POLICY_METHODS):
        for h in CAPTURE_HORIZONS:
            stats = [
                r for r in rotation_quality_summary
                if r["method"] == method and r["horizon"] == h
            ]
            observed = [r for r in stats if r["switch_precision"] is not None]
            macro_precision = (
                float(np.mean([r["switch_precision"] for r in observed]))
                if observed else None
            )
            rotation_log(
                f"decision_quality method={method} H{h:02d} "
                f"mean_fold_precision={macro_precision}"
            )

    # Escala minima fixada antes da execucao: taxa perfeita baseada
    # em 1 troca nao pode vencer dezenas de decisoes comparaveis.
    rotation_leaderboard: list[dict[str, Any]] = []
    for row in autonomous_financial:
        method = row["method"]
        fold_stats = [
            r for r in rotation_quality_summary
            if r["method"] == method and r["horizon"] == 5
        ]
        adequate = all(
            r["matured_switches"] >= AUTONOMOUS_MIN_MATURE_SWITCHES_PER_FOLD
            for r in fold_stats
        ) and len(fold_stats) == len(folds)
        total_switches = sum(r["matured_switches"] for r in fold_stats)
        wins = sum(
            (r["switch_precision"] or 0.0) * r["matured_switches"]
            for r in fold_stats
        )
        macro = [
            r["switch_precision"] for r in fold_stats
            if r["switch_precision"] is not None
        ]
        rotation_leaderboard.append({
            "method": method,
            "evidence_adequate_each_fold": bool(adequate),
            "minimum_matured_switches_per_fold":
                AUTONOMOUS_MIN_MATURE_SWITCHES_PER_FOLD,
            "total_matured_H05_switches": total_switches,
            "macro_fold_H05_switch_precision": (
                float(np.mean(macro)) if macro else None
            ),
            "pooled_H05_switch_precision": (
                float(wins / total_switches)
                if total_switches else None
            ),
            "ending_capital": row["ending_capital"],
            "sharpe": row["sharpe"],
            "maximum_drawdown": row["maximum_drawdown"],
            "worst_fold_return": row["worst_fold_return"],
            "total_financial_rotations": row["capital_rotations"],
            "total_fees": row["fees"],
            "is_exploratory": True,
        })
    rotation_leaderboard.sort(key=lambda r: (
        not r["evidence_adequate_each_fold"],
        -(r["macro_fold_H05_switch_precision"]
          if r["macro_fold_H05_switch_precision"] is not None
          else -1.0),
        r["method"],
    ))
    for n, row in enumerate(rotation_leaderboard, 1):
        row["descriptive_rank"] = (
            n if row["evidence_adequate_each_fold"] else None
        )
        rotation_log(
            f"quality_leaderboard rank={row['descriptive_rank']} "
            f"method={row['method']} "
            f"matured_H05={row['total_matured_H05_switches']} "
            f"precision={row['macro_fold_H05_switch_precision']} "
            f"sample_adequate={row['evidence_adequate_each_fold']}"
        )
    (OUT / "rotation_policy.log").write_text(
        "\n".join(rotation_logs) + "\n", encoding="utf-8",
    )

    # Joins dos outcomes APOS os rankings, somente se t+H amadureceu
    # dentro do proprio fold. Nenhum retorno OOS treinou o modelo.
    evaluated_raw: list[dict[str, Any]] = []
    evaluated_confident: list[dict[str, Any]] = []
    for fid, fold in folds_by_id.items():
        boundary = (
            common_dates[int(fold["test_end_index"])]
            if int(fold["test_end_index"]) < len(common_dates)
            else common_dates[-1] + pd.Timedelta(days=7)
        )
        evaluated_raw.extend(assess_pairwise_fixed_outcomes(
            [r for r in pair_raw if r["fold_id"] == fid],
            frames=frames, calendar=common_dates,
            horizon_end_exclusive=boundary,
        ))
        evaluated_confident.extend(assess_pairwise_fixed_outcomes(
            [r for r in pair_confident if r["fold_id"] == fid],
            frames=frames, calendar=common_dates,
            horizon_end_exclusive=boundary,
        ))

    folds_maturity = {
        fid: {
            t: int(fold["test_end_index"]) - 1 - date_to_index[t]
            for t, assigned in decision_to_fold.items()
            if int(assigned) == fid
        } for fid, fold in folds_by_id.items()
    }
    audit_ranked_eval, audit_selected_eval = attach_fixed_horizon_outcomes(
        audit_ranked, audit_selected,
        horizons=EXTENDED_AUDIT_HORIZONS,
        folds_maturity=folds_maturity, frames=frames,
        extra_switch_cost=0.0,
    )
    for row in audit_per_horizon:
        fid = int(row["fold_id"])
        h = int(row["horizon"])
        t = pd.Timestamp(row["decision_date"])
        realized = None
        if folds_maturity[fid].get(t, 0) >= h:
            label = f"forward_horizon_net_return_pct_{h}"
            candidate = float(frames[row["top_candidate"]].at[t, label])
            incumbent = float(frames[row["incumbent"]].at[t, label])
            if np.isfinite(candidate) and np.isfinite(incumbent):
                realized = candidate - incumbent - float(
                    row["estimated_incremental_cost_fraction"]
                )
        row["realized_gap_fraction"] = realized
        row["realized_outperformance"] = (
            bool(realized > 0) if realized is not None else None
        )
    audit_quality = []
    for fid in sorted(folds_by_id):
        for h in EXTENDED_AUDIT_HORIZONS:
            vals = np.asarray([
                float(r["realized_gap_fraction"])
                for r in audit_per_horizon
                if r["fold_id"] == fid and r["horizon"] == h
                and r["realized_gap_fraction"] is not None
            ])
            audit_quality.append({
                "fold_id": fid, "horizon": h,
                "matured_evaluable": len(vals),
                "mean_realized_gap_fraction": (
                    float(vals.mean()) if len(vals) else None
                ),
                "hit_rate": float(np.mean(vals > 0)) if len(vals) else None,
            })

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        # CSVs vazios continuam legiveis e identificaveis por ferramentas
        # externas, especialmente quando nenhuma escolha passa na confianca.
        schema = {
            "capture_direct_pair_confident_selected.csv": [
                "fold_id", "decision_date", "incumbent", "control_target",
                "candidate", "calibrated_qualifies",
            ],
            "capture_direct_pair_confident_quality.csv": [
                "fold_id", "horizon", "selected", "matured_evaluable",
                "realized_win_rate", "prediction_mae", "prediction_rmse",
            ],
            "capture_direct_pair_raw_selected.csv": [
                "fold_id", "decision_date", "incumbent",
                "control_target", "candidate", "raw_qualifies",
            ],
        }.get(name, ["fold_id", "decision_date"])
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=schema)
        frame.to_csv(OUT / name, index=False)

    write_csv("capture_direct_pair_ranked_top5.csv", pair_top5)
    write_csv("capture_direct_pair_raw_selected.csv", evaluated_raw)
    write_csv("capture_direct_pair_confident_selected.csv", evaluated_confident)
    write_csv("capture_direct_pair_calibration_by_fold.csv", calibrations)
    write_csv("capture_direct_pair_historical_validation.csv", pair_validation)
    write_csv("capture_direct_pair_validation_quality.csv",
              aggregate_pairwise_quality(pair_validation))
    write_csv("capture_direct_pair_raw_quality.csv",
              aggregate_pairwise_quality(evaluated_raw))
    write_csv("capture_direct_pair_confident_quality.csv",
              aggregate_pairwise_quality(evaluated_confident))
    write_csv("expanded_audit_ranked_top5.csv", audit_ranked_eval)
    write_csv("expanded_audit_selected.csv", audit_selected_eval)
    write_csv("expanded_audit_quality_by_fold_horizon.csv",
              summarize_selection_quality(
                  audit_selected_eval, EXTENDED_AUDIT_HORIZONS
              ))
    write_csv("expanded_audit_best_per_horizon.csv", audit_per_horizon)
    write_csv("expanded_audit_best_horizon_quality.csv", audit_quality)
    # Auditoria da correcao de cobertura por H, junto a cada fold.
    label_coverage = []
    for fid, fold in sorted(folds_by_id.items()):
        fold_decisions = [
            t for t, owner in decision_to_fold.items()
            if int(owner) == fid and t in seen_oos_days
        ]
        for h in EXTENDED_AUDIT_HORIZONS:
            potential = 0
            mature = 0
            for t in fold_decisions:
                row = control_guard_audit[t]
                current = str(row["current_asset"])
                if current == "CASH":
                    continue
                if date_to_index[t] + h >= int(fold["test_end_index"]):
                    continue
                potential += 1
                label = f"forward_horizon_net_return_pct_{h}"
                if t in frames[current].index:
                    observed = float(frames[current].at[t, label])
                    mature += int(np.isfinite(observed))
            label_coverage.append({
                "fold_id": fid,
                "horizon": h,
                "eligible_reference_days": potential,
                "valid_matured_incumbent_labels": mature,
                "missing_labels": potential - mature,
            })
    write_csv("horizon_label_coverage.csv", label_coverage)
    write_csv("ranking_oos_quality_by_session.csv", ranking_quality)
    write_csv("ranking_historical_error_calibration.csv", ranking_calibrations)
    write_csv("ranking_methods_leaderboard.csv", [
        {key: val for key, val in row.items() if key != "fold_details"}
        for row in ranking_leaderboard
    ])
    write_csv("ranking_methods_by_fold.csv", [
        {"method": row["method"], **fold}
        for row in ranking_leaderboard
        for fold in row.get("fold_details", [])
    ])
    write_csv("ranking_methods_by_horizon.csv", ranking_by_horizon)
    write_csv("autonomous_rotation_decisions.csv", autonomous_audit)
    write_csv("autonomous_rotation_realized_quality.csv", rotation_quality)
    write_csv("autonomous_rotation_quality_by_fold.csv",
              rotation_quality_summary)
    write_csv("autonomous_financial_summary.csv", [
        {k: v for k, v in row.items() if k != "folds"}
        for row in autonomous_financial
    ])
    write_csv("autonomous_quality_leaderboard.csv", rotation_leaderboard)
    write_csv("utility_ablation_financial.csv", [
        {key: value for key, value in row.items() if key != "folds"}
        for row in utility_ablation_financial
    ])
    write_csv("utility_ablation_quality_by_fold.csv", ablation_quality_fold)
    write_csv("utility_ablation_quality_by_decision.csv", ablation_quality)
    write_csv("utility_ablation_policy_decisions.csv",
              utility_ablation_decisions)
    write_csv("utility_ablation_one_step.csv", one_step_rows)
    write_csv("horizon_model_training.csv", model_audit)
    write_csv("control_decision_reasons.csv", [
        {"decision_date": d.isoformat(), **row}
        for d, row in sorted(control_guard_audit.items())
    ])
    result_2.predictions.to_csv(
        OUT / "control_holding_2_equity.csv", index_label="timestamp"
    )
    result_2.trades.to_csv(OUT / "control_holding_2_trades.csv", index=False)
    write_csv("comparison.csv", [
        {
            "variant": row["variant"],
            "ending_capital": row["ending_capital"],
            "sharpe": row["sharpe"],
            "maximum_drawdown": row["maximum_drawdown"],
            "capital_rotations": row["capital_rotations"],
            "fees": row["fees"],
            "control_reproduced": True,
            "policy_uses_control_action": False,
        }
        for row in autonomous_financial
    ] + [
        {
            "variant": row["method"],
            "ending_capital": row["ending_capital"],
            "sharpe": row["sharpe"],
            "maximum_drawdown": row["maximum_drawdown"],
            "capital_rotations": row["capital_rotations"],
            "fees": row["fees"],
            "control_reproduced": True,
            "policy_uses_control_action": False,
        }
        for row in utility_ablation_financial
    ])
    fixed_hashes = {
        sym: {
            "raw_sha256": sha256_file(FULL_DATA.raw_bars / f"{sym}.csv"),
            "actions_sha256": sha256_file(
                FULL_DATA.corporate_actions / f"{sym}.csv"
            ),
        } for sym in needed
    }
    payload = {
        "script_version": VERSION,
        "status": "completed",
        "snapshot_u67_sha256": snapshot.get("snapshot_sha256"),
        "external_asset_hashes": fixed_hashes,
        "effective_asset_count": len(symbols),
        "structural_exclusions": sorted(U67_EXPECTED_EXCLUSIONS),
        "capture_horizons": list(CAPTURE_HORIZONS),
        "audit_horizons": list(EXTENDED_AUDIT_HORIZONS),
        "pair_feature_columns": pair_feature_columns(),
        "reference_sampling": "two deterministic rotating anchors per day, with reverse pairs",
        "pair_models": 2 * len(CAPTURE_HORIZONS) * len(folds),
        "utility_ablation_factors": {
            "minimum_holding": list(UTILITY_ABLATION_HOLD),
            "margin_mode": list(UTILITY_ABLATION_MARGINS),
            "primary_horizon": UTILITY_ABLATION_PRIMARY_HORIZON,
            "primary_metric": UTILITY_ABLATION_PRIMARY_METRIC,
            "signal_source": "frozen_Control_fold_specific_risk_adjusted_utility",
            "same_signal_all_variants": True,
            "same_state_policy_comparison": True,
            "oos_tuning": False,
        },
        "utility_ablation_financial": utility_ablation_financial,
        "utility_ablation_quality_by_fold": ablation_quality_fold,
        "utility_ablation_same_state_summary": [{
            "method": row["method"],
            "one_step_action_disagreements_vs_control":
                row["one_step_action_disagreements_vs_control"],
            "one_step_dates": row["one_step_dates"],
        } for row in utility_ablation_financial],
        "autonomous_policy_methods": list(AUTONOMOUS_POLICY_METHODS),
        "autonomous_financial_comparison": autonomous_financial,
        "autonomous_rotation_precision_by_fold": rotation_quality_summary,
        "autonomous_quality_leaderboard": rotation_leaderboard,
        "autonomous_primary_diagnostic": "H05_switch_precision_vs_hold",
        "autonomous_min_matured_switches_per_fold":
            AUTONOMOUS_MIN_MATURE_SWITCHES_PER_FOLD,
        "autonomous_no_control_action_in_policy": True,
        "autonomous_live_orders": False,
        "autonomous_reuses_frozen_individual_forecasts": True,
        "autonomous_cash_option": True,
        "autonomous_minimum_holding_sessions": AUTONOMOUS_MIN_HOLD,
        "ranking_methods": list(RANKING_METHODS),
        "ranking_primary_metric": RANKING_PRIMARY_METRIC,
        "ranking_evaluation_horizons": list(
            RANKING_EVALUATION_HORIZONS
        ),
        "ranking_method_leaderboard": ranking_leaderboard,
        "ranking_method_leaderboard_by_horizon": ranking_by_horizon,
        "ranking_descriptive_top_method": (
            ranking_leaderboard[0]["method"]
            if ranking_leaderboard else None
        ),
        "ranking_candidates_full_export": "ranking_all_assets.csv",
        "ranking_diagnostics_log": "ranking_methods.log",
        "rankings_do_not_promote_strategy": True,
        "learning_to_rank_is_extra_supervised_training": True,
        "historical_calibration": calibrations,
        "capture_raw_selected": len(pair_raw),
        "capture_confident_selected": len(pair_confident),
        "audit_independent_of_pair_selection": True,
        "unique_oos_decision_days": len(seen_oos_days),
        "checkpoints": {
            "control_capital": actual,
            "control_checkpoint_reference": expected,
            "reproduces_checkpoint_to_cent": True,
        },
        "candidate_backtest_performed": True,
        "candidate_backtest_methods": list(AUTONOMOUS_POLICY_METHODS),
        "operational_orders": False,
        "oos_used_for_parameter_selection": False,
        "method_limitations": (
            "Mathematical rotation policies are independently simulated "
            "with the historical simulator and frozen execution fees, "
            "but its simplistic fills do not establish market impact "
            "or deployability at large compound capital. Decision-"
            "precision labels are post-hoc next-open-to-H-close "
            "relative returns, not realized strategy PnL. "
            "The OOS folds were repeatedly studied before v1.0.10; "
            "comparisons are exploratory until an untouched future period."
        ),
        "export_to_operational": False,
    }
    write_json(OUT / "experiment.json", payload)
    package_path = OUT / "pacote_persistencia_horizontes_u77_v111.zip"
    with zipfile.ZipFile(
        package_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for artifact in sorted(OUT.iterdir()):
            if artifact.is_file() and artifact != package_path:
                archive.write(artifact, arcname=artifact.name)
    print(f"[package] {package_path}", flush=True)
    print(f"[output] {OUT}", flush=True)
    return payload



def audible_completion(*, succeeded: bool) -> str:
    """Sinal sonoro no Spyder/Windows; em outros SO tenta bell ASCII."""
    try:
        import winsound
    except ImportError:
        print("\a", end="", flush=True)
        return "terminal_bell_fallback"
    if succeeded:
        winsound.Beep(880, 160)
        winsound.Beep(1046, 220)
        return "windows_success_two_tones"
    winsound.Beep(440, 420)
    winsound.Beep(330, 470)
    return "windows_error_two_tones"


class ConsoleLogMirror:
    """Captura stdout/stderr completo sem ocultar logs no Spyder."""

    def __init__(self, console: Any, stream: Any) -> None:
        self.console = console
        self.stream = stream

    def write(self, value: str) -> int:
        self.console.write(value)
        self.stream.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.stream.flush()

    def isatty(self) -> bool:
        return bool(self.console.isatty())


def run_spyder_experiment() -> dict[str, Any]:
    """Captura logs tambem em falha; emite som ao concluir."""
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.iterdir():
        if stale.is_file() and stale.suffix.lower() in {
            ".csv", ".json", ".zip", ".log",
        }:
            stale.unlink()
    logfile = OUT / "execution_full.log"
    with logfile.open("w", encoding="utf-8") as stream:
        stdout = ConsoleLogMirror(sys.stdout, stream)
        stderr = ConsoleLogMirror(sys.stderr, stream)
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            print(f"[experiment] START {VERSION}", flush=True)
            try:
                completed = main()
            except BaseException:
                print(f"[experiment] ERROR {VERSION}", flush=True)
                traceback.print_exc()
                try:
                    mode = audible_completion(succeeded=False)
                    print(f"[audio] ERROR {mode}", flush=True)
                except Exception as audio_error:
                    print(f"[audio] ERROR beep failed: {audio_error}", flush=True)
                raise
            print(f"[experiment] COMPLETED {VERSION}", flush=True)
            try:
                mode = audible_completion(succeeded=True)
                print(f"[audio] SUCCESS {mode}", flush=True)
            except Exception as audio_error:
                print(f"[audio] SUCCESS beep failed: {audio_error}", flush=True)
            return completed


if __name__ == "__main__":
    run_spyder_experiment()

