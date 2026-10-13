"""Pesquisa U67+10 v1.0.8: regressao direta de pares e auditoria.

Control e unico backtest; selecao de candidatos e diagnostica.
Reproduz o checkpoint, corrige cobertura de labels H01..H60, treina
modelo direto de diferencas e preserva auditoria independente.
Sem download, sem alteracao da reproducao oficial ou de sistemas externos.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
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
OUT = ROOT / "output" / "persistencia_horizontes_u77_v108"
VERSION = "horizon-persistence-oos-v1.0.8"
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
            gap = cand - base - float(row["incremental_cost_fraction"])
            row[key + "_realized_gap_fraction"] = gap
            row[key + "_hit"] = bool(gap > 0.0)
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
            rows.append({
                "fold_id": fold,
                "horizon": h,
                "selected": len(selected),
                "matured_evaluable": len(available),
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




    # Nenhuma regra de pesquisa pode alterar a execucao financeira.
    build_horizon_return_labels(frames, common_dates, configuration)
    model_audit: list[dict[str, Any]] = []
    calibration_audit: list[dict[str, Any]] = []
    validation_audit: list[dict[str, Any]] = []
    capture_ranked: list[dict[str, Any]] = []
    raw_selected: list[dict[str, Any]] = []
    confident_selected: list[dict[str, Any]] = []
    expanded_rankings: list[dict[str, Any]] = []
    expanded_selections: list[dict[str, Any]] = []
    expanded_per_horizon: list[dict[str, Any]] = []
    used_decisions: set[pd.Timestamp] = set()
    folds_by_id = {int(f["fold_id"]): f for f in folds}

    for state in states:
        fold_id = int(state.fold_id)
        fold = folds_by_id[fold_id]
        decision_days = pd.DatetimeIndex(state.decision_dates)
        cal_days = pd.DatetimeIndex(state.calibration_dates)
        split_index = int(len(cal_days) * PAIR_CALIBRATION_FRACTION)
        split_date = pd.Timestamp(cal_days[split_index])
        # Exclui da calibracao de residuos labels que atravessariam
        # a divisao treino/validacao. A validacao nao escolhe parametros.
        forecast_cal: dict[int, dict[pd.Timestamp, np.ndarray]] = {}
        forecast_oos: dict[int, dict[pd.Timestamp, np.ndarray]] = {}
        for h in ALL_RESEARCH_HORIZONS:
            target = f"forward_horizon_net_return_pct_{h}"
            if h in CAPTURE_HORIZONS:
                safe_initial = select_purged_horizon_training_dates(
                    calendar=common_dates,
                    training_end_index=int(fold["train_end_index"]),
                    evaluation_start_index=int(fold["calibration_start_index"]),
                    horizon=h,
                    min_training_rows=int(configuration.rotation_minimum_training_rows),
                )
                first_models = _ajustar_modelos_lightgbm(
                    frames, symbols, safe_initial, configuration,
                    phase=f"v107_cal_H{h}_fold{fold_id}",
                    target_column=target,
                )
                if len(first_models) != len(symbols):
                    raise RuntimeError("Pair calibration models incomplete")
                preds, _ = _precalcular_utilidades_modelo(
                    first_models, frames, symbols, cal_days, configuration
                )
                forecast_cal[h] = preds
                del first_models

            safe_final = select_purged_horizon_training_dates(
                calendar=common_dates,
                training_end_index=int(fold["final_fit_end_index"]),
                evaluation_start_index=int(fold["test_start_index"]),
                horizon=h,
                min_training_rows=int(configuration.rotation_minimum_training_rows),
            )
            final_models = _ajustar_modelos_lightgbm(
                frames, symbols, safe_final, configuration,
                phase=f"v107_final_H{h}_fold{fold_id}",
                target_column=target,
            )
            if len(final_models) != len(symbols):
                raise RuntimeError("Final horizon models incomplete")
            forecast_oos[h], diagnostics = _precalcular_utilidades_modelo(
                final_models, frames, symbols, decision_days, configuration,
            )
            model_audit.append({
                "fold_id": fold_id,
                "horizon": h,
                "train_calibration_models": (
                    len(symbols) if h in CAPTURE_HORIZONS else 0
                ),
                "train_final_models": len(final_models),
                "calibration_start": cal_days[0].isoformat(),
                "calibration_end": cal_days[-1].isoformat(),
                "calibration_split": split_date.isoformat(),
                "final_training_last_date": safe_final[-1].isoformat(),
                "test_first": decision_days[0].isoformat(),
                "diagnostics": diagnostics,
            })
            del final_models
            print(
                f"[v107] fold={fold_id} horizon=H{h} "
                f"calibration={'yes' if h in CAPTURE_HORIZONS else 'no'}",
                flush=True,
            )

        cal_policy = _politica_utilidade(
            {}, frames, symbols, configuration,
            float(fold_caches[fold_id]["margin"]),
            utility_cache=fold_caches[fold_id]["calibration"],
        )
        historical_state = historical_control_pair_context(
            days=cal_days, policy=cal_policy
        )
        training_pairs = []
        validation_pairs = []
        for state_row in historical_state:
            if (state_row["incumbent_position"] == 0
                or state_row["control_position"] == 0
                or state_row["holding_days"]
                < int(configuration.rotation_min_holding_days)):
                continue
            day = pd.Timestamp(state_row["decision_date"])
            inc = symbols[state_row["incumbent_position"] - 1]
            base = symbols[state_row["control_position"] - 1]
            options = rank_direct_pair_gap(
                day=day, incumbent=inc, control_target=base,
                symbols=symbols, forecasts=forecast_cal,
                frames=frames, config=configuration, limit=0,
            )
            if not options:
                continue
            leader = next(
                (r for r in options if r["raw_qualifies"]), None
            )
            if leader is None:
                continue
            enriched = {"fold_id": fold_id, **leader}
            if day < split_date:
                training_pairs.append(enriched)
            else:
                validation_pairs.append(enriched)

        calibration = train_pairwise_winner_calibration(
            selected=training_pairs, frames=frames, calendar=common_dates,
            split_date=split_date,
            cutoff_date=pd.Timestamp(fold["test_start"]),
        )
        calibration_audit.append({
            "fold_id": fold_id,
            "train_selected_pair_days": len(training_pairs),
            "validation_selected_pair_days": len(validation_pairs),
            **calibration,
        })
        for vrow in validation_pairs:
            picked, calibrated = calibrated_pair_choice(
                [vrow], calibration
            )
            if not calibrated:
                continue
            r = {"fold_id": fold_id, **calibrated[0]}
            r["would_pass_confidence"] = picked is not None
            validation_audit.extend(assess_pairwise_fixed_outcomes(
                [r], frames=frames, calendar=common_dates,
                horizon_end_exclusive=pd.Timestamp(fold["test_start"]),
            ))

        for day in decision_days:
            t = pd.Timestamp(day)
            if decision_to_fold.get(t) != fold_id:
                continue
            if t in used_decisions:
                raise RuntimeError(f"Decision duplication across folds {t}")
            used_decisions.add(t)
            control = control_guard_audit.get(t)
            if control is None or control.get("current_asset") in (None, "CASH"):
                continue
            incumbent = str(control["current_asset"])
            destination = str(control["final_action_asset"])
            metadata = {
                "fold_id": fold_id,
                "holding_days": int(control["holding_days_at_decision"]),
                "control_decision_reason": control["decision_reason"],
            }
            # Auditoria expandida v1.0.6 permanece independente.
            audit_rank, audit_choice = select_horizon_candidates(
                day=t, incumbent=incumbent, symbols=symbols,
                forecast=forecast_oos,
                horizons=EXTENDED_AUDIT_HORIZONS,
                extra_switch_cost=0.0,
                capture_require_short_medium_advantage=False,
                candidate_cost_fn=lambda symbol: estimated_incremental_pair_cost(
                    day=t, incumbent=incumbent, control_target=incumbent,
                    candidate=symbol, frames=frames, config=configuration,
                ),
            )
            expanded_rankings.extend([
                {**metadata, "rank": k, **r}
                for k, r in enumerate(audit_rank, 1)
            ])
            expanded_selections.append({**metadata, **audit_choice})
            idx = symbols.index(incumbent) + 1
            for h in EXTENDED_AUDIT_HORIZONS:
                prediction = forecast_oos[h][t]
                leader = sorted([
                    (
                        symbols[pos - 1],
                        float(prediction[pos] - prediction[idx])
                        - estimated_incremental_pair_cost(
                            day=t, incumbent=incumbent,
                            control_target=incumbent,
                            candidate=symbols[pos - 1],
                            frames=frames, config=configuration,
                        ),
                    )
                    for pos in range(1, len(symbols) + 1)
                    if pos != idx and np.isfinite(prediction[pos])
                ], key=lambda r: (-r[1], r[0]))
                if leader:
                    expanded_per_horizon.append({
                        "decision_date": t.isoformat(),
                        "fold_id": fold_id,
                        "horizon": h,
                        "incumbent": incumbent,
                        "top_candidate": leader[0][0],
                        "pred_gap_fraction": leader[0][1],
                        "estimated_incremental_cost_fraction":
                            estimated_incremental_pair_cost(
                                day=t, incumbent=incumbent,
                                control_target=incumbent,
                                candidate=leader[0][0], frames=frames,
                                config=configuration,
                            ),
                        "audit_only": True,
                    })

            if (destination == "CASH"
                or int(control["holding_days_at_decision"])
                < int(configuration.rotation_min_holding_days)):
                continue
            pairs = rank_direct_pair_gap(
                day=t, incumbent=incumbent, control_target=destination,
                symbols=symbols, forecasts=forecast_oos,
                frames=frames, config=configuration, limit=0,
            )
            selected, calibrated = calibrated_pair_choice(pairs, calibration)
            capture_ranked.extend([
                {**metadata, "rank": rank, **entry}
                for rank, entry in enumerate(calibrated[:5], 1)
            ])
            raw = next(
                (r for r in pairs if r["raw_qualifies"]), None
            )
            if raw is not None:
                raw_selected.append({
                    **metadata, "calibration_status": "raw_only",
                    **raw,
                })
            if selected is not None:
                confident_selected.append({
                    **metadata, "calibration_status": "confidence_passed",
                    **selected,
                })
        print(
            f"[fold] {fold_id} train_pairs={len(training_pairs)} "
            f"val_pairs={len(validation_pairs)} "
            f"confidence_total={len(confident_selected)}",
            flush=True,
        )


    # Mesmas observacoes dos folds do Control; nao somar fronteiras
    # duplicadas nem resultados t+H fora do proprio fold.
    folds_maturity: dict[int, dict[pd.Timestamp, int]] = {}
    date_index = {pd.Timestamp(t): i for i, t in enumerate(common_dates)}
    for fid, fold in folds_by_id.items():
        folds_maturity[fid] = {
            timestamp: int(fold["test_end_index"]) - 1 - date_index[timestamp]
            for timestamp, assigned in decision_to_fold.items()
            if int(assigned) == fid
        }
    expanded_ranked_eval, expanded_selected_eval = attach_fixed_horizon_outcomes(
        expanded_rankings, expanded_selections,
        horizons=EXTENDED_AUDIT_HORIZONS,
        folds_maturity=folds_maturity,
        frames=frames,
        extra_switch_cost=0.0,
    )
    for row in expanded_per_horizon:
        t = pd.Timestamp(row["decision_date"])
        fid, h = int(row["fold_id"]), int(row["horizon"])
        candidate, incumbent = row["top_candidate"], row["incumbent"]
        realized = None
        if folds_maturity[fid].get(t, 0) >= h:
            label = f"forward_horizon_net_return_pct_{h}"
            new = float(frames[candidate].at[t, label])
            old = float(frames[incumbent].at[t, label])
            if np.isfinite(new) and np.isfinite(old):
                realized = new - old - float(
                    row["estimated_incremental_cost_fraction"]
                )
        row["realized_gap_fraction"] = realized
        row["realized_outperformance"] = (
            bool(realized > 0.0) if realized is not None else None
        )
    expanded_quality = []
    for fid in sorted(folds_by_id):
        for h in EXTENDED_AUDIT_HORIZONS:
            group = [
                r["realized_gap_fraction"] for r in expanded_per_horizon
                if int(r["fold_id"]) == fid and int(r["horizon"]) == h
                and r["realized_gap_fraction"] is not None
            ]
            values = np.asarray(group, dtype=float)
            expanded_quality.append({
                "fold_id": fid,
                "horizon": h,
                "matured_evaluable": len(values),
                "mean_realized_gap_fraction": (
                    float(values.mean()) if len(values) else None
                ),
                "hit_rate": (
                    float((values > 0).mean()) if len(values) else None
                ),
            })
    def export_rows(name: str, rows: list[dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(OUT / name, index=False)

    # A avaliacao realizada entra SOMENTE depois da selecao congelada.
    evaluated_raw: list[dict[str, Any]] = []
    evaluated_confident: list[dict[str, Any]] = []
    for fold_id, fold in folds_by_id.items():
        raw_part = [r for r in raw_selected if r["fold_id"] == fold_id]
        calibrated_part = [
            r for r in confident_selected if r["fold_id"] == fold_id
        ]
        evaluated_raw.extend(assess_pairwise_fixed_outcomes(
            raw_part, frames=frames, calendar=common_dates,
            horizon_end_exclusive=common_dates[int(fold["test_end_index"])]
            if int(fold["test_end_index"]) < len(common_dates)
            else common_dates[-1] + pd.Timedelta(days=7),
        ))
        evaluated_confident.extend(assess_pairwise_fixed_outcomes(
            calibrated_part, frames=frames, calendar=common_dates,
            horizon_end_exclusive=common_dates[int(fold["test_end_index"])]
            if int(fold["test_end_index"]) < len(common_dates)
            else common_dates[-1] + pd.Timedelta(days=7),
        ))

    export_rows("capture_pair_ranked_top5.csv", capture_ranked)
    export_rows("capture_pair_raw_selected.csv", evaluated_raw)
    export_rows("capture_pair_confident_selected.csv", evaluated_confident)
    export_rows("capture_pair_calibration_by_fold.csv", calibration_audit)
    export_rows("capture_pair_historical_validation.csv", validation_audit)
    export_rows("capture_pair_validation_quality.csv",
                aggregate_pairwise_quality(validation_audit))
    export_rows("capture_pair_raw_quality.csv",
                aggregate_pairwise_quality(evaluated_raw))
    export_rows("capture_pair_confident_quality.csv",
                aggregate_pairwise_quality(evaluated_confident))
    export_rows("expanded_audit_ranked_top5.csv", expanded_ranked_eval)
    export_rows("expanded_audit_selected.csv", expanded_selected_eval)
    export_rows("expanded_audit_quality_by_fold_horizon.csv",
                summarize_selection_quality(
                    expanded_selected_eval, EXTENDED_AUDIT_HORIZONS
                ))
    export_rows("expanded_audit_best_per_horizon.csv", expanded_per_horizon)
    export_rows("expanded_audit_per_horizon_quality.csv", expanded_quality)
    export_rows("horizon_model_training.csv", model_audit)
    export_rows("control_decision_reasons.csv", [
        {"decision_date": d.isoformat(), **r}
        for d, r in sorted(control_guard_audit.items())
    ])
    result_2.predictions.to_csv(
        OUT / "control_holding_2_equity.csv", index_label="timestamp"
    )
    result_2.trades.to_csv(OUT / "control_holding_2_trades.csv", index=False)
    export_rows("comparison.csv", [{
        "variant": "control_holding_2",
        "ending_capital": metrics_2["ending_capital"],
        "sharpe": metrics_2.get("sharpe"),
        "maximum_drawdown": metrics_2.get("maximum_drawdown"),
        "candidate_backtest_performed": False,
        "control_reproduced": True,
    }])
    hashes = {
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
        "analysis_end": ANALYSIS_END_DATE,
        "effective_asset_count": len(symbols),
        "structural_exclusions": sorted(U67_EXPECTED_EXCLUSIONS),
        "snapshot_u67_sha256": snapshot.get("snapshot_sha256"),
        "external_asset_hashes": hashes,
        "capture_horizons": list(CAPTURE_HORIZONS),
        "audit_horizons": list(EXTENDED_AUDIT_HORIZONS),
        "calibration_fit_fraction": PAIR_CALIBRATION_FRACTION,
        "calibration_quantile": PAIR_ERROR_QUANTILE,
        "calibration": calibration_audit,
        "capture_raw_selected": len(raw_selected),
        "capture_confident_selected": len(confident_selected),
        "no_duplicate_oos_decisions": True,
        "OOS_used_for_parameter_selection": False,
        "control_capital": actual,
        "control_checkpoint_reference": expected,
        "reproduces_checkpoint_to_cent": True,
        "operational_orders": False,
        "candidate_backtest_performed": False,
        "scientific_limitation": (
            "Pairwise shrinkage calibrates differences of independent "
            "horizon forecasts on historical winning pairs. It is not "
            "a separately trained pairwise LightGBM, nor a portfolio "
            "backtest. Overlapping outcomes and repeatedly examined "
            "OOS folds limit statistical independence."
        ),
        "export_to_operational": False,
    }
    write_json(OUT / "experiment.json", payload)
    package_path = OUT / "pacote_persistencia_horizontes_u77_v107.zip"
    with zipfile.ZipFile(
        package_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for artifact in sorted(OUT.iterdir()):
            if artifact.is_file() and artifact != package_path:
                archive.write(artifact, arcname=artifact.name)
    print(f"[package] {package_path}", flush=True)
    return payload


if __name__ == "__main__":
    main()
