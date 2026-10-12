"""Experimento isolado U67+10: permanencia 1/2/3/4 e selecao cronologica.

Mantem modelos, snapshots, margens e custos da politica Control v1.0.3.
Nao faz download nem altera a reproducao oficial ou o MCT.
Uso no Spyder: execute este arquivo (F5), com os caches U67+10 presentes.
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
    _construir_contexto_execucao,
    selecionar_switch_margin,
)
from engine.rotacao import (
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
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
OUT = ROOT / "output" / "persistencia_horizontes_u77_v104"
VERSION = "horizon-persistence-oos-v1.0.4"
CONFIDENCE_MULTIPLIERS = (0.0, 0.5, 1.0)  # historico v1.0.3
ANCHOR_BLEND = 0.5  # historico v1.0.3
HOLDING_CANDIDATES = (1, 2, 3, 4)
CHECKPOINT_TOLERANCE_DOLLARS = 0.01


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

    result_0 = _simular_exato(
        "horizon_research_holding_0",
        _politica_agendada(policies_0, decision_to_fold),
        frames, symbols, all_decision_dates, configuration_without_hold,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="U67+10 hold=0 (ablation)",
    )
    metrics_0 = scenario_metrics(result_0, folds, float(configuration.initial_capital))
    print(f"[ablation] US$ {metrics_0['ending_capital']:,.2f}", flush=True)


    # A mesma utilidade agregada e o mesmo modelo de previsao alimentam
    # todos os cenarios: a unica mudanca sera o minimo de permanencia.
    configs = {
        h: configuration.copiar_modelo(
            update={"rotation_min_holding_days": h}
        )
        for h in HOLDING_CANDIDATES
    }
    policies_by_holding: dict[int, dict[int, Any]] = {
        h: {} for h in HOLDING_CANDIDATES
    }
    selected_fold_policies: dict[int, Any] = {}
    fold_selection: list[dict[str, Any]] = []

    for state in states:
        fold_id = int(state.fold_id)
        margin = float(fold_caches[fold_id]["margin"])
        calibrated_cache = fold_caches[fold_id]["calibration"]
        oos_cache = fold_caches[fold_id]["oos"]
        policies_by_holding[2][fold_id] = policies_2[fold_id]

        holdout_start = (2 * len(state.calibration_dates)) // 3
        selection_dates = state.calibration_dates[holdout_start:]
        if len(selection_dates) < 20:
            raise RuntimeError(
                f"Fold {fold_id}: menos de 20 datas na calibracao final."
            )

        scores: dict[int, float] = {}
        for h in HOLDING_CANDIDATES:
            # Calibracao historica. OOS nao participa da escolha.
            validation_policy = _politica_utilidade(
                {}, frames, symbols, configs[h], margin,
                utility_cache=calibrated_cache,
            )
            scores[h] = float(_crescimento_politica_simples(
                validation_policy, frames, symbols,
                selection_dates, configs[h],
            ))
            if h != 2:
                policies_by_holding[h][fold_id] = _politica_utilidade(
                    {}, frames, symbols, configs[h], margin,
                    utility_cache=oos_cache,
                )
        winner = select_calibrated_holding(scores)
        selected_fold_policies[fold_id] = policies_by_holding[winner][fold_id]
        fold_selection.append({
            "fold_id": fold_id,
            "calibration_total_sessions": len(state.calibration_dates),
            "calibration_validation_sessions": len(selection_dates),
            "calibration_validation_first": selection_dates[0].isoformat(),
            "calibration_validation_last": selection_dates[-1].isoformat(),
            "margin_frozen_from_control": margin,
            "hold_scores": {str(h): scores[h] for h in HOLDING_CANDIDATES},
            "chosen_holding": winner,
            "control_holding": 2,
            "score_difference_vs_control": scores[winner] - scores[2],
        })
        print(
            f"[selection] fold={fold_id} holding={winner} "
            f"score={scores[winner]:.6f} versus control={scores[2]:.6f}",
            flush=True,
        )

    outcomes: dict[str, Any] = {
        "control_holding_2": result_2,
        "ablation_holding_0": result_0,
    }
    for h in (1, 3, 4):
        result = _simular_exato(
            f"horizon_research_holding_{h}",
            _politica_agendada(policies_by_holding[h], decision_to_fold),
            frames, symbols, all_decision_dates, configs[h],
            calcular_taxas_referencia, aplicar_deslizamento,
            decision_metadata=decision_metadata,
            model_label=f"U67+10 minimum holding={h} (research)",
        )
        outcomes[f"holding_{h}"] = result
        print(
            f"[holding] {h} ending_equity="
            f"{scenario_metrics(result, folds, float(configuration.initial_capital))['ending_capital']:,.2f}",
            flush=True,
        )

    result_selected = _simular_exato(
        "horizon_research_calibration_selected_holding",
        _politica_agendada(selected_fold_policies, decision_to_fold),
        frames, symbols, all_decision_dates, configuration,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="U67+10 calibrated holding per fold (research)",
    )
    outcomes["calibration_selected_holding"] = result_selected

    variants = {
        name: scenario_metrics(result, folds, float(configuration.initial_capital))
        for name, result in outcomes.items()
    }
    for name, result in outcomes.items():
        result.predictions.to_csv(OUT / f"{name}_equity.csv", index_label="timestamp")
        result.trades.to_csv(OUT / f"{name}_trades.csv", index=False)

    pd.DataFrame(fold_selection).to_csv(
        OUT / "calibrated_holding_by_fold.csv", index=False
    )
    pd.DataFrame([
        {"decision_date": day.isoformat(), **record}
        for day, record in sorted(control_guard_audit.items())
    ]).to_csv(OUT / "control_decision_reasons.csv", index=False)
    causal_hold_guard_features(
        control_guard_audit, frames
    ).to_csv(OUT / "hold_guard_point_in_time_features.csv", index=False)

    rows_comparison = []
    fold_comparison = []
    asset_comparison = []
    month_comparison = []
    baseline = float(metrics_2["ending_capital"])
    for name, values in variants.items():
        rows_comparison.append({
            "variant": name,
            **{k: values.get(k) for k in (
                "ending_capital", "cagr", "sharpe", "maximum_drawdown",
                "worst_fold_return", "capital_rotations",
                "total_transaction_fees",
            )},
            "final_delta_vs_control": float(values["ending_capital"]) - baseline,
            "final_ratio_vs_control": (
                float(values["ending_capital"]) / baseline - 1.0
            ),
        })
        for fold in values.get("folds", []):
            fold_comparison.append({
                "variant": name,
                "fold_id": fold["fold_id"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "starting_capital": fold["strategy_starting_capital"],
                "ending_capital": fold["strategy_ending_capital"],
                "fold_return": fold["strategy_return"],
                "fold_maximum_drawdown": fold["maximum_drawdown"],
                "sessions": fold["sessions"],
            })
        trades = outcomes[name].trades
        if not trades.empty:
            trade_data = trades.copy()
            trade_data["month"] = pd.to_datetime(
                trade_data["timestamp"], utc=True
            ).dt.strftime("%Y-%m")
            for dimensions, output in (
                (["walk_forward_fold", "asset", "action"], asset_comparison),
                (["month", "walk_forward_fold"], month_comparison),
            ):
                for key, group in trade_data.groupby(dimensions, dropna=False):
                    key = key if isinstance(key, tuple) else (key,)
                    output.append({
                        "variant": name,
                        **dict(zip(dimensions, key)),
                        "trade_count": len(group),
                        "total_fee": float(group["total_fee"].fillna(0).sum()),
                        "realized_pnl": float(group["realized_pnl"].fillna(0).sum()),
                    })

    pd.DataFrame(rows_comparison).to_csv(
        OUT / "comparison.csv", index=False
    )
    pd.DataFrame(fold_comparison).to_csv(
        OUT / "fold_comparison.csv", index=False
    )
    pd.DataFrame(asset_comparison).to_csv(
        OUT / "trades_by_asset.csv", index=False
    )
    pd.DataFrame(month_comparison).to_csv(
        OUT / "trades_by_month.csv", index=False
    )

    external_hashes = {
        symbol: {
            "raw_sha256": sha256_file(FULL_DATA.raw_bars / f"{symbol}.csv"),
            "actions_sha256": sha256_file(
                FULL_DATA.corporate_actions / f"{symbol}.csv"
            ),
        }
        for symbol in needed
    }
    payload = {
        "script_version": VERSION,
        "status": "completed",
        "analysis_end": ANALYSIS_END_DATE,
        "snapshot_u67_sha256": snapshot.get("snapshot_sha256"),
        "external_asset_hashes": external_hashes,
        "requested_asset_count": len(U67_REQUESTED_ASSETS) + len(needed),
        "effective_asset_count": len(symbols),
        "structural_exclusions": sorted(U67_EXPECTED_EXCLUSIONS),
        "holding_candidates": list(HOLDING_CANDIDATES),
        "holding_selection": fold_selection,
        "policy_rule": "static_holding_1_2_3_4_and_fold_calibration_selection",
        "calibration_protocol": (
            "Fold-local baseline models and switch margins are preserved. "
            "Holding candidates are scored only in last third of the "
            "126-session historical calibration window. No OOS labels "
            "enter parameter selection. Selection may overfit calibration."
        ),
        "features_note": (
            "Only same-day features at Control minimum-holding guards; "
            "descriptive, not proof of causal advantage or a trained regime model."
        ),
        "metrics": variants,
        "checkpoint_reference": expected,
        "reproduces_checkpoint_to_cent": True,
        "export_to_operational": False,
    }
    write_json(OUT / "experiment.json", payload)
    package_path = OUT / "pacote_persistencia_horizontes_u77_v104.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(OUT.iterdir()):
            if artifact.is_file() and artifact != package_path:
                archive.write(artifact, arcname=artifact.name)
    print(f"[package] {package_path}", flush=True)
    print("[final] " + json.dumps({
        name: values["ending_capital"] for name, values in variants.items()
    }), flush=True)
    print(f"[output] {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    main()
