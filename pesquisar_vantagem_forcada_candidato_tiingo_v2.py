"""Forca bruta contrafactual em todo OOS com target ate reconvergencia.

Versao: tiingo-forced-candidate-episode-advantage-v2.0.0

Para cada decisao OOS e para cada um dos 56 ativos:
  1. parte do estado real do baseline em t;
  2. forca o candidato apenas na decisao t;
  3. a partir de t+1 volta para a politica normal;
  4. acompanha a trajetoria ate o estado da politica forcada reconvergir ao
     estado do baseline (mesma posicao e mesmos dias de holding);
  5. mede a vantagem marginal acumulada causada pela unica intervencao.

O target principal e:
  episode_delta_log_capital

Features sao somente informacoes disponiveis em t. O futuro aparece apenas no
TARGET diagnostico. Qualquer modelo aprendido depois precisa de validacao
cronologica walk-forward.

Esta v2 tambem:
- cobre TODO o OOS, nao apenas os oito meses mais fracos;
- testa os 56 ativos, inclusive candidatos sem score LightGBM naquele instante,
  quando ha precos validos para executar o contrafactual;
- diagnostica cobertura/eligibilidade de score por ativo (incluindo CLMT);
- reutiliza o baseline congelado da v1 quando dataset/config sao identicos;
- usa arrays precomputados de retorno para acelerar dezenas de milhares de
  rollouts;
- salva progresso e pode ser retomada.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import (
    ROTATION_FEATURES,
    _proportional_switch_cost,
)
from tcc_engine.config import ASSETS, CALENDAR_ANCHOR_ASSETS, CONFIG
from tcc_engine.research_challengers import _build_execution_context
from tunar_lightgbm_tiingo import dataset_signature, load_tiingo_split_causal

from pesquisar_contribuicao_marginal_periodica_tiingo import (
    make_replay_policy,
    period_statistics,
    replay_control,
    score_lookup,
)
from pesquisar_vantagem_forcada_candidato_tiingo import (
    compact_baseline,
    config_hash,
    experiment_config,
    finite_float,
    read_json,
    write_json,
)
from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.execution import apply_slippage, calculate_reference_fees


VERSION = "tiingo-forced-candidate-episode-advantage-v2.0.0"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "tiingo_forced_candidate_episode_advantage_v2"

MANIFEST = OUT / "experiment_manifest.json"
BASELINE_MATRIX = OUT / "baseline_score_matrix.csv"
BASELINE_METRICS = OUT / "baseline_metrics.json"
PERIOD_STATS = OUT / "period_stats.csv"
SCORE_COVERAGE = OUT / "asset_score_coverage.csv"
EPISODES = OUT / "forced_candidate_episode_advantage_all_oos.csv"
MONTHLY = OUT / "candidate_month_summary.csv"
YEARLY = OUT / "candidate_year_summary.csv"
GLOBAL = OUT / "candidate_global_summary.csv"
SUMMARY = OUT / "summary.json"

SAVE_EVERY = 250
DEFAULT_MAX_RECONVERGENCE_BARS = 252

PREVIOUS_BASELINE_DIRS = (
    ROOT / "output" / "tiingo_forced_candidate_episode_advantage_v1",
    ROOT / "output" / "tiingo_periodic_marginal_asset_search_v1",
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], utc=True)
    frame["asset_scores"] = frame["asset_scores_json"].map(json.loads)
    return frame.drop(columns=["asset_scores_json"])


def save_manifest(dataset_sha: str, cfg_hash: str, max_bars: int) -> None:
    write_json(
        MANIFEST,
        {
            "schema_version": 2,
            "script_version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "forced_candidate_episode_advantage_all_oos",
            "diagnostic_only": True,
            "future_information_used_as_target": True,
            "promotion_without_future_walk_forward_validation_forbidden": True,
            "dataset_sha256": dataset_sha,
            "config_sha256": cfg_hash,
            "universe_assets": list(ASSETS),
            "calendar_anchor_assets": list(CALENDAR_ANCHOR_ASSETS),
            "max_reconvergence_bars": int(max_bars),
            "target": "episode_delta_log_capital_until_policy_state_reconvergence",
            "reconvergence_state": ["position", "holding_days"],
            "rollout_design": (
                "paired baseline vs force candidate only at t; "
                "normal frozen-score policy resumes at t+1"
            ),
        },
    )


def manifest_matches(dataset_sha: str, cfg_hash: str, max_bars: int) -> bool:
    payload = read_json(MANIFEST)
    return bool(
        payload
        and payload.get("script_version") == VERSION
        and payload.get("dataset_sha256") == dataset_sha
        and payload.get("config_sha256") == cfg_hash
        and int(payload.get("max_reconvergence_bars", -1)) == int(max_bars)
    )


def try_reuse_previous_baseline(
    dataset_sha: str,
    cfg_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    for directory in PREVIOUS_BASELINE_DIRS:
        source_manifest = directory / "experiment_manifest.json"
        source_matrix = directory / "baseline_score_matrix.csv"
        source_metrics = directory / "baseline_metrics.json"
        if not (source_manifest.exists() and source_matrix.exists() and source_metrics.exists()):
            continue

        metadata = read_json(source_manifest) or {}
        if (
            metadata.get("dataset_sha256") != dataset_sha
            or metadata.get("config_sha256") != cfg_hash
        ):
            continue

        log(f"Reutilizando baseline congelado compativel de: {directory.name}")
        shutil.copy2(source_matrix, BASELINE_MATRIX)
        shutil.copy2(source_metrics, BASELINE_METRICS)
        return load_baseline(BASELINE_MATRIX), read_json(BASELINE_METRICS) or {}
    return None


def train_baseline(
    series: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()

    def progress(percent: float, stage: str, completed: int) -> None:
        log(f"[baseline] {percent:5.1f}% | {stage}")

    def technical(message: str) -> None:
        log(f"[motor] {message}")

    results = run_rotation_models(
        series,
        config,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=progress,
        technical_log_callback=technical,
    )
    if len(results) != 1:
        raise RuntimeError(f"Esperava um baseline; recebidos={len(results)}")

    result = results[0]
    matrix = compact_baseline(result.predictions)
    matrix.to_csv(BASELINE_MATRIX, index=False)

    metrics = dict(result.metrics)
    compact = {
        "strategy_ending_capital": float(metrics["strategy_ending_capital"]),
        "strategy_return": float(metrics["strategy_return"]),
        "strategy_cagr": float(metrics["strategy_cagr"]),
        "strategy_sharpe": float(metrics["strategy_sharpe"]),
        "strategy_maximum_drawdown": float(metrics["strategy_maximum_drawdown"]),
        "capital_rotations": int(metrics.get("capital_rotations") or 0),
        "walk_forward_folds": metrics.get("walk_forward_folds"),
        "effective_compute_device": metrics.get("effective_compute_device"),
        "effective_compute_device_note": metrics.get("effective_compute_device_note"),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(BASELINE_METRICS, compact)
    return load_baseline(BASELINE_MATRIX), compact


def ensure_baseline(
    series: dict[str, pd.DataFrame],
    config: Any,
    dataset_sha: str,
    cfg_hash: str,
    max_bars: int,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (
        not refresh
        and manifest_matches(dataset_sha, cfg_hash, max_bars)
        and BASELINE_MATRIX.exists()
        and BASELINE_METRICS.exists()
    ):
        log("Baseline v2 congelado encontrado; nao sera retreinado")
        return load_baseline(BASELINE_MATRIX), read_json(BASELINE_METRICS) or {}

    if not refresh:
        reused = try_reuse_previous_baseline(dataset_sha, cfg_hash)
        if reused is not None:
            save_manifest(dataset_sha, cfg_hash, max_bars)
            return reused

    log("Treinando baseline LightGBM uma unica vez")
    baseline, metrics = train_baseline(series, config)
    save_manifest(dataset_sha, cfg_hash, max_bars)
    return baseline, metrics


def position_for_asset(mapping: dict[str, int], asset: Any) -> int:
    value = str(asset or "CASH").strip().upper()
    if not value or value == "CASH":
        return 0
    return int(mapping.get(value, 0))


def build_baseline_state(
    baseline: pd.DataFrame,
    symbol_to_position: dict[str, int],
) -> tuple[
    dict[pd.Timestamp, int],
    dict[pd.Timestamp, int],
    dict[pd.Timestamp, int],
]:
    position_before: dict[pd.Timestamp, int] = {}
    holding_before: dict[pd.Timestamp, int] = {}
    action: dict[pd.Timestamp, int] = {}
    for row in baseline.itertuples(index=False):
        date = pd.Timestamp(row.decision_date)
        position_before[date] = position_for_asset(
            symbol_to_position, getattr(row, "previous_asset")
        )
        holding_raw = getattr(row, "holding_days_at_decision")
        holding_before[date] = int(holding_raw) if pd.notna(holding_raw) else 0
        action[date] = position_for_asset(
            symbol_to_position, getattr(row, "selected_asset")
        )
    return position_before, holding_before, action


def precompute_transition_arrays(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count_dates = max(0, len(decision_dates) - 1)
    count_positions = len(symbols) + 1
    overnight = np.zeros((count_dates, count_positions), dtype=np.float64)
    intraday = np.zeros((count_dates, count_positions), dtype=np.float64)

    for idx in range(count_dates):
        now = pd.Timestamp(decision_dates[idx])
        nxt = pd.Timestamp(decision_dates[idx + 1])
        for position, symbol in enumerate(symbols, start=1):
            frame = frames[symbol]
            try:
                close_now = float(frame.loc[now, "close"])
                open_next = float(frame.loc[nxt, "open"])
                close_next = float(frame.loc[nxt, "close"])
            except (KeyError, TypeError, ValueError):
                overnight[idx, position] = np.nan
                intraday[idx, position] = np.nan
                continue

            overnight[idx, position] = (
                math.log(open_next / close_now)
                if np.isfinite(close_now)
                and close_now > 0
                and np.isfinite(open_next)
                and open_next > 0
                else np.nan
            )
            intraday[idx, position] = (
                math.log(close_next / open_next)
                if np.isfinite(open_next)
                and open_next > 0
                and np.isfinite(close_next)
                and close_next > 0
                else np.nan
            )

    cost_log = np.zeros((count_positions, count_positions), dtype=np.float64)
    for from_position in range(count_positions):
        for to_position in range(count_positions):
            cost = _proportional_switch_cost(config, from_position, to_position)
            cost_log[from_position, to_position] = math.log(max(1e-8, 1.0 - cost))

    return overnight, intraday, cost_log


def transition_log(
    idx: int,
    from_position: int,
    to_position: int,
    overnight: np.ndarray,
    intraday: np.ndarray,
    cost_log: np.ndarray,
) -> float | None:
    value = (
        overnight[idx, int(from_position)]
        + cost_log[int(from_position), int(to_position)]
        + intraday[idx, int(to_position)]
    )
    return float(value) if np.isfinite(value) else None


def build_baseline_transition_prefix(
    decision_dates: pd.DatetimeIndex,
    baseline_position_before: dict[pd.Timestamp, int],
    baseline_action: dict[pd.Timestamp, int],
    overnight: np.ndarray,
    intraday: np.ndarray,
    cost_log: np.ndarray,
) -> np.ndarray:
    values = np.full(len(decision_dates) - 1, np.nan, dtype=np.float64)
    for idx in range(len(values)):
        date = pd.Timestamp(decision_dates[idx])
        if date not in baseline_position_before or date not in baseline_action:
            continue
        value = transition_log(
            idx,
            baseline_position_before[date],
            baseline_action[date],
            overnight,
            intraday,
            cost_log,
        )
        values[idx] = value if value is not None else np.nan

    prefix = np.zeros(len(values) + 1, dtype=np.float64)
    valid = np.isfinite(values)
    if not bool(valid.all()):
        bad = int((~valid).sum())
        raise RuntimeError(
            f"Baseline possui {bad} transicoes OOS sem retorno valido; abortando."
        )
    prefix[1:] = np.cumsum(values)
    return prefix


def baseline_log_between(prefix: np.ndarray, start: int, stop_exclusive: int) -> float:
    return float(prefix[stop_exclusive] - prefix[start])


def score_coverage_diagnostics(
    baseline: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    all_decision_dates: pd.DatetimeIndex,
    folds: list[dict[str, Any]],
    config: Any,
) -> pd.DataFrame:
    score_counts = {symbol: 0 for symbol in symbols}
    first_score: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
    last_score: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}

    for row in baseline.itertuples(index=False):
        date = pd.Timestamp(row.decision_date)
        scores = dict(getattr(row, "asset_scores") or {})
        for symbol in symbols:
            if finite_float(scores.get(symbol)) is None:
                continue
            score_counts[symbol] += 1
            if first_score[symbol] is None:
                first_score[symbol] = date
            last_score[symbol] = date

    rows: list[dict[str, Any]] = []
    minimum = int(config.rotation_minimum_training_rows)
    first_test_start = min(pd.Timestamp(fold["test_start"]) for fold in folds)
    oos_dates = pd.DatetimeIndex(all_decision_dates[:-1])

    for symbol in symbols:
        frame = frames[symbol]
        complete = frame[list(ROTATION_FEATURES) + ["forward_risk_adjusted_utility"]].notna().all(axis=1)
        training_rows_before_first_oos = int(
            complete.loc[complete.index < first_test_start].sum()
        )
        complete_oos = int(
            complete.reindex(oos_dates, fill_value=False).sum()
        )
        raw_start = pd.Timestamp(frame.index.min()) if len(frame.index) else None
        raw_end = pd.Timestamp(frame.index.max()) if len(frame.index) else None
        count = int(score_counts[symbol])

        reason = "ok"
        if count == 0:
            if training_rows_before_first_oos < minimum:
                reason = (
                    "insufficient_training_rows_before_first_oos"
                    f" ({training_rows_before_first_oos} < {minimum})"
                )
            elif complete_oos == 0:
                reason = "no_complete_feature_rows_in_oos"
            else:
                reason = "no_fitted_or_finite_oos_score; inspect_fold_training_coverage"
        elif count < len(baseline):
            reason = "partial_oos_score_coverage"

        rows.append(
            {
                "asset": symbol,
                "is_calendar_anchor": symbol in set(CALENDAR_ANCHOR_ASSETS),
                "frame_start": raw_start,
                "frame_end": raw_end,
                "frame_rows": int(len(frame)),
                "minimum_training_rows": minimum,
                "training_rows_before_first_oos": training_rows_before_first_oos,
                "complete_feature_rows_oos": complete_oos,
                "score_observations_oos": count,
                "score_coverage_pct": count / max(1, len(baseline)),
                "first_score_date": first_score[symbol],
                "last_score_date": last_score[symbol],
                "diagnosis": reason,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["score_observations_oos", "asset"],
        ascending=[True, True],
    )
    result.to_csv(SCORE_COVERAGE, index=False)
    return result


def build_date_context(
    date: pd.Timestamp,
    baseline_action_asset: str,
    previous_asset: str,
    scores: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, Any]:
    ranked = []
    for asset, value in scores.items():
        number = finite_float(value)
        if number is not None:
            ranked.append((str(asset).upper(), number))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    score_map = dict(ranked)
    rank_map = {asset: rank for rank, (asset, _) in enumerate(ranked, start=1)}

    top_asset = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else None
    baseline_score = score_map.get(baseline_action_asset)
    previous_score = score_map.get(previous_asset)

    universe_stats: dict[str, tuple[float | None, float | None]] = {}
    baseline_features: dict[str, float | None] = {}

    baseline_frame = frames.get(baseline_action_asset)
    baseline_row = (
        baseline_frame.loc[date]
        if baseline_frame is not None and date in baseline_frame.index
        else None
    )

    for feature in ROTATION_FEATURES:
        values: list[float] = []
        for symbol in symbols:
            frame = frames[symbol]
            if date not in frame.index:
                continue
            number = finite_float(frame.loc[date].get(feature))
            if number is not None:
                values.append(number)
        mean = float(np.mean(values)) if values else None
        std = float(np.std(values)) if values else None
        universe_stats[feature] = (mean, std)
        baseline_features[feature] = (
            finite_float(baseline_row.get(feature))
            if baseline_row is not None
            else None
        )

    return {
        "score_map": score_map,
        "rank_map": rank_map,
        "top_asset": top_asset,
        "top_score": top_score,
        "baseline_score": baseline_score,
        "previous_score": previous_score,
        "universe_stats": universe_stats,
        "baseline_features": baseline_features,
    }


def candidate_context(
    *,
    date: pd.Timestamp,
    candidate: str,
    baseline_action_asset: str,
    previous_asset: str,
    date_context: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    score_map = date_context["score_map"]
    rank_map = date_context["rank_map"]
    candidate_score = score_map.get(candidate)
    top_score = date_context["top_score"]
    baseline_score = date_context["baseline_score"]

    output: dict[str, Any] = {
        "candidate_model_score": candidate_score,
        "candidate_model_rank": rank_map.get(candidate),
        "candidate_has_model_score": candidate_score is not None,
        "top1_asset": date_context["top_asset"],
        "top1_score": top_score,
        "score_gap_candidate_to_top1": (
            float(candidate_score - top_score)
            if candidate_score is not None and top_score is not None
            else None
        ),
        "baseline_action_score": baseline_score,
        "score_gap_candidate_to_baseline_action": (
            float(candidate_score - baseline_score)
            if candidate_score is not None and baseline_score is not None
            else None
        ),
        "previous_asset_score": date_context["previous_score"],
        "candidate_is_top1": candidate == date_context["top_asset"],
        "candidate_is_baseline_action": candidate == baseline_action_asset,
        "candidate_is_previous_asset": candidate == previous_asset,
    }

    frame = frames.get(candidate)
    candidate_row = (
        frame.loc[date] if frame is not None and date in frame.index else None
    )

    for feature in ROTATION_FEATURES:
        candidate_value = (
            finite_float(candidate_row.get(feature))
            if candidate_row is not None
            else None
        )
        mean, std = date_context["universe_stats"][feature]
        baseline_value = date_context["baseline_features"][feature]
        output[f"candidate__{feature}"] = candidate_value
        output[f"relative__{feature}"] = (
            float(candidate_value - mean)
            if candidate_value is not None and mean is not None
            else None
        )
        output[f"candidate_vs_baseline__{feature}"] = (
            float(candidate_value - baseline_value)
            if candidate_value is not None and baseline_value is not None
            else None
        )
        output[f"candidate_z__{feature}"] = (
            float((candidate_value - mean) / std)
            if candidate_value is not None
            and mean is not None
            and std is not None
            and std > 1e-12
            else None
        )

    return output


def rollout_until_reconvergence(
    *,
    start_index: int,
    candidate_position: int,
    current_position: int,
    current_holding: int,
    decision_dates: pd.DatetimeIndex,
    baseline_policy,
    baseline_position_before: dict[pd.Timestamp, int],
    baseline_holding_before: dict[pd.Timestamp, int],
    baseline_prefix: np.ndarray,
    overnight: np.ndarray,
    intraday: np.ndarray,
    cost_log: np.ndarray,
    max_bars: int,
) -> dict[str, Any] | None:
    available = len(decision_dates) - 1 - start_index
    limit = min(int(max_bars), int(available))
    if limit <= 0:
        return None

    forced_position = int(current_position)
    forced_holding = int(current_holding)
    forced_log = 0.0
    peak_delta = float("-inf")
    trough_delta = float("inf")
    reconverged = False
    reconvergence_bars: int | None = None
    reconvergence_date: pd.Timestamp | None = None
    realized_bars = 0

    for step in range(1, limit + 1):
        idx = start_index + step - 1
        now = pd.Timestamp(decision_dates[idx])
        nxt = pd.Timestamp(decision_dates[idx + 1])

        if step == 1:
            forced_action = int(candidate_position)
        else:
            forced_action, _ = baseline_policy(
                now, forced_position, forced_holding
            )
            forced_action = int(forced_action)

        forced_step = transition_log(
            idx,
            forced_position,
            forced_action,
            overnight,
            intraday,
            cost_log,
        )
        if forced_step is None:
            return None

        forced_log += forced_step
        realized_bars = step

        if forced_action == forced_position:
            forced_holding = forced_holding + 1 if forced_action > 0 else 0
        else:
            forced_position = forced_action
            forced_holding = 1 if forced_position > 0 else 0

        baseline_log = baseline_log_between(
            baseline_prefix, start_index, start_index + step
        )
        delta = forced_log - baseline_log
        peak_delta = max(peak_delta, delta)
        trough_delta = min(trough_delta, delta)

        baseline_next_position = baseline_position_before.get(nxt)
        baseline_next_holding = baseline_holding_before.get(nxt)
        if (
            baseline_next_position is not None
            and baseline_next_holding is not None
            and forced_position == int(baseline_next_position)
            and forced_holding == int(baseline_next_holding)
        ):
            reconverged = True
            reconvergence_bars = step
            reconvergence_date = nxt
            break

    baseline_log = baseline_log_between(
        baseline_prefix, start_index, start_index + realized_bars
    )
    delta = forced_log - baseline_log

    return {
        "episode_bars": int(realized_bars),
        "reconverged": bool(reconverged),
        "reconvergence_bars": reconvergence_bars,
        "reconvergence_date": reconvergence_date,
        "target_censored": not bool(reconverged),
        "baseline_episode_log_return": float(baseline_log),
        "forced_episode_log_return": float(forced_log),
        "episode_delta_log_capital": float(delta),
        "episode_delta_capital_pct": float(math.exp(delta) - 1.0),
        "episode_peak_delta_log_capital": float(peak_delta),
        "episode_peak_advantage_pct": float(math.exp(peak_delta) - 1.0),
        "episode_trough_delta_log_capital": float(trough_delta),
        "episode_trough_advantage_pct": float(math.exp(trough_delta) - 1.0),
        "candidate_beats_baseline": bool(delta > 0.0),
    }


def load_rows() -> list[dict[str, Any]]:
    if not EPISODES.exists():
        return []
    return pd.read_csv(EPISODES).to_dict("records")


def save_rows(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(EPISODES, index=False)


def aggregate(rows: pd.DataFrame) -> None:
    if rows.empty:
        return

    valid = rows.loc[rows["forced_action_changed"].astype(bool)].copy()
    valid["decision_date"] = pd.to_datetime(valid["decision_date"], utc=True)
    valid["month"] = valid["decision_date"].dt.strftime("%Y-%m")
    valid["year"] = valid["decision_date"].dt.year.astype(int)

    def summarize(group_columns: list[str]) -> pd.DataFrame:
        return (
            valid.groupby(group_columns, as_index=False)
            .agg(
                episodes=("candidate", "size"),
                reconverged_rate=("reconverged", "mean"),
                mean_reconvergence_bars=("reconvergence_bars", "mean"),
                median_reconvergence_bars=("reconvergence_bars", "median"),
                mean_episode_delta_log_capital=("episode_delta_log_capital", "mean"),
                median_episode_delta_log_capital=("episode_delta_log_capital", "median"),
                positive_rate=("candidate_beats_baseline", "mean"),
                mean_peak_advantage_pct=("episode_peak_advantage_pct", "mean"),
                mean_trough_advantage_pct=("episode_trough_advantage_pct", "mean"),
            )
        )

    monthly = summarize(["month", "candidate"]).sort_values(
        ["month", "mean_episode_delta_log_capital"],
        ascending=[True, False],
    )
    yearly = summarize(["year", "candidate"]).sort_values(
        ["year", "mean_episode_delta_log_capital"],
        ascending=[True, False],
    )
    global_summary = summarize(["candidate"]).sort_values(
        ["mean_episode_delta_log_capital", "positive_rate"],
        ascending=[False, False],
    )

    monthly.to_csv(MONTHLY, index=False)
    yearly.to_csv(YEARLY, index=False)
    global_summary.to_csv(GLOBAL, index=False)

    changed = int(valid.shape[0])
    reconverged_count = int(valid["reconverged"].astype(bool).sum())
    uncensored = valid.loc[~valid["target_censored"].astype(bool)]
    write_json(
        SUMMARY,
        {
            "schema_version": 2,
            "script_version": VERSION,
            "rows_total": int(len(rows)),
            "forced_action_changed_rows": changed,
            "candidate_count": int(valid["candidate"].nunique()),
            "decision_count": int(valid["decision_date"].nunique()),
            "oos_start": valid["decision_date"].min(),
            "oos_end": valid["decision_date"].max(),
            "reconverged_rows": reconverged_count,
            "reconverged_rate": reconverged_count / max(1, changed),
            "median_reconvergence_bars": (
                float(uncensored["reconvergence_bars"].median())
                if not uncensored.empty
                else None
            ),
            "p90_reconvergence_bars": (
                float(uncensored["reconvergence_bars"].quantile(0.90))
                if not uncensored.empty
                else None
            ),
            "positive_rate_uncensored": (
                float(uncensored["candidate_beats_baseline"].mean())
                if not uncensored.empty
                else None
            ),
            "top_global_candidates": global_summary.head(20).to_dict("records"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forca candidato em cada decisao OOS e mede vantagem ate reconvergencia"
    )
    parser.add_argument(
        "--max-reconvergence-bars",
        type=int,
        default=DEFAULT_MAX_RECONVERGENCE_BARS,
        help="Limite maximo para esperar reconvergencia do estado da politica",
    )
    parser.add_argument(
        "--limit-assets",
        type=int,
        default=None,
        help="Limite de ativos para smoke test",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Limite de datas OOS para smoke test",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="Forca novo treinamento LightGBM em vez de reutilizar baseline compativel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.max_reconvergence_bars) < 1:
        raise SystemExit("--max-reconvergence-bars deve ser >= 1")

    OUT.mkdir(parents=True, exist_ok=True)
    config = experiment_config()
    signature = dataset_signature()
    dataset_sha = str(signature["combined_sha256"])
    cfg_hash = config_hash(config)

    if EPISODES.exists() and MANIFEST.exists() and not manifest_matches(
        dataset_sha, cfg_hash, int(args.max_reconvergence_bars)
    ):
        raise RuntimeError(
            "Ha resultados v2 existentes com outro dataset/config/max-bars. "
            "Mova ou apague a pasta output/tiingo_forced_candidate_episode_advantage_v2 "
            "antes de iniciar uma campanha diferente."
        )

    log(f"Experimento: {VERSION}")
    log("Escopo: TODO o OOS, 56 candidatos, target ate reconvergencia")
    log(f"Dataset SHA-256: {dataset_sha}")
    log(f"Config  SHA-256: {cfg_hash}")

    log("Carregando Tiingo split-causal em memoria")
    series = load_tiingo_split_causal()

    baseline, baseline_metrics = ensure_baseline(
        series,
        config,
        dataset_sha,
        cfg_hash,
        int(args.max_reconvergence_bars),
        bool(args.refresh_baseline),
    )
    log(
        "Baseline | capital=US$ "
        f"{float(baseline_metrics.get('strategy_ending_capital', float('nan'))):,.2f}"
    )

    stats = period_statistics(baseline, float(config.initial_capital))
    stats.to_csv(PERIOD_STATS, index=False)

    (
        frames,
        common_dates,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _build_execution_context(series, config)
    del common_dates, decision_to_fold

    score_by_date, margin_by_date = score_lookup(baseline)
    baseline_policy = make_replay_policy(
        symbols,
        config,
        score_by_date,
        margin_by_date,
        lambda _: set(symbols),
    )

    log("Validando replay congelado contra o baseline")
    replay_control(
        baseline,
        frames,
        symbols,
        all_decision_dates,
        decision_metadata,
        config,
        score_by_date,
        margin_by_date,
    )

    coverage = score_coverage_diagnostics(
        baseline,
        frames,
        symbols,
        all_decision_dates,
        folds,
        config,
    )
    missing = coverage.loc[coverage["score_observations_oos"].eq(0)]
    if not missing.empty:
        for row in missing.itertuples(index=False):
            log(
                f"[cobertura] {row.asset}: 0 scores OOS | "
                f"train_rows_pre_oos={row.training_rows_before_first_oos} | "
                f"diagnostico={row.diagnosis}"
            )

    symbol_to_position = {symbol: idx + 1 for idx, symbol in enumerate(symbols)}
    baseline_position_before, baseline_holding_before, baseline_action = (
        build_baseline_state(baseline, symbol_to_position)
    )

    log("Precomputando retornos de transicao em RAM")
    overnight, intraday, cost_log = precompute_transition_arrays(
        frames, symbols, all_decision_dates, config
    )
    baseline_prefix = build_baseline_transition_prefix(
        all_decision_dates,
        baseline_position_before,
        baseline_action,
        overnight,
        intraday,
        cost_log,
    )

    baseline_by_date = {
        pd.Timestamp(row.decision_date): row
        for row in baseline.itertuples(index=False)
    }
    decision_index = {
        pd.Timestamp(date): idx
        for idx, date in enumerate(all_decision_dates)
    }

    selected_dates = [
        pd.Timestamp(date)
        for date in all_decision_dates[:-1]
        if pd.Timestamp(date) in baseline_by_date
    ]
    if args.limit_episodes is not None:
        selected_dates = selected_dates[: max(0, int(args.limit_episodes))]

    candidates = list(symbols)
    if args.limit_assets is not None:
        candidates = candidates[: max(0, int(args.limit_assets))]

    rows = load_rows()
    completed = {
        (iso(row.get("decision_date")), str(row.get("candidate")).upper())
        for row in rows
    }

    total = len(selected_dates) * len(candidates)
    processed = 0
    new_since_save = 0
    started = time.perf_counter()

    for date in selected_dates:
        baseline_row = baseline_by_date[date]
        previous_asset = str(
            getattr(baseline_row, "previous_asset") or "CASH"
        ).upper()
        baseline_action_asset = str(
            getattr(baseline_row, "selected_asset") or "CASH"
        ).upper()
        current_position = int(baseline_position_before[date])
        current_holding = int(baseline_holding_before[date])
        start_index = int(decision_index[date])
        scores = dict(getattr(baseline_row, "asset_scores") or {})

        expected_position, _ = baseline_policy(
            date, current_position, current_holding
        )
        expected_asset = (
            symbols[int(expected_position) - 1]
            if int(expected_position) > 0
            else "CASH"
        )
        if expected_asset != baseline_action_asset:
            raise RuntimeError(
                "Replay local divergiu do baseline em "
                f"{date.date()}: baseline={baseline_action_asset}, replay={expected_asset}"
            )

        date_context = build_date_context(
            date,
            baseline_action_asset,
            previous_asset,
            scores,
            frames,
            symbols,
        )

        for candidate in candidates:
            processed += 1
            key = (iso(date), candidate)
            if key in completed:
                continue

            candidate_position = int(symbol_to_position[candidate])
            # Para a intervencao em t, o candidato precisa ter preco valido na
            # proxima sessao. Score LightGBM NAO e requisito na v2.
            if not np.isfinite(intraday[start_index, candidate_position]):
                continue

            rollout = rollout_until_reconvergence(
                start_index=start_index,
                candidate_position=candidate_position,
                current_position=current_position,
                current_holding=current_holding,
                decision_dates=all_decision_dates,
                baseline_policy=baseline_policy,
                baseline_position_before=baseline_position_before,
                baseline_holding_before=baseline_holding_before,
                baseline_prefix=baseline_prefix,
                overnight=overnight,
                intraday=intraday,
                cost_log=cost_log,
                max_bars=int(args.max_reconvergence_bars),
            )
            if rollout is None:
                continue

            context = candidate_context(
                date=date,
                candidate=candidate,
                baseline_action_asset=baseline_action_asset,
                previous_asset=previous_asset,
                date_context=date_context,
                frames=frames,
            )

            row = {
                "schema_version": 2,
                "script_version": VERSION,
                "decision_date": iso(date),
                "period": date.strftime("%Y-%m"),
                "year": int(date.year),
                "walk_forward_fold": getattr(baseline_row, "walk_forward_fold"),
                "previous_asset": previous_asset,
                "holding_days_at_decision": current_holding,
                "baseline_action": baseline_action_asset,
                "candidate": candidate,
                "forced_action_changed": candidate != baseline_action_asset,
                **context,
                **rollout,
            }
            rows.append(row)
            completed.add(key)
            new_since_save += 1

            if new_since_save >= SAVE_EVERY:
                save_rows(rows)
                new_since_save = 0

            if processed % 500 == 0:
                elapsed = time.perf_counter() - started
                done_pct = 100.0 * processed / max(1, total)
                eta = (
                    elapsed * (total - processed) / processed
                    if processed > 0
                    else float("nan")
                )
                log(
                    f"{processed:,}/{total:,} ({done_pct:5.1f}%) | "
                    f"{date.date()} | {candidate} | "
                    f"delta={float(row['episode_delta_capital_pct']):+.2%} | "
                    f"reconv={row['reconvergence_bars']} | "
                    f"ETA={eta/60:.1f}m"
                )

    save_rows(rows)
    frame = pd.DataFrame(rows)
    aggregate(frame)

    save_manifest(dataset_sha, cfg_hash, int(args.max_reconvergence_bars))

    log(
        f"Concluido | linhas={len(frame):,} | "
        f"datas={frame['decision_date'].nunique() if not frame.empty else 0} | "
        f"candidatos={frame['candidate'].nunique() if not frame.empty else 0}"
    )
    log(f"Dataset: {EPISODES}")
    log(f"Cobertura de scores: {SCORE_COVERAGE}")
    log(f"Resumo: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
