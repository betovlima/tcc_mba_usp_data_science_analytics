"""Vantagem contrafactual por episodio: forcar candidato agora e voltar a politica.

Versao: tiingo-forced-candidate-episode-advantage-v1.0.0

Esta pesquisa usa o baseline LightGBM Tiingo 56 e constroi rollouts pareados:

    politica normal
    versus
    forcar candidato A na decisao t e voltar a politica normal em t+1.

O objetivo nao e promover a escolha ex-post. O objetivo e gerar um dataset
supervisionado limpo, por data x candidato, contendo somente contexto conhecido
em t como features e a vantagem marginal futura observada como target.

Os horizontes padrao sao os mesmos da estrategia: 5, 10, 20, 40 e 60 sessoes.

IMPORTANTE
----------
- O LightGBM e treinado uma unica vez por fold para produzir os scores OOS.
- Os rollouts reutilizam a matriz congelada de scores OOS.
- A forca bruta e diagnostica/ex-post.
- Qualquer seletor aprendido depois precisa ser treinado em janelas anteriores
  e testado em walk-forward futuro, sem usar o proprio periodo como resposta.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import (
    LEGACY_ROTATION_MODE,
    ROTATION_FEATURES,
    _training_transition_log_return,
    run_rotation_models,
)
from tcc_engine.config import ASSETS, CALENDAR_ANCHOR_ASSETS, CONFIG
from tcc_engine.execution import apply_slippage, calculate_reference_fees
from tcc_engine.research_challengers import _build_execution_context
from tunar_lightgbm_tiingo import dataset_signature, load_tiingo_split_causal
from pesquisar_contribuicao_marginal_periodica_tiingo import (
    choose_weak_periods,
    make_replay_policy,
    period_statistics,
    replay_control,
    score_lookup,
)

VERSION = "tiingo-forced-candidate-episode-advantage-v1.0.0"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "tiingo_forced_candidate_episode_advantage_v1"
MANIFEST = OUT / "experiment_manifest.json"
BASELINE_MATRIX = OUT / "baseline_score_matrix.csv"
BASELINE_METRICS = OUT / "baseline_metrics.json"
PERIOD_STATS = OUT / "period_stats.csv"
WEAK_PERIODS = OUT / "weak_periods.csv"
EPISODES = OUT / "forced_candidate_episode_advantage.csv"
PERIOD_CANDIDATE = OUT / "candidate_period_summary.csv"
GLOBAL_CANDIDATE = OUT / "candidate_global_summary.csv"
PERIOD_MATRIX = OUT / "candidate_period_weighted_delta_matrix.csv"
SUMMARY = OUT / "summary.json"

DEFAULT_PERIODS = 8
HORIZONS = (5, 10, 20, 40, 60)
SAVE_EVERY = 100


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_config():
    config = CONFIG.model_copy(
        update={
            "assets": tuple(ASSETS),
            "calendar_anchor_assets": tuple(CALENDAR_ANCHOR_ASSETS),
            "research_reference_assets": tuple(ASSETS),
            "research_candidate_assets": (),
            "research_capture_full_score_matrix": True,
        }
    )
    if str(config.strategy_mode) != LEGACY_ROTATION_MODE:
        raise RuntimeError(
            "Este experimento v1 foi validado somente para "
            f"{LEGACY_ROTATION_MODE}; recebido={config.strategy_mode}"
        )
    if len(ASSETS) != 56:
        raise RuntimeError(f"Esperados 56 ativos; encontrados={len(ASSETS)}")
    if len(CALENDAR_ANCHOR_ASSETS) != 37:
        raise RuntimeError(
            f"Esperados 37 calendar anchors; encontrados={len(CALENDAR_ANCHOR_ASSETS)}"
        )
    return config


def config_hash(config: Any) -> str:
    payload = asdict(config)
    payload.pop("research_capture_full_score_matrix", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_baseline(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "timestamp" not in frame.columns:
        frame = frame.reset_index()

    required = {
        "timestamp",
        "decision_date",
        "strategy_equity",
        "selected_asset",
        "previous_asset",
        "holding_days_at_decision",
        "effective_switch_margin",
        "walk_forward_fold",
        "asset_scores",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "Baseline sem campos para rollout contrafactual: " + ", ".join(missing)
        )

    optional = [
        "trade_action",
        "decision_score",
        "top_1_asset",
        "top_1_score",
        "top_2_asset",
        "top_2_score",
        "top_3_asset",
        "top_3_score",
        "current_asset",
        "current_score",
        "current_asset_rank",
        "universe_score_mean",
        "universe_score_std",
        "best_vs_second_gap",
        "best_vs_current_gap",
        "spy_return_5",
        "spy_return_20",
        "spy_realized_volatility_20",
        "universe_breadth_5",
        "universe_breadth_20",
    ]
    columns = [
        "timestamp",
        "decision_date",
        "strategy_equity",
        "selected_asset",
        "previous_asset",
        "holding_days_at_decision",
        "effective_switch_margin",
        "walk_forward_fold",
        "asset_scores",
        *[name for name in optional if name in frame.columns],
    ]
    frame = frame[columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], utc=True)
    frame["asset_scores_json"] = frame["asset_scores"].map(
        lambda value: json.dumps(value or {}, sort_keys=True, separators=(",", ":"))
    )
    return frame.drop(columns=["asset_scores"])


def load_baseline_matrix() -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_MATRIX)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], utc=True)
    frame["asset_scores"] = frame["asset_scores_json"].map(json.loads)
    return frame.drop(columns=["asset_scores_json"])


def baseline_cache_valid(dataset_sha: str, cfg_hash: str) -> bool:
    manifest = read_json(MANIFEST)
    return bool(
        manifest
        and manifest.get("script_version") == VERSION
        and manifest.get("dataset_sha256") == dataset_sha
        and manifest.get("config_sha256") == cfg_hash
        and BASELINE_MATRIX.exists()
        and BASELINE_METRICS.exists()
    )


def save_manifest(dataset_sha: str, cfg_hash: str) -> None:
    write_json(
        MANIFEST,
        {
            "schema_version": 1,
            "script_version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "forced_candidate_episode_advantage",
            "diagnostic_only": True,
            "future_information_used_as_target": True,
            "promotion_without_future_walk_forward_validation_forbidden": True,
            "dataset_sha256": dataset_sha,
            "config_sha256": cfg_hash,
            "universe_assets": list(ASSETS),
            "calendar_anchor_assets": list(CALENDAR_ANCHOR_ASSETS),
            "horizons": list(HORIZONS),
            "rollout_design": (
                "paired normal policy vs force candidate at t; "
                "normal policy resumes at next decision"
            ),
        },
    )


def ensure_baseline(
    series: dict[str, pd.DataFrame],
    config: Any,
    dataset_sha: str,
    cfg_hash: str,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not refresh and baseline_cache_valid(dataset_sha, cfg_hash):
        log("Baseline OOS congelado encontrado; nao sera retreinado")
        return load_baseline_matrix(), read_json(BASELINE_METRICS) or {}

    log("Treinando baseline LightGBM uma unica vez e capturando scores OOS completos")
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
        raise RuntimeError(f"Esperava 1 resultado baseline; recebidos={len(results)}")

    result = results[0]
    matrix = compact_baseline(result.predictions)
    matrix.to_csv(BASELINE_MATRIX, index=False)

    metrics = dict(result.metrics)
    compact_metrics = {
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
    write_json(BASELINE_METRICS, compact_metrics)
    save_manifest(dataset_sha, cfg_hash)

    loaded = matrix.assign(
        asset_scores=matrix["asset_scores_json"].map(json.loads)
    ).drop(columns=["asset_scores_json"])
    log(
        "Baseline pronto | capital=US$ "
        f"{compact_metrics['strategy_ending_capital']:,.2f} | "
        f"device={compact_metrics.get('effective_compute_device')}"
    )
    return loaded, compact_metrics


def position_for_asset(symbol_to_position: dict[str, int], asset: str | None) -> int:
    if asset is None:
        return 0
    value = str(asset).strip().upper()
    if not value or value == "CASH":
        return 0
    return int(symbol_to_position.get(value, 0))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def rank_scores(scores: dict[str, Any]) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for asset, value in scores.items():
        score = finite_float(value)
        if score is not None:
            ranked.append((str(asset).upper(), score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def rollout_pair(
    *,
    start_index: int,
    candidate_position: int,
    current_position: int,
    current_holding_days: int,
    decision_dates: pd.DatetimeIndex,
    baseline_policy,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
) -> dict[int, dict[str, float]]:
    max_horizon = max(HORIZONS)
    available = len(decision_dates) - 1 - start_index
    steps = min(max_horizon, available)
    if steps <= 0:
        return {}

    normal_position = int(current_position)
    normal_holding = int(current_holding_days)
    forced_position = int(current_position)
    forced_holding = int(current_holding_days)
    normal_log = 0.0
    forced_log = 0.0
    outputs: dict[int, dict[str, float]] = {}

    for step in range(1, steps + 1):
        now = pd.Timestamp(decision_dates[start_index + step - 1])
        nxt = pd.Timestamp(decision_dates[start_index + step])

        normal_action, _ = baseline_policy(now, normal_position, normal_holding)

        if step == 1:
            forced_action = int(candidate_position)
        else:
            forced_action, _ = baseline_policy(now, forced_position, forced_holding)

        normal_step_log = _training_transition_log_return(
            frames,
            symbols,
            now,
            nxt,
            normal_position,
            int(normal_action),
            config,
        )
        forced_step_log = _training_transition_log_return(
            frames,
            symbols,
            now,
            nxt,
            forced_position,
            int(forced_action),
            config,
        )

        normal_log += float(normal_step_log)
        forced_log += float(forced_step_log)

        if int(normal_action) == normal_position:
            normal_holding = normal_holding + 1 if int(normal_action) > 0 else 0
        else:
            normal_position = int(normal_action)
            normal_holding = 1 if normal_position > 0 else 0

        if int(forced_action) == forced_position:
            forced_holding = forced_holding + 1 if int(forced_action) > 0 else 0
        else:
            forced_position = int(forced_action)
            forced_holding = 1 if forced_position > 0 else 0

        if step in HORIZONS:
            delta_log = forced_log - normal_log
            outputs[step] = {
                "baseline_log_return": float(normal_log),
                "forced_log_return": float(forced_log),
                "delta_log_capital": float(delta_log),
                "delta_capital_pct": float(math.exp(delta_log) - 1.0),
            }

    return outputs


def context_features(
    *,
    decision_date: pd.Timestamp,
    candidate: str,
    baseline_action: str,
    previous_asset: str,
    scores: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, Any]:
    ranked = rank_scores(scores)
    rank_map = {asset: rank for rank, (asset, _) in enumerate(ranked, start=1)}
    score_map = dict(ranked)
    top_asset = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else None
    candidate_score = score_map.get(candidate)
    baseline_score = score_map.get(baseline_action)
    previous_score = score_map.get(previous_asset)

    output: dict[str, Any] = {
        "candidate_model_score": candidate_score,
        "candidate_model_rank": rank_map.get(candidate),
        "top1_asset": top_asset,
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
        "previous_asset_score": previous_score,
        "candidate_is_top1": bool(candidate == top_asset),
        "candidate_is_baseline_action": bool(candidate == baseline_action),
        "candidate_is_previous_asset": bool(candidate == previous_asset),
    }

    # Features locais do candidato e deltas relativos ao universo naquele t.
    candidate_frame = frames.get(candidate)
    if candidate_frame is not None and decision_date in candidate_frame.index:
        candidate_row = candidate_frame.loc[decision_date]
    else:
        candidate_row = None

    baseline_frame = frames.get(baseline_action)
    baseline_row = (
        baseline_frame.loc[decision_date]
        if baseline_frame is not None and decision_date in baseline_frame.index
        else None
    )

    for feature in ROTATION_FEATURES:
        candidate_value = (
            finite_float(candidate_row.get(feature)) if candidate_row is not None else None
        )
        baseline_value = (
            finite_float(baseline_row.get(feature)) if baseline_row is not None else None
        )
        universe_values: list[float] = []
        for symbol in symbols:
            frame = frames.get(symbol)
            if frame is None or decision_date not in frame.index:
                continue
            value = finite_float(frame.loc[decision_date].get(feature))
            if value is not None:
                universe_values.append(value)
        universe_mean = float(np.mean(universe_values)) if universe_values else None
        universe_std = float(np.std(universe_values)) if universe_values else None

        output[f"candidate__{feature}"] = candidate_value
        output[f"relative__{feature}"] = (
            float(candidate_value - universe_mean)
            if candidate_value is not None and universe_mean is not None
            else None
        )
        output[f"candidate_vs_baseline__{feature}"] = (
            float(candidate_value - baseline_value)
            if candidate_value is not None and baseline_value is not None
            else None
        )
        output[f"candidate_z__{feature}"] = (
            float((candidate_value - universe_mean) / universe_std)
            if candidate_value is not None
            and universe_mean is not None
            and universe_std is not None
            and universe_std > 1e-12
            else None
        )

    return output


def save_episode_rows(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(EPISODES, index=False)


def load_episode_rows() -> list[dict[str, Any]]:
    if not EPISODES.exists():
        return []
    return pd.read_csv(EPISODES).to_dict("records")


def aggregate_outputs(rows: pd.DataFrame) -> None:
    if rows.empty:
        return

    horizon_weights = dict(
        zip(
            [int(value) for value in CONFIG.rotation_target_horizons],
            [float(value) for value in CONFIG.rotation_target_horizon_weights],
        )
    )
    weighted_columns = []
    for horizon in HORIZONS:
        column = f"delta_log_capital_{horizon}"
        if column in rows.columns:
            weighted_columns.append((column, float(horizon_weights.get(horizon, 0.0))))

    if weighted_columns:
        total_weight = sum(weight for _, weight in weighted_columns)
        rows = rows.copy()
        rows["weighted_delta_log_capital"] = sum(
            rows[column].fillna(0.0) * weight
            for column, weight in weighted_columns
        ) / max(total_weight, 1e-12)
        rows["weighted_delta_capital_pct"] = np.exp(
            rows["weighted_delta_log_capital"]
        ) - 1.0
        rows.to_csv(EPISODES, index=False)

    agg_spec: dict[str, tuple[str, str]] = {
        "episodes": ("candidate", "size"),
        "mean_weighted_delta_log_capital": ("weighted_delta_log_capital", "mean"),
        "median_weighted_delta_log_capital": ("weighted_delta_log_capital", "median"),
        "positive_weighted_rate": (
            "weighted_delta_log_capital",
            lambda x: float((x > 0).mean()),
        ),
        "mean_delta_5": ("delta_log_capital_5", "mean"),
        "mean_delta_10": ("delta_log_capital_10", "mean"),
        "mean_delta_20": ("delta_log_capital_20", "mean"),
        "mean_delta_40": ("delta_log_capital_40", "mean"),
        "mean_delta_60": ("delta_log_capital_60", "mean"),
    }

    period_candidate = (
        rows.groupby(["period", "candidate"], as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["period", "mean_weighted_delta_log_capital"],
            ascending=[True, False],
        )
    )
    period_candidate.to_csv(PERIOD_CANDIDATE, index=False)

    global_candidate = (
        rows.groupby(["candidate"], as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["mean_weighted_delta_log_capital", "positive_weighted_rate"],
            ascending=[False, False],
        )
    )
    global_candidate.to_csv(GLOBAL_CANDIDATE, index=False)

    period_candidate.pivot_table(
        index="candidate",
        columns="period",
        values="mean_weighted_delta_log_capital",
        aggfunc="first",
    ).to_csv(PERIOD_MATRIX)

    top_period = (
        period_candidate.groupby("period", group_keys=False)
        .head(5)
        .to_dict("records")
    )
    write_json(
        SUMMARY,
        {
            "schema_version": 1,
            "script_version": VERSION,
            "rows": int(len(rows)),
            "periods": sorted(rows["period"].astype(str).unique().tolist()),
            "candidate_count": int(rows["candidate"].nunique()),
            "top_candidates_by_period": top_period,
            "top_global_candidates": global_candidate.head(20).to_dict("records"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rollout pareado: politica normal vs forcar candidato agora"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=DEFAULT_PERIODS,
        help="Quantidade de meses OOS mais fracos a investigar",
    )
    parser.add_argument(
        "--limit-assets",
        type=int,
        default=None,
        help="Limite opcional de candidatos por episodio para teste rapido",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Limite opcional de datas de decisao para teste rapido",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="Ignora baseline congelado e retreina LightGBM",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = experiment_config()
    signature = dataset_signature()
    dataset_sha = str(signature["combined_sha256"])
    cfg_hash = config_hash(config)

    log(f"Experimento: {VERSION}")
    log(f"Dataset SHA-256: {dataset_sha}")
    log(f"Config  SHA-256: {cfg_hash}")
    log("Carregando Tiingo 56 split-causal em memoria")
    series = load_tiingo_split_causal()

    baseline, baseline_metrics = ensure_baseline(
        series,
        config,
        dataset_sha,
        cfg_hash,
        bool(args.refresh_baseline),
    )

    stats = period_statistics(baseline, float(config.initial_capital))
    stats.to_csv(PERIOD_STATS, index=False)
    weak = choose_weak_periods(stats, int(args.periods))
    weak.to_csv(WEAK_PERIODS, index=False)
    periods = weak["period"].astype(str).tolist()
    log(
        "Meses investigados: "
        + ", ".join(
            f"{row.period} ({float(row.period_return):+.2%})"
            for row in weak.itertuples(index=False)
        )
    )

    (
        frames,
        common_dates,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _build_execution_context(series, config)
    del common_dates, folds, decision_to_fold

    score_by_date, margin_by_date = score_lookup(baseline)
    all_assets = set(symbols)
    baseline_policy = make_replay_policy(
        symbols,
        config,
        score_by_date,
        margin_by_date,
        lambda _: all_assets,
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

    baseline_by_date = {
        pd.Timestamp(row.decision_date): row
        for row in baseline.itertuples(index=False)
    }
    decision_index = {
        pd.Timestamp(date): idx
        for idx, date in enumerate(all_decision_dates)
    }
    symbol_to_position = {symbol: idx + 1 for idx, symbol in enumerate(symbols)}

    selected_dates = [
        pd.Timestamp(date)
        for date in all_decision_dates[:-1]
        if pd.Timestamp(date).strftime("%Y-%m") in periods
        and pd.Timestamp(date) in baseline_by_date
    ]
    if args.limit_episodes is not None:
        selected_dates = selected_dates[: max(0, int(args.limit_episodes))]

    candidates = list(symbols)
    if args.limit_assets is not None:
        candidates = candidates[: max(0, int(args.limit_assets))]

    existing_rows = load_episode_rows()
    completed = {
        (str(row.get("decision_date")), str(row.get("candidate")))
        for row in existing_rows
    }
    rows = list(existing_rows)

    total = len(selected_dates) * len(candidates)
    counter = 0
    new_since_save = 0
    started_all = time.perf_counter()

    for date in selected_dates:
        baseline_row = baseline_by_date[date]
        period = date.strftime("%Y-%m")
        previous_asset = str(getattr(baseline_row, "previous_asset") or "CASH").upper()
        baseline_action = str(getattr(baseline_row, "selected_asset") or "CASH").upper()
        holding = int(getattr(baseline_row, "holding_days_at_decision") or 0)
        current_position = position_for_asset(symbol_to_position, previous_asset)

        # Protecao: a politica congelada precisa reproduzir a acao observada em t.
        expected_position, _ = baseline_policy(date, current_position, holding)
        expected_asset = (
            symbols[int(expected_position) - 1] if int(expected_position) > 0 else "CASH"
        )
        if expected_asset != baseline_action:
            raise RuntimeError(
                "Politica congelada divergiu do baseline em "
                f"{date.date()}: esperado={baseline_action}, replay={expected_asset}"
            )

        scores = dict(getattr(baseline_row, "asset_scores") or {})
        start_index = int(decision_index[date])

        for candidate in candidates:
            counter += 1
            key = (str(date), candidate)
            if key in completed:
                continue

            candidate_score = finite_float(scores.get(candidate))
            if candidate_score is None:
                continue

            candidate_position = symbol_to_position[candidate]
            rollout = rollout_pair(
                start_index=start_index,
                candidate_position=candidate_position,
                current_position=current_position,
                current_holding_days=holding,
                decision_dates=all_decision_dates,
                baseline_policy=baseline_policy,
                frames=frames,
                symbols=symbols,
                config=config,
            )
            if not rollout:
                continue

            context = context_features(
                decision_date=date,
                candidate=candidate,
                baseline_action=baseline_action,
                previous_asset=previous_asset,
                scores=scores,
                frames=frames,
                symbols=symbols,
            )

            row: dict[str, Any] = {
                "schema_version": 1,
                "script_version": VERSION,
                "period": period,
                "decision_date": date,
                "walk_forward_fold": getattr(baseline_row, "walk_forward_fold"),
                "previous_asset": previous_asset,
                "holding_days_at_decision": holding,
                "baseline_action": baseline_action,
                "candidate": candidate,
                "forced_action_changed": bool(candidate != baseline_action),
                **context,
            }

            positive_count = 0
            available_count = 0
            weighted_sum = 0.0
            weighted_total = 0.0
            horizon_weights = dict(
                zip(
                    [int(value) for value in config.rotation_target_horizons],
                    [float(value) for value in config.rotation_target_horizon_weights],
                )
            )
            for horizon in HORIZONS:
                metrics = rollout.get(horizon)
                if metrics is None:
                    row[f"baseline_log_return_{horizon}"] = None
                    row[f"forced_log_return_{horizon}"] = None
                    row[f"delta_log_capital_{horizon}"] = None
                    row[f"delta_capital_pct_{horizon}"] = None
                    continue
                delta = float(metrics["delta_log_capital"])
                row[f"baseline_log_return_{horizon}"] = metrics["baseline_log_return"]
                row[f"forced_log_return_{horizon}"] = metrics["forced_log_return"]
                row[f"delta_log_capital_{horizon}"] = delta
                row[f"delta_capital_pct_{horizon}"] = metrics["delta_capital_pct"]
                available_count += 1
                positive_count += int(delta > 0.0)
                weight = float(horizon_weights.get(horizon, 0.0))
                weighted_sum += delta * weight
                weighted_total += weight

            row["positive_horizon_count"] = positive_count
            row["available_horizon_count"] = available_count
            row["positive_horizon_rate"] = (
                positive_count / available_count if available_count else None
            )
            row["weighted_delta_log_capital"] = (
                weighted_sum / weighted_total if weighted_total > 0 else None
            )
            row["weighted_delta_capital_pct"] = (
                math.exp(row["weighted_delta_log_capital"]) - 1.0
                if row["weighted_delta_log_capital"] is not None
                else None
            )
            row["target_positive"] = bool(
                row["weighted_delta_log_capital"] is not None
                and row["weighted_delta_log_capital"] > 0.0
            )

            rows.append(row)
            completed.add(key)
            new_since_save += 1

            if new_since_save >= SAVE_EVERY:
                save_episode_rows(rows)
                new_since_save = 0

            if counter % 100 == 0:
                elapsed = time.perf_counter() - started_all
                log(
                    f"{counter}/{total} | {period} | {date.date()} | "
                    f"forca={candidate} | "
                    f"weighted_delta={row['weighted_delta_capital_pct']:+.2%} | "
                    f"elapsed={elapsed:.1f}s"
                )

    save_episode_rows(rows)
    episode_frame = pd.DataFrame(rows)
    aggregate_outputs(episode_frame)

    log(
        f"Concluido | linhas={len(episode_frame):,} | "
        f"periodos={len(periods)} | candidatos={len(candidates)}"
    )
    log(f"Dataset supervisionado: {EPISODES}")
    log(f"Resumo candidato x periodo: {PERIOD_CANDIDATE}")
    log(f"Matriz: {PERIOD_MATRIX}")
    log(f"Resumo: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
