"""Pesquisa de contribuicao marginal periodica de ativos sobre Tiingo.

Versao: tiingo-periodic-marginal-asset-search-v1.0.0

Objetivo
--------
Mapear "buracos" temporais do baseline LightGBM Tiingo 56 e testar, por forca
bruta diagnostica, se a elegibilidade de cada ativo altera positivamente a
trajetoria naquele periodo.

Esta etapa e EX-POST e diagnostica. Nenhum resultado deste script deve ser
promovido diretamente para a estrategia. Os deltas encontrados servem para
construir exemplos de aprendizado contextual que depois precisam ser avaliados
em walk-forward fora da amostra.

Dois contrafactuais sao medidos em cada mes fraco:

1. leave-one-out:
   baseline 56 vs baseline com um ativo impedido de disputar NOVAS decisoes
   somente naquele mes. Um delta positivo indica que a presenca daquele ativo
   prejudicou a trajetoria naquele periodo.

2. core-plus-one:
   durante o mes, restringe a disputa aos 37 ativos do nucleo temporal e mede
   o efeito de adicionar individualmente cada um dos 19 ativos adicionais.
   Um delta positivo indica que o candidato conseguiu "tapar o buraco" em
   relacao ao nucleo naquele periodo.

Os modelos NAO sao retreinados em cada contrafactual. O LightGBM e treinado
uma unica vez por fold, e a matriz completa de scores OOS e congelada em RAM/
CSV. Como os modelos sao independentes por ativo, a mudanca de elegibilidade
atua apenas sobre a competicao/ranking, que e justamente a hipotese testada.
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
from typing import Any, Callable

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import (
    LEGACY_ROTATION_MODE,
    _simulate_exact,
)
from tcc_engine.config import ASSETS, CALENDAR_ANCHOR_ASSETS, CONFIG
from tcc_engine.execution import apply_slippage, calculate_reference_fees
from tcc_engine.research_challengers import (
    _build_execution_context,
)
from tcc_engine.capital_rotation import run_rotation_models
from tunar_lightgbm_tiingo import (
    dataset_signature,
    load_tiingo_split_causal,
)

VERSION = "tiingo-periodic-marginal-asset-search-v1.0.0"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "tiingo_periodic_marginal_asset_search_v1"
MANIFEST_PATH = OUT / "experiment_manifest.json"
BASELINE_MATRIX_PATH = OUT / "baseline_score_matrix.csv"
BASELINE_METRICS_PATH = OUT / "baseline_metrics.json"
PERIOD_STATS_PATH = OUT / "period_stats.csv"
WEAK_PERIODS_PATH = OUT / "weak_periods.csv"
ASSET_RANK_PATH = OUT / "asset_period_rank.csv"
LEAVE_ONE_OUT_PATH = OUT / "leave_one_out.csv"
CORE_PLUS_ONE_PATH = OUT / "core_plus_one.csv"
LEAVE_MATRIX_PATH = OUT / "leave_one_out_matrix.csv"
CORE_MATRIX_PATH = OUT / "core_plus_one_matrix.csv"
SUMMARY_PATH = OUT / "summary.json"

DEFAULT_WEAK_PERIODS = 8
CAPITAL_TOLERANCE = 1e-6


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
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def config_for_experiment():
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
            "Este diagnostico v1 foi validado somente para "
            f"{LEGACY_ROTATION_MODE}; recebido={config.strategy_mode}"
        )
    if len(ASSETS) != 56:
        raise RuntimeError(f"Esperados 56 ativos; encontrados={len(ASSETS)}")
    if len(CALENDAR_ANCHOR_ASSETS) != 37:
        raise RuntimeError(
            f"Esperados 37 calendar anchors; encontrados={len(CALENDAR_ANCHOR_ASSETS)}"
        )
    return config


def config_sha256(config: Any) -> str:
    payload = asdict(config)
    # Captura da matriz e dispositivo mudam auditoria/performance, nao a
    # especificacao economica do baseline.
    payload.pop("research_capture_full_score_matrix", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def compact_baseline_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "decision_date",
        "strategy_equity",
        "selected_asset",
        "previous_asset",
        "effective_switch_margin",
        "walk_forward_fold",
        "asset_scores",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RuntimeError(
            "Baseline sem colunas necessarias para replay: " + ", ".join(missing)
        )

    frame = predictions[
        [
            "timestamp",
            "decision_date",
            "strategy_equity",
            "selected_asset",
            "previous_asset",
            "trade_action",
            "decision_score",
            "effective_switch_margin",
            "walk_forward_fold",
            "asset_scores",
        ]
    ].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], utc=True)
    frame["asset_scores_json"] = frame["asset_scores"].map(
        lambda value: json.dumps(value or {}, sort_keys=True, separators=(",", ":"))
    )
    return frame.drop(columns=["asset_scores"])


def load_baseline_matrix() -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_MATRIX_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], utc=True)
    frame["asset_scores"] = frame["asset_scores_json"].map(json.loads)
    return frame.drop(columns=["asset_scores_json"])


def baseline_cache_valid(
    signature: dict[str, Any],
    config_hash: str,
) -> bool:
    manifest = read_json(MANIFEST_PATH)
    return bool(
        manifest
        and manifest.get("script_version") == VERSION
        and manifest.get("dataset_sha256") == signature.get("combined_sha256")
        and manifest.get("config_sha256") == config_hash
        and BASELINE_MATRIX_PATH.exists()
        and BASELINE_METRICS_PATH.exists()
    )


def save_manifest(signature: dict[str, Any], config_hash: str) -> None:
    write_json(
        MANIFEST_PATH,
        {
            "schema_version": 1,
            "script_version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "tiingo_periodic_marginal_asset_search",
            "diagnostic_only": True,
            "future_information_used_for_period_selection": True,
            "promotion_without_new_walk_forward_test_forbidden": True,
            "universe_assets": list(ASSETS),
            "core_assets": list(CALENDAR_ANCHOR_ASSETS),
            "extension_assets": [
                asset for asset in ASSETS if asset not in set(CALENDAR_ANCHOR_ASSETS)
            ],
            "dataset_sha256": signature.get("combined_sha256"),
            "config_sha256": config_hash,
            "method": {
                "weak_period_unit": "calendar_month_by_decision_date",
                "weak_period_ranking": "lowest_baseline_period_return",
                "leave_one_out": "remove_asset_from_new_decision_competition_in_period",
                "core_plus_one": "37_core_assets_plus_one_extension_candidate_in_period",
                "models_retrained_per_counterfactual": False,
                "score_matrix": "same_frozen_OOS_LightGBM_scores",
            },
        },
    )


def ensure_baseline(
    series: dict[str, pd.DataFrame],
    config: Any,
    signature: dict[str, Any],
    config_hash: str,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not refresh and baseline_cache_valid(signature, config_hash):
        log("Baseline cache valido; treinamento LightGBM nao sera repetido")
        return load_baseline_matrix(), read_json(BASELINE_METRICS_PATH) or {}

    log("Executando baseline LightGBM uma unica vez e capturando todos os scores OOS")
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
        raise RuntimeError(f"Esperava uma execucao baseline; recebidas={len(results)}")

    result = results[0]
    matrix = compact_baseline_matrix(result.predictions)
    matrix.to_csv(BASELINE_MATRIX_PATH, index=False)

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
    write_json(BASELINE_METRICS_PATH, compact_metrics)
    save_manifest(signature, config_hash)

    log(
        "Baseline capturado | capital=US$ "
        f"{compact_metrics['strategy_ending_capital']:,.2f} | "
        f"device={compact_metrics.get('effective_compute_device')}"
    )
    return matrix.assign(
        asset_scores=matrix["asset_scores_json"].map(json.loads)
    ).drop(columns=["asset_scores_json"]), compact_metrics


def period_statistics(
    baseline: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    frame = baseline.sort_values("decision_date").reset_index(drop=True).copy()
    frame["period"] = frame["decision_date"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []

    for period, group in frame.groupby("period", sort=True):
        first_idx = int(group.index.min())
        start_equity = (
            float(initial_capital)
            if first_idx == 0
            else float(frame.loc[first_idx - 1, "strategy_equity"])
        )
        end_equity = float(group.iloc[-1]["strategy_equity"])
        curve = pd.Series(
            [start_equity, *group["strategy_equity"].astype(float).tolist()],
            dtype=float,
        )
        max_dd = float((curve / curve.cummax() - 1.0).min())
        period_return = float(end_equity / start_equity - 1.0)
        rows.append(
            {
                "period": period,
                "decision_start": group["decision_date"].min(),
                "decision_end": group["decision_date"].max(),
                "decision_count": int(len(group)),
                "starting_capital": start_equity,
                "ending_capital": end_equity,
                "period_return": period_return,
                "period_log_return": float(math.log(max(end_equity / start_equity, 1e-12))),
                "period_max_drawdown": max_dd,
                "rotations": int((group["trade_action"].astype(str) == "ROTATE").sum()),
                "selected_asset_count": int(group["selected_asset"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def choose_weak_periods(stats: pd.DataFrame, count: int) -> pd.DataFrame:
    count = max(1, min(int(count), len(stats)))
    ranked = stats.sort_values(
        ["period_return", "period_max_drawdown", "period"],
        ascending=[True, True, True],
    ).head(count).copy()
    ranked["weak_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def asset_period_ranking(
    baseline: pd.DataFrame,
    periods: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in periods:
        group = baseline.loc[
            baseline["decision_date"].dt.strftime("%Y-%m").eq(period)
        ].copy()
        accum: dict[str, dict[str, Any]] = {
            asset: {
                "ranks": [],
                "scores": [],
                "top1": 0,
                "top3": 0,
                "selected": 0,
            }
            for asset in ASSETS
        }
        for row in group.itertuples(index=False):
            scores = getattr(row, "asset_scores") or {}
            finite = [
                (asset, float(score))
                for asset, score in scores.items()
                if score is not None and np.isfinite(float(score))
            ]
            finite.sort(key=lambda item: (-item[1], item[0]))
            rank_map = {asset: rank for rank, (asset, _) in enumerate(finite, start=1)}
            for asset, score in finite:
                accum[asset]["ranks"].append(rank_map[asset])
                accum[asset]["scores"].append(score)
                if rank_map[asset] == 1:
                    accum[asset]["top1"] += 1
                if rank_map[asset] <= 3:
                    accum[asset]["top3"] += 1
            selected = str(getattr(row, "selected_asset") or "")
            if selected in accum:
                accum[selected]["selected"] += 1

        for asset, values in accum.items():
            ranks = values["ranks"]
            scores = values["scores"]
            rows.append(
                {
                    "period": period,
                    "asset": asset,
                    "is_core_asset": asset in set(CALENDAR_ANCHOR_ASSETS),
                    "score_observations": len(scores),
                    "mean_model_rank": float(np.mean(ranks)) if ranks else None,
                    "median_model_rank": float(np.median(ranks)) if ranks else None,
                    "mean_model_score": float(np.mean(scores)) if scores else None,
                    "top1_count": int(values["top1"]),
                    "top3_count": int(values["top3"]),
                    "selected_days": int(values["selected"]),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["period", "mean_model_rank", "asset"],
            ascending=[True, True, True],
            na_position="last",
        )
    return out


def score_lookup(baseline: pd.DataFrame) -> tuple[
    dict[pd.Timestamp, dict[str, float | None]],
    dict[pd.Timestamp, float],
]:
    score_by_date: dict[pd.Timestamp, dict[str, float | None]] = {}
    margin_by_date: dict[pd.Timestamp, float] = {}
    for row in baseline.itertuples(index=False):
        date = pd.Timestamp(getattr(row, "decision_date"))
        score_by_date[date] = dict(getattr(row, "asset_scores") or {})
        margin = getattr(row, "effective_switch_margin")
        margin_by_date[date] = (
            float(margin)
            if margin is not None and pd.notna(margin)
            else float(CONFIG.rotation_switch_margin)
        )
    return score_by_date, margin_by_date


def make_replay_policy(
    symbols: list[str],
    config: Any,
    score_by_date: dict[pd.Timestamp, dict[str, float | None]],
    margin_by_date: dict[pd.Timestamp, float],
    allowed_assets: Callable[[pd.Timestamp], set[str]],
):
    symbol_to_position = {symbol: idx + 1 for idx, symbol in enumerate(symbols)}
    minimum = float(config.rotation_cash_threshold)
    entry_threshold = minimum + float(config.rotation_min_expected_edge)

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        key = pd.Timestamp(timestamp)
        raw = score_by_date.get(key)
        if raw is None:
            raise KeyError(f"Scores OOS ausentes para {key}")

        allowed = allowed_assets(key)
        ranked: list[tuple[str, float]] = []
        for symbol in symbols:
            if symbol not in allowed:
                continue
            value = raw.get(symbol)
            if value is None:
                continue
            score = float(value)
            if np.isfinite(score):
                ranked.append((symbol, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))

        if not ranked:
            return (0, 0.0)

        best_symbol, best_value = ranked[0]
        best_position = symbol_to_position[best_symbol]
        current_symbol = symbols[current_position - 1] if current_position > 0 else None
        current_raw = raw.get(current_symbol) if current_symbol else 0.0
        current_value = (
            float(current_raw)
            if current_raw is not None and np.isfinite(float(current_raw))
            else float("-inf")
        )
        required = max(
            float(config.rotation_switch_margin),
            float(margin_by_date.get(key, config.rotation_switch_margin)),
        )

        if (
            current_position > 0
            and np.isfinite(current_value)
            and holding_days < int(config.rotation_min_holding_days)
        ):
            return (current_position, current_value)

        if best_value <= minimum:
            return (0, 0.0)

        if current_position == 0:
            if best_value >= entry_threshold:
                return (best_position, best_value)
            return (0, 0.0)

        if best_position == current_position:
            return (current_position, current_value)

        if best_value >= current_value + required:
            return (best_position, best_value)

        return (current_position, current_value)

    return policy


def decision_dates_through_period(
    all_decision_dates: pd.DatetimeIndex,
    period: str,
) -> pd.DatetimeIndex:
    labels = pd.DatetimeIndex(all_decision_dates).strftime("%Y-%m")
    positions = np.flatnonzero(labels == period)
    if len(positions) == 0:
        raise RuntimeError(f"Periodo {period} nao existe no calendario OOS")
    last_decision_position = int(positions[-1])
    # O simulador precisa da proxima data para executar a ultima decisao.
    stop = min(len(all_decision_dates), last_decision_position + 2)
    return pd.DatetimeIndex(all_decision_dates[:stop])


def period_rows(predictions: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = predictions.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], utc=True)
    return frame.loc[frame["decision_date"].dt.strftime("%Y-%m").eq(period)].copy()


def scenario_metrics(
    period: str,
    reference: pd.DataFrame,
    counterfactual: pd.DataFrame,
    reference_ending_capital: float,
) -> dict[str, Any]:
    ref = period_rows(reference, period).sort_values("decision_date")
    cf = period_rows(counterfactual, period).sort_values("decision_date")
    if ref.empty or cf.empty:
        raise RuntimeError(f"Periodo {period} sem linhas para comparacao")

    merged = ref[["decision_date", "selected_asset"]].merge(
        cf[["decision_date", "selected_asset"]],
        on="decision_date",
        how="inner",
        suffixes=("_reference", "_counterfactual"),
    )
    changed = merged["selected_asset_reference"].astype(str) != merged[
        "selected_asset_counterfactual"
    ].astype(str)
    first_divergence = (
        merged.loc[changed, "decision_date"].min() if bool(changed.any()) else None
    )
    ending = float(cf.iloc[-1]["strategy_equity"])
    delta = ending - float(reference_ending_capital)
    ratio = ending / float(reference_ending_capital)
    return {
        "counterfactual_ending_capital": ending,
        "delta_capital": delta,
        "delta_capital_pct": ratio - 1.0,
        "delta_log_capital": float(math.log(max(ratio, 1e-12))),
        "changed_decisions": int(changed.sum()),
        "compared_decisions": int(len(merged)),
        "first_divergence": first_divergence,
    }


def run_replay(
    period: str,
    allowed_inside_period: set[str],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    all_decision_dates: pd.DatetimeIndex,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]],
    config: Any,
    score_by_date: dict[pd.Timestamp, dict[str, float | None]],
    margin_by_date: dict[pd.Timestamp, float],
):
    all_assets = set(symbols)

    def allowed(date: pd.Timestamp) -> set[str]:
        return (
            allowed_inside_period
            if pd.Timestamp(date).strftime("%Y-%m") == period
            else all_assets
        )

    policy = make_replay_policy(
        symbols,
        config,
        score_by_date,
        margin_by_date,
        allowed,
    )
    dates = decision_dates_through_period(all_decision_dates, period)
    return _simulate_exact(
        "periodic_marginal_replay",
        policy,
        frames,
        symbols,
        dates,
        config,
        calculate_reference_fees,
        apply_slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=None,
        model_label="LightGBM Utility · frozen OOS score replay",
        method_line="- Frozen OOS score replay with temporary asset eligibility mask.",
    )


def replay_control(
    baseline: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    all_decision_dates: pd.DatetimeIndex,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]],
    config: Any,
    score_by_date: dict[pd.Timestamp, dict[str, float | None]],
    margin_by_date: dict[pd.Timestamp, float],
) -> None:
    all_assets = set(symbols)
    policy = make_replay_policy(
        symbols,
        config,
        score_by_date,
        margin_by_date,
        lambda _: all_assets,
    )
    replay = _simulate_exact(
        "periodic_marginal_replay_control",
        policy,
        frames,
        symbols,
        all_decision_dates,
        config,
        calculate_reference_fees,
        apply_slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=None,
        model_label="LightGBM Utility · replay control",
        method_line="- Control replay from frozen OOS scores.",
    )

    base = baseline.sort_values("decision_date").reset_index(drop=True)
    cf = replay.predictions.sort_values("decision_date").reset_index(drop=True)
    merged = base[["decision_date", "selected_asset", "strategy_equity"]].merge(
        cf[["decision_date", "selected_asset", "strategy_equity"]],
        on="decision_date",
        suffixes=("_baseline", "_replay"),
        how="inner",
    )
    divergences = (
        merged["selected_asset_baseline"].astype(str)
        != merged["selected_asset_replay"].astype(str)
    )
    baseline_end = float(base.iloc[-1]["strategy_equity"])
    replay_end = float(cf.iloc[-1]["strategy_equity"])
    delta = replay_end - baseline_end

    log(
        "Replay control | "
        f"capital baseline=US$ {baseline_end:,.2f} | "
        f"replay=US$ {replay_end:,.2f} | "
        f"divergencias={int(divergences.sum())}"
    )
    if int(divergences.sum()) != 0 or abs(delta) > CAPITAL_TOLERANCE:
        raise RuntimeError(
            "Replay congelado nao reproduziu exatamente o baseline. "
            f"divergencias={int(divergences.sum())}, delta_capital={delta:.12f}. "
            "A forca bruta foi abortada para evitar atribuicao invalida."
        )


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "first_divergence" in frame.columns:
        frame["first_divergence"] = pd.to_datetime(
            frame["first_divergence"], utc=True, errors="coerce"
        )
    return frame


def save_results(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def run_leave_one_out(
    periods: list[str],
    baseline: pd.DataFrame,
    period_stats: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    all_decision_dates: pd.DatetimeIndex,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]],
    config: Any,
    score_by_date: dict[pd.Timestamp, dict[str, float | None]],
    margin_by_date: dict[pd.Timestamp, float],
    limit_assets: int | None,
) -> pd.DataFrame:
    existing = load_existing(LEAVE_ONE_OUT_PATH)
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        (str(row["period"]), str(row["asset"]))
        for row in rows
    }
    assets = list(symbols)
    if limit_assets is not None:
        assets = assets[: max(0, int(limit_assets))]

    total = len(periods) * len(assets)
    done = 0
    all_assets = set(symbols)
    stats_by_period = period_stats.set_index("period")

    for period in periods:
        reference_end = float(stats_by_period.loc[period, "ending_capital"])
        for asset in assets:
            done += 1
            key = (period, asset)
            if key in completed:
                continue
            started = time.perf_counter()
            allowed = set(all_assets)
            allowed.discard(asset)
            result = run_replay(
                period,
                allowed,
                frames,
                symbols,
                all_decision_dates,
                decision_metadata,
                config,
                score_by_date,
                margin_by_date,
            )
            metrics = scenario_metrics(
                period,
                baseline,
                result.predictions,
                reference_end,
            )
            row = {
                "scenario": "leave_one_out",
                "period": period,
                "asset": asset,
                "is_core_asset": asset in set(CALENDAR_ANCHOR_ASSETS),
                "reference_ending_capital": reference_end,
                **metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            completed.add(key)
            save_results(LEAVE_ONE_OUT_PATH, rows)
            log(
                f"[leave-one-out] {done}/{total} | {period} | -{asset} | "
                f"delta=US$ {metrics['delta_capital']:,.2f} "
                f"({metrics['delta_capital_pct']:+.2%}) | "
                f"mudancas={metrics['changed_decisions']}"
            )

    return save_results(LEAVE_ONE_OUT_PATH, rows)


def run_core_plus_one(
    periods: list[str],
    baseline: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    all_decision_dates: pd.DatetimeIndex,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]],
    config: Any,
    score_by_date: dict[pd.Timestamp, dict[str, float | None]],
    margin_by_date: dict[pd.Timestamp, float],
    limit_assets: int | None,
) -> pd.DataFrame:
    existing = load_existing(CORE_PLUS_ONE_PATH)
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        (str(row["period"]), str(row["asset"]))
        for row in rows
    }

    core = set(CALENDAR_ANCHOR_ASSETS)
    extensions = [asset for asset in symbols if asset not in core]
    if limit_assets is not None:
        extensions = extensions[: max(0, int(limit_assets))]

    total = len(periods) * len(extensions)
    done = 0

    for period in periods:
        log(f"[core-plus-one] {period} | executando controle core-only")
        core_result = run_replay(
            period,
            core,
            frames,
            symbols,
            all_decision_dates,
            decision_metadata,
            config,
            score_by_date,
            margin_by_date,
        )
        core_rows = period_rows(core_result.predictions, period).sort_values("decision_date")
        if core_rows.empty:
            raise RuntimeError(f"Core-only sem resultado para {period}")
        core_end = float(core_rows.iloc[-1]["strategy_equity"])

        for asset in extensions:
            done += 1
            key = (period, asset)
            if key in completed:
                continue
            started = time.perf_counter()
            result = run_replay(
                period,
                core | {asset},
                frames,
                symbols,
                all_decision_dates,
                decision_metadata,
                config,
                score_by_date,
                margin_by_date,
            )
            metrics = scenario_metrics(
                period,
                core_result.predictions,
                result.predictions,
                core_end,
            )
            full_baseline_rows = period_rows(baseline, period).sort_values("decision_date")
            full_end = float(full_baseline_rows.iloc[-1]["strategy_equity"])
            candidate_end = float(
                period_rows(result.predictions, period)
                .sort_values("decision_date")
                .iloc[-1]["strategy_equity"]
            )
            row = {
                "scenario": "core_plus_one",
                "period": period,
                "asset": asset,
                "core_ending_capital": core_end,
                "full56_ending_capital": full_end,
                **metrics,
                "delta_vs_full56_capital": candidate_end - full_end,
                "delta_vs_full56_pct": candidate_end / full_end - 1.0,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            completed.add(key)
            save_results(CORE_PLUS_ONE_PATH, rows)
            log(
                f"[core-plus-one] {done}/{total} | {period} | +{asset} | "
                f"delta_vs_core=US$ {metrics['delta_capital']:,.2f} "
                f"({metrics['delta_capital_pct']:+.2%}) | "
                f"mudancas={metrics['changed_decisions']}"
            )

    return save_results(CORE_PLUS_ONE_PATH, rows)


def write_matrices(
    leave: pd.DataFrame,
    core: pd.DataFrame,
) -> None:
    if not leave.empty:
        leave.pivot_table(
            index="asset",
            columns="period",
            values="delta_capital_pct",
            aggfunc="first",
        ).to_csv(LEAVE_MATRIX_PATH)
    if not core.empty:
        core.pivot_table(
            index="asset",
            columns="period",
            values="delta_capital_pct",
            aggfunc="first",
        ).to_csv(CORE_MATRIX_PATH)


def write_summary(
    baseline_metrics: dict[str, Any],
    weak: pd.DataFrame,
    leave: pd.DataFrame,
    core: pd.DataFrame,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "script_version": VERSION,
        "diagnostic_only": True,
        "baseline": baseline_metrics,
        "weak_periods": weak.to_dict("records"),
        "leave_one_out_best": [],
        "core_plus_one_best": [],
    }
    if not leave.empty:
        best_leave = leave.sort_values(
            ["delta_capital_pct", "changed_decisions"],
            ascending=[False, True],
        ).head(20)
        payload["leave_one_out_best"] = best_leave.to_dict("records")
    if not core.empty:
        best_core = core.sort_values(
            ["delta_capital_pct", "changed_decisions"],
            ascending=[False, True],
        ).head(20)
        payload["core_plus_one_best"] = best_core.to_dict("records")
    write_json(SUMMARY_PATH, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forca bruta periodica de contribuicao marginal de ativos no Tiingo"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=DEFAULT_WEAK_PERIODS,
        help="Quantidade de meses mais fracos do baseline a investigar",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "leave-one-out", "core-plus-one"),
        default="both",
        help="Familia de contrafactuais",
    )
    parser.add_argument(
        "--limit-assets",
        type=int,
        default=None,
        help="Limite opcional de ativos por familia para teste rapido",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="Ignora score matrix congelada e retreina o baseline",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = config_for_experiment()
    log(f"Experimento: {VERSION}")
    log(
        f"Universo={len(ASSETS)} | core={len(CALENDAR_ANCHOR_ASSETS)} | "
        f"extensoes={len(ASSETS) - len(CALENDAR_ANCHOR_ASSETS)}"
    )
    log("Fase EX-POST diagnostica: resultados nao podem ser promovidos diretamente")

    log("Validando assinatura SHA-256 do snapshot Tiingo")
    signature = dataset_signature()
    config_hash = config_sha256(config)
    log(f"Dataset: {signature['combined_sha256']}")
    log(f"Config : {config_hash}")

    log("Carregando Tiingo split-causal em memoria")
    series = load_tiingo_split_causal()

    baseline, baseline_metrics = ensure_baseline(
        series,
        config,
        signature,
        config_hash,
        bool(args.refresh_baseline),
    )

    stats = period_statistics(baseline, float(config.initial_capital))
    stats.to_csv(PERIOD_STATS_PATH, index=False)
    weak = choose_weak_periods(stats, int(args.periods))
    weak.to_csv(WEAK_PERIODS_PATH, index=False)
    periods = weak["period"].astype(str).tolist()
    log(
        "Periodos fracos selecionados: "
        + ", ".join(
            f"{row.period} ({float(row.period_return):+.2%})"
            for row in weak.itertuples(index=False)
        )
    )

    ranking = asset_period_ranking(baseline, periods)
    ranking.to_csv(ASSET_RANK_PATH, index=False)
    log(f"Ranking por periodo salvo: {ASSET_RANK_PATH}")

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

    log("Validando replay congelado antes da forca bruta")
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

    leave = load_existing(LEAVE_ONE_OUT_PATH)
    core = load_existing(CORE_PLUS_ONE_PATH)

    if args.mode in ("both", "leave-one-out"):
        leave = run_leave_one_out(
            periods,
            baseline,
            stats,
            frames,
            symbols,
            all_decision_dates,
            decision_metadata,
            config,
            score_by_date,
            margin_by_date,
            args.limit_assets,
        )

    if args.mode in ("both", "core-plus-one"):
        core = run_core_plus_one(
            periods,
            baseline,
            frames,
            symbols,
            all_decision_dates,
            decision_metadata,
            config,
            score_by_date,
            margin_by_date,
            args.limit_assets,
        )

    write_matrices(leave, core)
    write_summary(baseline_metrics, weak, leave, core)

    log("Pesquisa concluida")
    log(f"Resumo: {SUMMARY_PATH}")
    log(f"Matriz leave-one-out: {LEAVE_MATRIX_PATH}")
    log(f"Matriz core-plus-one: {CORE_MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
