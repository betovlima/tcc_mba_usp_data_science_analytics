"""TCC v1.24.6 — labels completos de utilidade, RAW sem alteracoes.

Auditoria offline (Spyder F5), mantendo unico universo fixo de 75 ativos.

Reconstrói rótulos de utilidade multihorizonte do motor usando OHLC RAW
e multiplicadores de QUANTIDADE exclusivamente nos caminhos econômicos
entre a abertura da compra em t+1 e cada sessão futura. O CSV RAW nunca
é ajustado ou reescrito. A série split_normalize legada é construída
somente EM MEMORIA para medir equivalência dos labels de treinamento.

Nao executa modelos, calibracao, backtest, ordens, ou altera snapshots.
Saida: output/utilidades_economicas_raw/<ID>/pacote_utilidades_economicas_raw.zip
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.rotacao import construir_quadro_rotacao
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import load_actions_file, load_raw_bar_file, split_normalize
from reproducao.universo_fixo75 import universo_75
from auditar_retornos_economicos_splits_spyder import split_events

ROOT = Path(__file__).resolve().parent
VERSION = "1.24.6"
SNAPSHOT_END = "2026-10-22"
WINDOW = 60
RTOL = 1e-9
ATOL = 1e-9


def utility_names(config=CONFIG) -> tuple[str, ...]:
    h = tuple(int(n) for n in config.rotation_target_horizons)
    return (
        *(f"forward_horizon_utility_{n}" for n in h),
        *(f"forward_horizon_net_log_return_{n}" for n in h),
        "forward_net_log_return", "forward_cash_edge",
        "forward_movement_capture", "forward_trend_persistence",
        "forward_risk_adjusted_utility",
    )


def economic_utility_at_decision(
    raw: pd.DataFrame,
    factors: tuple[tuple[pd.Timestamp, float], ...],
    *,
    decision_index: int,
    config=CONFIG,
) -> dict[str, float]:
    """Reproduz a fórmula exata do motor para utilidade e risco por horizonte.

    Quando uma posição foi aberta na ex-date, as novas unidades já são
    negociadas na abertura; jamais aplicar aquele split novamente.
    Para os dias posteriores, valorização=preco_RAW * fator_de_quantidade.
    Só rótulos ex-post: nenhuma informação futura é retornada como feature.
    """
    horizons = tuple(int(value) for value in config.rotation_target_horizons)
    weights = np.asarray(config.rotation_target_horizon_weights, dtype=float)
    weights = weights / weights.sum()
    maximum = max(horizons)
    if decision_index < 0 or decision_index + maximum >= len(raw):
        raise ValueError("Rótulo t+h ainda indisponível no histórico RAW.")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Calendário RAW inválido.")
    start = decision_index + 1
    entry = float(raw.iloc[start]["open"])
    if not np.isfinite(entry) or entry <= 0:
        raise ValueError("Preço de abertura RAW inválido.")
    indices = pd.DatetimeIndex(raw.index).normalize()
    entry_date = indices[start]

    qty = np.ones(maximum, dtype=np.float64)
    for event_date, factor in factors:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Fator de split inválido.")
        if event_date <= entry_date:
            continue
        qty[indices[start:start+maximum] >= event_date] *= factor

    end = start + maximum
    close = raw.iloc[start:end]["close"].to_numpy(dtype=np.float64) * qty
    high = raw.iloc[start:end]["high"].to_numpy(dtype=np.float64) * qty
    low = raw.iloc[start:end]["low"].to_numpy(dtype=np.float64) * qty
    if not (
        np.isfinite(close).all() and np.isfinite(high).all()
        and np.isfinite(low).all() and (low > 0).all()
    ):
        raise ValueError("Preços RAW futuros inválidos.")

    max_upside = max(0., float(np.nanmax(high) / entry - 1.))
    positive_share = float(np.mean(close > entry))
    steps = np.diff(close)
    positive_steps = float(np.mean(steps > 0)) if len(steps) else 0.
    movement_capture = math.log1p(max_upside)
    trend_persistence = 0.5 * positive_share + 0.5 * positive_steps

    round_trip = min(
        .25, 2. * (
            max(0., float(config.slippage_bps)) / 10000.
            + max(0., float(config.commission_rate))
        ),
    )
    net_cost_log = math.log(max(1e-12, 1. - round_trip))
    utilities = []
    net_returns = []
    results: dict[str, float] = {}
    for horizon in horizons:
        min_low = float(np.nanmin(low[:horizon]))
        gross_log = math.log(max(float(close[horizon-1] / entry), 1e-12))
        downside = max(0., 1. - min_low / entry)
        path = np.concatenate(([entry], close[:horizon]))
        peaks = np.maximum.accumulate(path)
        dd = 1. - np.divide(
            path, peaks, out=np.ones_like(path), where=peaks > 0,
        )
        path_drawdown = max(0., float(np.nanmax(dd)))
        net_log = gross_log + net_cost_log
        utility = (
            net_log
            - float(config.rotation_downside_penalty) * downside
            - float(config.rotation_drawdown_penalty) * path_drawdown
        )
        results[f"forward_horizon_net_log_return_{horizon}"] = net_log
        results[f"forward_horizon_utility_{horizon}"] = utility
        utilities.append(utility)
        net_returns.append(net_log)

    weighted_util = float(np.dot(np.asarray(utilities), weights))
    weighted_net = float(np.dot(np.asarray(net_returns), weights))
    results.update({
        "forward_net_log_return": weighted_net,
        "forward_cash_edge": weighted_util,
        "forward_movement_capture": movement_capture,
        "forward_trend_persistence": trend_persistence,
        "forward_risk_adjusted_utility": (
            weighted_util
            + float(config.rotation_movement_capture_weight) * movement_capture
            + float(config.rotation_trend_persistence_weight) * trend_persistence
        ),
    })
    return results


def compare_labels_near_splits(
    symbol: str, raw: pd.DataFrame, actions: list[dict], *,
    config=CONFIG, window: int=WINDOW,
) -> tuple[dict, list[dict]]:
    original_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy(
            dtype=np.uint64,
        ).tobytes(),
    ).hexdigest()
    events = split_events(actions)
    summary = {
        "symbol": symbol, "split_events": len(events),
        "decisions_compared": 0, "label_cells_compared": 0,
        "decisions_crossing_split": 0,
        "materially_different_cells": 0,
        "max_abs_error": 0.,
    }
    if not events:
        return summary, []
    adjusted, _ = split_normalize(raw, actions)  # apenas RAM
    legacy = construir_quadro_rotacao(adjusted, config)
    names = utility_names(config)
    maxh = max(map(int, config.rotation_target_horizons))
    days = pd.DatetimeIndex(raw.index).normalize()
    candidates: set[int] = set()
    for day, _ in events:
        position = int(days.searchsorted(day, side="left"))
        if not 0 < position < len(days):
            continue
        candidates.update(range(
            max(0, position-window), min(len(days)-maxh, position+window),
        ))

    deviations: list[dict] = []
    for position in sorted(candidates):
        date = raw.index[position]
        if date not in legacy.index:
            continue  # Aquecimento inicial do modelo
        measured = economic_utility_at_decision(
            raw, events, decision_index=position, config=config,
        )
        matured = raw.index[position+maxh]
        crossing = any(
            days[position+1] < day <= days[position+maxh] for day, _ in events
        )
        summary["decisions_compared"] += 1
        summary["decisions_crossing_split"] += int(crossing)
        for name in names:
            actual = float(measured[name])
            expected = float(legacy.loc[date, name])
            if not (np.isfinite(actual) and np.isfinite(expected)):
                raise RuntimeError(
                    f"{symbol} {date}: rótulo não finito em {name}."
                )
            summary["label_cells_compared"] += 1
            delta = abs(actual-expected)
            summary["max_abs_error"] = max(
                summary["max_abs_error"], float(delta),
            )
            if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
                summary["materially_different_cells"] += 1
                deviations.append({
                    "symbol": symbol,
                    "decision_day": date.date().isoformat(),
                    "label_matures_after_close_on": matured.date().isoformat(),
                    "name": name,
                    "economic_raw_quantity_value": actual,
                    "legacy_split_normalized_value": expected,
                    "absolute_difference": float(delta),
                    "decision_crosses_split": crossing,
                })
    if hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy(
            dtype=np.uint64,
        ).tobytes(),
    ).hexdigest() != original_fingerprint:
        raise RuntimeError(f"{symbol}: OHLCV RAW modificado.")
    return summary, deviations


def audit(*, root: Path=ROOT) -> dict:
    root = Path(root).resolve()
    contract = json.loads(
        (root/"evidencias"/"manifest_universo_congelado.json").read_text(
            encoding="utf-8",
        )
    )
    assets = universo_75(contract, ())
    paths = None
    snapshot = None
    for folder in sorted(
        (root/"dados"/".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_END}_*/universe_75"
        ), reverse=True,
    ):
        candidate = SnapshotPaths.from_root(folder)
        if candidate.manifest.is_file():
            check = validate_snapshot(candidate)
            if tuple(check.get("assets") or ()) == assets:
                paths, snapshot = candidate, check
                break
    if paths is None or snapshot is None:
        raise FileNotFoundError(
            "Snapshot U75 validado não encontrado, sem downloads."
        )
    output_rows = []
    all_diffs: list[dict] = []
    for symbol in assets:
        raw = load_raw_bar_file(
            paths.raw_bars/f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        )
        actions = load_actions_file(
            paths.corporate_actions/f"{symbol}.csv",
        )
        result, diffs = compare_labels_near_splits(symbol, raw, actions)
        output_rows.append(result)
        all_diffs.extend(diffs)
        if result["split_events"]:
            print(
                f"[utility] {symbol} eventos={result['split_events']} "
                f"rotulos={result['label_cells_compared']} "
                f"divergencias={result['materially_different_cells']}",
                flush=True,
            )
    table = pd.DataFrame(output_rows)
    folder = (
        root/"output"/"utilidades_economicas_raw"
        /datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    folder.mkdir(parents=True, exist_ok=False)
    table.to_csv(folder/"por_ativo.csv", index=False)
    pd.DataFrame(
        all_diffs, columns=[
            "symbol", "decision_day", "label_matures_after_close_on",
            "name", "economic_raw_quantity_value",
            "legacy_split_normalized_value", "absolute_difference",
            "decision_crosses_split",
        ],
    ).to_csv(folder/"divergencias_labels.csv", index=False)
    summary = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "assets": len(assets),
        "assets_with_split_events": int((table["split_events"] > 0).sum()),
        "split_events": int(table["split_events"].sum()),
        "horizons": list(CONFIG.rotation_target_horizons),
        "label_columns": len(utility_names()),
        "decisions_compared": int(table["decisions_compared"].sum()),
        "cells_compared": int(table["label_cells_compared"].sum()),
        "materially_different_cells": int(
            table["materially_different_cells"].sum()
        ),
        "maximum_abs_difference": float(table["max_abs_error"].max()),
        "raw_modified": False,
        "model_retrained": False,
        "backtest_or_capital": False,
        "scope": (
            "Equivalência de utilidade ex-post em janelas próximas aos splits, "
            "não testa índices temporais usados no ajuste, scores, custo real "
            "da carteira ou disponibilidade histórica de anúncios."
        ),
    }
    (folder/"resumo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    archive = folder/"pacote_utilidades_economicas_raw.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("resumo.json", "por_ativo.csv", "divergencias_labels.csv"):
            z.write(folder/name, arcname=name)
    print(f"[utility] relatório {archive}", flush=True)
    return summary


if __name__ == "__main__":
    audit()
