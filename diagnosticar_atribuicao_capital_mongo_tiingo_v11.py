"""Atribuicao exata de capital Mongo 43M vs Tiingo usando snapshots locais.

Versao: mongo-tiingo-exact-capital-attribution-v1.1.0

Base de codigo: certified-43m-spyder. O motor tcc_engine permanece intacto.
Depois de preparar os snapshots, este script nao acessa internet nem MongoDB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import ASSETS, CONFIG, END_DATE, START_DATE
from tcc_engine.execution import apply_slippage, calculate_reference_fees

VERSION = "mongo-tiingo-exact-capital-attribution-v1.1.0"
CERTIFIED_TAG = "certified-43m-spyder"
CERTIFIED_COMMIT = "726cf883489a403fae5699cce715e59863e4a368"
CERTIFIED_CAPITAL = 43_759_854.82
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dados"
MONGO_DIR = DATA_ROOT / "referencia_mongo_43m"
TIINGO_SERIES_DIR = DATA_ROOT / "series_historicas"
TIINGO_EVENTS_DIR = DATA_ROOT / "eventos_corporativos"
OUTPUT_DIR = ROOT / "output" / "mongo_tiingo_exact_capital_attribution_v1"
OHLCV = ["open", "high", "low", "close", "volume"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame(symbol: str, table: pd.DataFrame, source: str) -> pd.DataFrame:
    required = ["timestamp", *OHLCV]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}/{source}: colunas ausentes: {', '.join(missing)}")
    result = table[required].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    for column in OHLCV:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=required)
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    valid = (
        (result["open"] > 0) & (result["high"] > 0) & (result["low"] > 0)
        & (result["close"] > 0) & (result["volume"] >= 0)
    )
    result = result.loc[valid].set_index("timestamp")
    if result.empty:
        raise RuntimeError(f"{symbol}/{source}: serie vazia")
    return result[OHLCV].copy()


def load_snapshot_frames(directory: Path, source: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not directory.exists():
        raise RuntimeError(
            f"Snapshot ausente: {directory}. Rode primeiro: %run preparar_snapshots_mongo_tiingo.py"
        )
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, dict[str, str]] = {}
    for index, symbol in enumerate(ASSETS, start=1):
        path = directory / f"{symbol}.csv"
        if not path.exists():
            raise RuntimeError(
                f"{source}: faltando {path}. Rode novamente preparar_snapshots_mongo_tiingo.py"
            )
        frame = validate_frame(symbol, pd.read_csv(path), source)
        frames[symbol] = frame
        hashes[symbol] = {"ohlcv": sha256_file(path)}
        if index == 1 or index % 5 == 0 or index == len(ASSETS):
            log(f"{source} {index:02d}/{len(ASSETS)} {symbol} | candles={len(frame)}")
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return frames, {"combined_sha256": signature, "files": hashes}


def load_events(symbol: str) -> pd.DataFrame:
    path = TIINGO_EVENTS_DIR / f"{symbol}.csv"
    if not path.exists():
        raise RuntimeError(
            f"Tiingo: faltando {path}. Rode novamente preparar_snapshots_mongo_tiingo.py"
        )
    table = pd.read_csv(path)
    required = ["timestamp", "dividendo", "fator_split"]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}/eventos: colunas ausentes: {', '.join(missing)}")
    table = table[required].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    table["dividendo"] = pd.to_numeric(table["dividendo"], errors="coerce").fillna(0.0)
    table["fator_split"] = pd.to_numeric(table["fator_split"], errors="coerce").fillna(1.0)
    return table.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def normalize_tiingo_causally(symbol: str, raw: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    split_on_session = pd.Series(1.0, index=raw.index, dtype=float)
    split_events = events.loc[events["fator_split"] != 1.0]
    for event in split_events.itertuples(index=False):
        event_date = pd.Timestamp(event.timestamp).normalize()
        sessions = raw.index[raw.index.normalize() == event_date]
        if len(sessions) != 1:
            raise RuntimeError(f"{symbol}: split {event_date.date()} sem sessao unica")
        factor = float(event.fator_split)
        if not np.isfinite(factor) or factor <= 0:
            raise RuntimeError(f"{symbol}: fator split invalido em {event_date.date()}: {factor}")
        split_on_session.loc[sessions[0]] *= factor

    split_cumulative = split_on_session.cumprod()
    split_series = raw.copy()
    for column in ("open", "high", "low", "close"):
        split_series[column] = split_series[column] * split_cumulative
    split_series["volume"] = split_series["volume"] / split_cumulative

    dividend_on_session = pd.Series(1.0, index=split_series.index, dtype=float)
    dividend_events = events.loc[events["dividendo"] != 0.0]
    for event in dividend_events.itertuples(index=False):
        event_date = pd.Timestamp(event.timestamp).normalize()
        sessions = split_series.index[split_series.index.normalize() == event_date]
        if not len(sessions):
            continue
        if len(sessions) != 1:
            raise RuntimeError(f"{symbol}: dividendo {event_date.date()} sem sessao unica")
        session = sessions[0]
        position = int(split_series.index.get_loc(session))
        if position == 0:
            continue
        previous = split_series.index[position - 1]
        previous_close = float(split_series.loc[previous, "close"])
        normalized_dividend = float(event.dividendo) * float(split_cumulative.loc[session])
        denominator = previous_close - normalized_dividend
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError(f"{symbol}: dividendo invalido em {event_date.date()}")
        dividend_on_session.loc[session] *= previous_close / denominator

    dividend_cumulative = dividend_on_session.cumprod()
    result = split_series.copy()
    for column in ("open", "high", "low", "close"):
        result[column] = result[column] * dividend_cumulative
    return result


def load_tiingo_frames() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not TIINGO_SERIES_DIR.exists() or not TIINGO_EVENTS_DIR.exists():
        raise RuntimeError(
            "Snapshot Tiingo ausente. Rode primeiro: %run preparar_snapshots_mongo_tiingo.py"
        )
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, dict[str, str]] = {}
    for index, symbol in enumerate(ASSETS, start=1):
        raw_path = TIINGO_SERIES_DIR / f"{symbol}.csv"
        events_path = TIINGO_EVENTS_DIR / f"{symbol}.csv"
        if not raw_path.exists() or not events_path.exists():
            raise RuntimeError(f"Tiingo incompleta para {symbol}; rode preparar_snapshots_mongo_tiingo.py")
        raw = validate_frame(symbol, pd.read_csv(raw_path), "tiingo_raw")
        events = load_events(symbol)
        total = normalize_tiingo_causally(symbol, raw, events)
        frames[symbol] = total
        hashes[symbol] = {"raw": sha256_file(raw_path), "events": sha256_file(events_path)}
        if index == 1 or index % 5 == 0 or index == len(ASSETS):
            log(f"Tiingo {index:02d}/{len(ASSETS)} {symbol} | candles={len(total)} | eventos={len(events)}")
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return frames, {"combined_sha256": signature, "files": hashes}


def progress_callback(percent: float, stage: str, completed_runs: int) -> None:
    log(f"{percent:5.1f}% | {stage}")


def technical_log_callback(message: str) -> None:
    log(f"[motor] {message}")


def run_engine(frames: dict[str, pd.DataFrame]) -> Any:
    results = run_rotation_models(
        frames, CONFIG, calculate_reference_fees, apply_slippage,
        progress_callback=progress_callback,
        technical_log_callback=technical_log_callback,
    )
    if not results or len(results) != 1:
        raise RuntimeError(f"Motor retornou {len(results) if results else 0} resultados")
    return results[0]


def predictions_for_compare(result: Any) -> pd.DataFrame:
    table = result.predictions.copy()
    if not isinstance(table.index, pd.DatetimeIndex):
        table.index = pd.to_datetime(table.index, utc=True, errors="coerce")
    elif table.index.tz is None:
        table.index = table.index.tz_localize("UTC")
    else:
        table.index = table.index.tz_convert("UTC")
    return table.sort_index()


def compare_decisions(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    if "selected_asset" not in reference.columns or "selected_asset" not in candidate.columns:
        return {"available": False}
    common = reference.index.intersection(candidate.index)
    if common.empty:
        return {"available": False, "reason": "no_common_sessions"}
    left = reference.loc[common, "selected_asset"].astype(str)
    right = candidate.loc[common, "selected_asset"].astype(str)
    different = left != right
    count = int(different.sum())
    first = common[different.to_numpy()][0] if count else None
    payload: dict[str, Any] = {
        "available": True,
        "common_sessions": int(len(common)),
        "divergent_sessions": count,
        "divergence_rate": float(count / len(common)),
        "first_divergence": first,
    }
    if first is not None:
        payload["reference_selected_asset"] = str(left.loc[first])
        payload["candidate_selected_asset"] = str(right.loc[first])
        for column in ("selected_score", "decision_score", "walk_forward_fold"):
            if column in reference.columns:
                payload[f"reference_{column}"] = json_default(reference.loc[first, column])
            if column in candidate.columns:
                payload[f"candidate_{column}"] = json_default(candidate.loc[first, column])
    return payload


def scenario_frames(mongo: dict[str, pd.DataFrame], tiingo: dict[str, pd.DataFrame], direction: str, symbol: str | None) -> dict[str, pd.DataFrame]:
    if direction == "mongo":
        return {a: mongo[a].copy() for a in ASSETS}
    if direction == "tiingo":
        return {a: tiingo[a].copy() for a in ASSETS}
    if symbol is None:
        raise ValueError("symbol obrigatorio")
    if direction == "mongo-to-tiingo":
        result = {a: mongo[a].copy() for a in ASSETS}
        result[symbol] = tiingo[symbol].copy()
        return result
    if direction == "tiingo-to-mongo":
        result = {a: tiingo[a].copy() for a in ASSETS}
        result[symbol] = mongo[symbol].copy()
        return result
    raise ValueError(direction)


def scenario_name(direction: str, symbol: str | None) -> str:
    return f"all_{direction}" if direction in ("mongo", "tiingo") else f"{direction}_{symbol}"


def load_baseline_predictions() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "all_mongo_predictions.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, index_col=0)
    table.index = pd.to_datetime(table.index, utc=True, errors="coerce")
    return table.sort_index()


def execute_scenario(
    mongo: dict[str, pd.DataFrame], tiingo: dict[str, pd.DataFrame],
    mongo_signature: dict[str, Any], tiingo_signature: dict[str, Any],
    direction: str, symbol: str | None, *, force: bool,
) -> dict[str, Any]:
    name = scenario_name(direction, symbol)
    result_path = OUTPUT_DIR / f"{name}.json"
    if result_path.exists() and not force:
        log(f"[resume] {name} ja existe; pulando")
        return json.loads(result_path.read_text(encoding="utf-8"))

    baseline_predictions = load_baseline_predictions()
    log(f"=== CENARIO {name} ===")
    started = time.perf_counter()
    result = run_engine(scenario_frames(mongo, tiingo, direction, symbol))
    elapsed = time.perf_counter() - started
    metrics = dict(result.metrics)
    predictions = predictions_for_compare(result)
    capital = float(metrics["strategy_ending_capital"])
    comparison = (
        compare_decisions(baseline_predictions, predictions)
        if baseline_predictions is not None and direction != "mongo" else None
    )
    payload = {
        "schema_version": 2,
        "script_version": VERSION,
        "certified_tag": CERTIFIED_TAG,
        "certified_commit": CERTIFIED_COMMIT,
        "certified_reference_capital": CERTIFIED_CAPITAL,
        "scenario": name,
        "direction": direction,
        "replaced_symbol": symbol,
        "history_start": START_DATE,
        "history_end": END_DATE,
        "asset_count": len(ASSETS),
        "mongo_snapshot_signature": mongo_signature["combined_sha256"],
        "tiingo_snapshot_signature": tiingo_signature["combined_sha256"],
        "strategy_ending_capital": capital,
        "delta_vs_certified_capital": capital - CERTIFIED_CAPITAL,
        "ratio_vs_certified_capital": capital / CERTIFIED_CAPITAL,
        "elapsed_seconds": elapsed,
        "decision_divergence_vs_mongo_control": comparison,
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    predictions.to_csv(OUTPUT_DIR / f"{name}_predictions.csv")
    log(f"{name} concluido | capital=US$ {capital:,.2f} | delta_43m=US$ {capital-CERTIFIED_CAPITAL:,.2f}")
    if comparison and comparison.get("available"):
        log(f"divergencia vs Mongo: {comparison['divergent_sessions']}/{comparison['common_sessions']} | primeira={comparison['first_divergence']}")
    return payload


def write_leaderboard() -> None:
    rows = []
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        if path.name == "campaign.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "scenario" not in payload:
            continue
        divergence = payload.get("decision_divergence_vs_mongo_control") or {}
        rows.append({
            "scenario": payload["scenario"],
            "direction": payload["direction"],
            "symbol": payload.get("replaced_symbol"),
            "capital": payload["strategy_ending_capital"],
            "delta_vs_43m": payload["delta_vs_certified_capital"],
            "ratio_vs_43m": payload["ratio_vs_certified_capital"],
            "divergent_sessions": divergence.get("divergent_sessions"),
            "divergence_rate": divergence.get("divergence_rate"),
            "first_divergence": divergence.get("first_divergence"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })
    if rows:
        pd.DataFrame(rows).sort_values("capital", ascending=False).to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "pair", "mongo-to-tiingo", "tiingo-to-mongo", "both"), default="pair")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(VERSION)
    log(f"Base certificada: {CERTIFIED_TAG} @ {CERTIFIED_COMMIT[:12]}")
    log("Entrada: snapshots locais. Sem internet e sem Mongo durante o backtest.")

    mongo, mongo_signature = load_snapshot_frames(MONGO_DIR, "Mongo")
    tiingo, tiingo_signature = load_tiingo_frames()
    campaign = {
        "script_version": VERSION,
        "certified_tag": CERTIFIED_TAG,
        "certified_commit": CERTIFIED_COMMIT,
        "certified_reference_capital": CERTIFIED_CAPITAL,
        "mongo_snapshot_signature": mongo_signature["combined_sha256"],
        "tiingo_snapshot_signature": tiingo_signature["combined_sha256"],
        "assets": list(ASSETS),
        "purpose": "diagnostic_attribution_only",
        "selection_uses_43m_reference": False,
    }
    (OUTPUT_DIR / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    execute_scenario(mongo, tiingo, mongo_signature, tiingo_signature, "mongo", None, force=args.force)
    if args.mode == "baseline":
        write_leaderboard(); return 0
    if args.mode == "pair":
        execute_scenario(mongo, tiingo, mongo_signature, tiingo_signature, "tiingo", None, force=args.force)
        write_leaderboard(); return 0

    symbols = [str(args.symbol).upper()] if args.symbol else list(ASSETS)
    invalid = [s for s in symbols if s not in ASSETS]
    if invalid:
        raise RuntimeError("Ativo fora do universo certificado: " + ", ".join(invalid))
    if args.limit > 0:
        symbols = symbols[:args.limit]
    if args.mode in ("mongo-to-tiingo", "both"):
        for symbol in symbols:
            execute_scenario(mongo, tiingo, mongo_signature, tiingo_signature, "mongo-to-tiingo", symbol, force=args.force)
    if args.mode in ("tiingo-to-mongo", "both"):
        for symbol in symbols:
            execute_scenario(mongo, tiingo, mongo_signature, tiingo_signature, "tiingo-to-mongo", symbol, force=args.force)
    write_leaderboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
