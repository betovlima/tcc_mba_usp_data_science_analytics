"""Standalone historical rotation backtest for the USP MBA TCC.

Runtime inputs are intentionally restricted to raw daily OHLCV candles stored
in the local MongoDB collection ``alpaca_market_bars``.  The strategy
configuration is declared in ``tcc_engine.config`` and every feature, target,
walk-forward fold, LightGBM model, policy decision, trade and capital metric is
computed by the vendored local ``tcc_engine``.

No Market Cycle Trader API, Strategy document, persisted model, prediction or
previous backtest result is read during execution.

Open this file in Spyder and press F5, or run:

    python backtest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import (
    ASSETS,
    CONFIG,
    END_DATE,
    START_DATE,
    HISTORICAL_ENDING_CAPITAL,
    HISTORICAL_EXECUTION_REQUEST_SHA256,
    HISTORICAL_MARKET_OHLCV_SHA256,
    HISTORICAL_MODEL_SETTINGS_SHA256,
    HISTORICAL_SOURCE_COMMIT,
    HISTORICAL_STRATEGY_CONFIGURATION_SHA256,
    HISTORICAL_STRATEGY_ID,
    HISTORICAL_STRATEGY_REVISION_AT_CERTIFICATION,
    HISTORICAL_STRATEGY_SEQUENCE,
)
from tcc_engine.execution import apply_slippage, calculate_reference_fees


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DATABASE = "extrema_backtest"
MARKET_COLLECTION = "alpaca_market_bars"
ENDING_CAPITAL_RELATIVE_TOLERANCE = 1e-6


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return str(value)


def _local_mongo_settings() -> tuple[str, str]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    uri = str(os.getenv("TCC_MONGO_URI") or DEFAULT_MONGO_URI).strip()
    database = str(os.getenv("TCC_MONGO_DATABASE") or DEFAULT_MONGO_DATABASE).strip()
    lowered = uri.lower()
    if not any(host in lowered for host in ("localhost", "127.0.0.1", "::1")):
        raise RuntimeError(
            "O experimento do TCC aceita somente MongoDB local. "
            "Defina TCC_MONGO_URI apontando para localhost."
        )
    return uri, database


def load_market_data() -> dict[str, pd.DataFrame]:
    """Read only the frozen 37-asset OHLCV snapshot from local MongoDB."""
    mongo_uri, database_name = _local_mongo_settings()
    start = pd.Timestamp(START_DATE, tz="UTC").to_pydatetime()
    end = (pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()

    log(f"[1/7] MongoDB local: {database_name}.{MARKET_COLLECTION}")
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        collection = client[database_name][MARKET_COLLECTION]
        rows = list(
            collection.find(
                {
                    "symbol": {"$in": list(ASSETS)},
                    "interval": CONFIG.timeframe,
                    "feed": CONFIG.alpaca_historical_feed,
                    "adjustment": CONFIG.alpaca_adjustment,
                    "timestamp": {"$gte": start, "$lt": end},
                },
                {
                    "_id": 0,
                    "symbol": 1,
                    "timestamp": 1,
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "volume": 1,
                },
            ).sort([("symbol", 1), ("timestamp", 1)])
        )
    finally:
        client.close()

    if not rows:
        raise RuntimeError(
            "Nenhum candle foi encontrado em alpaca_market_bars para o período congelado."
        )

    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
        valid = (
            (frame["open"] > 0)
            & (frame["high"] > 0)
            & (frame["low"] > 0)
            & (frame["close"] > 0)
            & (frame["volume"] >= 0)
        )
        frames[str(symbol)] = frame.loc[valid].copy()

    missing = [symbol for symbol in ASSETS if symbol not in frames or frames[symbol].empty]
    if missing:
        raise RuntimeError("Sem histórico OHLCV local para: " + ", ".join(missing))

    for position, symbol in enumerate(ASSETS, start=1):
        if position == 1 or position % 5 == 0 or position == len(ASSETS):
            frame = frames[symbol]
            log(
                f"[1/7] Dados {position:02d}/{len(ASSETS)} | {symbol} | "
                f"rows={len(frame)} | {frame.index.min().date()} -> {frame.index.max().date()}"
            )
    return frames


def market_frames_sha256(frames: dict[str, pd.DataFrame]) -> str:
    """Mirror the OHLCV fingerprint used by the certified historical replay."""
    digest = hashlib.sha256()
    for symbol in sorted(frames):
        frame = frames[symbol].sort_index()
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\n")
        for timestamp, row in frame.iterrows():
            stamp = pd.Timestamp(timestamp)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            else:
                stamp = stamp.tz_convert("UTC")
            values = [
                stamp.isoformat(),
                *[
                    "" if pd.isna(row.get(column)) else format(float(row.get(column)), ".17g")
                    for column in ("open", "high", "low", "close", "volume")
                ],
            ]
            digest.update(("|".join(values) + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_runtime_fingerprints(frames: dict[str, pd.DataFrame]) -> tuple[str, str]:
    """Fail before training if request or raw market snapshot differs from certification."""
    request_hash = CONFIG.execution_request_sha256()
    market_hash = market_frames_sha256(frames)

    log(f"[2/7] Model snapshot SHA-256 : {CONFIG.model_settings_sha256()}")
    log(f"[2/7] Execution request SHA-256: {request_hash}")
    log(f"[2/7] Market OHLCV SHA-256     : {market_hash}")

    if request_hash != HISTORICAL_EXECUTION_REQUEST_SHA256:
        raise RuntimeError(
            "O request standalone não corresponde ao request certificado: "
            f"{request_hash} != {HISTORICAL_EXECUTION_REQUEST_SHA256}."
        )
    if market_hash != HISTORICAL_MARKET_OHLCV_SHA256:
        raise RuntimeError(
            "O snapshot OHLCV local não corresponde ao snapshot certificado: "
            f"{market_hash} != {HISTORICAL_MARKET_OHLCV_SHA256}."
        )
    log("[2/7] Fingerprints certificados: OK")
    return request_hash, market_hash


def progress_callback(percent: float, stage: str, completed_runs: int) -> None:
    log(f"[4/7] {percent:5.1f}% | {stage}")


def technical_log_callback(message: str) -> None:
    log(f"[motor] {message}")


def _frame_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if output.index.name is not None or not isinstance(output.index, pd.RangeIndex):
        output = output.reset_index()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, default=_json_default)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return output


def write_artifacts(
    result: Any,
    elapsed_seconds: float,
    *,
    request_hash: str,
    market_hash: str,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = result.predictions.copy()
    trades = result.trades.copy()
    folds = list(result.metrics.get("walk_forward_folds") or [])

    equity_columns = [
        column
        for column in (
            "strategy_equity",
            "buy_hold_equity",
            "selected_asset",
            "selected_score",
            "decision_score",
            "trade_action",
            "trade_reason",
            "walk_forward_fold",
            "cash_weight",
            "market_exposure_weight",
            "assets_held",
        )
        if column in predictions.columns
    ]
    equity = predictions[equity_columns].copy() if equity_columns else predictions.copy()

    _frame_for_csv(equity).to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    _frame_for_csv(trades).to_csv(OUTPUT_DIR / "trades.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTPUT_DIR / "folds.csv", index=False)

    metrics = dict(result.metrics)
    ending = float(metrics["strategy_ending_capital"])
    relative_error = abs(ending / HISTORICAL_ENDING_CAPITAL - 1.0)
    certified_capital_match = bool(relative_error <= ENDING_CAPITAL_RELATIVE_TOLERANCE)

    payload = {
        "experiment": "standalone_historical_rotation_engine",
        "runtime_source": "tcc_engine",
        "market_data_source": "local_mongodb_alpaca_market_bars",
        "history_start": START_DATE,
        "snapshot_end": END_DATE,
        "asset_count": len(ASSETS),
        "assets": list(ASSETS),
        "mct_runtime_dependency": False,
        "strategy_document_dependency": False,
        "persisted_model_dependency": False,
        "persisted_prediction_dependency": False,
        "runtime_fingerprint": {
            "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
            "historical_strategy_id": HISTORICAL_STRATEGY_ID,
            "historical_strategy_sequence": HISTORICAL_STRATEGY_SEQUENCE,
            "historical_strategy_revision_at_certification": HISTORICAL_STRATEGY_REVISION_AT_CERTIFICATION,
            "historical_strategy_configuration_sha256": HISTORICAL_STRATEGY_CONFIGURATION_SHA256,
            "model_settings_sha256": CONFIG.model_settings_sha256(),
            "expected_model_settings_sha256": HISTORICAL_MODEL_SETTINGS_SHA256,
            "execution_request_sha256": request_hash,
            "expected_execution_request_sha256": HISTORICAL_EXECUTION_REQUEST_SHA256,
            "market_ohlcv_sha256": market_hash,
            "expected_market_ohlcv_sha256": HISTORICAL_MARKET_OHLCV_SHA256,
        },
        "historical_reference": {
            "ending_capital": HISTORICAL_ENDING_CAPITAL,
            "ending_capital_relative_tolerance": ENDING_CAPITAL_RELATIVE_TOLERANCE,
            "ending_capital_relative_error": relative_error,
            "ending_capital_match": certified_capital_match,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "metrics": metrics,
    }
    (OUTPUT_DIR / "backtest_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(str(result.summary).rstrip() + "\n", encoding="utf-8")
    return payload


def run_backtest() -> dict[str, Any]:
    started = time.perf_counter()
    log("USP MBA Data Science & Analytics — standalone historical engine")
    log("Runtime: somente código do TCC + OHLCV bruto do MongoDB local")
    log(
        f"Config: {len(ASSETS)} ativos | {START_DATE} -> {END_DATE} | "
        f"modelo={CONFIG.research_model_family}"
    )

    frames = load_market_data()
    request_hash, market_hash = validate_runtime_fingerprints(frames)
    log("[3/7] Iniciando features, targets e folds pelo tcc_engine")

    results = run_rotation_models(
        frames,
        CONFIG,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=progress_callback,
        technical_log_callback=technical_log_callback,
    )
    if not results:
        raise RuntimeError("O motor histórico não retornou nenhuma execução.")
    if len(results) != 1:
        raise RuntimeError(f"Esperava uma execução histórica; o motor retornou {len(results)}.")

    result = results[0]
    elapsed = time.perf_counter() - started
    log("[5/7] Simulação concluída; gravando artefatos")
    payload = write_artifacts(
        result,
        elapsed,
        request_hash=request_hash,
        market_hash=market_hash,
    )

    metrics = result.metrics
    reference = payload["historical_reference"]
    log("[6/7] Resultado")
    log(f"Capital inicial : US$ {CONFIG.initial_capital:,.2f}")
    log(f"Capital final   : US$ {float(metrics['strategy_ending_capital']):,.2f}")
    log(f"Referência      : US$ {HISTORICAL_ENDING_CAPITAL:,.2f}")
    log(f"Erro relativo   : {float(reference['ending_capital_relative_error']):.8%}")
    log(f"Capital certific.: {'OK' if reference['ending_capital_match'] else 'DIVERGIU'}")
    log(f"Retorno         : {float(metrics['strategy_return']):.2%}")
    log(f"CAGR            : {float(metrics['strategy_cagr']):.2%}")
    log(f"Sharpe          : {float(metrics['strategy_sharpe']):.3f}")
    log(f"Max Drawdown    : {float(metrics['strategy_maximum_drawdown']):.2%}")
    log(f"Rotações        : {int(metrics.get('capital_rotations') or 0)}")
    log(f"CASH days       : {int(metrics.get('cash_days') or 0)}")
    log(f"Tempo           : {elapsed:.2f}s")
    log(f"[7/7] Artefatos: {OUTPUT_DIR}")

    if platform.system().lower() == "windows":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass
    return payload


if __name__ == "__main__":
    run_backtest()
