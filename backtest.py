"""Backtest academico reproduzivel para o TCC MBA USP.

A unica entrada externa do experimento e o historico diario OHLCV armazenado
na collection local ``alpaca_market_bars``. Features, targets, folds
walk-forward, treinamento LightGBM, politica de rotacao, operacoes, curva de
capital e metricas sao reconstruidos a cada execucao.

Nao sao lidos Strategy documents, modelos treinados, previsoes persistidas,
resultados anteriores ou configuracoes do Market Cycle Trader.

Spyder: abra este arquivo e pressione F5.
Terminal: python backtest.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import ASSETS, CONFIG, END_DATE, START_DATE
from tcc_engine.execution import apply_slippage, calculate_reference_fees

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DATABASE = "extrema_backtest"
MARKET_COLLECTION = "alpaca_market_bars"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
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
    return str(value)


def mongo_settings() -> tuple[str, str]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    uri = str(os.getenv("TCC_MONGO_URI") or DEFAULT_MONGO_URI).strip()
    database = str(os.getenv("TCC_MONGO_DATABASE") or DEFAULT_MONGO_DATABASE).strip()

    lowered = uri.lower()
    if not any(host in lowered for host in ("localhost", "127.0.0.1", "::1")):
        raise RuntimeError(
            "O TCC aceita somente MongoDB local. Defina TCC_MONGO_URI para localhost."
        )
    return uri, database


def load_raw_ohlcv() -> dict[str, pd.DataFrame]:
    """Carrega exclusivamente candles OHLCV diarios do MongoDB local."""
    mongo_uri, database_name = mongo_settings()
    start = pd.Timestamp(START_DATE, tz="UTC").to_pydatetime()
    end = (pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()

    log(f"[1/8] Carregando OHLCV bruto de {len(ASSETS)} ativos")
    log(f"[1/8] Fonte: {database_name}.{MARKET_COLLECTION}")

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
        raise RuntimeError("Nenhum candle OHLCV foi encontrado no MongoDB local.")

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
        raise RuntimeError("Historico ausente para: " + ", ".join(missing))

    for position, symbol in enumerate(ASSETS, start=1):
        if position == 1 or position % 5 == 0 or position == len(ASSETS):
            frame = frames[symbol]
            log(
                f"[1/8] {position:02d}/{len(ASSETS)} {symbol} | "
                f"{len(frame)} candles | {frame.index.min().date()} -> {frame.index.max().date()}"
            )

    return frames


def progress_callback(percent: float, stage: str, completed_runs: int) -> None:
    log(f"[5/8] {percent:5.1f}% | {stage}")


def technical_log_callback(message: str) -> None:
    log(f"[motor] {message}")


def frame_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if output.index.name is not None or not isinstance(output.index, pd.RangeIndex):
        output = output.reset_index()

    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, default=json_default)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return output


def save_results(result: Any, elapsed_seconds: float) -> dict[str, Any]:
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

    frame_for_csv(equity).to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    frame_for_csv(trades).to_csv(OUTPUT_DIR / "trades.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTPUT_DIR / "folds.csv", index=False)

    payload = {
        "experiment": "mba_usp_reproducible_backtest",
        "input_source": "local_mongodb_alpaca_market_bars_ohlcv_only",
        "history_start": START_DATE,
        "history_end": END_DATE,
        "assets": list(ASSETS),
        "asset_count": len(ASSETS),
        "model_family": CONFIG.research_model_family,
        "target_horizons": list(CONFIG.rotation_target_horizons),
        "walk_forward": {
            "minimum_training_rows": CONFIG.rotation_minimum_training_rows,
            "calibration_days": CONFIG.rotation_walk_forward_calibration_days,
            "test_days": CONFIG.rotation_walk_forward_test_days,
            "minimum_test_days": CONFIG.rotation_walk_forward_min_test_days,
            "purge_days": CONFIG.rotation_purge_days,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "metrics": dict(result.metrics),
    }

    (OUTPUT_DIR / "backtest_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(
        str(result.summary).rstrip() + "\n",
        encoding="utf-8",
    )
    return payload


def run_backtest() -> dict[str, Any]:
    started = time.perf_counter()

    log("TCC MBA USP — backtest reconstruido passo a passo")
    log("Entrada externa permitida: somente candles OHLCV do MongoDB local")
    log(f"Periodo: {START_DATE} -> {END_DATE} | ativos={len(ASSETS)}")

    frames = load_raw_ohlcv()

    log("[2/8] Construindo features tecnicas a partir do OHLCV")
    log(
        "[3/8] Construindo targets multi-horizonte: "
        + ", ".join(str(value) for value in CONFIG.rotation_target_horizons)
    )
    log("[4/8] Criando folds temporais walk-forward com purge")
    log("[5/8] Treinando LightGBM do zero em cada fold e ativo")

    results = run_rotation_models(
        frames,
        CONFIG,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=progress_callback,
        technical_log_callback=technical_log_callback,
    )
    if not results:
        raise RuntimeError("O motor nao retornou resultado.")
    if len(results) != 1:
        raise RuntimeError(f"Esperava uma execucao; foram retornadas {len(results)}.")

    result = results[0]

    log("[6/8] Aplicando a politica de rotacao somente nas sessoes fora da amostra")
    log("[7/8] Reconstruindo trades, custos e curva de capital")

    elapsed = time.perf_counter() - started
    payload = save_results(result, elapsed)

    metrics = result.metrics
    log("[8/8] Calculando e salvando metricas finais")
    log(f"Capital inicial : US$ {CONFIG.initial_capital:,.2f}")
    log(f"Capital final   : US$ {float(metrics['strategy_ending_capital']):,.2f}")
    log(f"Retorno         : {float(metrics['strategy_return']):.2%}")
    log(f"CAGR            : {float(metrics['strategy_cagr']):.2%}")
    log(f"Sharpe          : {float(metrics['strategy_sharpe']):.3f}")
    log(f"Max Drawdown    : {float(metrics['strategy_maximum_drawdown']):.2%}")
    log(f"Rotacoes        : {int(metrics.get('capital_rotations') or 0)}")
    log(f"CASH days       : {int(metrics.get('cash_days') or 0)}")
    log(f"Tempo total     : {elapsed:.2f}s")
    log(f"Resultados      : {OUTPUT_DIR}")

    return payload


def main() -> int:
    run_backtest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
