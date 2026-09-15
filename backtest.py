"""Backtest acadêmico reproduzível para o TCC MBA USP.

A entrada externa do experimento é o histórico diário OHLCV baixado do Yahoo
Finance via yfinance. Features, targets, folds walk-forward, treinamento
LightGBM, política de rotação, operações, curva de capital e métricas são
reconstruídos a cada execução.

O arquivo é organizado em células Spyder ``# %%``. F5 executa o script completo;
Ctrl+Enter executa somente a célula atual e mantém as variáveis disponíveis no
Variable Explorer.
"""

# %% 0 - Imports e configuração
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import ASSETS, CONFIG, END_DATE, START_DATE
from tcc_engine.execution import apply_slippage, calculate_reference_fees

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

YAHOO_INTERVAL = "1d"
YAHOO_AUTO_ADJUST = True


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


# %% 1 - Início da execução
started = time.perf_counter()

print("TCC MBA USP — backtest reconstruído passo a passo")
print("Fonte de mercado: Yahoo Finance via yfinance")
print(f"Período solicitado: {START_DATE} -> {END_DATE} | ativos={len(ASSETS)}")

# yfinance trata `end` como exclusivo. Somamos um dia para incluir END_DATE.
yahoo_end = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# %% 2 - Download e validação das séries temporais OHLCV
frames: dict[str, pd.DataFrame] = {}

for position, symbol in enumerate(ASSETS, start=1):
    print(f"[{position:02d}/{len(ASSETS)}] Yahoo Finance: {symbol}")

    frame = yf.download(
        symbol,
        start=START_DATE,
        end=yahoo_end,
        interval=YAHOO_INTERVAL,
        auto_adjust=YAHOO_AUTO_ADJUST,
        actions=False,
        progress=False,
        threads=False,
        repair=False,
        keepna=False,
        multi_level_index=False,
    )

    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance não retornou histórico para {symbol}.")

    frame = frame.rename(columns={str(column): str(column).lower() for column in frame.columns})
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(
            f"{symbol}: colunas ausentes no Yahoo Finance: {', '.join(missing_columns)}"
        )

    frame = frame[required_columns].copy()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "timestamp"
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=required_columns)
    valid = (
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] >= 0)
    )
    frame = frame.loc[valid].copy()

    if frame.empty:
        raise RuntimeError(f"{symbol}: histórico vazio após validação OHLCV.")

    frames[symbol] = frame
    print(
        f"    {len(frame)} sessões | "
        f"{frame.index.min().date()} -> {frame.index.max().date()}"
    )

# Snapshot exato das séries usadas pelo motor.
market_parts: list[pd.DataFrame] = []
for symbol in ASSETS:
    part = frames[symbol].reset_index().copy()
    part.insert(0, "symbol", symbol)
    market_parts.append(part)

market_data = pd.concat(market_parts, ignore_index=True)
market_data = market_data[
    ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
market_csv = market_data.to_csv(index=False)
(OUTPUT_DIR / "market_data.csv").write_text(market_csv, encoding="utf-8")
market_data_sha256 = hashlib.sha256(market_csv.encode("utf-8")).hexdigest()

print(f"Snapshot: {len(market_data):,} linhas")
print(f"SHA-256: {market_data_sha256}")


# %% 3 - Preparação metodológica
print("Construindo features técnicas a partir do OHLCV")
print(
    "Targets multi-horizonte: "
    + ", ".join(str(value) for value in CONFIG.rotation_target_horizons)
)
print("Validação walk-forward com purge")

experiment_context = {
    "assets": list(ASSETS),
    "asset_count": len(ASSETS),
    "history_start": START_DATE,
    "history_end": END_DATE,
    "market_data": {
        "provider": "yahoo_finance",
        "library": "yfinance",
        "interval": YAHOO_INTERVAL,
        "auto_adjust": YAHOO_AUTO_ADJUST,
        "snapshot_file": "market_data.csv",
        "snapshot_rows": len(market_data),
        "snapshot_sha256": market_data_sha256,
    },
    "model_family": CONFIG.research_model_family,
    "target_horizons": list(CONFIG.rotation_target_horizons),
    "minimum_training_rows": CONFIG.rotation_minimum_training_rows,
    "calibration_days": CONFIG.rotation_walk_forward_calibration_days,
    "test_days": CONFIG.rotation_walk_forward_test_days,
    "minimum_test_days": CONFIG.rotation_walk_forward_min_test_days,
    "purge_days": CONFIG.rotation_purge_days,
}


# %% 4 - Treinamento LightGBM, previsões OOS e política de rotação
print("Treinando LightGBM do zero em cada fold e ativo")

results = run_rotation_models(
    frames,
    CONFIG,
    calculate_reference_fees,
    apply_slippage,
)

if not results:
    raise RuntimeError("O motor não retornou resultado.")
if len(results) != 1:
    raise RuntimeError(f"Esperava uma execução; foram retornadas {len(results)}.")

result = results[0]


# %% 5 - Objetos de resultado para inspeção no Spyder
predictions = result.predictions.copy()
trades = result.trades.copy()
folds = list(result.metrics.get("walk_forward_folds") or [])
metrics = dict(result.metrics)

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

elapsed = time.perf_counter() - started

payload = {
    "experiment": "mba_usp_yahoo_backtest",
    "input_source": "yahoo_finance_ohlcv",
    **experiment_context,
    "walk_forward": {
        "minimum_training_rows": CONFIG.rotation_minimum_training_rows,
        "calibration_days": CONFIG.rotation_walk_forward_calibration_days,
        "test_days": CONFIG.rotation_walk_forward_test_days,
        "minimum_test_days": CONFIG.rotation_walk_forward_min_test_days,
        "purge_days": CONFIG.rotation_purge_days,
    },
    "elapsed_seconds": float(elapsed),
    "metrics": metrics,
}


# %% 6 - Gravação dos artefatos

equity_csv = equity.copy()
if equity_csv.index.name is not None or not isinstance(equity_csv.index, pd.RangeIndex):
    equity_csv = equity_csv.reset_index()

trades_csv = trades.copy()
if trades_csv.index.name is not None or not isinstance(trades_csv.index, pd.RangeIndex):
    trades_csv = trades_csv.reset_index()

for dataframe in (equity_csv, trades_csv):
    for column in dataframe.columns:
        has_nested_values = dataframe[column].map(
            lambda value: isinstance(value, (dict, list, tuple))
        ).any()
        if has_nested_values:
            dataframe[column] = dataframe[column].map(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                    default=json_default,
                )
                if isinstance(value, (dict, list, tuple))
                else value
            )

equity_csv.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
trades_csv.to_csv(OUTPUT_DIR / "trades.csv", index=False)
pd.DataFrame(folds).to_csv(OUTPUT_DIR / "folds.csv", index=False)

(OUTPUT_DIR / "backtest_result.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
    encoding="utf-8",
)
(OUTPUT_DIR / "summary.txt").write_text(
    str(result.summary).rstrip() + "\n",
    encoding="utf-8",
)


# %% 7 - Métricas finais
print(f"Capital inicial : US$ {CONFIG.initial_capital:,.2f}")
print(f"Capital final   : US$ {float(metrics['strategy_ending_capital']):,.2f}")
print(f"Retorno         : {float(metrics['strategy_return']):.2%}")
print(f"CAGR            : {float(metrics['strategy_cagr']):.2%}")
print(f"Sharpe          : {float(metrics['strategy_sharpe']):.3f}")
print(f"Max Drawdown    : {float(metrics['strategy_maximum_drawdown']):.2%}")
print(f"Rotações        : {int(metrics.get('capital_rotations') or 0)}")
print(f"CASH days       : {int(metrics.get('cash_days') or 0)}")
print(f"Tempo total     : {elapsed:.2f}s")
print(f"Resultados      : {OUTPUT_DIR}")
