"""Backtest de controle sobre a referencia historica Mongo/Alpaca dos 43M.

Versao: mongo43-tiingo-audit-v1.0.0

Usa exatamente o motor e a configuracao atuais do TCC, que preservam o mesmo
capital_rotation.py da tag certified-43m-standalone. A unica entrada e o OHLCV
congelado em dados/referencia_mongo_43m.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import ASSETS, CONFIG, START_DATE, END_DATE
from tcc_engine.execution import apply_slippage, calculate_reference_fees

VERSION = "mongo43-tiingo-audit-v1.0.0"
ROOT = Path(__file__).resolve().parent
DIR_SERIES = ROOT / "dados" / "referencia_mongo_43m"
DIR_OUT = ROOT / "output" / "mongo_43m_control"
OHLCV = ["open", "high", "low", "close", "volume"]
CERTIFIED_CAPITAL = 43_759_854.82


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


def write_frame(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            out[col] = out[col].map(
                lambda v: json.dumps(v, ensure_ascii=False, default=json_default)
                if isinstance(v, (dict, list, tuple)) else v
            )
    out.to_csv(path, index=False)


if not DIR_SERIES.exists():
    raise RuntimeError(f"Referencia Mongo 43M ausente: {DIR_SERIES}")

started = time.perf_counter()
series: dict[str, pd.DataFrame] = {}
log(f"Controle historico 43M | versao={VERSION}")
log(f"Periodo: {START_DATE} -> {END_DATE} | ativos={len(ASSETS)}")

for pos, asset in enumerate(ASSETS, start=1):
    path = DIR_SERIES / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"Snapshot 43M incompleto: {path}")
    df = pd.read_csv(path)
    required = ["timestamp", *OHLCV]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{asset}: colunas ausentes: {', '.join(missing)}")
    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last")
    df = df.loc[
        (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)
        & (df["close"] > 0) & (df["volume"] >= 0)
    ].copy()
    if df.empty:
        raise RuntimeError(f"{asset}: serie vazia apos validacao")
    series[asset] = df.set_index("timestamp")[OHLCV].copy()
    log(f"[dados] {pos:02d}/{len(ASSETS)} {asset} | candles={len(df)}")


def progress(percent: float, stage: str, completed: int) -> None:
    log(f"[motor] {percent:5.1f}% | {stage}")

results = run_rotation_models(
    series,
    CONFIG,
    calculate_reference_fees,
    apply_slippage,
    progress_callback=progress,
    technical_log_callback=lambda msg: log(f"[motor] {msg}"),
)
if len(results) != 1:
    raise RuntimeError(f"Esperava uma execucao; recebidas={len(results)}")

result = results[0]
metrics = dict(result.metrics)
predictions = result.predictions.copy()
trades = result.trades.copy()
folds = list(metrics.get("walk_forward_folds") or [])
DIR_OUT.mkdir(parents=True, exist_ok=True)

curve_cols = [c for c in (
    "strategy_equity", "buy_hold_equity", "selected_asset", "selected_score",
    "decision_score", "trade_action", "trade_reason", "walk_forward_fold",
    "cash_weight", "market_exposure_weight", "assets_held",
) if c in predictions.columns]
curve = predictions[curve_cols].copy() if curve_cols else predictions.copy()
write_frame(curve, DIR_OUT / "equity_curve.csv")
write_frame(trades, DIR_OUT / "trades.csv")
pd.DataFrame(folds).to_csv(DIR_OUT / "folds.csv", index=False)

capital = float(metrics["strategy_ending_capital"])
elapsed = time.perf_counter() - started
payload = {
    "schema_version": 1,
    "script_version": VERSION,
    "experiment": "certified_mongo_alpaca_43m_control",
    "historical_tag": "certified-43m-standalone",
    "input_source": "dados/referencia_mongo_43m",
    "certified_ending_capital": CERTIFIED_CAPITAL,
    "reproduced_ending_capital": capital,
    "difference_vs_certified_usd": capital - CERTIFIED_CAPITAL,
    "ratio_vs_certified": capital / CERTIFIED_CAPITAL,
    "elapsed_seconds": elapsed,
    "metrics": metrics,
}
(DIR_OUT / "backtest_result.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
    encoding="utf-8",
)
(DIR_OUT / "summary.txt").write_text(str(result.summary).rstrip() + "\n", encoding="utf-8")

log(f"Capital certificado : US$ {CERTIFIED_CAPITAL:,.2f}")
log(f"Capital reproduzido : US$ {capital:,.2f}")
log(f"Diferenca           : US$ {capital - CERTIFIED_CAPITAL:,.2f}")
log(f"Resultados          : {DIR_OUT}")
