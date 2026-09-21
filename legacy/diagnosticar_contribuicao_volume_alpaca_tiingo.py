"""Experimento pareado para isolar a contribuicao do volume Alpaca x Tiingo.

Versao: alpaca-tiingo-volume-ablation-v1.0.0

Nao altera o modelo nem a politica. Constroi dois contrafactuais:

1) precos Tiingo total-causal + volume Alpaca
2) precos Alpaca adjustment=all + volume Tiingo total-causal

Os controles completos permanecem:
- output/alpaca_atual_22m_control
- output/tiingo_total_causal_v1

Objetivo: medir se a diferenca de volume entre fornecedores explica materialmente
as divergencias de features, calibracao da margem de troca e capital composto.
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
from tcc_engine.config import ASSETS, CONFIG
from tcc_engine.execution import apply_slippage, calculate_reference_fees

VERSION = "alpaca-tiingo-volume-ablation-v1.0.0"
ROOT = Path(__file__).resolve().parent
DIR_ALPACA = ROOT / "dados" / "referencia_alpaca_atual"
DIR_TIINGO = ROOT / "dados" / "series_historicas"
DIR_EVENTS = ROOT / "dados" / "eventos_corporativos"
DIR_SPLITS = ROOT / "dados" / "desdobramentos"
DIR_OUT = ROOT / "output" / "diagnostico_contribuicao_volume"
BASELINE_ALPACA = ROOT / "output" / "alpaca_atual_22m_control" / "backtest_result.json"
BASELINE_TIINGO = ROOT / "output" / "tiingo_total_causal_v1" / "backtest_result.json"
OHLCV = ["open", "high", "low", "close", "volume"]


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


def read_ohlcv(directory: Path, asset: str) -> pd.DataFrame:
    path = directory / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"Arquivo ausente: {path}")
    df = pd.read_csv(path)
    required = ["timestamp", *OHLCV]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{asset}: colunas ausentes em {path.name}: {', '.join(missing)}")
    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp")
    df["session"] = df["timestamp"].dt.normalize()
    df = df.drop_duplicates("session", keep="last")
    df = df.loc[
        (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)
        & (df["close"] > 0) & (df["volume"] >= 0)
    ].copy()
    return df.set_index("session")[OHLCV].sort_index()


def apply_tiingo_splits(asset: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    path = DIR_SPLITS / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"{asset}: desdobramentos ausentes: {path}")
    events = pd.read_csv(path)
    session_factor = pd.Series(1.0, index=raw.index, dtype=float)
    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        events["fator_split"] = pd.to_numeric(events["fator_split"], errors="coerce")
        if "status" in events.columns:
            events = events.loc[events["status"].astype(str).str.lower().str.strip() == "a"]
        events = events.dropna(subset=["timestamp", "fator_split"])
        events = events.loc[events["fator_split"] > 0]
        for event in events.itertuples(index=False):
            session = pd.Timestamp(event.timestamp).normalize()
            if session in session_factor.index:
                session_factor.loc[session] *= float(event.fator_split)
    cumulative = session_factor.cumprod()
    out = raw.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * cumulative
    out["volume"] = out["volume"] / cumulative
    return out, cumulative


def apply_tiingo_dividends(asset: str, split_series: pd.DataFrame, split_factor: pd.Series) -> pd.DataFrame:
    path = DIR_EVENTS / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"{asset}: eventos corporativos ausentes: {path}")
    events = pd.read_csv(path)
    session_factor = pd.Series(1.0, index=split_series.index, dtype=float)
    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        events["dividendo"] = pd.to_numeric(events["dividendo"], errors="coerce").fillna(0.0)
        events = events.dropna(subset=["timestamp"])
        events = events.loc[events["dividendo"] != 0.0].sort_values("timestamp")
        for event in events.itertuples(index=False):
            session = pd.Timestamp(event.timestamp).normalize()
            if session not in split_series.index:
                continue
            pos = int(split_series.index.get_loc(session))
            if pos == 0:
                continue
            previous_close = float(split_series.iloc[pos - 1]["close"])
            dividend = float(event.dividendo) * float(split_factor.loc[session])
            denominator = previous_close - dividend
            if not np.isfinite(denominator) or denominator <= 0:
                raise RuntimeError(f"{asset}: dividendo invalido em {session.date()}")
            session_factor.loc[session] *= previous_close / denominator
    cumulative = session_factor.cumprod()
    out = split_series.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * cumulative
    return out


def load_tiingo_total_causal(asset: str) -> pd.DataFrame:
    raw = read_ohlcv(DIR_TIINGO, asset)
    split, split_factor = apply_tiingo_splits(asset, raw)
    return apply_tiingo_dividends(asset, split, split_factor)


def build_hybrids() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, Any]]]:
    tiingo_prices_alpaca_volume: dict[str, pd.DataFrame] = {}
    alpaca_prices_tiingo_volume: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []

    for pos, asset in enumerate(ASSETS, start=1):
        alpaca = read_ohlcv(DIR_ALPACA, asset)
        tiingo = load_tiingo_total_causal(asset)
        common = alpaca.index.intersection(tiingo.index).sort_values()
        if len(common) < 700:
            raise RuntimeError(f"{asset}: historico comum insuficiente: {len(common)}")

        a = alpaca.loc[common].copy()
        t = tiingo.loc[common].copy()

        ta = t.copy()
        ta["volume"] = a["volume"]
        at = a.copy()
        at["volume"] = t["volume"]

        tiingo_prices_alpaca_volume[asset] = ta
        alpaca_prices_tiingo_volume[asset] = at

        vol_ratio = (t["volume"] / a["volume"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        diagnostics.append({
            "asset": asset,
            "sessions": int(len(common)),
            "mean_abs_volume_pct_diff": float(((t["volume"] - a["volume"]).abs() / a["volume"].replace(0, np.nan)).dropna().mean()),
            "median_tiingo_alpaca_volume_ratio": float(vol_ratio.dropna().median()),
            "corr_volume": float(a["volume"].corr(t["volume"])),
        })
        log(f"[dados] {pos:02d}/{len(ASSETS)} {asset} | sessoes={len(common)}")

    return tiingo_prices_alpaca_volume, alpaca_prices_tiingo_volume, diagnostics


def write_frame(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            out[c] = out[c].map(
                lambda v: json.dumps(v, ensure_ascii=False, default=json_default)
                if isinstance(v, (dict, list, tuple)) else v
            )
    out.to_csv(path, index=False)


def run_scenario(name: str, series: dict[str, pd.DataFrame]) -> dict[str, Any]:
    started = time.perf_counter()
    log(f"[{name}] iniciando backtest")

    def progress(percent: float, stage: str, completed: int) -> None:
        log(f"[{name}] {percent:5.1f}% | {stage}")

    results = run_rotation_models(
        series,
        CONFIG,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=progress,
        technical_log_callback=lambda msg: log(f"[{name}][motor] {msg}"),
    )
    if len(results) != 1:
        raise RuntimeError(f"{name}: esperava uma execucao; recebidas={len(results)}")
    result = results[0]
    metrics = dict(result.metrics)
    predictions = result.predictions.copy()
    trades = result.trades.copy()
    scenario_dir = DIR_OUT / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    curve_cols = [c for c in (
        "strategy_equity", "buy_hold_equity", "selected_asset", "selected_score",
        "decision_score", "trade_action", "trade_reason", "walk_forward_fold",
    ) if c in predictions.columns]
    curve = predictions[curve_cols].copy() if curve_cols else predictions.copy()
    write_frame(curve, scenario_dir / "equity_curve.csv")
    write_frame(trades, scenario_dir / "trades.csv")
    pd.DataFrame(list(metrics.get("walk_forward_folds") or [])).to_csv(scenario_dir / "folds.csv", index=False)

    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "script_version": VERSION,
        "scenario": name,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    (scenario_dir / "backtest_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "summary.txt").write_text(str(result.summary).rstrip() + "\n", encoding="utf-8")
    log(f"[{name}] capital final = US$ {float(metrics['strategy_ending_capital']):,.2f}")
    return payload


def read_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


for directory in (DIR_ALPACA, DIR_TIINGO, DIR_EVENTS, DIR_SPLITS):
    if not directory.exists():
        raise RuntimeError(f"Diretorio ausente: {directory}")
DIR_OUT.mkdir(parents=True, exist_ok=True)

log(f"Experimento: {VERSION}")
log("Hipotese: diferencas de volume podem explicar parte relevante da divergencia Alpaca x Tiingo")

scenario_ta, scenario_at, diagnostics = build_hybrids()
pd.DataFrame(diagnostics).sort_values("mean_abs_volume_pct_diff", ascending=False).to_csv(
    DIR_OUT / "volume_por_ativo.csv", index=False
)

result_ta = run_scenario("tiingo_precos_alpaca_volume", scenario_ta)
result_at = run_scenario("alpaca_precos_tiingo_volume", scenario_at)

baseline_alpaca = read_baseline(BASELINE_ALPACA)
baseline_tiingo = read_baseline(BASELINE_TIINGO)

def capital(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    metrics = payload.get("metrics") or {}
    value = metrics.get("strategy_ending_capital")
    return float(value) if value is not None else None

summary = {
    "schema_version": 1,
    "script_version": VERSION,
    "capital_alpaca_control": capital(baseline_alpaca),
    "capital_tiingo_total_causal": capital(baseline_tiingo),
    "capital_tiingo_prices_alpaca_volume": capital(result_ta),
    "capital_alpaca_prices_tiingo_volume": capital(result_at),
    "interpretation": {
        "if_tiingo_prices_alpaca_volume_moves_toward_alpaca": "volume e um fator material",
        "if_alpaca_prices_tiingo_volume_stays_near_alpaca": "precos/OHLC dominam e volume e secundario",
        "if_both_move_strongly": "volume e OHLC interagem na aprendizagem e calibracao",
    },
}
(DIR_OUT / "relatorio.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
    encoding="utf-8",
)
log(f"Resultados: {DIR_OUT}")
