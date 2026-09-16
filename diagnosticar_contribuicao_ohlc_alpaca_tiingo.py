"""Experimento pareado para decompor OHLC Alpaca x Tiingo.

Versao: alpaca-tiingo-price-shape-ablation-v1.0.0

Depois da ablacao de volume, este experimento separa:
1) trajetoria de fechamento (close path)
2) forma intradiaria do candle, representada por open/close, high/close e low/close

Cenarios:
- Tiingo close + forma intradiaria Alpaca + volume Tiingo
- Alpaca close + forma intradiaria Tiingo + volume Alpaca

A reconstrucao preserva candles validos porque aplica os ratios intradiarios
do fornecedor doador sobre o fechamento do fornecedor base.
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

VERSION = "alpaca-tiingo-price-shape-ablation-v1.0.0"
ROOT = Path(__file__).resolve().parent
DIR_ALPACA = ROOT / "dados" / "referencia_alpaca_atual"
DIR_TIINGO = ROOT / "dados" / "series_historicas"
DIR_EVENTS = ROOT / "dados" / "eventos_corporativos"
DIR_SPLITS = ROOT / "dados" / "desdobramentos"
DIR_OUT = ROOT / "output" / "diagnostico_contribuicao_ohlc"
BASELINE_ALPACA = ROOT / "output" / "alpaca_atual_22m_control"
BASELINE_TIINGO = ROOT / "output" / "tiingo_total_causal_v1"
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
        raise RuntimeError(f"{asset}: colunas ausentes: {', '.join(missing)}")
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


def load_tiingo_total_causal(asset: str) -> pd.DataFrame:
    raw = read_ohlcv(DIR_TIINGO, asset)

    split_path = DIR_SPLITS / f"{asset}.csv"
    split_events = pd.read_csv(split_path)
    split_session = pd.Series(1.0, index=raw.index, dtype=float)
    if not split_events.empty:
        split_events["timestamp"] = pd.to_datetime(
            split_events["timestamp"], utc=True, errors="coerce"
        )
        split_events["fator_split"] = pd.to_numeric(
            split_events["fator_split"], errors="coerce"
        )
        if "status" in split_events.columns:
            split_events = split_events.loc[
                split_events["status"].astype(str).str.lower().str.strip() == "a"
            ]
        split_events = split_events.dropna(subset=["timestamp", "fator_split"])
        split_events = split_events.loc[split_events["fator_split"] > 0]
        for event in split_events.itertuples(index=False):
            session = pd.Timestamp(event.timestamp).normalize()
            if session in split_session.index:
                split_session.loc[session] *= float(event.fator_split)

    split_factor = split_session.cumprod()
    split = raw.copy()
    for c in ("open", "high", "low", "close"):
        split[c] = split[c] * split_factor
    split["volume"] = split["volume"] / split_factor

    event_path = DIR_EVENTS / f"{asset}.csv"
    events = pd.read_csv(event_path)
    dividend_session = pd.Series(1.0, index=split.index, dtype=float)
    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        events["dividendo"] = pd.to_numeric(events["dividendo"], errors="coerce").fillna(0.0)
        events = events.dropna(subset=["timestamp"])
        events = events.loc[events["dividendo"] != 0.0].sort_values("timestamp")
        for event in events.itertuples(index=False):
            session = pd.Timestamp(event.timestamp).normalize()
            if session not in split.index:
                continue
            pos = int(split.index.get_loc(session))
            if pos == 0:
                continue
            previous_close = float(split.iloc[pos - 1]["close"])
            dividend = float(event.dividendo) * float(split_factor.loc[session])
            denominator = previous_close - dividend
            if not np.isfinite(denominator) or denominator <= 0:
                raise RuntimeError(f"{asset}: dividendo invalido em {session.date()}")
            dividend_session.loc[session] *= previous_close / denominator

    dividend_factor = dividend_session.cumprod()
    total = split.copy()
    for c in ("open", "high", "low", "close"):
        total[c] = total[c] * dividend_factor
    return total


def intraday_shape(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].replace(0, np.nan)
    return pd.DataFrame(
        {
            "open_to_close": df["open"] / close,
            "high_to_close": df["high"] / close,
            "low_to_close": df["low"] / close,
        },
        index=df.index,
    )


def apply_shape(close_source: pd.DataFrame, shape_source: pd.DataFrame) -> pd.DataFrame:
    out = close_source.copy()
    shape = intraday_shape(shape_source)
    out["open"] = out["close"] * shape["open_to_close"]
    out["high"] = out["close"] * shape["high_to_close"]
    out["low"] = out["close"] * shape["low_to_close"]

    valid = (
        (out["high"] >= out[["open", "close"]].max(axis=1))
        & (out["low"] <= out[["open", "close"]].min(axis=1))
        & (out["low"] > 0)
    )
    if not bool(valid.all()):
        bad = int((~valid).sum())
        raise RuntimeError(f"Reconstrucao OHLC gerou {bad} candles invalidos.")
    return out


def build_scenarios():
    tiingo_close_alpaca_shape: dict[str, pd.DataFrame] = {}
    alpaca_close_tiingo_shape: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []

    for pos, asset in enumerate(ASSETS, start=1):
        alpaca = read_ohlcv(DIR_ALPACA, asset)
        tiingo = load_tiingo_total_causal(asset)
        common = alpaca.index.intersection(tiingo.index).sort_values()
        if len(common) < 700:
            raise RuntimeError(f"{asset}: historico comum insuficiente: {len(common)}")

        a = alpaca.loc[common].copy()
        t = tiingo.loc[common].copy()

        tc_as = apply_shape(t, a)
        ac_ts = apply_shape(a, t)

        tiingo_close_alpaca_shape[asset] = tc_as
        alpaca_close_tiingo_shape[asset] = ac_ts

        sa = intraday_shape(a)
        st = intraday_shape(t)
        close_ret_a = a["close"].pct_change()
        close_ret_t = t["close"].pct_change()

        diagnostics.append(
            {
                "asset": asset,
                "sessions": int(len(common)),
                "mean_abs_close_return_diff_bps": float(
                    (close_ret_t - close_ret_a).abs().dropna().mean() * 10_000
                ),
                "mean_abs_open_close_ratio_diff": float(
                    (st["open_to_close"] - sa["open_to_close"]).abs().mean()
                ),
                "mean_abs_high_close_ratio_diff": float(
                    (st["high_to_close"] - sa["high_to_close"]).abs().mean()
                ),
                "mean_abs_low_close_ratio_diff": float(
                    (st["low_to_close"] - sa["low_to_close"]).abs().mean()
                ),
            }
        )
        log(f"[dados] {pos:02d}/{len(ASSETS)} {asset} | sessoes={len(common)}")

    return tiingo_close_alpaca_shape, alpaca_close_tiingo_shape, diagnostics


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

    curve_cols = [
        c for c in (
            "strategy_equity", "buy_hold_equity", "selected_asset",
            "decision_score", "trade_action", "trade_reason", "walk_forward_fold",
        )
        if c in predictions.columns
    ]
    curve = predictions[curve_cols].copy() if curve_cols else predictions.copy()
    write_frame(curve, scenario_dir / "equity_curve.csv")
    write_frame(trades, scenario_dir / "trades.csv")
    pd.DataFrame(list(metrics.get("walk_forward_folds") or [])).to_csv(
        scenario_dir / "folds.csv", index=False
    )

    payload = {
        "schema_version": 1,
        "script_version": VERSION,
        "scenario": name,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    (scenario_dir / "backtest_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "summary.txt").write_text(
        str(result.summary).rstrip() + "\n", encoding="utf-8"
    )
    log(f"[{name}] capital final = US$ {float(metrics['strategy_ending_capital']):,.2f}")
    return payload


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ending_capital(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    value = (payload.get("metrics") or {}).get("strategy_ending_capital")
    return float(value) if value is not None else None


def compare_policy(baseline_curve: Path, scenario_curve: Path) -> dict[str, Any] | None:
    if not baseline_curve.exists() or not scenario_curve.exists():
        return None
    b = pd.read_csv(baseline_curve)
    s = pd.read_csv(scenario_curve)
    for df in (b, s):
        df["session"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.normalize()
    p = b[["session", "selected_asset"]].merge(
        s[["session", "selected_asset"]],
        on="session",
        suffixes=("_baseline", "_scenario"),
    )
    diff = p["selected_asset_baseline"].astype(str) != p["selected_asset_scenario"].astype(str)
    first = p.loc[diff, "session"].min() if bool(diff.any()) else None
    return {
        "common_sessions": int(len(p)),
        "divergent_sessions": int(diff.sum()),
        "divergent_fraction": float(diff.mean()) if len(p) else 0.0,
        "first_divergence": (
            pd.Timestamp(first).date().isoformat() if first is not None and not pd.isna(first) else None
        ),
    }


def main() -> None:
    for directory in (DIR_ALPACA, DIR_TIINGO, DIR_EVENTS, DIR_SPLITS):
        if not directory.exists():
            raise RuntimeError(f"Diretorio ausente: {directory}")
    DIR_OUT.mkdir(parents=True, exist_ok=True)

    log(f"Experimento: {VERSION}")
    log("Separando trajetoria de fechamento da forma intradiaria OHLC")

    scenario_ta, scenario_at, diagnostics = build_scenarios()
    pd.DataFrame(diagnostics).sort_values(
        "mean_abs_close_return_diff_bps", ascending=False
    ).to_csv(DIR_OUT / "ohlc_shape_por_ativo.csv", index=False)

    result_ta = run_scenario("tiingo_close_alpaca_shape", scenario_ta)
    result_at = run_scenario("alpaca_close_tiingo_shape", scenario_at)

    baseline_alpaca = read_json(BASELINE_ALPACA / "backtest_result.json")
    baseline_tiingo = read_json(BASELINE_TIINGO / "backtest_result.json")

    policy_ta_vs_alpaca = compare_policy(
        BASELINE_ALPACA / "equity_curve.csv",
        DIR_OUT / "tiingo_close_alpaca_shape" / "equity_curve.csv",
    )
    policy_at_vs_alpaca = compare_policy(
        BASELINE_ALPACA / "equity_curve.csv",
        DIR_OUT / "alpaca_close_tiingo_shape" / "equity_curve.csv",
    )

    summary = {
        "schema_version": 1,
        "script_version": VERSION,
        "capital_alpaca_control": ending_capital(baseline_alpaca),
        "capital_tiingo_total_causal": ending_capital(baseline_tiingo),
        "capital_tiingo_close_alpaca_shape": ending_capital(result_ta),
        "capital_alpaca_close_tiingo_shape": ending_capital(result_at),
        "policy_vs_alpaca": {
            "tiingo_close_alpaca_shape": policy_ta_vs_alpaca,
            "alpaca_close_tiingo_shape": policy_at_vs_alpaca,
        },
        "interpretation": {
            "close_path_dominates": (
                "tiingo_close_alpaca_shape permanece perto do Tiingo e "
                "alpaca_close_tiingo_shape permanece perto da Alpaca"
            ),
            "intraday_shape_matters": (
                "trocar open/high/low relativos ao close desloca fortemente "
                "os dois cenarios em direcao ao fornecedor da forma intradiaria"
            ),
        },
    }
    (DIR_OUT / "relatorio.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    log(f"Resultados: {DIR_OUT}")


if __name__ == "__main__":
    main()
