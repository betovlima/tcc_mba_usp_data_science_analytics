"""Auditoria completa da referencia historica 43M versus Tiingo total-causal.

Versao: mongo43-tiingo-audit-v1.0.0

Compara:
- OHLCV e forma intradiaria;
- features e targets usados pelo modelo;
- politica de rotacao, quando os dois backtests ja foram executados.

A referencia historica vem exclusivamente de dados/referencia_mongo_43m.
A serie Tiingo e reconstruida apenas com dados Tiingo congelados, splits e
dividendos causais. Nenhum dado Alpaca atual entra neste diagnostico.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import ROTATION_FEATURES, build_rotation_frame
from tcc_engine.config import ASSETS, CONFIG

VERSION = "mongo43-tiingo-audit-v1.0.0"
ROOT = Path(__file__).resolve().parent
DIR_REF = ROOT / "dados" / "referencia_mongo_43m"
DIR_TIINGO = ROOT / "dados" / "series_historicas"
DIR_EVENTS = ROOT / "dados" / "eventos_corporativos"
DIR_SPLITS = ROOT / "dados" / "desdobramentos"
DIR_OUT = ROOT / "output" / "auditoria_mongo43_vs_tiingo"
CURVE_REF = ROOT / "output" / "mongo_43m_control" / "equity_curve.csv"
CURVE_TIINGO = ROOT / "output" / "tiingo_total_causal_v1" / "equity_curve.csv"
FOLDS_REF = ROOT / "output" / "mongo_43m_control" / "folds.csv"
FOLDS_TIINGO = ROOT / "output" / "tiingo_total_causal_v1" / "folds.csv"
OHLCV = ["open", "high", "low", "close", "volume"]
TARGETS = [
    "forward_net_log_return",
    "forward_cash_edge",
    "forward_movement_capture",
    "forward_trend_persistence",
    "forward_risk_adjusted_utility",
]
WINDOWS = {
    "fold1_treino": (None, pd.Timestamp("2019-07-30", tz="UTC")),
    "fold1_calibracao": (
        pd.Timestamp("2019-10-24", tz="UTC"),
        pd.Timestamp("2020-04-24", tz="UTC"),
    ),
    "historico_completo": (None, None),
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


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
    return df.set_index("session")[OHLCV].sort_index()


def apply_splits(asset: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    path = DIR_SPLITS / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"{asset}: splits ausentes: {path}")
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


def apply_dividends(asset: str, split_series: pd.DataFrame, split_factor: pd.Series) -> pd.DataFrame:
    path = DIR_EVENTS / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"{asset}: eventos ausentes: {path}")
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


def load_tiingo(asset: str) -> pd.DataFrame:
    raw = read_ohlcv(DIR_TIINGO, asset)
    split, split_factor = apply_splits(asset, raw)
    return apply_dividends(asset, split, split_factor)


def cut(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


def diff_summary(asset: str, window: str, variable: str, cls: str, ref: pd.Series, tiingo: pd.Series) -> dict[str, Any] | None:
    data = pd.concat([ref.rename("ref43"), tiingo.rename("tiingo")], axis=1, join="inner").dropna()
    if data.empty:
        return None
    diff = data["tiingo"] - data["ref43"]
    abs_diff = diff.abs()
    scale = float(data["ref43"].abs().median())
    normalizer = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    max_idx = abs_diff.idxmax()
    return {
        "asset": asset,
        "window": window,
        "class": cls,
        "variable": variable,
        "observations": int(len(data)),
        "mean_abs_error": float(abs_diff.mean()),
        "p95_abs_error": float(abs_diff.quantile(0.95)),
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error_normalized": float(abs_diff.mean() / normalizer),
        "correlation": float(data["ref43"].corr(data["tiingo"])) if len(data) >= 2 else float("nan"),
        "max_error_date": pd.Timestamp(max_idx).date().isoformat(),
        "reference_value_at_max": float(data.loc[max_idx, "ref43"]),
        "tiingo_value_at_max": float(data.loc[max_idx, "tiingo"]),
    }


for directory in (DIR_REF, DIR_TIINGO, DIR_EVENTS, DIR_SPLITS):
    if not directory.exists():
        raise RuntimeError(f"Diretorio ausente: {directory}")
DIR_OUT.mkdir(parents=True, exist_ok=True)

summaries: list[dict[str, Any]] = []
ohlcv_rows: list[dict[str, Any]] = []
frames_ref: dict[str, pd.DataFrame] = {}
frames_tiingo: dict[str, pd.DataFrame] = {}

log("Auditoria referencia historica 43M -> Tiingo total-causal")
for pos, asset in enumerate(ASSETS, start=1):
    ref = read_ohlcv(DIR_REF, asset)
    tiingo = load_tiingo(asset)
    common = ref.index.intersection(tiingo.index).sort_values()
    ref = ref.loc[common].copy()
    tiingo = tiingo.loc[common].copy()

    shape_ref = pd.DataFrame(index=common)
    shape_t = pd.DataFrame(index=common)
    for c in ("open", "high", "low"):
        shape_ref[f"{c}_over_close"] = ref[c] / ref["close"] - 1.0
        shape_t[f"{c}_over_close"] = tiingo[c] / tiingo["close"] - 1.0
    shape_ref["close_return"] = ref["close"].pct_change()
    shape_t["close_return"] = tiingo["close"].pct_change()
    shape_ref["volume_change"] = ref["volume"].pct_change()
    shape_t["volume_change"] = tiingo["volume"].pct_change()

    for variable in shape_ref.columns:
        item = diff_summary(asset, "historico_completo", variable, "ohlcv", shape_ref[variable], shape_t[variable])
        if item:
            ohlcv_rows.append(item)

    fr = build_rotation_frame(ref, CONFIG)
    ft = build_rotation_frame(tiingo, CONFIG)
    idx = fr.index.intersection(ft.index)
    fr = fr.loc[idx]
    ft = ft.loc[idx]
    frames_ref[asset] = fr
    frames_tiingo[asset] = ft

    for window, (start, end) in WINDOWS.items():
        a = cut(fr, start, end)
        b = cut(ft, start, end)
        idxw = a.index.intersection(b.index)
        a = a.loc[idxw]
        b = b.loc[idxw]
        for variable in [*ROTATION_FEATURES, *TARGETS]:
            if variable not in a.columns or variable not in b.columns:
                continue
            cls = "target" if variable in TARGETS else "feature"
            item = diff_summary(asset, window, variable, cls, a[variable], b[variable])
            if item:
                summaries.append(item)

    log(f"{pos:02d}/{len(ASSETS)} {asset} | sessoes={len(common)} | frame={len(idx)}")

summary_df = pd.DataFrame(summaries)
ohlcv_df = pd.DataFrame(ohlcv_rows)
if not summary_df.empty:
    summary_df = summary_df.sort_values(["window", "mean_abs_error_normalized"], ascending=[True, False])
    summary_df.to_csv(DIR_OUT / "comparacao_features_targets_resumo.csv", index=False)
if not ohlcv_df.empty:
    ohlcv_df = ohlcv_df.sort_values("mean_abs_error_normalized", ascending=False)
    ohlcv_df.to_csv(DIR_OUT / "comparacao_ohlcv_resumo.csv", index=False)

policy: dict[str, Any] = {"available": False}
if CURVE_REF.exists() and CURVE_TIINGO.exists():
    def prep_curve(path: Path, suffix: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        date_col = next((c for c in ("timestamp", "date", "data_sessao", "index") if c in df.columns), None)
        if date_col is None or "selected_asset" not in df.columns:
            raise RuntimeError(f"Curva invalida: {path}")
        df["session"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.normalize()
        cols = ["session", "selected_asset"]
        if "strategy_equity" in df.columns:
            cols.append("strategy_equity")
        df = df[cols].dropna(subset=["session"]).drop_duplicates("session", keep="last")
        rename = {"selected_asset": f"selected_asset_{suffix}"}
        if "strategy_equity" in df.columns:
            rename["strategy_equity"] = f"strategy_equity_{suffix}"
        return df.rename(columns=rename)

    r = prep_curve(CURVE_REF, "ref43")
    t = prep_curve(CURVE_TIINGO, "tiingo")
    p = r.merge(t, on="session", how="inner").sort_values("session")
    p["decision_divergent"] = p["selected_asset_ref43"].astype(str) != p["selected_asset_tiingo"].astype(str)
    divergent = p.loc[p["decision_divergent"]]
    first = divergent.iloc[0] if not divergent.empty else None
    if "strategy_equity_ref43" in p.columns and "strategy_equity_tiingo" in p.columns:
        p["equity_ratio_tiingo_ref43"] = p["strategy_equity_tiingo"] / p["strategy_equity_ref43"]
    p.to_csv(DIR_OUT / "comparacao_politica.csv", index=False)
    policy = {
        "available": True,
        "common_sessions": int(len(p)),
        "divergent_sessions": int(p["decision_divergent"].sum()),
        "divergent_fraction": float(p["decision_divergent"].mean()) if len(p) else 0.0,
        "first_divergence": pd.Timestamp(first["session"]).date().isoformat() if first is not None else None,
        "reference_asset_first_divergence": str(first["selected_asset_ref43"]) if first is not None else None,
        "tiingo_asset_first_divergence": str(first["selected_asset_tiingo"]) if first is not None else None,
    }

fold_comparison: list[dict[str, Any]] = []
if FOLDS_REF.exists() and FOLDS_TIINGO.exists():
    rf = pd.read_csv(FOLDS_REF)
    tf = pd.read_csv(FOLDS_TIINGO)
    for fold_id in sorted(set(rf.get("fold_id", [])) & set(tf.get("fold_id", []))):
        a = rf.loc[rf["fold_id"] == fold_id].iloc[0]
        b = tf.loc[tf["fold_id"] == fold_id].iloc[0]
        fold_comparison.append({
            "fold_id": int(fold_id),
            "reference_ending_capital": float(a["strategy_ending_capital"]),
            "tiingo_ending_capital": float(b["strategy_ending_capital"]),
            "reference_switch_margin": float(a["effective_switch_margin"]),
            "tiingo_switch_margin": float(b["effective_switch_margin"]),
        })

report = {
    "schema_version": 1,
    "script_version": VERSION,
    "reference": "certified historical Mongo/Alpaca snapshot 43M",
    "tiingo": "raw EOD + causal splits + causal dividends",
    "policy": policy,
    "fold_comparison": fold_comparison,
    "top_features_fold1_train": (
        summary_df.loc[(summary_df["window"] == "fold1_treino") & (summary_df["class"] == "feature")].head(30).to_dict(orient="records")
        if not summary_df.empty else []
    ),
    "top_targets_fold1_train": (
        summary_df.loc[(summary_df["window"] == "fold1_treino") & (summary_df["class"] == "target")].head(30).to_dict(orient="records")
        if not summary_df.empty else []
    ),
    "top_features_fold1_calibration": (
        summary_df.loc[(summary_df["window"] == "fold1_calibracao") & (summary_df["class"] == "feature")].head(30).to_dict(orient="records")
        if not summary_df.empty else []
    ),
    "top_ohlcv": ohlcv_df.head(30).to_dict(orient="records") if not ohlcv_df.empty else [],
}
(DIR_OUT / "relatorio.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
log(f"Resultados: {DIR_OUT}")
