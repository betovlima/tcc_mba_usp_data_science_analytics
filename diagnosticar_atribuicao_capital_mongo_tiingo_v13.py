"""Atribuicao Mongo 43M vs Tiingo com sessao alinhada e splits validados.

Versao: mongo-tiingo-exact-capital-attribution-v1.3.0

Sobre v1.2:
- mantem o alinhamento temporal Tiingo -> meia-noite America/New_York -> UTC;
- deixa de aplicar todo splitFactor EOD cegamente;
- aplica somente eventos presentes em dados/desdobramentos, gerados pela
  validacao de ruptura mecanica do bootstrap v2.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

import diagnosticar_atribuicao_capital_mongo_tiingo_v11 as v11
import diagnosticar_atribuicao_capital_mongo_tiingo_v12 as v12

VERSION = "mongo-tiingo-exact-capital-attribution-v1.3.0"
SPLITS_DIR = v11.DATA_ROOT / "desdobramentos"


def load_validated_splits(symbol: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{symbol}.csv"
    if not path.exists():
        raise RuntimeError(
            f"Splits validados ausentes: {path}. Rode primeiro: "
            "%run preparar_snapshots_mongo_tiingo.py"
        )
    table = pd.read_csv(path)
    required = ["timestamp", "split_de", "split_para", "fator_split", "status"]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}/splits: colunas ausentes: {', '.join(missing)}")
    if table.empty:
        return pd.DataFrame(columns=required)
    table = table[required].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    for c in ("split_de", "split_para", "fator_split"):
        table[c] = pd.to_numeric(table[c], errors="coerce")
    table["status"] = table["status"].astype(str).str.lower().str.strip()
    table = table.dropna(subset=["timestamp", "split_de", "split_para", "fator_split"])
    table = table.loc[
        (table["split_de"] > 0)
        & (table["split_para"] > 0)
        & (table["fator_split"] > 0)
        & (table["status"] == "a")
    ].copy()
    return table.sort_values("timestamp").drop_duplicates(
        ["timestamp", "split_de", "split_para"], keep="last"
    ).reset_index(drop=True)


def normalize_tiingo_causally(
    symbol: str,
    raw: pd.DataFrame,
    events: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    split_on_session = pd.Series(1.0, index=raw.index, dtype=float)
    for event in splits.itertuples(index=False):
        event_date = pd.Timestamp(event.timestamp).normalize()
        sessions = raw.index[raw.index.normalize() == event_date]
        if len(sessions) != 1:
            raise RuntimeError(f"{symbol}: split {event_date.date()} sem sessao unica")
        factor = float(event.fator_split)
        expected = float(event.split_para) / float(event.split_de)
        if not np.isclose(factor, expected, rtol=1e-8, atol=1e-10):
            raise RuntimeError(
                f"{symbol}: split inconsistente em {event_date.date()} | "
                f"fonte={factor} calculado={expected}"
            )
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
    if not v11.TIINGO_SERIES_DIR.exists() or not v11.TIINGO_EVENTS_DIR.exists() or not SPLITS_DIR.exists():
        raise RuntimeError(
            "Snapshot Tiingo/splits validados ausentes. Rode primeiro: "
            "%run preparar_snapshots_mongo_tiingo.py"
        )

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, dict[str, str]] = {}
    accepted_split_count = 0

    for index, symbol in enumerate(v11.ASSETS, start=1):
        raw_path = v11.TIINGO_SERIES_DIR / f"{symbol}.csv"
        events_path = v11.TIINGO_EVENTS_DIR / f"{symbol}.csv"
        splits_path = SPLITS_DIR / f"{symbol}.csv"
        if not raw_path.exists() or not events_path.exists() or not splits_path.exists():
            raise RuntimeError(f"Snapshot Tiingo incompleto para {symbol}")

        raw = v11.validate_frame(symbol, pd.read_csv(raw_path), "tiingo_raw")
        raw = v12.canonicalize_tiingo_session_clock(raw)
        events = v11.load_events(symbol)
        splits = load_validated_splits(symbol)
        total = normalize_tiingo_causally(symbol, raw, events, splits)
        frames[symbol] = total
        accepted_split_count += len(splits)
        hashes[symbol] = {
            "raw": v11.sha256_file(raw_path),
            "events": v11.sha256_file(events_path),
            "validated_splits": v11.sha256_file(splits_path),
        }
        if index == 1 or index % 5 == 0 or index == len(v11.ASSETS):
            v11.log(
                f"Tiingo {index:02d}/{len(v11.ASSETS)} {symbol} | "
                f"candles={len(total)} | eventos={len(events)} | splits_aceitos={len(splits)}"
            )

    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    v11.log(f"Tiingo total-causal pronta | splits validados aplicados={accepted_split_count}")
    return frames, {
        "combined_sha256": signature,
        "files": hashes,
        "session_timestamp_policy": "tiingo_date_as_midnight_America/New_York_then_UTC",
        "split_policy": "validated_mechanical_break_only",
        "accepted_split_count": accepted_split_count,
    }


def main() -> int:
    v11.VERSION = VERSION
    v11.OUTPUT_DIR = v11.ROOT / "output" / "mongo_tiingo_exact_capital_attribution_v13"
    v11.load_tiingo_frames = load_tiingo_frames
    v11.log("Correcao v1.3: relogio de sessao alinhado + splitFactor validado")
    return v11.main()


if __name__ == "__main__":
    raise SystemExit(main())
