"""Atribuicao exata Mongo 43M vs Tiingo com relogio de sessao alinhado.

Versao: mongo-tiingo-exact-capital-attribution-v1.2.0

Correcao metodologica sobre v1.1:
- Mongo/Alpaca representa cada sessao como meia-noite America/New_York convertida
  para UTC (04:00Z no horario de verao, 05:00Z no horario padrao);
- Tiingo EOD rotula a mesma sessao como 00:00Z.

Para testes hibridos por ativo, os indices precisam representar a mesma sessao.
A v1.2 preserva os CSVs Tiingo RAW e converte SOMENTE em memoria o rotulo da
data Tiingo para meia-noite America/New_York -> UTC. Assim os timestamps passam
a coincidir exatamente com a referencia Mongo certificada sem alterar OHLCV.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

import diagnosticar_atribuicao_capital_mongo_tiingo_v11 as v11

VERSION = "mongo-tiingo-exact-capital-attribution-v1.2.0"
SESSION_TIMEZONE = "America/New_York"


def canonicalize_tiingo_session_clock(frame: pd.DataFrame) -> pd.DataFrame:
    """Mapeia a data EOD Tiingo para o mesmo timestamp usado pelo Mongo/Alpaca.

    Exemplo:
      Tiingo  2026-07-01 00:00:00Z
      Mongo   2026-07-01 04:00:00Z  (00:00 America/New_York)

    Nenhum valor OHLCV e alterado; apenas o identificador temporal da sessao.
    """
    result = frame.copy()
    dates = pd.DatetimeIndex(result.index).tz_convert("UTC").date
    canonical = (
        pd.DatetimeIndex(pd.to_datetime(dates))
        .tz_localize(SESSION_TIMEZONE)
        .tz_convert("UTC")
    )
    result.index = canonical
    result.index.name = "timestamp"
    if result.index.has_duplicates:
        raise RuntimeError("Tiingo: timestamps duplicados apos alinhamento de sessao")
    return result.sort_index()


def load_tiingo_frames() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not v11.TIINGO_SERIES_DIR.exists() or not v11.TIINGO_EVENTS_DIR.exists():
        raise RuntimeError(
            "Snapshot Tiingo ausente. Rode primeiro: %run preparar_snapshots_mongo_tiingo.py"
        )

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, dict[str, str]] = {}
    for index, symbol in enumerate(v11.ASSETS, start=1):
        raw_path = v11.TIINGO_SERIES_DIR / f"{symbol}.csv"
        events_path = v11.TIINGO_EVENTS_DIR / f"{symbol}.csv"
        if not raw_path.exists() or not events_path.exists():
            raise RuntimeError(
                f"Tiingo incompleta para {symbol}; rode preparar_snapshots_mongo_tiingo.py"
            )

        raw = v11.validate_frame(symbol, pd.read_csv(raw_path), "tiingo_raw")
        raw = canonicalize_tiingo_session_clock(raw)
        events = v11.load_events(symbol)
        total = v11.normalize_tiingo_causally(symbol, raw, events)
        frames[symbol] = total
        hashes[symbol] = {
            "raw": v11.sha256_file(raw_path),
            "events": v11.sha256_file(events_path),
        }
        if index == 1 or index % 5 == 0 or index == len(v11.ASSETS):
            v11.log(
                f"Tiingo {index:02d}/{len(v11.ASSETS)} {symbol} | "
                f"candles={len(total)} | eventos={len(events)} | sessao=NY->UTC"
            )

    signature = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    return frames, {
        "combined_sha256": signature,
        "files": hashes,
        "session_timestamp_policy": "tiingo_date_as_midnight_America/New_York_then_UTC",
    }


def main() -> int:
    # Mantem todo o motor e protocolo da v1.1, trocando apenas o carregamento
    # temporal da Tiingo e isolando os resultados para impedir resume de runs
    # potencialmente produzidos com timestamps desalinhados.
    v11.VERSION = VERSION
    v11.OUTPUT_DIR = v11.ROOT / "output" / "mongo_tiingo_exact_capital_attribution_v12"
    v11.load_tiingo_frames = load_tiingo_frames
    v11.log(f"Correcao v1.2: alinhando sessoes Tiingo ao relogio Mongo ({SESSION_TIMEZONE})")
    return v11.main()


if __name__ == "__main__":
    raise SystemExit(main())
