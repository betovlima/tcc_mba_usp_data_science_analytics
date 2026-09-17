"""Prepara snapshots locais para o experimento Mongo 43M vs Tiingo.

Versao: mongo-tiingo-snapshot-bootstrap-v1.0.0

O script faz duas operacoes de forma reproduzivel:
1. exporta o OHLCV certificado do MongoDB local para dados/referencia_mongo_43m;
2. baixa Tiingo EOD RAW + eventos corporativos para dados/series_historicas e
   dados/eventos_corporativos.

Depois da preparacao, o diagnostico de atribuicao pode rodar sem internet e sem
consultar o MongoDB novamente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient

from tcc_engine.config import ASSETS, CONFIG, END_DATE, START_DATE

VERSION = "mongo-tiingo-snapshot-bootstrap-v1.0.0"
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dados"
MONGO_DIR = DATA_ROOT / "referencia_mongo_43m"
TIINGO_SERIES_DIR = DATA_ROOT / "series_historicas"
TIINGO_EVENTS_DIR = DATA_ROOT / "eventos_corporativos"
MONGO_MANIFEST = DATA_ROOT / "manifesto_mongo_43m.json"
TIINGO_MANIFEST = DATA_ROOT / "manifesto_tiingo.json"

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DATABASE = "extrema_backtest"
MARKET_COLLECTION = "alpaca_market_bars"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily"
OHLCV = ["open", "high", "low", "close", "volume"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_signature(files: dict[str, dict[str, str]]) -> str:
    payload = json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)


def validate_ohlcv_table(symbol: str, table: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", *OHLCV]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}: OHLCV sem colunas: {', '.join(missing)}")
    result = table[required].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    for column in OHLCV:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=required)
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    valid = (
        (result["open"] > 0)
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
        & (result["volume"] >= 0)
    )
    result = result.loc[valid].copy()
    if result.empty:
        raise RuntimeError(f"{symbol}: OHLCV vazio depois da validacao")
    return result


def export_mongo(force: bool) -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MONGO_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env", override=False)
    mongo_uri = str(os.getenv("TCC_MONGO_URI") or DEFAULT_MONGO_URI).strip()
    database_name = str(os.getenv("TCC_MONGO_DATABASE") or DEFAULT_MONGO_DATABASE).strip()
    lowered = mongo_uri.lower()
    if not any(host in lowered for host in ("localhost", "127.0.0.1", "::1")):
        raise RuntimeError("Exportacao aceita somente MongoDB local")

    start_ts = pd.Timestamp(START_DATE, tz="UTC").to_pydatetime()
    end_ts = (pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()
    log(f"Mongo: exportando {database_name}.{MARKET_COLLECTION}")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5_000,
        connectTimeoutMS=5_000,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        collection = client[database_name][MARKET_COLLECTION]
        file_hashes: dict[str, dict[str, str]] = {}
        for index, symbol in enumerate(ASSETS, start=1):
            path = MONGO_DIR / f"{symbol}.csv"
            if path.exists() and not force:
                table = validate_ohlcv_table(symbol, pd.read_csv(path))
                log(f"Mongo {index:02d}/{len(ASSETS)} {symbol} | existente | {len(table)} candles")
            else:
                rows = list(
                    collection.find(
                        {
                            "symbol": symbol,
                            "interval": CONFIG.timeframe,
                            "feed": CONFIG.alpaca_historical_feed,
                            "adjustment": CONFIG.alpaca_adjustment,
                            "timestamp": {"$gte": start_ts, "$lt": end_ts},
                        },
                        {
                            "_id": 0,
                            "timestamp": 1,
                            "open": 1,
                            "high": 1,
                            "low": 1,
                            "close": 1,
                            "volume": 1,
                        },
                    ).sort("timestamp", 1)
                )
                if not rows:
                    raise RuntimeError(f"Mongo sem historico para {symbol}")
                table = validate_ohlcv_table(symbol, pd.DataFrame(rows))
                atomic_csv(table, path)
                log(f"Mongo {index:02d}/{len(ASSETS)} {symbol} | exportado | {len(table)} candles")
            file_hashes[symbol] = {"ohlcv": sha256_file(path)}
    finally:
        client.close()

    manifest = {
        "schema_version": 1,
        "script_version": VERSION,
        "source": "local_mongodb_alpaca_market_bars",
        "database": database_name,
        "collection": MARKET_COLLECTION,
        "feed": CONFIG.alpaca_historical_feed,
        "adjustment": CONFIG.alpaca_adjustment,
        "timeframe": CONFIG.timeframe,
        "history_start": START_DATE,
        "history_end": END_DATE,
        "asset_count": len(ASSETS),
        "assets": list(ASSETS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_hashes,
        "combined_sha256": combined_signature(file_hashes),
    }
    MONGO_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Mongo congelado | SHA-256={manifest['combined_sha256'][:16]}...")
    return manifest


def tiingo_token() -> str:
    load_dotenv(ROOT / ".env", override=False)
    for key in ("TIINGO_API_KEY", "TIINGO_TOKEN", "TIINGO_API_TOKEN"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    raise RuntimeError("Token Tiingo ausente. Defina TIINGO_API_KEY ou TIINGO_TOKEN no .env")


def validate_events(symbol: str, table: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "dividendo", "fator_split"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}: eventos sem colunas: {', '.join(missing)}")
    result = table[required].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result["dividendo"] = pd.to_numeric(result["dividendo"], errors="coerce").fillna(0.0)
    result["fator_split"] = pd.to_numeric(result["fator_split"], errors="coerce").fillna(1.0)
    result = result.dropna(subset=["timestamp"])
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return result


def download_tiingo(force: bool) -> dict[str, Any]:
    TIINGO_SERIES_DIR.mkdir(parents=True, exist_ok=True)
    TIINGO_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    token = tiingo_token()
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "Authorization": f"Token {token}"})
    file_hashes: dict[str, dict[str, str]] = {}

    log("Tiingo: baixando/conferindo snapshot EOD RAW")
    for index, symbol in enumerate(ASSETS, start=1):
        series_path = TIINGO_SERIES_DIR / f"{symbol}.csv"
        events_path = TIINGO_EVENTS_DIR / f"{symbol}.csv"
        if series_path.exists() and events_path.exists() and not force:
            series = validate_ohlcv_table(symbol, pd.read_csv(series_path))
            events = validate_events(symbol, pd.read_csv(events_path))
            log(f"Tiingo {index:02d}/{len(ASSETS)} {symbol} | existente | {len(series)} candles")
        else:
            response = session.get(
                f"{TIINGO_URL}/{symbol}/prices",
                params={"startDate": START_DATE, "endDate": END_DATE, "resampleFreq": "daily"},
                timeout=60,
            )
            if response.status_code == 429:
                raise RuntimeError(
                    "Tiingo retornou HTTP 429. Os ativos ja baixados foram preservados; "
                    "execute o mesmo comando novamente mais tarde para continuar."
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise RuntimeError(f"Tiingo nao retornou dados para {symbol}")
            raw = pd.DataFrame(payload)
            required = ["date", *OHLCV]
            missing = [column for column in required if column not in raw.columns]
            if missing:
                raise RuntimeError(f"{symbol}: resposta Tiingo sem colunas: {', '.join(missing)}")

            series = raw[required].rename(columns={"date": "timestamp"})
            series = validate_ohlcv_table(symbol, series)
            dividends = (
                pd.to_numeric(raw["divCash"], errors="coerce").fillna(0.0)
                if "divCash" in raw.columns else pd.Series(0.0, index=raw.index)
            )
            split_factors = (
                pd.to_numeric(raw["splitFactor"], errors="coerce").fillna(1.0)
                if "splitFactor" in raw.columns else pd.Series(1.0, index=raw.index)
            )
            events = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(raw["date"], utc=True, errors="coerce"),
                    "dividendo": dividends,
                    "fator_split": split_factors,
                }
            )
            events = validate_events(symbol, events)
            events = events.loc[
                (events["dividendo"] != 0.0) | (events["fator_split"] != 1.0)
            ].copy()
            atomic_csv(series, series_path)
            atomic_csv(events, events_path)
            log(
                f"Tiingo {index:02d}/{len(ASSETS)} {symbol} | baixado | "
                f"{len(series)} candles | eventos={len(events)}"
            )
        file_hashes[symbol] = {
            "raw": sha256_file(series_path),
            "events": sha256_file(events_path),
        }

    manifest = {
        "schema_version": 1,
        "script_version": VERSION,
        "source": "Tiingo EOD",
        "price_basis": "raw",
        "corporate_actions": "divCash_and_splitFactor_separate",
        "history_start": START_DATE,
        "history_end": END_DATE,
        "asset_count": len(ASSETS),
        "assets": list(ASSETS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_hashes,
        "combined_sha256": combined_signature(file_hashes),
    }
    TIINGO_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Tiingo congelada | SHA-256={manifest['combined_sha256'][:16]}...")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "mongo", "tiingo"), default="all")
    parser.add_argument("--force-mongo", action="store_true")
    parser.add_argument("--force-tiingo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log(VERSION)
    log(f"Destino: {DATA_ROOT}")
    if args.only in ("all", "mongo"):
        export_mongo(args.force_mongo)
    if args.only in ("all", "tiingo"):
        download_tiingo(args.force_tiingo)
    log("Snapshots prontos. Agora rode diagnosticar_atribuicao_capital_mongo_tiingo.py --mode pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
