"""Exporta para CSV as series historicas certificadas que estao no Mongo local.

Este script existe apenas para migrar a entrada historica que reproduziu o
backtest certificado para arquivos estaticos. Depois da exportacao, o backtest
continua lendo somente ``dados/series_historicas`` e nao depende de MongoDB.

O script nao recalcula, ajusta ou baixa candles. Ele apenas copia para CSV os
mesmos registros OHLCV armazenados na collection ``alpaca_market_bars``.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import CONFIG as CONFIGURACAO
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"
ARQUIVO_ENV = RAIZ_PROJETO / ".env"

URI_MONGO_PADRAO = "mongodb://localhost:27017"
BANCO_MONGO_PADRAO = "extrema_backtest"
COLLECTION_MERCADO = "alpaca_market_bars"


# %% 1 - Configuracao do Mongo local
load_dotenv(ARQUIVO_ENV, override=True)

uri_mongo = str(os.getenv("TCC_MONGO_URI") or URI_MONGO_PADRAO).strip()
banco_mongo = str(os.getenv("TCC_MONGO_DATABASE") or BANCO_MONGO_PADRAO).strip()

uri_minuscula = uri_mongo.lower()
if not any(host in uri_minuscula for host in ("localhost", "127.0.0.1", "::1")):
    raise RuntimeError(
        "A exportacao certificada aceita somente o MongoDB local. "
        "Defina TCC_MONGO_URI para localhost."
    )

inicio = pd.Timestamp(DATA_INICIO, tz="UTC").to_pydatetime()
fim_exclusivo = (
    pd.Timestamp(DATA_FIM, tz="UTC") + pd.Timedelta(days=1)
).to_pydatetime()

print("Exportacao das series historicas certificadas")
print(f"MongoDB: {banco_mongo}.{COLLECTION_MERCADO}")
print(f"Periodo: {DATA_INICIO} -> {DATA_FIM}")
print(
    "Filtro: "
    f"interval={CONFIGURACAO.timeframe} | "
    f"feed={CONFIGURACAO.alpaca_historical_feed} | "
    f"adjustment={CONFIGURACAO.alpaca_adjustment}"
)
print(f"Ativos: {len(ATIVOS)}")


# %% 2 - Leitura exata dos candles armazenados
cliente = MongoClient(
    uri_mongo,
    serverSelectionTimeoutMS=3_000,
    connectTimeoutMS=3_000,
    retryWrites=False,
)

try:
    cliente.admin.command("ping")
    collection = cliente[banco_mongo][COLLECTION_MERCADO]

    registros = list(
        collection.find(
            {
                "symbol": {"$in": list(ATIVOS)},
                "interval": CONFIGURACAO.timeframe,
                "feed": CONFIGURACAO.alpaca_historical_feed,
                "adjustment": CONFIGURACAO.alpaca_adjustment,
                "timestamp": {"$gte": inicio, "$lt": fim_exclusivo},
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
    cliente.close()

if not registros:
    raise RuntimeError(
        "Nenhum candle foi encontrado no MongoDB local com o filtro certificado."
    )

historico = pd.DataFrame(registros)
historico["symbol"] = historico["symbol"].astype(str).str.upper()
historico["timestamp"] = pd.to_datetime(historico["timestamp"], utc=True)

print(f"Candles encontrados: {len(historico):,}")


# %% 3 - Validacao e gravacao de um CSV por ativo
DIRETORIO_SERIES.mkdir(parents=True, exist_ok=True)
colunas = ["timestamp", "open", "high", "low", "close", "volume"]

for posicao, ativo in enumerate(ATIVOS, start=1):
    serie = historico.loc[historico["symbol"].eq(ativo), colunas].copy()
    serie = serie.sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")

    for coluna in ("open", "high", "low", "close", "volume"):
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=colunas)
    valida = (
        (serie["open"] > 0)
        & (serie["high"] > 0)
        & (serie["low"] > 0)
        & (serie["close"] > 0)
        & (serie["volume"] >= 0)
    )
    serie = serie.loc[valida].copy()

    if serie.empty:
        raise RuntimeError(f"{ativo}: serie certificada ausente ou vazia no MongoDB.")

    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    serie.to_csv(arquivo, index=False)

    print(
        f"[{posicao:02d}/{len(ATIVOS)}] {ativo} | "
        f"{len(serie)} candles | "
        f"{serie['timestamp'].min().date()} -> {serie['timestamp'].max().date()} | "
        f"{arquivo.relative_to(RAIZ_PROJETO)}"
    )

print("Series certificadas exportadas com sucesso.")
print("Agora execute: python backtest.py")
