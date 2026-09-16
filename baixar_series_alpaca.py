"""Baixa as series historicas da Alpaca e grava um CSV por ativo.

Este arquivo e executado somente quando for necessario criar ou atualizar a base
historica. O backtest nao acessa a Alpaca diretamente e nao depende de MongoDB.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"


# %% 1 - Credenciais da Alpaca
load_dotenv(RAIZ_PROJETO / ".env", override=False)

chave_api = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
    or ""
).strip()
segredo_api = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or ""
).strip()

if not chave_api or not segredo_api:
    raise RuntimeError(
        "Credenciais da Alpaca ausentes. Preencha ALPACA_API_KEY e "
        "ALPACA_SECRET_KEY no arquivo .env."
    )

cliente = StockHistoricalDataClient(
    api_key=chave_api,
    secret_key=segredo_api,
)

inicio = pd.Timestamp(DATA_INICIO, tz="UTC").to_pydatetime()
fim = (
    pd.Timestamp(DATA_FIM, tz="UTC")
    + pd.Timedelta(days=1)
    - pd.Timedelta(microseconds=1)
).to_pydatetime()

DIRETORIO_SERIES.mkdir(parents=True, exist_ok=True)

print("Alpaca - download das series historicas")
print(f"Periodo: {DATA_INICIO} -> {DATA_FIM}")
print("Timeframe: 1Day | feed: SIP | adjustment: all")
print(f"Ativos: {len(ATIVOS)}")


# %% 2 - Download e gravacao de um arquivo por ativo
for posicao, ativo in enumerate(ATIVOS, start=1):
    print(f"[{posicao:02d}/{len(ATIVOS)}] {ativo}")

    solicitacao = StockBarsRequest(
        symbol_or_symbols=ativo,
        timeframe=TimeFrame.Day,
        start=inicio,
        end=fim,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
        limit=10_000,
    )

    resposta = cliente.get_stock_bars(solicitacao)
    serie = resposta.df.copy()

    if serie.empty:
        raise RuntimeError(f"A Alpaca nao retornou dados para {ativo}.")

    if isinstance(serie.index, pd.MultiIndex):
        nomes_indice = list(serie.index.names)
        if "symbol" in nomes_indice:
            serie = serie.xs(ativo, level="symbol")

    serie = serie.reset_index()
    serie.columns = [str(coluna).lower() for coluna in serie.columns]

    if "symbol" in serie.columns:
        serie = serie.drop(columns=["symbol"])

    colunas = ["timestamp", "open", "high", "low", "close", "volume"]
    ausentes = [coluna for coluna in colunas if coluna not in serie.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes na resposta da Alpaca: {', '.join(ausentes)}"
        )

    serie = serie[colunas].copy()
    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True)
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
        raise RuntimeError(f"{ativo}: serie vazia depois da validacao.")

    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    serie.to_csv(arquivo, index=False)

    print(
        f"    {len(serie)} candles | "
        f"{serie['timestamp'].min().date()} -> {serie['timestamp'].max().date()} | "
        f"{arquivo.relative_to(RAIZ_PROJETO)}"
    )

print("Series historicas gravadas com sucesso.")
