"""Exporta uma única vez o histórico Alpaca certificado para um arquivo CSV.

Este utilitário existe apenas para retirar a dependência do MongoDB da execução
normal do TCC. Depois que ``dados/mercado_certificado.csv`` for criado, o
``backtest.py`` trabalha somente com esse arquivo.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from tcc_engine.configuracao import ATIVOS, DATA_FIM, DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_DADOS = RAIZ_PROJETO / "dados"
ARQUIVO_DESTINO = DIRETORIO_DADOS / "mercado_certificado.csv"

URI_MONGO = str(os.getenv("TCC_MONGO_URI") or "mongodb://localhost:27017").strip()
BANCO_MONGO = str(os.getenv("TCC_MONGO_DATABASE") or "extrema_backtest").strip()
COLECAO_MERCADO = "alpaca_market_bars"
INTERVALO = "1Day"
FONTE = "sip"
AJUSTE = "all"
COLUNAS_MERCADO = ["open", "high", "low", "close", "volume"]


def carregar_dados_certificados() -> pd.DataFrame:
    """Lê do Mongo local somente as barras Alpaca usadas pelo baseline certificado."""
    inicio = pd.Timestamp(DATA_INICIO, tz="UTC").to_pydatetime()
    fim = (pd.Timestamp(DATA_FIM, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()

    cliente = MongoClient(
        URI_MONGO,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        retryWrites=False,
    )
    try:
        cliente.admin.command("ping")
        colecao = cliente[BANCO_MONGO][COLECAO_MERCADO]
        registros = list(
            colecao.find(
                {
                    "symbol": {"$in": list(ATIVOS)},
                    "interval": INTERVALO,
                    "feed": FONTE,
                    "adjustment": AJUSTE,
                    "timestamp": {"$gte": inicio, "$lt": fim},
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
            "Nenhuma barra Alpaca foi encontrada no Mongo local. "
            "A coleção esperada é extrema_backtest.alpaca_market_bars."
        )

    dados = pd.DataFrame(registros)
    dados["symbol"] = dados["symbol"].astype(str).str.upper().str.strip()
    dados["timestamp"] = pd.to_datetime(dados["timestamp"], utc=True)

    for coluna in COLUNAS_MERCADO:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados = dados.dropna(subset=["symbol", "timestamp", *COLUNAS_MERCADO])
    dados = dados.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    dados = dados.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    faltantes = [
        ativo
        for ativo in ATIVOS
        if dados.loc[dados["symbol"].eq(ativo)].empty
    ]
    if faltantes:
        raise RuntimeError(
            "O histórico certificado está ausente para: " + ", ".join(faltantes)
        )

    return dados[["symbol", "timestamp", *COLUNAS_MERCADO]]


def principal() -> None:
    """Exporta as séries certificadas e mostra um resumo por ativo."""
    print("Exportando histórico Alpaca certificado")
    print(f"MongoDB: {BANCO_MONGO}.{COLECAO_MERCADO}")
    print(f"Critérios: intervalo={INTERVALO}, fonte={FONTE}, ajuste={AJUSTE}")

    dados = carregar_dados_certificados()
    DIRETORIO_DADOS.mkdir(parents=True, exist_ok=True)
    dados.to_csv(ARQUIVO_DESTINO, index=False)

    for posicao, ativo in enumerate(ATIVOS, start=1):
        tabela = dados.loc[dados["symbol"].eq(ativo)]
        print(
            f"[{posicao:02d}/{len(ATIVOS)}] {ativo}: "
            f"{len(tabela)} sessões | "
            f"{tabela['timestamp'].min().date()} -> {tabela['timestamp'].max().date()}"
        )

    print(f"Total: {len(dados):,} linhas")
    print(f"Arquivo: {ARQUIVO_DESTINO}")
    print("A partir de agora, execute normalmente: python backtest.py")


if __name__ == "__main__":
    principal()
