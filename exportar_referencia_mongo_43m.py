"""Exporta, sem alterar o banco, a referencia historica que reproduz os 43M.

O objetivo deste script nao e alimentar o backtest atual. Ele cria uma copia
separada das series armazenadas no Mongo para comparacao direta com o snapshot
Tiingo congelado.

A leitura e somente leitura. Nenhum documento do MongoDB e criado, alterado ou
apagado.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import CONFIG as CONFIGURACAO
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_REFERENCIA = RAIZ_PROJETO / "dados" / "referencia_mongo_43m"
ARQUIVO_MANIFESTO = DIRETORIO_REFERENCIA / "manifesto_mongo_43m.json"
ARQUIVO_ENV = RAIZ_PROJETO / ".env"

URI_MONGO_PADRAO = "mongodb://localhost:27017"
BANCO_MONGO_PADRAO = "extrema_backtest"
COLECAO_MERCADO = "alpaca_market_bars"
COLUNAS = ["timestamp", "open", "high", "low", "close", "volume"]


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


# %% 1 - Conexao somente leitura
load_dotenv(ARQUIVO_ENV, override=True)

uri_mongo = str(os.getenv("TCC_MONGO_URI") or URI_MONGO_PADRAO).strip()
banco_mongo = str(os.getenv("TCC_MONGO_DATABASE") or BANCO_MONGO_PADRAO).strip()

inicio = pd.Timestamp(DATA_INICIO, tz="UTC").to_pydatetime()
fim_exclusivo = (
    pd.Timestamp(DATA_FIM, tz="UTC") + pd.Timedelta(days=1)
).to_pydatetime()

registrar("Exportando referencia historica Mongo que reproduz os 43M")
registrar(f"Banco/colecao: {banco_mongo}.{COLECAO_MERCADO}")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar(
    "Filtro historico: "
    f"interval={CONFIGURACAO.timeframe} | "
    f"feed={CONFIGURACAO.alpaca_historical_feed} | "
    f"adjustment={CONFIGURACAO.alpaca_adjustment}"
)
registrar("Operacao: somente leitura")

cliente = MongoClient(
    uri_mongo,
    serverSelectionTimeoutMS=10_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=60_000,
    retryWrites=False,
)

try:
    cliente.admin.command("ping")
    colecao = cliente[banco_mongo][COLECAO_MERCADO]
    registros = list(
        colecao.find(
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
        "Nenhum candle foi encontrado com o filtro historico certificado."
    )

historico = pd.DataFrame(registros)
historico["symbol"] = historico["symbol"].astype(str).str.upper()
historico["timestamp"] = pd.to_datetime(historico["timestamp"], utc=True)

# %% 2 - Congelamento separado da referencia
DIRETORIO_REFERENCIA.mkdir(parents=True, exist_ok=True)
resumo_ativos: list[dict[str, object]] = []

for posicao, ativo in enumerate(ATIVOS, start=1):
    serie = historico.loc[historico["symbol"].eq(ativo), COLUNAS].copy()
    serie = serie.sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")

    for coluna in ("open", "high", "low", "close", "volume"):
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=COLUNAS)
    serie = serie.loc[
        (serie["open"] > 0)
        & (serie["high"] > 0)
        & (serie["low"] > 0)
        & (serie["close"] > 0)
        & (serie["volume"] >= 0)
    ].copy()

    if serie.empty:
        raise RuntimeError(f"{ativo}: serie de referencia ausente ou vazia.")

    arquivo = DIRETORIO_REFERENCIA / f"{ativo}.csv"
    serie.to_csv(arquivo, index=False)

    resumo = {
        "ativo": ativo,
        "candles": len(serie),
        "inicio": serie["timestamp"].min().date().isoformat(),
        "fim": serie["timestamp"].max().date().isoformat(),
    }
    resumo_ativos.append(resumo)
    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | {len(serie)} candles | "
        f"{resumo['inicio']} -> {resumo['fim']}"
    )

manifesto = {
    "descricao": "Referencia historica Mongo que reproduz o backtest de aproximadamente 43M",
    "data_exportacao_utc": datetime.now(timezone.utc).isoformat(),
    "banco": banco_mongo,
    "colecao": COLECAO_MERCADO,
    "periodo_inicio": DATA_INICIO,
    "periodo_fim": DATA_FIM,
    "intervalo": CONFIGURACAO.timeframe,
    "feed": CONFIGURACAO.alpaca_historical_feed,
    "ajuste": CONFIGURACAO.alpaca_adjustment,
    "quantidade_ativos": len(ATIVOS),
    "ativos": list(ATIVOS),
    "resumo_ativos": resumo_ativos,
}
ARQUIVO_MANIFESTO.write_text(
    json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

registrar(f"Referencia congelada em: {DIRETORIO_REFERENCIA}")
registrar("Proximo passo: python comparar_referencia_mongo_tiingo.py")
