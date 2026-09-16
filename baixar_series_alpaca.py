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
from dotenv import dotenv_values, load_dotenv

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"


# %% 1 - Credenciais da Alpaca
candidatos_env = [
    RAIZ_PROJETO / ".env",
    Path.cwd() / ".env",
    RAIZ_PROJETO.parent / ".env",
]

arquivos_env_unicos = []
for candidato in candidatos_env:
    candidato = candidato.resolve()
    if candidato not in arquivos_env_unicos:
        arquivos_env_unicos.append(candidato)

arquivo_env = next(
    (candidato for candidato in arquivos_env_unicos if candidato.exists()),
    None,
)

if arquivo_env is None:
    caminhos = "\n".join(f"- {caminho}" for caminho in arquivos_env_unicos)
    raise RuntimeError(
        "Arquivo .env nao encontrado. Foram verificados:\n" + caminhos
    )

# No Spyder podem existir variaveis de ambiente antigas ou vazias no processo.
# O override=True garante que o arquivo .env encontrado seja a fonte usada nesta execucao.
load_dotenv(arquivo_env, override=True)
valores_env = {
    str(nome): str(valor or "").strip()
    for nome, valor in dotenv_values(arquivo_env).items()
}


def ler_credencial(*nomes: str) -> tuple[str, str | None]:
    for nome in nomes:
        valor = valores_env.get(nome) or os.getenv(nome) or ""
        valor = str(valor).strip()
        if valor:
            return valor, nome
    return "", None


chave_api, nome_chave_api = ler_credencial(
    "ALPACA_API_KEY",
    "ALPACA_API_KEY_ID",
    "APCA_API_KEY_ID",
)
segredo_api, nome_segredo_api = ler_credencial(
    "ALPACA_SECRET_KEY",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_SECRET_KEY",
)

print(f"Arquivo .env: {arquivo_env}")
print(f"Variavel da chave: {nome_chave_api or 'nao encontrada'}")
print(f"Variavel do segredo: {nome_segredo_api or 'nao encontrada'}")

if not chave_api or not segredo_api:
    nomes_encontrados = sorted(
        nome
        for nome, valor in valores_env.items()
        if valor and ("ALPACA" in nome.upper() or "APCA" in nome.upper())
    )
    encontrados = ", ".join(nomes_encontrados) if nomes_encontrados else "nenhuma"
    raise RuntimeError(
        "Credenciais da Alpaca nao foram reconhecidas no .env. "
        "Use um dos pares: "
        "ALPACA_API_KEY + ALPACA_SECRET_KEY, "
        "ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY, ou "
        "APCA_API_KEY_ID + APCA_API_SECRET_KEY. "
        f"Variaveis relacionadas encontradas: {encontrados}."
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
