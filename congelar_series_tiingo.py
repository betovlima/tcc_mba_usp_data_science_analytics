"""Congela uma fotografia reproduzivel das series EOD brutas da Tiingo.

O script consulta a Tiingo uma unica vez por ativo e grava um CSV local com
OHLCV bruto, dividendo em dinheiro e fator de split. O backtest passa a usar
somente esses arquivos, sem depender da API durante o experimento.

Campos ajustados da Tiingo nao sao gravados nem usados.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"
ARQUIVO_MANIFESTO = DIRETORIO_SERIES / "manifesto_tiingo.json"
URL_BASE_TIINGO = "https://api.tiingo.com/tiingo/daily"

COLUNAS_SNAPSHOT = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dividendo",
    "fator_split",
]


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


# %% 1 - Credencial da Tiingo
candidatos_env = [
    RAIZ_PROJETO / ".env",
    Path.cwd() / ".env",
    RAIZ_PROJETO.parent / ".env",
]

arquivo_env = next(
    (caminho.resolve() for caminho in candidatos_env if caminho.resolve().exists()),
    None,
)
if arquivo_env is None:
    raise RuntimeError("Arquivo .env nao encontrado.")

load_dotenv(arquivo_env, override=True)
valores_env = {
    str(nome): str(valor or "").strip()
    for nome, valor in dotenv_values(arquivo_env).items()
}

token_tiingo = str(
    valores_env.get("TIINGO_API_KEY")
    or valores_env.get("TIINGO_TOKEN")
    or os.getenv("TIINGO_API_KEY")
    or os.getenv("TIINGO_TOKEN")
    or ""
).strip()

if not token_tiingo:
    raise RuntimeError(
        "Token da Tiingo nao encontrado. Defina TIINGO_API_KEY no arquivo .env."
    )

sessao_tiingo = requests.Session()
sessao_tiingo.headers.update(
    {
        "Content-Type": "application/json",
        "Authorization": f"Token {token_tiingo}",
    }
)


# %% 2 - Download completo para memoria
registrar("Congelando fotografia Tiingo EOD RAW")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar("Os campos ajustados da Tiingo nao serao armazenados")

snapshot_por_ativo: dict[str, pd.DataFrame] = {}
resumo_ativos: list[dict[str, object]] = []
total_splits = 0
total_dividendos = 0

for posicao, ativo in enumerate(ATIVOS, start=1):
    resposta = sessao_tiingo.get(
        f"{URL_BASE_TIINGO}/{ativo}/prices",
        params={
            "startDate": DATA_INICIO,
            "endDate": DATA_FIM,
            "resampleFreq": "daily",
        },
        timeout=60,
    )

    try:
        resposta.raise_for_status()
    except requests.RequestException as erro:
        detalhe = str(resposta.text or "").strip()
        raise RuntimeError(
            f"Falha ao consultar {ativo} na Tiingo: {erro}"
            + (f" | resposta={detalhe[:300]}" if detalhe else "")
        ) from erro

    dados = resposta.json()
    if not isinstance(dados, list) or not dados:
        raise RuntimeError(f"A Tiingo nao retornou dados para {ativo}.")

    tabela = pd.DataFrame(dados)
    colunas_obrigatorias = ["date", "open", "high", "low", "close", "volume"]
    ausentes = [coluna for coluna in colunas_obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes na resposta da Tiingo: "
            + ", ".join(ausentes)
        )

    serie = tabela[colunas_obrigatorias].copy()
    serie = serie.rename(columns={"date": "timestamp"})
    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True, errors="coerce")

    for coluna in ("open", "high", "low", "close", "volume"):
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie["dividendo"] = (
        pd.to_numeric(tabela["divCash"], errors="coerce").fillna(0.0)
        if "divCash" in tabela.columns
        else 0.0
    )
    serie["fator_split"] = (
        pd.to_numeric(tabela["splitFactor"], errors="coerce").fillna(1.0)
        if "splitFactor" in tabela.columns
        else 1.0
    )

    serie = serie.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    serie = serie.sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")

    valores_validos = (
        (serie["open"] > 0)
        & (serie["high"] > 0)
        & (serie["low"] > 0)
        & (serie["close"] > 0)
        & (serie["volume"] >= 0)
        & (serie["fator_split"] > 0)
    )
    serie = serie.loc[valores_validos, COLUNAS_SNAPSHOT].copy()

    if serie.empty:
        raise RuntimeError(f"{ativo}: serie historica vazia depois da validacao.")

    quantidade_splits = int((serie["fator_split"] != 1.0).sum())
    quantidade_dividendos = int((serie["dividendo"] != 0.0).sum())
    total_splits += quantidade_splits
    total_dividendos += quantidade_dividendos

    snapshot_por_ativo[ativo] = serie
    resumo_ativos.append(
        {
            "ativo": ativo,
            "candles": len(serie),
            "inicio": serie["timestamp"].min().date().isoformat(),
            "fim": serie["timestamp"].max().date().isoformat(),
            "splits": quantidade_splits,
            "dividendos": quantidade_dividendos,
        }
    )

    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"{len(serie)} candles | "
        f"{serie['timestamp'].min().date()} -> {serie['timestamp'].max().date()} | "
        f"splits={quantidade_splits} | dividendos={quantidade_dividendos}"
    )

if len(snapshot_por_ativo) != len(ATIVOS):
    raise RuntimeError(
        f"Esperava {len(ATIVOS)} series; foram carregadas {len(snapshot_por_ativo)}."
    )


# %% 3 - Gravacao atomica da fotografia local
DIRETORIO_SERIES.mkdir(parents=True, exist_ok=True)

for ativo, serie in snapshot_por_ativo.items():
    serie.to_csv(DIRETORIO_SERIES / f"{ativo}.csv", index=False)

manifesto = {
    "fonte": "Tiingo EOD",
    "tipo_preco": "raw",
    "frequencia": "daily",
    "data_congelamento_utc": datetime.now(timezone.utc).isoformat(),
    "periodo_inicio": DATA_INICIO,
    "periodo_fim": DATA_FIM,
    "quantidade_ativos": len(ATIVOS),
    "ativos": list(ATIVOS),
    "colunas": COLUNAS_SNAPSHOT,
    "campos_ajustados_usados": False,
    "eventos_corporativos_aplicados": False,
    "total_splits": total_splits,
    "total_dividendos": total_dividendos,
    "resumo_ativos": resumo_ativos,
}

ARQUIVO_MANIFESTO.write_text(
    json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

registrar(
    f"Snapshot congelado: {len(ATIVOS)} ativos | "
    f"splits={total_splits} | dividendos={total_dividendos}"
)
registrar(f"Diretorio: {DIRETORIO_SERIES}")
registrar("Agora execute: python backtest.py")
