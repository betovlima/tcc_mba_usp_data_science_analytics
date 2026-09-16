"""Congela os desdobramentos reais informados pela API de Corporate Actions da Tiingo.

O endpoint EOD da Tiingo possui ``splitFactor``, mas a propria documentacao
informa que esse campo tambem pode representar distribuicoes. Por isso ele nao
e usado para normalizar a serie do modelo.

Este script consulta o endpoint especifico de splits/desdobramentos e grava uma
fotografia local separada. O backtest usa somente eventos ativos, na data em que
ocorrem, para remover a ruptura mecanica do split sem reescrever o passado.
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
DIRETORIO_DESDOBRAMENTOS = RAIZ_PROJETO / "dados" / "desdobramentos"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_desdobramentos_tiingo.json"
ARQUIVO_PROGRESSO = DIRETORIO_DESDOBRAMENTOS / ".tiingo_desdobramentos_parcial.json"
URL_BASE_DESDOBRAMENTOS = "https://api.tiingo.com/tiingo/corporate-actions"

COLUNAS_DESDOBRAMENTOS = [
    "timestamp",
    "split_de",
    "split_para",
    "fator_split",
    "status",
]


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


def carregar_progresso() -> set[str]:
    if not ARQUIVO_PROGRESSO.exists():
        return set()
    try:
        progresso = json.loads(ARQUIVO_PROGRESSO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if (
        progresso.get("fonte") != "Tiingo Corporate Actions Splits"
        or progresso.get("periodo_inicio") != DATA_INICIO
        or progresso.get("periodo_fim") != DATA_FIM
    ):
        return set()
    return {str(ativo).upper() for ativo in progresso.get("ativos_concluidos") or []}


def salvar_progresso(ativos_concluidos: set[str]) -> None:
    DIRETORIO_DESDOBRAMENTOS.mkdir(parents=True, exist_ok=True)
    progresso = {
        "fonte": "Tiingo Corporate Actions Splits",
        "periodo_inicio": DATA_INICIO,
        "periodo_fim": DATA_FIM,
        "atualizado_em_utc": datetime.now(timezone.utc).isoformat(),
        "ativos_concluidos": [ativo for ativo in ATIVOS if ativo in ativos_concluidos],
    }
    ARQUIVO_PROGRESSO.write_text(
        json.dumps(progresso, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ler_desdobramentos(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_DESDOBRAMENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"{ativo}: arquivo de desdobramentos nao encontrado: {arquivo}")
    tabela = pd.read_csv(arquivo)
    ausentes = [coluna for coluna in COLUNAS_DESDOBRAMENTOS if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes no arquivo de desdobramentos: "
            + ", ".join(ausentes)
        )
    if tabela.empty:
        return pd.DataFrame(columns=COLUNAS_DESDOBRAMENTOS)
    tabela = tabela[COLUNAS_DESDOBRAMENTOS].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in ("split_de", "split_para", "fator_split"):
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela["status"] = tabela["status"].astype(str).str.lower().str.strip()
    tabela = tabela.dropna(subset=["timestamp", "split_de", "split_para", "fator_split"])
    tabela = tabela.loc[
        (tabela["split_de"] > 0)
        & (tabela["split_para"] > 0)
        & (tabela["fator_split"] > 0)
        & (tabela["status"] == "a")
    ].copy()
    tabela = tabela.sort_values("timestamp").drop_duplicates(
        subset=["timestamp", "split_de", "split_para"], keep="last"
    )
    return tabela.reset_index(drop=True)


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


# %% 2 - Download dos desdobramentos reais
DIRETORIO_DESDOBRAMENTOS.mkdir(parents=True, exist_ok=True)
ativos_concluidos = carregar_progresso()

registrar("Congelando desdobramentos da Tiingo Corporate Actions")
registrar(f"Periodo do experimento: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar("O splitFactor do endpoint EOD nao sera usado para normalizar o modelo")

if ativos_concluidos:
    registrar(f"Retomando execucao: {len(ativos_concluidos)} ativo(s) ja concluidos")

inicio = pd.Timestamp(DATA_INICIO, tz="UTC")
fim = pd.Timestamp(DATA_FIM, tz="UTC")

for posicao, ativo in enumerate(ATIVOS, start=1):
    if ativo in ativos_concluidos:
        existentes = ler_desdobramentos(ativo)
        registrar(
            f"{posicao:02d}/{len(ATIVOS)} {ativo} | ja congelado | "
            f"desdobramentos={len(existentes)}"
        )
        continue

    resposta = sessao_tiingo.get(
        f"{URL_BASE_DESDOBRAMENTOS}/{ativo}/splits",
        timeout=60,
    )

    if resposta.status_code == 429:
        salvar_progresso(ativos_concluidos)
        registrar("Limite horario da Tiingo atingido (HTTP 429).")
        registrar("Os ativos ja concluidos foram preservados; execute novamente mais tarde.")
        detalhe = str(resposta.text or "").strip()
        if detalhe:
            registrar(f"Tiingo: {detalhe[:300]}")
        raise SystemExit(2)

    try:
        resposta.raise_for_status()
    except requests.RequestException as erro:
        salvar_progresso(ativos_concluidos)
        detalhe = str(resposta.text or "").strip()
        raise RuntimeError(
            f"Falha ao consultar desdobramentos de {ativo}: {erro}"
            + (f" | resposta={detalhe[:300]}" if detalhe else "")
        ) from erro

    dados = resposta.json()
    if not isinstance(dados, list):
        salvar_progresso(ativos_concluidos)
        raise RuntimeError(f"Resposta inesperada da Tiingo para {ativo}: esperado uma lista.")

    linhas: list[dict[str, object]] = []
    for item in dados:
        if not isinstance(item, dict):
            continue
        timestamp = pd.to_datetime(item.get("exDate"), utc=True, errors="coerce")
        if pd.isna(timestamp) or timestamp < inicio or timestamp > fim:
            continue

        status = str(item.get("splitStatus") or "").lower().strip()
        if status != "a":
            continue

        split_de = pd.to_numeric(item.get("splitFrom"), errors="coerce")
        split_para = pd.to_numeric(item.get("splitTo"), errors="coerce")
        fator_split = pd.to_numeric(item.get("splitFactor"), errors="coerce")
        if not all(pd.notna(valor) and float(valor) > 0 for valor in (split_de, split_para, fator_split)):
            continue

        fator_calculado = float(split_para) / float(split_de)
        if not abs(float(fator_split) - fator_calculado) <= max(1e-10, abs(fator_calculado) * 1e-8):
            raise RuntimeError(
                f"{ativo}: fator de split inconsistente em {timestamp.date()}: "
                f"API={float(fator_split)} calculado={fator_calculado}"
            )

        linhas.append(
            {
                "timestamp": timestamp,
                "split_de": float(split_de),
                "split_para": float(split_para),
                "fator_split": float(fator_split),
                "status": status,
            }
        )

    tabela = pd.DataFrame(linhas, columns=COLUNAS_DESDOBRAMENTOS)
    if not tabela.empty:
        tabela = tabela.sort_values("timestamp").drop_duplicates(
            subset=["timestamp", "split_de", "split_para"], keep="last"
        )

    arquivo_destino = DIRETORIO_DESDOBRAMENTOS / f"{ativo}.csv"
    arquivo_temporario = DIRETORIO_DESDOBRAMENTOS / f".{ativo}.csv.tmp"
    tabela.to_csv(arquivo_temporario, index=False)
    arquivo_temporario.replace(arquivo_destino)

    ativos_concluidos.add(ativo)
    salvar_progresso(ativos_concluidos)
    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"desdobramentos={len(tabela)}"
    )


# %% 3 - Validacao final e manifesto
resumo_ativos: list[dict[str, object]] = []
total_desdobramentos = 0

for ativo in ATIVOS:
    tabela = ler_desdobramentos(ativo)
    total_desdobramentos += len(tabela)
    resumo_ativos.append(
        {
            "ativo": ativo,
            "desdobramentos": len(tabela),
            "eventos": [
                {
                    "data": pd.Timestamp(linha.timestamp).date().isoformat(),
                    "split_de": float(linha.split_de),
                    "split_para": float(linha.split_para),
                    "fator_split": float(linha.fator_split),
                }
                for linha in tabela.itertuples(index=False)
            ],
        }
    )

manifesto = {
    "fonte": "Tiingo Corporate Actions Splits",
    "endpoint": "/tiingo/corporate-actions/<ticker>/splits",
    "data_congelamento_utc": datetime.now(timezone.utc).isoformat(),
    "periodo_inicio": DATA_INICIO,
    "periodo_fim": DATA_FIM,
    "quantidade_ativos": len(ATIVOS),
    "ativos": list(ATIVOS),
    "somente_status_ativo": True,
    "total_desdobramentos": total_desdobramentos,
    "resumo_ativos": resumo_ativos,
}
ARQUIVO_MANIFESTO.write_text(
    json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

if ARQUIVO_PROGRESSO.exists():
    ARQUIVO_PROGRESSO.unlink()

registrar(f"Snapshot concluido: {total_desdobramentos} desdobramento(s) ativo(s)")
registrar(f"Diretorio: {DIRETORIO_DESDOBRAMENTOS}")
registrar("Agora execute: python backtest.py")
