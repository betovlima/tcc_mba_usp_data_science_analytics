"""Congela uma fotografia reproduzivel das series EOD brutas da Tiingo.

O script consulta a Tiingo uma unica vez por ativo e grava imediatamente cada
serie concluida em CSV. Se o limite horario da API for atingido, a execucao pode
ser retomada depois sem repetir os ativos ja baixados.

O snapshot contem OHLCV bruto, dividendo em dinheiro e fator de split. Campos
ajustados da Tiingo nao sao gravados nem usados.
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
ARQUIVO_PROGRESSO = DIRETORIO_SERIES / ".tiingo_congelamento_parcial.json"
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


def carregar_progresso() -> set[str]:
    if not ARQUIVO_PROGRESSO.exists():
        return set()

    try:
        progresso = json.loads(ARQUIVO_PROGRESSO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    if (
        progresso.get("fonte") != "Tiingo EOD"
        or progresso.get("periodo_inicio") != DATA_INICIO
        or progresso.get("periodo_fim") != DATA_FIM
    ):
        return set()

    concluidos = progresso.get("ativos_concluidos") or []
    return {str(ativo).upper() for ativo in concluidos}


def salvar_progresso(ativos_concluidos: set[str]) -> None:
    DIRETORIO_SERIES.mkdir(parents=True, exist_ok=True)
    progresso = {
        "fonte": "Tiingo EOD",
        "tipo_preco": "raw",
        "periodo_inicio": DATA_INICIO,
        "periodo_fim": DATA_FIM,
        "atualizado_em_utc": datetime.now(timezone.utc).isoformat(),
        "ativos_concluidos": [ativo for ativo in ATIVOS if ativo in ativos_concluidos],
    }
    ARQUIVO_PROGRESSO.write_text(
        json.dumps(progresso, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ler_serie_congelada(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(
            f"{ativo}: marcado como concluido, mas o arquivo nao existe: {arquivo}"
        )

    serie = pd.read_csv(arquivo)
    ausentes = [coluna for coluna in COLUNAS_SNAPSHOT if coluna not in serie.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: snapshot parcial invalido; colunas ausentes: "
            + ", ".join(ausentes)
        )

    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True, errors="coerce")
    for coluna in ("open", "high", "low", "close", "volume", "dividendo", "fator_split"):
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=COLUNAS_SNAPSHOT)
    serie = serie.sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")
    return serie[COLUNAS_SNAPSHOT].copy()


def resumir_serie(ativo: str, serie: pd.DataFrame) -> dict[str, object]:
    quantidade_splits = int((serie["fator_split"] != 1.0).sum())
    quantidade_dividendos = int((serie["dividendo"] != 0.0).sum())
    return {
        "ativo": ativo,
        "candles": len(serie),
        "inicio": serie["timestamp"].min().date().isoformat(),
        "fim": serie["timestamp"].max().date().isoformat(),
        "splits": quantidade_splits,
        "dividendos": quantidade_dividendos,
    }


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


# %% 2 - Download e gravacao progressiva do snapshot
DIRETORIO_SERIES.mkdir(parents=True, exist_ok=True)
ativos_concluidos = carregar_progresso()

registrar("Congelando fotografia Tiingo EOD RAW")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar("Os campos ajustados da Tiingo nao serao armazenados")
if ativos_concluidos:
    registrar(
        f"Retomando execucao: {len(ativos_concluidos)} ativo(s) ja congelado(s)"
    )

for posicao, ativo in enumerate(ATIVOS, start=1):
    if ativo in ativos_concluidos:
        serie_existente = ler_serie_congelada(ativo)
        registrar(
            f"{posicao:02d}/{len(ATIVOS)} {ativo} | ja congelado | "
            f"{len(serie_existente)} candles"
        )
        continue

    resposta = sessao_tiingo.get(
        f"{URL_BASE_TIINGO}/{ativo}/prices",
        params={
            "startDate": DATA_INICIO,
            "endDate": DATA_FIM,
            "resampleFreq": "daily",
        },
        timeout=60,
    )

    if resposta.status_code == 429:
        salvar_progresso(ativos_concluidos)
        detalhe = str(resposta.text or "").strip()
        registrar("Limite horario da Tiingo atingido (HTTP 429).")
        registrar(
            "Os ativos ja baixados foram preservados. "
            "Execute este mesmo script novamente quando a janela da API liberar."
        )
        if detalhe:
            registrar(f"Tiingo: {detalhe[:300]}")
        raise SystemExit(2)

    try:
        resposta.raise_for_status()
    except requests.RequestException as erro:
        salvar_progresso(ativos_concluidos)
        detalhe = str(resposta.text or "").strip()
        raise RuntimeError(
            f"Falha ao consultar {ativo} na Tiingo: {erro}"
            + (f" | resposta={detalhe[:300]}" if detalhe else "")
        ) from erro

    dados = resposta.json()
    if not isinstance(dados, list) or not dados:
        salvar_progresso(ativos_concluidos)
        raise RuntimeError(f"A Tiingo nao retornou dados para {ativo}.")

    tabela = pd.DataFrame(dados)
    colunas_obrigatorias = ["date", "open", "high", "low", "close", "volume"]
    ausentes = [coluna for coluna in colunas_obrigatorias if coluna not in tabela.columns]
    if ausentes:
        salvar_progresso(ativos_concluidos)
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
        salvar_progresso(ativos_concluidos)
        raise RuntimeError(f"{ativo}: serie historica vazia depois da validacao.")

    arquivo_destino = DIRETORIO_SERIES / f"{ativo}.csv"
    arquivo_temporario = DIRETORIO_SERIES / f".{ativo}.csv.tmp"
    serie.to_csv(arquivo_temporario, index=False)
    arquivo_temporario.replace(arquivo_destino)

    ativos_concluidos.add(ativo)
    salvar_progresso(ativos_concluidos)

    resumo = resumir_serie(ativo, serie)
    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"{resumo['candles']} candles | "
        f"{resumo['inicio']} -> {resumo['fim']} | "
        f"splits={resumo['splits']} | dividendos={resumo['dividendos']}"
    )


# %% 3 - Validacao final e manifesto
resumo_ativos: list[dict[str, object]] = []
total_splits = 0
total_dividendos = 0

for ativo in ATIVOS:
    serie = ler_serie_congelada(ativo)
    resumo = resumir_serie(ativo, serie)
    resumo_ativos.append(resumo)
    total_splits += int(resumo["splits"])
    total_dividendos += int(resumo["dividendos"])

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

if ARQUIVO_PROGRESSO.exists():
    ARQUIVO_PROGRESSO.unlink()

registrar(
    f"Snapshot congelado: {len(ATIVOS)} ativos | "
    f"splits={total_splits} | dividendos={total_dividendos}"
)
registrar(f"Diretorio: {DIRETORIO_SERIES}")
registrar("Agora execute: python backtest.py")
