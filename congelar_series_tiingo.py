"""Congela uma fotografia reproduzivel dos dados EOD da Tiingo.

O snapshot separa deliberadamente duas entradas:

- dados/series_historicas/<ativo>.csv: somente OHLCV bruto, sem qualquer
  coluna ajustada, dividendo ou fator de split;
- dados/eventos_corporativos/<ativo>.csv: somente os eventos de dividendo e
  split informados pela Tiingo nas respectivas datas.

Assim, a serie de mercado permanece exatamente como negociada. Os eventos
corporativos ficam disponiveis para um tratamento causal posterior, mas nunca
reescrevem silenciosamente as observacoes historicas.
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
DIRETORIO_EVENTOS = RAIZ_PROJETO / "dados" / "eventos_corporativos"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_tiingo.json"
ARQUIVO_PROGRESSO = RAIZ_PROJETO / "dados" / ".tiingo_congelamento_parcial.json"
URL_BASE_TIINGO = "https://api.tiingo.com/tiingo/daily"

COLUNAS_SERIE_BRUTA = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]
COLUNAS_EVENTOS = ["timestamp", "dividendo", "fator_split"]


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
        or progresso.get("formato_snapshot") != "ohlcv_bruto_eventos_separados"
    ):
        return set()

    concluidos = progresso.get("ativos_concluidos") or []
    return {str(ativo).upper() for ativo in concluidos}


def salvar_progresso(ativos_concluidos: set[str]) -> None:
    ARQUIVO_PROGRESSO.parent.mkdir(parents=True, exist_ok=True)
    progresso = {
        "fonte": "Tiingo EOD",
        "tipo_preco": "raw",
        "formato_snapshot": "ohlcv_bruto_eventos_separados",
        "periodo_inicio": DATA_INICIO,
        "periodo_fim": DATA_FIM,
        "atualizado_em_utc": datetime.now(timezone.utc).isoformat(),
        "ativos_concluidos": [ativo for ativo in ATIVOS if ativo in ativos_concluidos],
    }
    ARQUIVO_PROGRESSO.write_text(
        json.dumps(progresso, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ler_serie_bruta(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"{ativo}: serie bruta nao encontrada: {arquivo}")

    serie = pd.read_csv(arquivo)
    ausentes = [coluna for coluna in COLUNAS_SERIE_BRUTA if coluna not in serie.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: serie bruta invalida; colunas ausentes: "
            + ", ".join(ausentes)
        )

    colunas_proibidas = {
        "adjopen",
        "adjhigh",
        "adjlow",
        "adjclose",
        "adjvolume",
        "dividendo",
        "fator_split",
        "divcash",
        "splitfactor",
    }
    encontradas = [
        coluna
        for coluna in serie.columns
        if str(coluna).lower() in colunas_proibidas
    ]
    if encontradas:
        raise RuntimeError(
            f"{ativo}: a serie bruta contem colunas que nao deveriam estar nela: "
            + ", ".join(encontradas)
        )

    serie = serie[COLUNAS_SERIE_BRUTA].copy()
    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True, errors="coerce")
    for coluna in ("open", "high", "low", "close", "volume"):
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=COLUNAS_SERIE_BRUTA)
    serie = serie.sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")
    return serie


def ler_eventos(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_EVENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"{ativo}: arquivo de eventos nao encontrado: {arquivo}")

    eventos = pd.read_csv(arquivo)
    ausentes = [coluna for coluna in COLUNAS_EVENTOS if coluna not in eventos.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: arquivo de eventos invalido; colunas ausentes: "
            + ", ".join(ausentes)
        )

    if eventos.empty:
        return eventos[COLUNAS_EVENTOS].copy()

    eventos = eventos[COLUNAS_EVENTOS].copy()
    eventos["timestamp"] = pd.to_datetime(eventos["timestamp"], utc=True, errors="coerce")
    eventos["dividendo"] = pd.to_numeric(eventos["dividendo"], errors="coerce").fillna(0.0)
    eventos["fator_split"] = pd.to_numeric(eventos["fator_split"], errors="coerce").fillna(1.0)
    eventos = eventos.dropna(subset=["timestamp"])
    eventos = eventos.sort_values("timestamp")
    eventos = eventos.drop_duplicates(subset=["timestamp"], keep="last")
    return eventos


def resumir_ativo(ativo: str) -> dict[str, object]:
    serie = ler_serie_bruta(ativo)
    eventos = ler_eventos(ativo)
    quantidade_splits = int((eventos["fator_split"] != 1.0).sum()) if not eventos.empty else 0
    quantidade_dividendos = int((eventos["dividendo"] != 0.0).sum()) if not eventos.empty else 0
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
DIRETORIO_EVENTOS.mkdir(parents=True, exist_ok=True)
ativos_concluidos = carregar_progresso()

registrar("Congelando fotografia Tiingo EOD")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar("Series: somente OHLCV bruto, sem ajustes e sem eventos")
registrar("Eventos: dividendos e splits gravados em diretorio separado")
registrar("Campos adjOpen/adjHigh/adjLow/adjClose/adjVolume nao serao armazenados")
if ativos_concluidos:
    registrar(f"Retomando execucao: {len(ativos_concluidos)} ativo(s) ja congelado(s)")

for posicao, ativo in enumerate(ATIVOS, start=1):
    if ativo in ativos_concluidos:
        resumo = resumir_ativo(ativo)
        registrar(
            f"{posicao:02d}/{len(ATIVOS)} {ativo} | ja congelado | "
            f"{resumo['candles']} candles"
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
            f"{ativo}: colunas ausentes na resposta da Tiingo: " + ", ".join(ausentes)
        )

    # Serie pura: somente os valores efetivamente negociados em cada data.
    serie = tabela[colunas_obrigatorias].copy()
    serie = serie.rename(columns={"date": "timestamp"})
    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True, errors="coerce")
    for coluna in ("open", "high", "low", "close", "volume"):
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=COLUNAS_SERIE_BRUTA)
    serie = serie.sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")
    valores_validos = (
        (serie["open"] > 0)
        & (serie["high"] > 0)
        & (serie["low"] > 0)
        & (serie["close"] > 0)
        & (serie["volume"] >= 0)
    )
    serie = serie.loc[valores_validos, COLUNAS_SERIE_BRUTA].copy()
    if serie.empty:
        salvar_progresso(ativos_concluidos)
        raise RuntimeError(f"{ativo}: serie historica vazia depois da validacao.")

    # Eventos ficam fora da serie de precos. Eles nao corrigem observacoes anteriores.
    dividendos = (
        pd.to_numeric(tabela["divCash"], errors="coerce").fillna(0.0)
        if "divCash" in tabela.columns
        else pd.Series(0.0, index=tabela.index)
    )
    fatores_split = (
        pd.to_numeric(tabela["splitFactor"], errors="coerce").fillna(1.0)
        if "splitFactor" in tabela.columns
        else pd.Series(1.0, index=tabela.index)
    )
    eventos = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(tabela["date"], utc=True, errors="coerce"),
            "dividendo": dividendos,
            "fator_split": fatores_split,
        }
    )
    eventos = eventos.dropna(subset=["timestamp"])
    eventos = eventos.loc[
        (eventos["dividendo"] != 0.0) | (eventos["fator_split"] != 1.0),
        COLUNAS_EVENTOS,
    ].copy()
    eventos = eventos.sort_values("timestamp")
    eventos = eventos.drop_duplicates(subset=["timestamp"], keep="last")

    arquivo_serie = DIRETORIO_SERIES / f"{ativo}.csv"
    arquivo_serie_tmp = DIRETORIO_SERIES / f".{ativo}.csv.tmp"
    serie.to_csv(arquivo_serie_tmp, index=False)
    arquivo_serie_tmp.replace(arquivo_serie)

    arquivo_eventos = DIRETORIO_EVENTOS / f"{ativo}.csv"
    arquivo_eventos_tmp = DIRETORIO_EVENTOS / f".{ativo}.csv.tmp"
    eventos.to_csv(arquivo_eventos_tmp, index=False)
    arquivo_eventos_tmp.replace(arquivo_eventos)

    ativos_concluidos.add(ativo)
    salvar_progresso(ativos_concluidos)

    resumo = resumir_ativo(ativo)
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
    resumo = resumir_ativo(ativo)
    resumo_ativos.append(resumo)
    total_splits += int(resumo["splits"])
    total_dividendos += int(resumo["dividendos"])

manifesto = {
    "fonte": "Tiingo EOD",
    "tipo_preco": "raw",
    "frequencia": "daily",
    "formato_snapshot": "ohlcv_bruto_eventos_separados",
    "data_congelamento_utc": datetime.now(timezone.utc).isoformat(),
    "periodo_inicio": DATA_INICIO,
    "periodo_fim": DATA_FIM,
    "quantidade_ativos": len(ATIVOS),
    "ativos": list(ATIVOS),
    "colunas_series_brutas": COLUNAS_SERIE_BRUTA,
    "colunas_eventos": COLUNAS_EVENTOS,
    "campos_ajustados_armazenados": False,
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
registrar(f"Series brutas: {DIRETORIO_SERIES}")
registrar(f"Eventos separados: {DIRETORIO_EVENTOS}")
registrar("Nenhum preco ajustado foi armazenado.")