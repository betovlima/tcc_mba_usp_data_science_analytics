"""Backtest academico reproduzivel para o TCC MBA USP.

As series historicas OHLCV sao obtidas diretamente da Tiingo no inicio da
execucao e permanecem em memoria ate o fim do processamento. Para este teste,
o modelo usa somente os campos brutos open, high, low, close e volume. Os
campos ajustados disponibilizados pela Tiingo nao entram no treinamento.

Features, targets, folds walk-forward, treinamento LightGBM, politica de
rotacao, operacoes, curva de capital e metricas sao reconstruidos a cada
execucao.

O arquivo e organizado em celulas Spyder ``# %%``. F5 executa o script completo;
Ctrl+Enter executa somente a celula atual, mantendo as variaveis no namespace
para inspecao no Variable Explorer.

Nao sao lidos Strategy documents, modelos treinados, previsoes persistidas,
resultados anteriores ou configuracoes de runtime do Market Cycle Trader.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv

from tcc_engine.capital_rotation import run_rotation_models as executar_modelos_rotacao
from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import CONFIG as CONFIGURACAO
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO
from tcc_engine.execution import apply_slippage as aplicar_deslizamento
from tcc_engine.execution import calculate_reference_fees as calcular_taxas_referencia

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output"
URL_BASE_TIINGO = "https://api.tiingo.com/tiingo/daily"


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


def converter_json(valor: Any) -> Any:
    if isinstance(valor, (pd.Timestamp, datetime)):
        return pd.Timestamp(valor).isoformat()
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        numero = float(valor)
        return numero if np.isfinite(numero) else None
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, Path):
        return str(valor)
    return str(valor)


def registrar_progresso(percentual: float, etapa: str, execucoes_concluidas: int) -> None:
    registrar(f"[5/8] {percentual:5.1f}% | {etapa}")


def registrar_detalhe_tecnico(mensagem: str) -> None:
    registrar(f"[motor] {mensagem}")


# %% 1 - Inicio da execucao e credencial da Tiingo
inicio_execucao = time.perf_counter()

registrar("TCC MBA USP - backtest reconstruido passo a passo")
registrar("Entrada: Tiingo EOD RAW -> memoria -> motor")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")

candidatos_env = [
    RAIZ_PROJETO / ".env",
    Path.cwd() / ".env",
    RAIZ_PROJETO.parent / ".env",
]

arquivos_env_unicos: list[Path] = []
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

registrar(f"Arquivo .env: {arquivo_env}")
registrar(
    "Variavel do token Tiingo: "
    + (
        "TIINGO_API_KEY"
        if str(valores_env.get("TIINGO_API_KEY") or os.getenv("TIINGO_API_KEY") or "").strip()
        else "TIINGO_TOKEN"
        if str(valores_env.get("TIINGO_TOKEN") or os.getenv("TIINGO_TOKEN") or "").strip()
        else "nao encontrada"
    )
)

if not token_tiingo:
    raise RuntimeError(
        "Token da Tiingo nao encontrado. "
        "Defina TIINGO_API_KEY no arquivo .env."
    )

sessao_tiingo = requests.Session()
sessao_tiingo.headers.update(
    {
        "Content-Type": "application/json",
        "Authorization": f"Token {token_tiingo}",
    }
)


# %% 2 - Download das series historicas brutas diretamente para a memoria
registrar(f"[1/8] Baixando {len(ATIVOS)} series historicas da Tiingo")
registrar("[1/8] Fonte: Tiingo EOD | frequencia=diaria | precos=RAW")
registrar("[1/8] Campos ajustados nao serao usados pelo modelo")

series_historicas: dict[str, pd.DataFrame] = {}
eventos_corporativos: dict[str, pd.DataFrame] = {}
colunas_ohlcv = ["open", "high", "low", "close", "volume"]

for posicao, ativo in enumerate(ATIVOS, start=1):
    url = f"{URL_BASE_TIINGO}/{ativo}/prices"
    parametros = {
        "startDate": DATA_INICIO,
        "endDate": DATA_FIM,
        "resampleFreq": "daily",
    }

    try:
        resposta = sessao_tiingo.get(
            url,
            params=parametros,
            timeout=60,
        )
        resposta.raise_for_status()
    except requests.RequestException as erro:
        detalhe = ""
        if getattr(erro, "response", None) is not None:
            detalhe = str(erro.response.text or "").strip()
        raise RuntimeError(
            f"Falha ao consultar {ativo} na Tiingo: {erro}"
            + (f" | resposta={detalhe[:300]}" if detalhe else "")
        ) from erro

    dados = resposta.json()
    if not isinstance(dados, list) or not dados:
        raise RuntimeError(f"A Tiingo nao retornou dados para {ativo}.")

    serie_completa = pd.DataFrame(dados)
    serie_completa.columns = [str(coluna) for coluna in serie_completa.columns]

    colunas_obrigatorias = ["date", *colunas_ohlcv]
    ausentes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in serie_completa.columns
    ]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes na resposta da Tiingo: "
            + ", ".join(ausentes)
        )

    # Guardamos os eventos informados pela fonte somente para auditoria.
    # Eles nao alteram a serie usada no treinamento deste teste.
    dividendos = (
        pd.to_numeric(serie_completa["divCash"], errors="coerce").fillna(0.0)
        if "divCash" in serie_completa.columns
        else pd.Series(0.0, index=serie_completa.index)
    )
    fatores_split = (
        pd.to_numeric(serie_completa["splitFactor"], errors="coerce").fillna(1.0)
        if "splitFactor" in serie_completa.columns
        else pd.Series(1.0, index=serie_completa.index)
    )

    eventos = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                serie_completa["date"],
                utc=True,
                errors="coerce",
            ),
            "dividendo": dividendos,
            "fator_split": fatores_split,
        }
    )
    eventos = eventos.dropna(subset=["timestamp"])
    eventos = eventos.loc[
        (eventos["dividendo"] != 0.0)
        | (eventos["fator_split"] != 1.0)
    ].copy()
    eventos_corporativos[ativo] = eventos.reset_index(drop=True)

    serie = serie_completa[colunas_obrigatorias].copy()
    serie = serie.rename(columns={"date": "timestamp"})
    serie["timestamp"] = pd.to_datetime(
        serie["timestamp"],
        utc=True,
        errors="coerce",
    )
    serie = serie.dropna(subset=["timestamp"])
    serie = serie.set_index("timestamp").sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]

    for coluna in colunas_ohlcv:
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=colunas_ohlcv)
    valores_validos = (
        (serie["open"] > 0)
        & (serie["high"] > 0)
        & (serie["low"] > 0)
        & (serie["close"] > 0)
        & (serie["volume"] >= 0)
    )
    serie = serie.loc[valores_validos].copy()

    if serie.empty:
        raise RuntimeError(
            f"{ativo}: serie historica vazia depois da validacao."
        )

    series_historicas[ativo] = serie

    quantidade_splits = int(
        (eventos_corporativos[ativo]["fator_split"] != 1.0).sum()
    )
    quantidade_dividendos = int(
        (eventos_corporativos[ativo]["dividendo"] != 0.0).sum()
    )

    registrar(
        f"[1/8] {posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"{len(serie)} candles | "
        f"{serie.index.min().date()} -> {serie.index.max().date()} | "
        f"splits={quantidade_splits} | dividendos={quantidade_dividendos}"
    )

if len(series_historicas) != len(ATIVOS):
    raise RuntimeError(
        f"Esperava {len(ATIVOS)} series em memoria; "
        f"foram carregadas {len(series_historicas)}."
    )

total_splits = sum(
    int((eventos["fator_split"] != 1.0).sum())
    for eventos in eventos_corporativos.values()
)
total_dividendos = sum(
    int((eventos["dividendo"] != 0.0).sum())
    for eventos in eventos_corporativos.values()
)

registrar(
    f"[1/8] {len(series_historicas)} series RAW mantidas em memoria; "
    "nenhum CSV intermediario foi usado"
)
registrar(
    f"[1/8] Eventos apenas para auditoria: "
    f"splits={total_splits} | dividendos={total_dividendos}"
)


# %% 3 - Preparacao metodologica
# As features, targets e folds sao construidos pelo tcc_engine diretamente a
# partir das series_historicas mantidas em memoria. Os eventos corporativos
# coletados acima nao sao aplicados ao OHLCV neste teste.
registrar("[2/8] Construindo features tecnicas a partir do OHLCV RAW")
registrar(
    "[3/8] Construindo targets multi-horizonte: "
    + ", ".join(str(valor) for valor in CONFIGURACAO.rotation_target_horizons)
)
registrar("[4/8] Criando folds temporais walk-forward com purge")

contexto_experimento = {
    "assets": list(ATIVOS),
    "asset_count": len(ATIVOS),
    "history_start": DATA_INICIO,
    "history_end": DATA_FIM,
    "market_data_source": "tiingo_eod_memoria",
    "market_data_feed": "eod",
    "market_data_adjustment": "raw",
    "corporate_actions_available": True,
    "corporate_actions_applied": False,
    "split_event_count": total_splits,
    "dividend_event_count": total_dividendos,
    "model_family": CONFIGURACAO.research_model_family,
    "target_horizons": list(CONFIGURACAO.rotation_target_horizons),
    "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
    "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
    "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
    "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
    "purge_days": CONFIGURACAO.rotation_purge_days,
}


# %% 4 - Treinamento LightGBM, previsoes OOS e politica de rotacao
registrar("[5/8] Treinando LightGBM do zero em cada fold e ativo")

resultados = executar_modelos_rotacao(
    series_historicas,
    CONFIGURACAO,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=registrar_progresso,
    technical_log_callback=registrar_detalhe_tecnico,
)

if not resultados:
    raise RuntimeError("O motor nao retornou resultado.")
if len(resultados) != 1:
    raise RuntimeError(
        f"Esperava uma execucao; foram retornadas {len(resultados)}."
    )

resultado = resultados[0]

registrar("[6/8] Aplicando a politica de rotacao somente nas sessoes fora da amostra")
registrar("[7/8] Reconstruindo trades, custos e curva de capital")


# %% 5 - Objetos de resultado para inspecao no Spyder
previsoes = resultado.predictions.copy()
operacoes = resultado.trades.copy()
folds = list(resultado.metrics.get("walk_forward_folds") or [])
metricas = dict(resultado.metrics)

colunas_curva = [
    coluna
    for coluna in (
        "strategy_equity",
        "buy_hold_equity",
        "selected_asset",
        "selected_score",
        "decision_score",
        "trade_action",
        "trade_reason",
        "walk_forward_fold",
        "cash_weight",
        "market_exposure_weight",
        "assets_held",
    )
    if coluna in previsoes.columns
]
curva_capital = (
    previsoes[colunas_curva].copy()
    if colunas_curva
    else previsoes.copy()
)

tempo_total = time.perf_counter() - inicio_execucao

resultado_serializado = {
    "experiment": "mba_usp_reproducible_backtest",
    "input_source": "tiingo_eod_raw_memoria",
    **contexto_experimento,
    "walk_forward": {
        "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
        "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
        "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
        "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
        "purge_days": CONFIGURACAO.rotation_purge_days,
    },
    "elapsed_seconds": float(tempo_total),
    "metrics": metricas,
}


# %% 6 - Gravacao somente dos artefatos finais
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

curva_csv = curva_capital.copy()
if curva_csv.index.name is not None or not isinstance(curva_csv.index, pd.RangeIndex):
    curva_csv = curva_csv.reset_index()

operacoes_csv = operacoes.copy()
if operacoes_csv.index.name is not None or not isinstance(operacoes_csv.index, pd.RangeIndex):
    operacoes_csv = operacoes_csv.reset_index()

for tabela in (curva_csv, operacoes_csv):
    for coluna in tabela.columns:
        possui_valores_aninhados = tabela[coluna].map(
            lambda valor: isinstance(valor, (dict, list, tuple))
        ).any()
        if possui_valores_aninhados:
            tabela[coluna] = tabela[coluna].map(
                lambda valor: json.dumps(
                    valor,
                    ensure_ascii=False,
                    default=converter_json,
                )
                if isinstance(valor, (dict, list, tuple))
                else valor
            )

curva_csv.to_csv(DIRETORIO_RESULTADOS / "equity_curve.csv", index=False)
operacoes_csv.to_csv(DIRETORIO_RESULTADOS / "trades.csv", index=False)
pd.DataFrame(folds).to_csv(DIRETORIO_RESULTADOS / "folds.csv", index=False)

(DIRETORIO_RESULTADOS / "backtest_result.json").write_text(
    json.dumps(
        resultado_serializado,
        indent=2,
        ensure_ascii=False,
        default=converter_json,
    )
    + "\n",
    encoding="utf-8",
)
(DIRETORIO_RESULTADOS / "summary.txt").write_text(
    str(resultado.summary).rstrip() + "\n",
    encoding="utf-8",
)


# %% 7 - Metricas finais
registrar("[8/8] Calculando e salvando metricas finais")
registrar(f"Capital inicial : US$ {CONFIGURACAO.initial_capital:,.2f}")
registrar(f"Capital final   : US$ {float(metricas['strategy_ending_capital']):,.2f}")
registrar(f"Retorno         : {float(metricas['strategy_return']):.2%}")
registrar(f"CAGR            : {float(metricas['strategy_cagr']):.2%}")
registrar(f"Sharpe          : {float(metricas['strategy_sharpe']):.3f}")
registrar(f"Max Drawdown    : {float(metricas['strategy_maximum_drawdown']):.2%}")
registrar(f"Rotacoes        : {int(metricas.get('capital_rotations') or 0)}")
registrar(f"CASH days       : {int(metricas.get('cash_days') or 0)}")
registrar(f"Series em memoria: {len(series_historicas)}")
registrar(f"Tempo total     : {tempo_total:.2f}s")
registrar(f"Resultados      : {DIRETORIO_RESULTADOS}")
