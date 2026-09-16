"""Backtest academico reproduzivel para o TCC MBA USP.

A unica entrada externa do experimento e o conjunto de series historicas OHLCV
armazenadas em ``dados/series_historicas``, com um arquivo CSV por ativo.
Features, targets, folds walk-forward, treinamento LightGBM, politica de
rotacao, operacoes, curva de capital e metricas sao reconstruidos a cada
execucao.

O arquivo e organizado em celulas Spyder ``# %%``. F5 executa o script completo;
Ctrl+Enter executa somente a celula atual, mantendo as variaveis no namespace
para inspecao no Variable Explorer.

O backtest nao acessa MongoDB, nao baixa dados de mercado e nao le Strategy,
modelo treinado, previsao persistida ou resultado anterior do Market Cycle
Trader.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import run_rotation_models as executar_modelos_rotacao
from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import CONFIG as CONFIGURACAO
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO
from tcc_engine.execution import apply_slippage as aplicar_deslizamento
from tcc_engine.execution import calculate_reference_fees as calcular_taxas_referencia

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output"


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


# %% 1 - Inicio da execucao
inicio_execucao = time.perf_counter()

registrar("TCC MBA USP - backtest reconstruido passo a passo")
registrar("Entrada: um arquivo CSV historico por ativo")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar(f"Diretorio das series: {DIRETORIO_SERIES}")

if not DIRETORIO_SERIES.exists():
    raise RuntimeError(
        "Diretorio de series historicas inexistente. Execute "
        "'python baixar_series_alpaca.py' antes do backtest."
    )

inicio_periodo = pd.Timestamp(DATA_INICIO, tz="UTC")
fim_periodo_exclusivo = pd.Timestamp(DATA_FIM, tz="UTC") + pd.Timedelta(days=1)


# %% 2 - Carregamento e validacao das series historicas
registrar(f"[1/8] Carregando {len(ATIVOS)} arquivos de series historicas")

series_historicas: dict[str, pd.DataFrame] = {}
arquivos_series: dict[str, Path] = {}
colunas_ohlcv = ["open", "high", "low", "close", "volume"]

for posicao, ativo in enumerate(ATIVOS, start=1):
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    arquivos_series[ativo] = arquivo

    if not arquivo.exists():
        raise RuntimeError(
            f"Serie historica ausente para {ativo}: {arquivo}. "
            "Execute 'python baixar_series_alpaca.py'."
        )

    serie = pd.read_csv(arquivo)
    serie.columns = [str(coluna).lower() for coluna in serie.columns]

    colunas_obrigatorias = ["timestamp", *colunas_ohlcv]
    ausentes = [coluna for coluna in colunas_obrigatorias if coluna not in serie.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes em {arquivo.name}: {', '.join(ausentes)}"
        )

    serie = serie[colunas_obrigatorias].copy()
    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True, errors="coerce")
    serie = serie.dropna(subset=["timestamp"])
    serie = serie.set_index("timestamp").sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]

    for coluna in colunas_ohlcv:
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")

    serie = serie.dropna(subset=colunas_ohlcv)
    serie = serie.loc[
        (serie.index >= inicio_periodo)
        & (serie.index < fim_periodo_exclusivo)
    ].copy()

    valores_validos = (
        (serie["open"] > 0)
        & (serie["high"] > 0)
        & (serie["low"] > 0)
        & (serie["close"] > 0)
        & (serie["volume"] >= 0)
    )
    serie = serie.loc[valores_validos].copy()

    if serie.empty:
        raise RuntimeError(f"{ativo}: serie historica vazia depois da validacao.")

    series_historicas[ativo] = serie

    registrar(
        f"[1/8] {posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"{len(serie)} candles | "
        f"{serie.index.min().date()} -> {serie.index.max().date()}"
    )


# %% 3 - Preparacao metodologica
# As features, targets e folds sao construidos pelo tcc_engine a partir das
# series_historicas. Esta celula deixa explicito o protocolo antes do processamento.
registrar("[2/8] Construindo features tecnicas a partir do OHLCV")
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
    "market_data_source": "alpaca_sip_csv_por_ativo",
    "market_data_directory": "dados/series_historicas",
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
    "input_source": "alpaca_sip_csv_por_ativo",
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


# %% 6 - Gravacao dos artefatos
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
registrar(f"Tempo total     : {tempo_total:.2f}s")
registrar(f"Resultados      : {DIRETORIO_RESULTADOS}")
