"""Backtest academico reproduzivel para o TCC MBA USP.

A entrada do experimento e uma fotografia local e congelada das series EOD
brutas da Tiingo. O backtest nao consulta a internet e nao depende de token.
Cada execucao reconstrói features, targets, folds walk-forward, treinamento
LightGBM, politica de rotacao, operacoes, curva de capital e metricas.

Os eventos corporativos ficam preservados nos CSVs, mas ainda nao sao aplicados
ao OHLCV. Isso permite estudar depois um tratamento causal sem reescrever o
passado com eventos futuros.
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
ARQUIVO_MANIFESTO = DIRETORIO_SERIES / "manifesto_tiingo.json"
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output"

COLUNAS_OHLCV = ["open", "high", "low", "close", "volume"]


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


# %% 1 - Inicio da execucao e identificacao do snapshot
inicio_execucao = time.perf_counter()

registrar("TCC MBA USP - backtest reconstruido passo a passo")
registrar("Entrada: snapshot local Tiingo EOD RAW -> memoria -> motor")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")

if not DIRETORIO_SERIES.exists():
    raise RuntimeError(
        "Diretorio de series historicas nao encontrado. "
        "Execute primeiro: python congelar_series_tiingo.py"
    )

manifesto: dict[str, Any] = {}
if ARQUIVO_MANIFESTO.exists():
    manifesto = json.loads(ARQUIVO_MANIFESTO.read_text(encoding="utf-8"))
    registrar(
        "Snapshot: "
        + str(manifesto.get("data_congelamento_utc") or "data nao informada")
    )
else:
    registrar("Aviso: manifesto_tiingo.json nao encontrado; usando somente os CSVs.")


# %% 2 - Carregamento das series congeladas para memoria
registrar(f"[1/8] Carregando {len(ATIVOS)} series congeladas")
registrar("[1/8] Fonte: Tiingo EOD | frequencia=diaria | precos=RAW")
registrar("[1/8] Nenhuma consulta externa sera realizada")

series_historicas: dict[str, pd.DataFrame] = {}
eventos_corporativos: dict[str, pd.DataFrame] = {}

for posicao, ativo in enumerate(ATIVOS, start=1):
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(
            f"Snapshot incompleto: arquivo ausente para {ativo}: {arquivo}"
        )

    tabela = pd.read_csv(arquivo)

    colunas_obrigatorias = ["timestamp", *COLUNAS_OHLCV]
    ausentes = [coluna for coluna in colunas_obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes no snapshot: " + ", ".join(ausentes)
        )

    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in COLUNAS_OHLCV:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")

    tabela["dividendo"] = (
        pd.to_numeric(tabela["dividendo"], errors="coerce").fillna(0.0)
        if "dividendo" in tabela.columns
        else 0.0
    )
    tabela["fator_split"] = (
        pd.to_numeric(tabela["fator_split"], errors="coerce").fillna(1.0)
        if "fator_split" in tabela.columns
        else 1.0
    )

    tabela = tabela.dropna(subset=["timestamp", *COLUNAS_OHLCV])
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")

    valores_validos = (
        (tabela["open"] > 0)
        & (tabela["high"] > 0)
        & (tabela["low"] > 0)
        & (tabela["close"] > 0)
        & (tabela["volume"] >= 0)
    )
    tabela = tabela.loc[valores_validos].copy()

    if tabela.empty:
        raise RuntimeError(f"{ativo}: serie congelada vazia depois da validacao.")

    eventos = tabela.loc[
        (tabela["dividendo"] != 0.0) | (tabela["fator_split"] != 1.0),
        ["timestamp", "dividendo", "fator_split"],
    ].copy()
    eventos_corporativos[ativo] = eventos.reset_index(drop=True)

    serie = tabela.set_index("timestamp")[COLUNAS_OHLCV].copy()
    series_historicas[ativo] = serie

    registrar(
        f"[1/8] {posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"{len(serie)} candles | "
        f"{serie.index.min().date()} -> {serie.index.max().date()} | "
        f"splits={int((eventos['fator_split'] != 1.0).sum())} | "
        f"dividendos={int((eventos['dividendo'] != 0.0).sum())}"
    )

if len(series_historicas) != len(ATIVOS):
    raise RuntimeError(
        f"Esperava {len(ATIVOS)} series; foram carregadas {len(series_historicas)}."
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
    f"[1/8] {len(series_historicas)} series RAW carregadas do snapshot local"
)
registrar(
    f"[1/8] Eventos preservados, ainda nao aplicados: "
    f"splits={total_splits} | dividendos={total_dividendos}"
)


# %% 3 - Preparacao metodologica
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
    "market_data_source": "tiingo_eod_snapshot_local",
    "market_data_feed": "eod",
    "market_data_adjustment": "raw",
    "market_data_snapshot_frozen": True,
    "market_data_snapshot_created_at": manifesto.get("data_congelamento_utc"),
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
    "input_source": "tiingo_eod_raw_snapshot_local",
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

# Remove artefato antigo que nao e produzido por esta versao.
arquivo_residual = DIRETORIO_RESULTADOS / "market_data.csv"
if arquivo_residual.exists():
    arquivo_residual.unlink()

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
