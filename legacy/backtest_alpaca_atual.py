"""Backtest de controle sobre o snapshot Alpaca atual congelado.

Versao: alpaca-tiingo-audit-v1.0.0
Mantem exatamente o motor/configuracao do TCC; muda apenas a fonte de OHLCV.
"""
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

VERSAO = "alpaca-tiingo-audit-v1.0.0"
RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "referencia_alpaca_atual"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_referencia_alpaca_atual.json"
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "alpaca_atual_22m_control"
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
    registrar(f"[4/7] {percentual:5.1f}% | {etapa}")


def registrar_detalhe_tecnico(mensagem: str) -> None:
    registrar(f"[motor] {mensagem}")


inicio_execucao = time.perf_counter()
if not DIRETORIO_SERIES.exists() or not ARQUIVO_MANIFESTO.exists():
    raise RuntimeError("Snapshot Alpaca atual ausente. Execute primeiro: python congelar_referencia_alpaca_atual.py")
manifesto = json.loads(ARQUIVO_MANIFESTO.read_text(encoding="utf-8"))
if manifesto.get("feed") != "sip" or manifesto.get("adjustment") != "all":
    raise RuntimeError("Manifesto Alpaca nao corresponde a feed=SIP + adjustment=all.")

registrar("TCC MBA USP - controle Alpaca atual congelado")
registrar(f"Versao: {VERSAO}")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar(f"Snapshot: {manifesto.get('data_congelamento_utc')}")

series_historicas: dict[str, pd.DataFrame] = {}
registrar(f"[1/7] Carregando {len(ATIVOS)} series Alpaca congeladas")
for posicao, ativo in enumerate(ATIVOS, start=1):
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"Snapshot incompleto: {arquivo}")
    tabela = pd.read_csv(arquivo)
    obrigatorias = ["timestamp", *COLUNAS_OHLCV]
    ausentes = [coluna for coluna in obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(f"{ativo}: colunas ausentes: {', '.join(ausentes)}")
    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in COLUNAS_OHLCV:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela = tabela.dropna(subset=obrigatorias).sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")
    tabela = tabela.loc[
        (tabela["open"] > 0) & (tabela["high"] > 0) & (tabela["low"] > 0)
        & (tabela["close"] > 0) & (tabela["volume"] >= 0)
    ].copy()
    if tabela.empty:
        raise RuntimeError(f"{ativo}: serie vazia depois da validacao.")
    serie = tabela.set_index("timestamp")[COLUNAS_OHLCV].copy()
    series_historicas[ativo] = serie
    registrar(f"[1/7] {posicao:02d}/{len(ATIVOS)} {ativo} | {len(serie)} candles")

registrar("[2/7] Construindo features e targets com a configuracao atual")
registrar("[3/7] Criando folds temporais walk-forward com purge")
resultados = executar_modelos_rotacao(
    series_historicas,
    CONFIGURACAO,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=registrar_progresso,
    technical_log_callback=registrar_detalhe_tecnico,
)
if len(resultados) != 1:
    raise RuntimeError(f"Esperava uma execucao; recebidas={len(resultados)}")
resultado = resultados[0]
registrar("[5/7] Reconstruindo operacoes e curva de capital")

previsoes = resultado.predictions.copy()
operacoes = resultado.trades.copy()
metricas = dict(resultado.metrics)
folds = list(metricas.get("walk_forward_folds") or [])
colunas_curva = [
    coluna for coluna in (
        "strategy_equity", "buy_hold_equity", "selected_asset", "selected_score",
        "decision_score", "trade_action", "trade_reason", "walk_forward_fold",
        "cash_weight", "market_exposure_weight", "assets_held",
    ) if coluna in previsoes.columns
]
curva = previsoes[colunas_curva].copy() if colunas_curva else previsoes.copy()
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)


def preparar_csv(tabela: pd.DataFrame) -> pd.DataFrame:
    saida = tabela.copy()
    if saida.index.name is not None or not isinstance(saida.index, pd.RangeIndex):
        saida = saida.reset_index()
    for coluna in saida.columns:
        if saida[coluna].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            saida[coluna] = saida[coluna].map(
                lambda v: json.dumps(v, ensure_ascii=False, default=converter_json)
                if isinstance(v, (dict, list, tuple)) else v
            )
    return saida

preparar_csv(curva).to_csv(DIRETORIO_RESULTADOS / "equity_curve.csv", index=False)
preparar_csv(operacoes).to_csv(DIRETORIO_RESULTADOS / "trades.csv", index=False)
pd.DataFrame(folds).to_csv(DIRETORIO_RESULTADOS / "folds.csv", index=False)

tempo_total = time.perf_counter() - inicio_execucao
serializado = {
    "schema_version": 1,
    "script_version": VERSAO,
    "experiment": "alpaca_current_sip_adjustment_all_control",
    "input_source": "frozen_alpaca_current_snapshot",
    "market_data_feed": "sip",
    "market_data_adjustment": "all",
    "snapshot_created_at": manifesto.get("data_congelamento_utc"),
    "assets": list(ATIVOS),
    "history_start": DATA_INICIO,
    "history_end": DATA_FIM,
    "elapsed_seconds": tempo_total,
    "metrics": metricas,
}
(DIRETORIO_RESULTADOS / "backtest_result.json").write_text(
    json.dumps(serializado, indent=2, ensure_ascii=False, default=converter_json) + "\n",
    encoding="utf-8",
)
(DIRETORIO_RESULTADOS / "summary.txt").write_text(str(resultado.summary).rstrip() + "\n", encoding="utf-8")

registrar("[6/7] Artefatos gravados")
registrar("[7/7] Metricas finais")
registrar(f"Capital inicial : US$ {CONFIGURACAO.initial_capital:,.2f}")
registrar(f"Capital final   : US$ {float(metricas['strategy_ending_capital']):,.2f}")
registrar(f"Retorno         : {float(metricas['strategy_return']):.2%}")
registrar(f"CAGR            : {float(metricas['strategy_cagr']):.2%}")
registrar(f"Sharpe          : {float(metricas['strategy_sharpe']):.3f}")
registrar(f"Max Drawdown    : {float(metricas['strategy_maximum_drawdown']):.2%}")
registrar(f"Rotacoes        : {int(metricas.get('capital_rotations') or 0)}")
registrar(f"Tempo total     : {tempo_total:.2f}s")
registrar(f"Resultados      : {DIRETORIO_RESULTADOS}")
