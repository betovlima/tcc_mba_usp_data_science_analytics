"""Métricas financeiras usadas na avaliação do backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_anualizado(curva: pd.Series, periodos_ano: float = 252.0) -> float:
    """Calcula o índice de Sharpe a partir dos retornos da curva de capital."""
    retornos = curva.pct_change().dropna()
    if retornos.empty or float(retornos.std()) <= 0:
        return float("nan")
    return float(np.sqrt(periodos_ano) * retornos.mean() / retornos.std())


def drawdown_maximo(curva: pd.Series) -> float:
    """Calcula o maior rebaixamento percentual da curva."""
    if curva.empty:
        return float("nan")
    pico = curva.cummax()
    return float((curva / pico - 1.0).min())


def cagr(curva: pd.Series, capital_inicial: float) -> float:
    """Calcula a taxa anual composta usando o período calendário observado."""
    if len(curva) < 2:
        return float("nan")
    inicio = pd.Timestamp(curva.index[0])
    fim = pd.Timestamp(curva.index[-1])
    anos = max((fim - inicio).days / 365.25, 1 / 365.25)
    final = float(curva.iloc[-1])
    if capital_inicial <= 0 or final <= 0:
        return float("nan")
    return float((final / float(capital_inicial)) ** (1.0 / anos) - 1.0)


def retorno_geometrico_operacoes(operacoes: pd.DataFrame) -> float:
    """Retorna a média geométrica dos retornos das posições encerradas."""
    if operacoes.empty or "position_return" not in operacoes.columns:
        return float("nan")
    retornos = pd.to_numeric(
        operacoes.loc[operacoes["action"].isin(["SELL", "FINAL_SELL"]), "position_return"],
        errors="coerce",
    ).dropna()
    if retornos.empty:
        return float("nan")
    bruto = np.prod(1.0 + retornos.clip(lower=-0.999999))
    return float(bruto ** (1.0 / len(retornos)) - 1.0)
