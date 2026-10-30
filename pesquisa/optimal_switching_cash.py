"""Pesquisa independente: dois estados de exposicao (investido/CASH).

Modelo causal em tres estados de retorno/volatilidade por ativo. A cada fold,
apenas precos conhecidos ate o fim do ajuste entram na estimacao de retornos
open(t+1)->open(t+2) e da matriz de transicao. Programacao dinamica finita
avalia investir versus esperar em CASH, sem empregar o score LightGBM.
O Control segue responsavel por qual ativo comprar/manter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwitchingSettings:
    lookback: int = 20
    horizon: int = 10
    downside_weight: float = 1.0
    minimum_observations: int = 126

    def __post_init__(self) -> None:
        if self.lookback < 2 or self.horizon < 1 or self.minimum_observations < 30:
            raise ValueError("Janelas de Switching invalidas.")
        if not np.isfinite(self.downside_weight) or self.downside_weight < 0:
            raise ValueError("Penalizacao de risco invalida.")


def _market_state_feature(frame: pd.DataFrame, lookback: int) -> pd.Series:
    """Momentum padronizado pela volatilidade, apenas com fechamentos ate t."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns = np.log(close / close.shift(1))
    volatility = returns.rolling(lookback, min_periods=lookback).std(ddof=1)
    feature = np.log(close / close.shift(lookback)) / (
        np.sqrt(lookback) * volatility.replace(0.0, np.nan)
    )
    return feature.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class SwitchingModel:
    symbol: str
    fit_end: pd.Timestamp
    boundaries: tuple[float, float]
    transitions: np.ndarray
    expected_returns: np.ndarray
    downside_losses: np.ndarray
    fit_observations: int
    lookback: int

    def state_at(self, frame: pd.DataFrame, decision_date: pd.Timestamp) -> int:
        date = pd.Timestamp(decision_date)
        if date not in frame.index:
            raise ValueError(f"{self.symbol}: sessao de decisao ausente.")
        # Nao acessar o passado usando as linhas posteriores a decision_date.
        past = frame.loc[:date, ["close"]]
        value = float(_market_state_feature(past, self.lookback).iloc[-1])
        if not np.isfinite(value):
            raise ValueError(f"{self.symbol}: estado sem historico suficiente.")
        return int(np.searchsorted(np.asarray(self.boundaries), value, side="right"))

    def values(
        self,
        state: int,
        invested: bool,
        *,
        settings: SwitchingSettings,
        buy_cost: float,
        sell_cost: float,
    ) -> tuple[float, float]:
        """Bellman finito: Q(CASH), Q(INVEST), no estado atual."""
        if state not in (0, 1, 2):
            raise ValueError("Estado fora do dominio.")
        if min(buy_cost, sell_cost) < 0 or not np.isfinite(buy_cost + sell_cost):
            raise ValueError("Custos invalidos.")
        rewards = self.expected_returns - settings.downside_weight * self.downside_losses
        # eixo 0: regime; eixo 1: posicao anterior (CASH=0, investido=1).
        value = np.zeros((3, 2), dtype=float)
        for _ in range(settings.horizon - 1):
            future_cash = self.transitions @ value[:, 0]
            future_invested = self.transitions @ value[:, 1]
            nxt = np.empty_like(value)
            nxt[:, 0] = np.maximum(
                future_cash, -buy_cost + rewards + future_invested,
            )
            nxt[:, 1] = np.maximum(
                -sell_cost + future_cash, rewards + future_invested,
            )
            value = nxt
        continuation_cash = float(self.transitions[state] @ value[:, 0])
        continuation_invest = float(self.transitions[state] @ value[:, 1])
        q_cash = continuation_cash - (sell_cost if invested else 0.0)
        q_invest = float(rewards[state]) + continuation_invest - (
            0.0 if invested else buy_cost
        )
        return q_cash, q_invest


def fit_switching_model(
    symbol: str,
    frame: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    settings: SwitchingSettings,
) -> SwitchingModel:
    """Recompensas somente quando ambos os opens futuros ja sao conhecidos.

    Um exemplo na data t possui o atributo de fechamento(t) e retorno
    realizavel entre abertura(t+1) e abertura(t+2). Nenhum retorno t+2 pode
    ser consultado alem de fit_end (inclusive se o DataFrame tiver futuro).
    """
    end = pd.Timestamp(fit_end)
    history = frame.loc[:end, ["open", "close"]].copy()
    if len(history) <= settings.lookback + settings.minimum_observations + 2:
        raise ValueError(f"{symbol}: treino insuficiente para Switching.")
    features = _market_state_feature(history, settings.lookback)
    opens = pd.to_numeric(history["open"], errors="coerce")
    # OPEN(t+1) e OPEN(t+2) sao conhecidos somente no trecho de ajuste.
    next_open_return = np.log(opens.shift(-2) / opens.shift(-1))
    sample = pd.DataFrame({"feature": features, "return": next_open_return})
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < settings.minimum_observations:
        raise ValueError(f"{symbol}: observacoes insuficientes.")
    boundaries = tuple(float(x) for x in np.quantile(sample["feature"], [1 / 3, 2 / 3]))
    if not boundaries[0] < boundaries[1]:
        raise ValueError(f"{symbol}: regimes nao identificaveis sem variacao.")
    states = np.searchsorted(boundaries, sample["feature"].to_numpy(), side="right")
    observations = sample["return"].to_numpy(dtype=float)
    counts = np.ones((3, 3), dtype=float)  # Suavizacao Dirichlet(1).
    rewards = np.zeros(3, dtype=float)
    downside = np.zeros(3, dtype=float)
    for state in range(3):
        state_returns = observations[states == state]
        if len(state_returns) < 10:
            raise ValueError(f"{symbol}: regime com amostra insuficiente.")
        rewards[state] = float(np.mean(state_returns))
        downside[state] = float(np.mean(np.maximum(-state_returns, 0.0)))
    # Transicoes observadas em decisoes diarias consecutivas; nao
    # inventar adjacencia quando algum atributo ficou ausente.
    locations = history.index.get_indexer(sample.index)
    for j in range(len(states) - 1):
        if locations[j + 1] == locations[j] + 1:
            counts[states[j], states[j + 1]] += 1.0
    transitions = counts / counts.sum(axis=1, keepdims=True)
    return SwitchingModel(
        symbol=symbol,
        fit_end=end,
        boundaries=boundaries,
        transitions=transitions,
        expected_returns=rewards,
        downside_losses=downside,
        fit_observations=len(sample),
        lookback=settings.lookback,
    )


def switching_costs(config: Any) -> tuple[float, float]:
    """Aproximacao proporcional para Bellman; o replay cobra taxas exatas.

    Taxas por acao e limites SEC/TAF dependem da quantidade negociada,
    desconhecida no momento da avaliacao. Nao sao inventadas aqui.
    """
    slippage = float(config.slippage_bps) / 10_000
    buy = slippage + float(config.commission_rate)
    sell = buy + float(config.sec_fee_rate)
    return buy, sell


def build_overlay(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_to_fold: dict[pd.Timestamp, int],
    models_by_fold: dict[int, dict[str, SwitchingModel]],
    settings: SwitchingSettings,
    config: Any,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    settings_by_fold: dict[int, SwitchingSettings] | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Mexe exclusivamente na exposicao, nunca no ranking nem no fit Control."""
    buy_cost, sell_cost = switching_costs(config)

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        date = pd.Timestamp(timestamp)
        control_target, control_score = base_policy(date, current_position, holding_days)
        fold = decision_to_fold.get(date)
        if fold is None:
            raise KeyError(f"Data {date} sem fold cronologico.")
        if control_target == 0:
            diagnostics[date] = {
                "switching_control_target": "CASH",
                "switching_final_target": "CASH",
                "switching_reason": "control_selected_cash",
            }
            return 0, 0.0
        candidate = symbols[control_target - 1]
        fold_settings = (settings_by_fold or {}).get(int(fold), settings)
        model = models_by_fold[int(fold)][candidate]
        state = model.state_at(frames[candidate], date)
        q_cash, q_invest = model.values(
            state, current_position > 0,
            settings=fold_settings,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
        )
        # Empate preserva a posicao atual para nao criar rotacao artificial.
        invest = (
            q_invest > q_cash if current_position == 0
            else q_invest >= q_cash
        )
        target = control_target if invest else 0
        diagnostics[date] = {
            "switching_fold": int(fold),
            "switching_state": state,
            "switching_downside_weight": float(fold_settings.downside_weight),
            "switching_fit_end": model.fit_end.isoformat(),
            "switching_fit_observations": model.fit_observations,
            "switching_control_target": candidate,
            "switching_final_target": symbols[target - 1] if target else "CASH",
            "switching_q_cash": q_cash,
            "switching_q_invest": q_invest,
            "switching_delta_invest_minus_cash": q_invest - q_cash,
            "switching_reason": "exposure_approved" if invest else "cash_preferred",
        }
        return (target, float(control_score)) if invest else (0, 0.0)

    return policy



def selecionar_peso_calibracao(
    resultados: list[tuple[float, float]],
    *,
    tolerance: float = 1e-12,
) -> float:
    """Escolhe peso por resultado APENAS de calibracao, desempate conservador.
    
    Os resultados ja precisam vir de datas de calibracao anteriores ao OOS;
    o caller e responsavel pelo isolamento cronologico.
    """
    if not resultados:
        raise ValueError("Calibracao de risco sem candidatos.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerancia invalida.")
    weights = [float(weight) for weight, _ in resultados]
    scores = [float(score) for _, score in resultados]
    if len(weights) != len(set(weights)) or not all(
        np.isfinite(weight) and weight >= 0 for weight in weights
    ) or not all(np.isfinite(score) for score in scores):
        raise ValueError("Calibracao de risco contem valor invalido ou duplicado.")
    best_score = max(scores)
    candidates = [
        weight for weight, score in zip(weights, scores)
        if score >= best_score - tolerance
    ]
    return min(candidates)
