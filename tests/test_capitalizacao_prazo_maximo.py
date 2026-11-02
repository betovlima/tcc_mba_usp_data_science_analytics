"""Regressao: capitalizacao com prazo maximo, sem rotacoes por ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from pesquisar_capitalizacao_prazo_maximo_spyder import simular_ate_alvo_com_prazo


DAYS = pd.bdate_range("2026-01-05", periods=28, tz="UTC")


@dataclass
class Config:
    initial_capital: float = 10_000.


def fees(side: str, quantity: float, price: float, config: Any) -> dict:
    return {"total_fee": 0.}


def slip(price: float, side: str, config: Any) -> float:
    return price


def bars(overrides: dict[int, dict[str, float]] | None = None) -> pd.DataFrame:
    source = pd.DataFrame(
        {k: [100.] * len(DAYS) for k in ("open", "high", "low", "close")},
        index=DAYS,
    )
    for idx, changes in (overrides or {}).items():
        for key, value in changes.items():
            source.loc[DAYS[idx], key] = value
    return source


def simulation(
    *, target: float = .03,
    max_holding_sessions: int | None = 5,
    overrides: dict[int, dict[str, float]] | None = None,
    policy=None,
    symbols=None,
    frames=None,
):
    calls = []
    def observed_policy(day, pos, held):
        calls.append((day, pos, held))
        if policy is not None:
            return policy(day, pos, held)
        return (1, .5)

    actual_symbols = symbols or ["AAA"]
    result = simular_ate_alvo_com_prazo(
        policy=observed_policy,
        frames=frames or {"AAA": bars(overrides)},
        symbols=actual_symbols,
        decision_dates=DAYS,
        config=Config(),
        fee_calculator=fees,
        slippage=slip,
        target=target,
        max_holding_sessions=max_holding_sessions,
        fold_map={day: 1 for day in DAYS},
    )
    return result, calls


@pytest.mark.parametrize("max_days", [5, 10, 20])
def test_exits_at_next_open_after_exact_number_of_completed_sessions(max_days):
    r, calls = simulation(max_holding_sessions=max_days)
    first_buy = r.trades.loc[r.trades.action.eq("BUY")].iloc[0]
    first_timeout = r.trades.loc[r.trades.reason.eq("MAX_HOLD_OPEN")].iloc[0]
    assert first_buy.timestamp == DAYS[1]
    assert first_timeout.timestamp == DAYS[max_days + 1]
    assert first_timeout.holding_sessions == max_days
    assert r.metrics["max_hold_sessions"] <= max_days
    assert r.metrics["timeout_exits"] >= 1
    assert all(pos == 0 and held == 0 for _, pos, held in calls)


def test_no_repurchase_on_expiry_day_and_same_asset_can_reenter_next_session():
    r, _ = simulation(max_holding_sessions=5)
    buys = r.trades.loc[r.trades.action.eq("BUY")].reset_index(drop=True)
    timeout = r.trades.loc[r.trades.reason.eq("MAX_HOLD_OPEN")].iloc[0]
    assert timeout.timestamp == DAYS[6]
    assert buys.iloc[1].timestamp == DAYS[7]
    assert buys.iloc[1].asset == "AAA"
    assert r.curve.loc[DAYS[6], "selected_asset"] == "CASH"
    assert r.metrics["cash_sessions"] >= 1


def test_forced_exit_realizes_loss_without_waiting_for_positive_result():
    r, _ = simulation(
        max_holding_sessions=5,
        overrides={
            6: {"open": 75., "high": 110., "low": 74., "close": 90.},
        },
    )
    first_exit = r.trades.loc[r.trades.reason.eq("MAX_HOLD_OPEN")].iloc[0]
    assert first_exit.price == pytest.approx(75.)
    assert first_exit.position_return == pytest.approx(-.25)
    assert r.metrics["timeout_exits_with_loss"] >= 1
    # A maxima do dia da expiracao nao adia a ordem programada de saida.
    assert r.trades.loc[r.trades.timestamp.eq(DAYS[6]), "reason"].tolist() == [
        "MAX_HOLD_OPEN"
    ]


def test_when_open_above_target_on_deadline_it_counts_as_profit_not_timeout():
    r, _ = simulation(
        max_holding_sessions=5,
        overrides={6: {"open": 108., "high": 110., "low": 101., "close": 108.}},
    )
    exit_row = r.trades.loc[r.trades.action.eq("SELL")].iloc[0]
    assert exit_row.reason == "TP_GAP_OPEN"
    assert exit_row.price == pytest.approx(108.)
    assert r.metrics["target_exits"] >= 1
    assert r.metrics["timeout_exits"] == 0 or (
        r.trades.reason.eq("MAX_HOLD_OPEN").any()
    )


@pytest.mark.parametrize("target", [.015, .03, .05])
def test_recapitalizes_each_realized_profit_and_never_exceeds_max_hold(target):
    r, calls = simulation(
        target=target,
        max_holding_sessions=5,
        overrides={
            i: {"high": 115.} for i in range(1, len(DAYS))
        },
    )
    assert r.metrics["timeout_exits"] == 0
    assert r.metrics["target_exits"] == len(DAYS) - 1
    assert r.metrics["ending_capital"] == pytest.approx(
        10000. * (1. + target) ** (len(DAYS) - 1),
        rel=1e-10,
    )
    assert all(pos == 0 for _, pos, _ in calls)


def test_reinvests_in_new_model_proposal_only_after_cash_close():
    state = {"entries": 0}
    def policy(day, pos, held):
        assert pos == 0
        state["entries"] += 1
        return (1, .5) if state["entries"] == 1 else (2, .6)
    r, _ = simulation(
        max_holding_sessions=5,
        symbols=["AAA", "BBB"],
        frames={"AAA": bars(), "BBB": bars()},
        policy=policy,
    )
    buys = r.trades.loc[r.trades.action.eq("BUY")].reset_index(drop=True)
    assert buys.iloc[0].asset == "AAA"
    assert buys.iloc[1].asset == "BBB"
    assert buys.iloc[1].timestamp == DAYS[7]
    assert r.metrics["timeout_exits"] >= 1


def test_without_timeout_retains_original_1500_day_risk_behavior():
    r, calls = simulation(max_holding_sessions=None)
    assert r.metrics["timeout_exits"] == 0
    assert r.metrics["target_exits"] == 0
    assert r.metrics["buys"] == 1
    assert r.metrics["final_liquidation"]
    assert r.metrics["max_hold_sessions"] == len(DAYS)-1
    assert len(calls) == 1


def test_buy_day_is_first_holding_session_and_can_hit_target():
    r, calls = simulation(
        max_holding_sessions=1,
        overrides={1: {"high": 106.}},
    )
    first_exit = r.trades.loc[r.trades.action.eq("SELL")].iloc[0]
    assert first_exit.timestamp == DAYS[1]
    assert first_exit.reason == "TP_LIMIT"
    assert first_exit.holding_sessions == 1
    assert r.trades.loc[r.trades.timestamp.eq(DAYS[1]), "action"].tolist() == [
        "BUY", "SELL"
    ]
    assert r.trades.loc[r.trades.action.eq("BUY")].iloc[1].timestamp == DAYS[2]


@pytest.mark.parametrize("invalid", [0, -1, 2.5, True, "5"])
def test_rejects_invalid_holding_periods(invalid):
    with pytest.raises(ValueError, match="max_holding_sessions"):
        simulation(max_holding_sessions=invalid)


def test_prices_only_referenced_at_or_after_known_signal_and_no_future_high_in_selection():
    dates_seen = []
    def policy(day, pos, held):
        assert pos == 0
        dates_seen.append(day)
        return 1, .5
    r, calls = simulation(max_holding_sessions=5, policy=policy)
    assert all(calls[i][0] <= DAYS[-2] for i in range(len(calls)))
    assert all(day in DAYS[:-1] for day in dates_seen)
    assert r.metrics["max_hold_sessions"] <= 5
