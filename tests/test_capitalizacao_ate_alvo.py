"""Testes da estrategia de capitalizacao sem rotacao antes do alvo."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import pytest

from pesquisar_capitalizacao_ate_alvo_spyder import simular_ate_alvo


DAYS = pd.date_range("2026-01-05", periods=6, freq="B", tz="UTC")


@dataclass
class Config:
    initial_capital: float = 10_000.0


def fees(side, quantity, price, config):
    return {"total_fee": 0.0}


def slip(price, side, config):
    return price


def frame(opens=None, highs=None, lows=None, closes=None):
    return pd.DataFrame(
        {
            "open": opens if opens is not None else [100.] * len(DAYS),
            "high": highs if highs is not None else [100.] * len(DAYS),
            "low": lows if lows is not None else [100.] * len(DAYS),
            "close": closes if closes is not None else [100.] * len(DAYS),
        },
        index=DAYS,
    )


def run(target=.03, policy=None, frames=None, symbols=None, config=None):
    calls = []
    def observed_policy(day, current_position, held):
        calls.append((day, current_position, held))
        return (policy or (lambda day, pos, n: (1, .5)))(
            day, current_position, held
        )
    replay = simular_ate_alvo(
        policy=observed_policy,
        frames=frames or {"AAA": frame()},
        symbols=symbols or ["AAA"],
        decision_dates=DAYS,
        config=config or Config(),
        fee_calculator=fees,
        slippage=slip,
        target=target,
        fold_map={d: 1 for d in DAYS},
    )
    return replay, calls


@pytest.mark.parametrize("target", [.015, .03, .05])
def test_compounds_actual_target_exits_even_with_same_asset(target):
    result, calls = run(
        target=target, frames={"AAA": frame(
            highs=[100., 110., 110., 110., 110., 110.],
        )},
    )
    assert result.metrics["target_exits"] == 5
    assert result.metrics["buys"] == 5
    assert result.metrics["losing_exits"] == 0
    assert result.metrics["same_asset_reentries_after_tp"] == 4
    assert result.metrics["final_liquidation"] is False
    assert result.metrics["cash_sessions"] == 5
    assert math.isclose(
        result.metrics["ending_capital"], 10_000. * (1. + target) ** 5,
        rel_tol=1e-10,
    )
    assert len(calls) == 5
    assert [pos for _, pos, _ in calls] == [0] * 5


def test_no_rotation_even_when_policy_would_choose_other_asset():
    def policy(day, pos, days):
        # Esta funcao jamais deve receber posicao investida.
        assert pos == 0
        return 1, 0.6
    f = frame(
        highs=[100., 108., 108., 108., 108., 108.],
        closes=[100., 98., 90., 105., 107., 95.],
        lows=[100., 98., 90., 100., 100., 95.],
    )
    result, calls = run(target=.10, policy=policy, frames={"AAA": f})
    assert len(calls) == 1
    assert result.metrics["buys"] == 1
    assert result.metrics["target_exits"] == 0
    assert result.metrics["final_liquidation"] is True
    assert result.trades.reason.iloc[-1] == "FINAL_LIQUIDATION"
    assert math.isclose(result.metrics["ending_capital"], 9500.)
    assert result.metrics["losing_exits"] == 1
    assert result.metrics["max_hold_sessions"] == 5


def test_keeps_unrealized_loss_in_equity_instead_of_faking_profit():
    result, _ = run(
        target=.05,
        frames={"AAA": frame(
            highs=[100., 100., 100., 100., 100., 100.],
            lows=[100., 100., 80., 65., 50., 40.],
            closes=[100., 100., 80., 65., 50., 40.],
        )},
    )
    assert result.metrics["target_exits"] == 0
    assert result.metrics["ending_capital"] == pytest.approx(4_000.)
    assert result.metrics["maximum_drawdown_pct"] == pytest.approx(-60.)
    assert result.metrics["other_exits"] == 1
    assert result.metrics["losing_exits"] == 1
    assert result.curve["strategy_equity"].iloc[2] == pytest.approx(6_500.)


def test_gap_above_target_sells_at_open():
    result, _ = run(
        target=.03,
        frames={"AAA": frame(
            opens=[100., 100., 106., 100., 100., 100.],
            highs=[100., 100., 108., 100., 100., 100.],
            lows=[100., 100., 105., 100., 100., 100.],
            closes=[100., 100., 106., 100., 100., 100.],
        )},
    )
    gapped = result.trades[result.trades.reason == "TP_GAP_OPEN"]
    assert len(gapped) == 1
    assert gapped.iloc[0].price == 106.
    assert gapped.iloc[0].position_return == pytest.approx(.06)


def test_model_can_choose_different_asset_after_target():
    names = ["AAA", "BBB"]
    cache = {"AAA": frame(
        highs=[100., 104., 100., 100., 100., 100.],
    ), "BBB": frame()}
    calls = []
    def policy(day, pos, held):
        assert pos == 0
        calls.append(day)
        return (1, .4) if day == DAYS[0] else (2, .8)
    result, traced = run(
        target=.03, frames=cache, symbols=names, policy=policy,
    )
    buy_rows = result.trades[result.trades.action == "BUY"]
    assert list(buy_rows.asset) == ["AAA", "BBB"]
    assert list(buy_rows.timestamp) == [DAYS[1], DAYS[2]]
    assert result.metrics["target_exits"] == 1
    assert result.metrics["final_liquidation"] is True
    assert result.metrics["ending_capital"] == pytest.approx(10_300.)
    assert len(calls) == 2
    assert len(traced) == 2


def test_remains_in_cash_without_valid_entry():
    result, calls = run(policy=lambda date, pos, held: (0, 0.))
    assert result.metrics["buys"] == 0
    assert result.metrics["target_exits"] == 0
    assert result.metrics["ending_capital"] == 10000.
    assert result.metrics["cash_sessions"] == len(DAYS) - 1
    assert len(calls) == len(DAYS) - 1


def test_no_stop_and_position_can_stay_many_sessions():
    result, calls = run(
        target=.03,
        frames={"AAA": frame(
            highs=[100., 101., 101., 101., 101., 101.],
            lows=[100., 70., 70., 70., 70., 70.],
            closes=[100., 70., 70., 70., 70., 70.],
        )},
    )
    assert result.metrics["target_exits"] == 0
    assert result.metrics["ending_capital"] == pytest.approx(7_000.)
    assert result.metrics["max_hold_sessions"] == 5
    assert len(calls) == 1


def test_invalid_target_and_duplicate_calendar():
    with pytest.raises(ValueError):
        run(target=0.)
    with pytest.raises(ValueError):
        run(target=1.)
    with pytest.raises(ValueError):
        simular_ate_alvo(
            policy=lambda *args: (1, .4),
            frames={"AAA": frame()},
            symbols=["AAA"],
            decision_dates=DAYS.append(DAYS[[-1]]),
            config=Config(),
            fee_calculator=fees, slippage=slip, target=.05,
        )
