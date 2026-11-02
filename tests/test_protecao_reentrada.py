"""Testes para confirmacao de reentrada e stop independente; barras sinteticas."""
from dataclasses import dataclass
import math

import pandas as pd
import pytest

from pesquisar_protecao_reentrada_spyder import simular_alvo


DAYS = pd.date_range("2026-01-05", periods=6, freq="D", tz="UTC")


@dataclass
class Config:
    initial_capital: float = 10000.0


def fees(side, quantity, price, config):
    return {"total_fee": 0.0}


def slip(price, side, config):
    return price


def replay(
    *, target=None, stop_loss=None, reentry_confirmations=1,
    opens=None, highs=None, lows=None, closes=None, policy=None,
):
    defaults = [100.] * len(DAYS)
    frame = pd.DataFrame(dict(
        open=opens or defaults, high=highs or defaults,
        low=lows or defaults, close=closes or defaults,
    ), index=DAYS)
    calls = []
    def instrumented(day, pos, held):
        calls.append((day, pos, held))
        return (policy or (lambda date, position, n: (1, 0.5)))(day, pos, held)
    result = simular_alvo(
        policy=instrumented, frames={"CAPR": frame}, symbols=["CAPR"],
        decision_dates=DAYS, config=Config(), fee_calculator=fees,
        slippage=slip, target=target, stop_loss=stop_loss,
        reentry_confirmations=reentry_confirmations,
        fold_map={day: 1 for day in DAYS},
    )
    return result, calls


def test_default_tp_behaviour_unchanged_and_next_session_reentry():
    result, calls = replay(target=0.03, highs=[100., 104., 104., 104., 104., 104.])
    assert result.metrics["tp_exits"] == 5
    assert result.metrics["stop_exits"] == 0
    assert result.metrics["same_asset_reentries_after_tp"] == 4
    assert math.isclose(result.metrics["ending_capital"], 10000. * 1.03 ** 5)
    assert all(calls[i][1] == 0 for i in range(1, len(calls)))


def test_confirmation_requires_two_closes_before_same_asset_reentry():
    result, calls = replay(
        target=0.03, reentry_confirmations=2,
        highs=[100., 104., 100., 104., 100., 104.],
    )
    buys = result.trades[result.trades.action == "BUY"]
    assert list(buys.timestamp) == [DAYS[1], DAYS[3], DAYS[5]]
    assert result.metrics["blocked_same_asset_reentries"] == 2
    assert result.metrics["same_asset_reentries_after_tp"] == 2


def test_stop_gap_open_executes_at_open_not_at_stop_threshold():
    result, _ = replay(
        stop_loss=0.10,
        opens=[100., 100., 75., 100., 100., 100.],
        highs=[100., 101., 80., 100., 100., 100.],
        lows=[100., 99., 70., 100., 100., 100.],
        closes=[100., 100., 75., 100., 100., 100.],
    )
    stop = result.trades[result.trades.reason == "STOP_GAP_OPEN"].iloc[0]
    assert math.isclose(stop.price, 75.)
    assert result.metrics["stop_exits"] == 1


def test_independent_stop_can_exit_during_minimum_hold():
    # A politica nao mandou sair (e teria imposto min_hold), mas o stop atua.
    def hold(day, pos, held):
        return (1, 0.5)
    result, _ = replay(
        target=0.05, stop_loss=0.10, policy=hold,
        highs=[100., 100., 100., 100., 100., 100.],
        lows=[100., 89., 100., 100., 100., 100.],
    )
    assert result.trades.reason.iloc[1] == "STOP_INTRADAY"
    assert result.metrics["stop_exits"] >= 1
    assert result.trades.timestamp.iloc[0] == result.trades.timestamp.iloc[1]


def test_ambiguous_intraday_bar_uses_conservative_stop_first():
    result, _ = replay(
        target=0.05, stop_loss=0.10,
        highs=[100., 110., 100., 100., 100., 100.],
        lows=[100., 85., 100., 100., 100., 100.],
    )
    first_sell = result.trades[result.trades.action == "SELL"].iloc[0]
    assert first_sell.reason == "STOP_INTRADAY"
    assert math.isclose(first_sell.price, 90.)
    assert result.metrics["ambiguous_intraday_bars"] == 1


def test_first_signal_after_stop_may_be_blocked():
    result, _ = replay(
        stop_loss=0.10, reentry_confirmations=2,
        highs=[100., 100., 100., 100., 100., 100.],
        lows=[100., 89., 100., 100., 100., 100.],
    )
    buys = result.trades[result.trades.action == "BUY"]
    assert list(buys.timestamp[:2]) == [DAYS[1], DAYS[3]]
    assert result.metrics["blocked_same_asset_reentries"] >= 1


def test_invalid_safety_parameters():
    for kwargs in [dict(stop_loss=0.), dict(stop_loss=1.), dict(reentry_confirmations=0)]:
        with pytest.raises(ValueError):
            replay(**kwargs)


def test_original_without_target_or_stop_unchanged():
    result, _ = replay()
    assert result.metrics["stop_exits"] == 0
    assert result.metrics["tp_exits"] == 0
    assert result.metrics["buys"] == 1
    assert result.metrics["cash_sessions"] == 0
    assert result.metrics["ending_capital"] == 10000.
