"""Testes artificiais da regra TP com saida baseada em utilidade OOS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pesquisar_capitalizacao_ate_alvo_spyder import simular_ate_alvo
from pesquisar_capitalizacao_saida_utilidade_spyder import (
    simular_com_saida_utilidade,
)

DAYS = pd.bdate_range("2026-01-05", periods=7, tz="UTC")


@dataclass
class Config:
    initial_capital: float = 10_000.0
    rotation_cash_threshold: float = 0.0


def fees(side, qty, price, config):
    return {"total_fee": 0.0}


def slip(price, side, config):
    return price


def prices(changes=None):
    base = pd.DataFrame(
        {key: [100.0] * len(DAYS) for key in ("open", "high", "low", "close")},
        index=DAYS,
    )
    for ix, fields in (changes or {}).items():
        for key, value in fields.items():
            base.loc[DAYS[ix], key] = value
    return base


def run(
    *, mode="negative", target=.03, cache=None, policy=None,
    changes=None, frames=None, symbols=None, folds=None,
):
    calls = []
    def recording(day, current, held):
        calls.append((day, current, held))
        return (policy or (lambda day, current, held: (1, .5)))(
            day, current, held
        )
    names = symbols or ["AAA"]
    actual_frames = frames or {"AAA": prices(changes)}
    fold_map = folds or {day: 1 for day in DAYS}
    actual_cache = cache
    if actual_cache is None:
        actual_cache = {1: {
            day: np.array([0., .12], dtype=float)
            for day in DAYS
        }}
    result = simular_com_saida_utilidade(
        recording, actual_frames, names, DAYS, Config(), fees, slip,
        target, fold_map, actual_cache, mode,
    )
    return result, calls


def test_negative_score_sells_on_next_open_not_prior_close():
    cache = {1: {d: np.array([0., -.2 if d == DAYS[1] else .1])
                 for d in DAYS}}
    result, calls = run(cache=cache)
    sell = result.trades[result.trades.reason.eq("UTILITY_NONPOSITIVE")]
    assert len(sell) == 1
    assert sell.iloc[0].timestamp == DAYS[2]
    assert result.curve.loc[DAYS[2], "selected_asset"] == "CASH"
    assert result.trades[result.trades.action.eq("BUY")].iloc[1].timestamp == DAYS[3]
    assert result.metrics["utility_exits"] == 1
    assert result.metrics["utility_nonpositive_exits"] == 1
    assert result.metrics["target_exits"] == 0
    assert len(calls) == 2


def test_positive_score_holds_until_final_liquidation():
    result, calls = run()
    assert result.metrics["utility_exits"] == 0
    assert result.metrics["target_exits"] == 0
    assert result.metrics["buys"] == 1
    assert result.metrics["final_liquidation"]
    assert len(calls) == 1


def test_native_switch_exits_to_cash_then_buys_next_day():
    def policy(day, pos, held):
        if pos == 0:
            return 1 if day == DAYS[0] else 2, .5
        return 2, .6
    assets = {"AAA": prices(), "BBB": prices()}
    result, calls = run(
        mode="native_switch", policy=policy, frames=assets,
        symbols=["AAA", "BBB"],
    )
    exit_row = result.trades[
        result.trades.reason.eq("UTILITY_NATIVE_SWITCH")
    ].iloc[0]
    assert exit_row.timestamp == DAYS[2]
    buys = result.trades[result.trades.action.eq("BUY")]
    assert buys.asset.tolist() == ["AAA", "BBB"]
    assert buys.timestamp.tolist() == [DAYS[1], DAYS[3]]
    assert result.curve.loc[DAYS[2], "selected_asset"] == "CASH"
    assert result.metrics["utility_native_switch_exits"] == 1


def test_target_open_gap_takes_precedence_over_utility_exit():
    cache = {1: {d: np.array([0., -.5]) for d in DAYS}}
    result, _ = run(
        cache=cache,
        changes={2: {"open": 108., "high": 110., "low": 105., "close": 108.}},
    )
    sells = result.trades[result.trades.action.eq("SELL")]
    assert sells.iloc[0].reason == "TP_GAP_OPEN"
    assert sells.iloc[0].price == pytest.approx(108.)
    assert result.metrics["target_exits"] >= 1
    assert sells.iloc[0].timestamp == DAYS[2]


def test_combined_negative_first():
    cache = {1: {d: np.array([0., -.3]) for d in DAYS}}
    def policy(day, pos, held):
        return (0, 0.) if pos > 0 else (1, .5)
    result, _ = run(mode="combined", cache=cache, policy=policy)
    assert result.trades.iloc[1].reason == "UTILITY_NONPOSITIVE"
    assert result.metrics["utility_nonpositive_exits"] >= 1


def test_negative_utility_exit_can_realize_loss():
    cache = {1: {d: np.array([0., -.5 if d == DAYS[1] else .5])
                 for d in DAYS}}
    result, _ = run(
        cache=cache,
        changes={2: {"open": 80., "high": 82., "low": 78., "close": 80.}},
    )
    exit_row = result.trades[
        result.trades.reason.eq("UTILITY_NONPOSITIVE")
    ].iloc[0]
    assert exit_row.position_return == pytest.approx(-.20)
    assert exit_row.timestamp == DAYS[2]


def test_missing_oos_score_is_error_not_silent_fallback():
    cache = {1: {d: np.array([0., .2]) for d in DAYS if d != DAYS[1]}}
    with pytest.raises(RuntimeError, match="Vetor OOS incompleto"):
        run(cache=cache)


def test_fold_specific_oos_cache_used_not_previous_or_future_fold():
    folds = {d: 1 if d <= DAYS[2] else 2 for d in DAYS}
    cache = {
        1: {d: np.array([0., .2]) for d in DAYS[:3]},
        2: {d: np.array([0., -.2]) for d in DAYS[3:]},
    }
    r, _ = run(mode="negative", cache=cache, folds=folds)
    sells = r.trades[r.trades.reason.eq("UTILITY_NONPOSITIVE")]
    assert len(sells) >= 1
    assert sells.iloc[0].timestamp == DAYS[4]


def test_none_mode_identical_to_prior_hold_until_target_engine():
    p = lambda day, pos, held: (1, .5)
    f = {"AAA": prices({1: {"high": 106.}, 4: {"high": 103.}})}
    common = dict(
        policy=p, frames=f, symbols=["AAA"], decision_dates=DAYS,
        config=Config(), fee_calculator=fees, slippage=slip,
        target=.05, fold_map={d: 1 for d in DAYS},
    )
    old = simular_ate_alvo(**common)
    new = simular_com_saida_utilidade(
        **common, utility_mode="none",
    )
    pd.testing.assert_series_equal(
        old.curve.strategy_equity, new.curve.strategy_equity
    )
    assert old.metrics["ending_capital"] == pytest.approx(
        new.metrics["ending_capital"]
    )
    assert old.metrics["target_exits"] == new.metrics["target_exits"]


def test_invalid_mode_or_missing_cache():
    with pytest.raises(ValueError, match="utility_mode"):
        run(mode="unsupported")
    with pytest.raises(ValueError, match="Cache"):
        simular_com_saida_utilidade(
            lambda *args: (1, .5),
            {"AAA": prices()}, ["AAA"], DAYS, Config(), fees, slip,
            .03, {d: 1 for d in DAYS}, None, "negative",
        )
