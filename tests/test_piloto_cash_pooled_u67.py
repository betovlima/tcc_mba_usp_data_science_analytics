"""Guards for causal, pooled CASH / HOLD / ROTATE decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from reproducao.piloto_cash_pooled_u67 import (
    FEATURES, decisions, cumulative_metrics, make_training_panels,
)


def _prices(days, multiplier=1.0):
    t = np.arange(len(days), dtype=float)
    close = (50 + .1 * t + 1.3 * np.sin(t / 9)) * multiplier
    frame = pd.DataFrame({
        "open": close * (1.001 + .0001 * np.sin(t)),
        "high": close * 1.015,
        "low": close * .99,
        "close": close,
        "volume": 100_000 + (t % 10) * 1000,
    }, index=days)
    return frame


def test_no_future_market_data_is_used_in_features():
    days = pd.date_range("2025-01-01", periods=165, freq="B", tz="UTC")
    frames = {"SPY": _prices(days), "AAA": _prices(days, 1.2)}
    first = make_training_panels(frames, days)
    frames["AAA"].loc[days[-1], ["open", "high", "low", "close"]] *= 2.
    after = make_training_panels(frames, days)
    first_t = first.loc[
        (first["symbol"] == "AAA") & (first["date"] == days[-3]),
        list(FEATURES),
    ]
    after_t = after.loc[
        (after["symbol"] == "AAA") & (after["date"] == days[-3]),
        list(FEATURES),
    ]
    pd.testing.assert_frame_equal(first_t, after_t)
    old_label = first.loc[
        (first["symbol"] == "AAA") & (first["date"] == days[-3]),
        "target_open_to_open",
    ].iloc[0]
    changed_label = after.loc[
        (after["symbol"] == "AAA") & (after["date"] == days[-3]),
        "target_open_to_open",
    ].iloc[0]
    assert changed_label != old_label


def test_cash_buy_sell_and_hold_decisions_are_distinct():
    days = pd.date_range("2026-02-02", periods=4, freq="B", tz="UTC")
    estimates = [.02, -.04, .01, .01]
    rows = []
    for d, mu in zip(days, estimates):
        rows.append({
            "date": d, "symbol": "AAA",
            "predicted_mean": mu, "predicted_q10": -.005,
            "raw_open_next": 100., "raw_open_after_next": 101.,
        })
    result, audit = decisions(pd.DataFrame(rows), days, aversion=0.)
    assert audit["action"].tolist() == [
        "CASH_BUY", "SELL_CASH", "CASH_BUY", "HOLD",
    ]
    assert result["cash_sessions"] == 1
    assert result["actions"]["HOLD"] == 1


def test_missing_future_open_fails_not_filled_with_zero():
    days = pd.date_range("2026-02-02", periods=1, freq="B", tz="UTC")
    data = pd.DataFrame([{
        "date": days[0], "symbol": "AAA",
        "predicted_mean": .05, "predicted_q10": 0.,
        "raw_open_next": float("nan"),
        "raw_open_after_next": 105.,
    }])
    import pytest
    with pytest.raises(RuntimeError, match="NEGOCIACAO_NAO_VERIFICAVEL"):
        decisions(data, days, aversion=0.)


def test_drawdown_is_measured_on_capital_curve():
    metrics = cumulative_metrics(np.array([.10, -.20, .05]))
    assert np.isclose(metrics["growth_multiple"], .924)
    assert np.isclose(metrics["maximum_drawdown"], -.20)


def test_no_ticker_identity_or_future_trade_labels_in_features():
    for forbidden in (
        "symbol", "target_open_to_open", "raw_open_next",
        "raw_open_after_next", "winner_utility",
        "strategy_equity", "future_return", "CAPR",
    ):
        assert forbidden not in FEATURES
