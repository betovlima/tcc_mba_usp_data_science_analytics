"""Regressoes: RAW sem dividendos e diagnostico de 11 pregoes sem lookahead."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from diagnosticar_decisoes_outubro_2026_spyder import (
    select_11_sessions, paired_completed_positions,
)
from reproducao.dados import download_raw_bars
from reproducao.preparacao import split_normalize


DAYS = [
    "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09",
    "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15",
    "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21",
]


def test_alpaca_request_is_raw_and_sip_not_dividend_or_all() -> None:
    source = inspect.getsource(download_raw_bars)
    assert "adjustment=Adjustment.RAW" in source
    assert "feed=DataFeed.SIP" in source
    assert "adjustment=Adjustment.ALL" not in source
    assert "adjustment=Adjustment.DIVIDEND" not in source


def test_cash_dividend_cannot_reprice_historical_bars_in_split_normalize() -> None:
    dates = pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30"], utc=True)
    original = pd.DataFrame(
        {"open": [9.0, 10.0, 11.0], "high": [10.0, 11.0, 12.0],
         "low": [8.0, 9.0, 10.0], "close": [9.5, 10.5, 11.5],
         "volume": [1000.0, 1100.0, 1200.0]}, index=dates,
    )
    dividend = [{"action_type": "cash_dividend", "ex_date": "2026-09-30",
                 "rate": 0.44, "symbol": "CRM"}]
    result, applied = split_normalize(original, dividend)
    pd.testing.assert_frame_equal(result, original)
    assert applied == []

    split = [{"action_type": "forward_split", "ex_date": "2026-09-30",
              "old_rate": 1.0, "new_rate": 2.0}]
    adjusted, applied_splits = split_normalize(original, split)
    assert len(applied_splits) == 1
    assert adjusted.loc[dates[0], "close"] == pytest.approx(9.5 / 2)
    assert adjusted.loc[dates[0], "volume"] == pytest.approx(2000.0)
    assert adjusted.loc[dates[2], "close"] == original.loc[dates[2], "close"]


def test_sessions_and_scores_are_not_assumed_to_be_probabilities() -> None:
    predictions = pd.DataFrame({
        "timestamp": [x + "T04:00:00Z" for x in DAYS],
        "selected_asset": ["CAPR"] * 12,
        "strategy_equity": np.linspace(10000.0, 9000.0, num=12),
        "decision_score": [0.5] * 12,
    })
    sessions = select_11_sessions(predictions)
    assert len(sessions) == 12
    assert sessions.iloc[0]["portfolio_return_pct"] != sessions.iloc[0]["portfolio_return_pct"]
    assert sessions.iloc[-1]["next_session_equity_return_pct"] != sessions.iloc[-1]["next_session_equity_return_pct"]
    assert sessions.iloc[-1]["decision_score"] == 0.5


def test_completed_position_matched_without_future_score_leakage() -> None:
    predictions = pd.DataFrame({
        "timestamp": [x + "T04:00:00Z" for x in DAYS],
        "selected_asset": ["WDAY"] * 12,
        "strategy_equity": np.linspace(10000.0, 10200.0, num=12),
        "decision_score": [0.387] * 12,
    })
    sessions = select_11_sessions(predictions)
    trades = pd.DataFrame({
        "timestamp": ["2026-10-08T04:00:00Z", "2026-10-12T04:00:00Z"],
        "action": ["BUY", "SELL"],
        "asset": ["WDAY", "WDAY"],
        "execution_price": [184.08, 189.45],
        "position_return": [0.0, 189.45 / 184.08 - 1],
        "realized_pnl": [0.0, 291.7],
    })
    completed = paired_completed_positions(trades, sessions)
    assert len(completed) == 1
    assert completed.iloc[0]["entry_day"] == "2026-10-08"
    assert completed.iloc[0]["score_at_entry"] == 0.387
    assert completed.iloc[0]["realized_position_return_pct"] == pytest.approx(2.9172, abs=0.0001)
    assert completed.iloc[0]["exit_day"] == "2026-10-12"


def test_rejects_incomplete_recent_sessions() -> None:
    predictions = pd.DataFrame({
        "timestamp": ["2026-10-20T04:00:00Z"],
        "strategy_equity": [10000.0], "selected_asset": ["CAPR"],
        "decision_score": [0.42],
    })
    with pytest.raises(RuntimeError, match="11 novas sessoes"):
        select_11_sessions(predictions)
