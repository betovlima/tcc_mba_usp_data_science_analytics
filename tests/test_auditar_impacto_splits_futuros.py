"""Regressoes do diagnostico de splits posteriores usando o motor de atributos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from auditar_impacto_splits_futuros_spyder import (
    measure_legacy_future_split_impact, partition_by_effective_date,
)


def series() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", periods=430, tz="UTC")
    t = np.arange(len(dates), dtype=float)
    close = 40.0 + t * .025 + 1.4 * np.sin(t * .21)
    return pd.DataFrame({
        "open": close * 1.003,
        "high": close * 1.018,
        "low": close * .985,
        "close": close,
        "volume": 120000.0 + 15000.0 * np.sin(t * .08),
    }, index=dates)


def future_split() -> dict:
    return {
        "action_type": "forward_split", "symbol": "AAA",
        "ex_date": "2021-03-01",
        "old_rate": 1, "new_rate": 10,
    }


def test_future_split_affects_legacy_history_without_editing_raw() -> None:
    raw = series()
    original = raw.copy(deep=True)
    record, features = measure_legacy_future_split_impact(
        "AAA", raw, [future_split()], decision_day="2020-07-28"
    )
    assert record["splits_effective_after_decision"] == 1
    assert record["splits_already_effective"] == 0
    assert record["legacy_ohlcv_changed_rows"] > 0
    assert record["legacy_price_changed_rows"] > 0
    assert record["legacy_volume_changed_rows"] > 0
    assert record["max_abs_close_difference"] > 0
    assert record["future_actions_with_verified_first_seen"] == 0
    assert record["feature_rows_common"] > 10
    assert record["features_compared"] == 52
    assert not record["raw_modified"]
    pd.testing.assert_frame_equal(original, raw, check_exact=True)
    assert all(row["symbol"] == "AAA" for row in features)


def test_past_only_split_is_not_future_leakage() -> None:
    raw = series()
    previous = {
        **future_split(),
        "ex_date": "2019-04-15",
    }
    result, features = measure_legacy_future_split_impact(
        "AAA", raw, [previous], decision_day="2020-07-28"
    )
    assert result["splits_already_effective"] == 1
    assert result["splits_effective_after_decision"] == 0
    assert result["legacy_ohlcv_changed_rows"] == 0
    assert result["features_materially_changed_cells"] == 0
    assert features == []


def test_multiple_future_events_are_counted_without_proving_announcement() -> None:
    actions = [
        future_split(),
        {**future_split(), "ex_date": "2024-06-10"},
        {**future_split(), "action_type": "cash_dividend", "ex_date": "2019-01-01"},
        {**future_split(), "ex_date": "2019-06-06"},
    ]
    past, future = partition_by_effective_date(
        actions, decision_day="2020-07-28"
    )
    assert len(past) == 1
    assert len(future) == 2


def test_no_events_no_raw_or_feature_change() -> None:
    raw = series()
    result, details = measure_legacy_future_split_impact(
        "AAA", raw, [], decision_day="2020-07-28"
    )
    assert result["splits_effective_after_decision"] == 0
    assert result["legacy_ohlcv_changed_rows"] == 0
    assert result["features_materially_changed_cells"] == 0
    assert details == []
