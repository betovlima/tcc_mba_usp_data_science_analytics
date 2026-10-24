"""Testes v1.24.4: retorno de preço com split na custódia, RAW intacto."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auditar_retornos_economicos_splits_spyder import (
    forward_price_return, measure_around_split, split_events,
)


def stable_bars(*, factor_after: float = 0.1, event_idx: int = 270) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-02", periods=450, tz="UTC")
    p = np.full(len(idx), 100.0)
    p[event_idx:] *= factor_after
    v = np.full(len(idx), 1000.0)
    v[event_idx:] /= factor_after
    return pd.DataFrame({
        "open": p, "high": p * 1.01,
        "low": p * .99, "close": p, "volume": v,
    }, index=idx)


def event_at(bars: pd.DataFrame, index: int, *,
             old: int = 1, new: int = 10) -> dict:
    return {
        "symbol": "AAA", "id": f"split-{index}",
        "action_type": "forward_split" if new >= old else "reverse_split",
        "ex_date": bars.index[index].date().isoformat(),
        "old_rate": old, "new_rate": new,
    }


def test_forward_10_for_1_preserves_price_wealth_crossing_split() -> None:
    bars = stable_bars()
    original = bars.copy(deep=True)
    event = event_at(bars, 270)
    factors = split_events([event])
    item = forward_price_return(
        bars, factors, decision_index=267, horizon=5,
    )
    assert item["entry_day"] == bars.index[268].date().isoformat()
    assert item["exit_day"] == bars.index[272].date().isoformat()
    assert item["crossed_split_count"] == 1
    assert item["split_quantity_multiplier"] == 10
    assert item["naive_raw_return"] == pytest.approx(-.9)
    assert item["custody_economic_price_return"] == pytest.approx(0.)
    assert item["label_matures_after_close_on"] == item["exit_day"]
    pd.testing.assert_frame_equal(bars, original, check_exact=True)


def test_entry_on_exdate_must_not_receive_split_twice() -> None:
    bars = stable_bars()
    item = forward_price_return(
        bars, split_events([event_at(bars, 270)]),
        decision_index=269, horizon=5,
    )
    assert item["entry_day"] == bars.index[270].date().isoformat()
    assert item["crossed_split_count"] == 0
    assert item["split_quantity_multiplier"] == 1.0
    assert item["custody_economic_price_return"] == pytest.approx(0.)


def test_reverse_1_for_10_preserves_wealth() -> None:
    bars = stable_bars(factor_after=10.)
    item = forward_price_return(
        bars, split_events([event_at(bars, 270, old=10, new=1)]),
        decision_index=267, horizon=5,
    )
    assert item["naive_raw_return"] == pytest.approx(9.)
    assert item["custody_economic_price_return"] == pytest.approx(0.)


def test_two_splits_compose_multipliers_correctly() -> None:
    bars = stable_bars()
    bars = bars.copy()
    bars.loc[bars.index[272]:, ["open", "high", "low", "close"]] /= 2
    events = [event_at(bars, 270),
              event_at(bars, 272, old=1, new=2)]
    result = forward_price_return(
        bars, split_events(events), decision_index=267, horizon=6,
    )
    assert result["crossed_split_count"] == 2
    assert result["split_quantity_multiplier"] == 20.
    assert result["custody_economic_price_return"] == pytest.approx(0.)


def test_economic_price_returns_equal_legacy_normalization_as_control() -> None:
    bars = stable_bars()
    events = [event_at(bars, 270)]
    rows = measure_around_split("AAA", bars, events, horizon_set=(5, 10, 60))
    assert rows
    assert all(row["economic_equals_legacy"] for row in rows)
    crossing = [row for row in rows if row["crossed_split_count"]]
    assert crossing
    assert all(row["raw_vs_economic_abs_difference"] > .5
               for row in crossing)


def test_dividend_not_treated_as_share_multiplier() -> None:
    bars = stable_bars()
    cash = {
        "action_type": "cash_dividend", "symbol": "AAA",
        "ex_date": bars.index[270].date().isoformat(),
        "old_rate": 1, "new_rate": 50,
    }
    assert split_events([cash]) == ()


def test_invalid_split_rejected() -> None:
    bars = stable_bars()
    bad = event_at(bars, 270, old=0, new=10)
    with pytest.raises(ValueError, match="proporção válida"):
        split_events([bad])


def test_no_future_price_data_available_fails_closed() -> None:
    bars = stable_bars()
    with pytest.raises(ValueError, match="indisponível"):
        forward_price_return(
            bars, (), decision_index=len(bars) - 2, horizon=5,
        )
