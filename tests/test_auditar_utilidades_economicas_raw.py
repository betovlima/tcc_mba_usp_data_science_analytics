"""Regressão v1.24.6: utilidade multihorizonte por quantidade, RAW intacto."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from engine.rotacao import construir_quadro_rotacao
from reproducao.preparacao import split_normalize
from auditar_retornos_economicos_splits_spyder import split_events
from auditar_utilidades_economicas_raw_spyder import (
    compare_labels_near_splits, economic_utility_at_decision, utility_names,
)


def series() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=650, tz="UTC")
    t = np.arange(len(dates), dtype=float)
    close = 80.0 + .018*t + 1.9*np.sin(.1*t)
    volume = 100000.0 + 25000*np.sin(.21*t)
    close[330:] /= 10
    volume[330:] *= 10
    return pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.025,
        "low": close * .98,
        "close": close,
        "volume": volume,
    }, index=dates)


def action(bars: pd.DataFrame) -> dict:
    return {
        "symbol": "AAA", "action_type": "forward_split",
        "ex_date": bars.index[330].date().isoformat(),
        "old_rate": 1, "new_rate": 10,
    }


@pytest.mark.parametrize("decision_idx", [215, 270, 305, 315, 328, 329, 330, 340, 390])
def test_economic_full_utility_equals_legacy_only_in_memory(decision_idx: int) -> None:
    bars = series()
    source_copy = bars.copy(deep=True)
    event = action(bars)
    calculated = economic_utility_at_decision(
        bars, split_events([event]), decision_index=decision_idx,
    )
    adjusted, _ = split_normalize(bars, [event])
    comparison = construir_quadro_rotacao(adjusted, CONFIG)
    stamp = bars.index[decision_idx]
    assert stamp in comparison.index
    assert len(calculated) == 15
    for name in utility_names():
        assert calculated[name] == pytest.approx(
            float(comparison.loc[stamp, name]), rel=1e-9, abs=1e-9,
        ), (decision_idx, name)
    pd.testing.assert_frame_equal(bars, source_copy, check_exact=True)


def test_comparison_around_split_covers_forward_labels_and_no_raw_change() -> None:
    bars = series()
    original = bars.copy(deep=True)
    result, diffs = compare_labels_near_splits(
        "AAA", bars, [action(bars)],
    )
    assert result["decisions_compared"] >= 90
    assert result["label_cells_compared"] == 15*result["decisions_compared"]
    assert result["decisions_crossing_split"] > 0
    assert result["materially_different_cells"] == 0
    assert diffs == []
    pd.testing.assert_frame_equal(bars, original, check_exact=True)


def test_multiple_split_factor_composition_matches_legacy() -> None:
    bars = series()
    bars = bars.copy()
    bars.loc[bars.index[410]:, ["open","high","low","close"]] /= 2
    bars.loc[bars.index[410]:, "volume"] *= 2
    events = [action(bars), {
        **action(bars), "ex_date": bars.index[410].date().isoformat(),
        "old_rate": 1, "new_rate": 2,
    }]
    current = economic_utility_at_decision(
        bars, split_events(events), decision_index=385,
    )
    normalized, _ = split_normalize(bars, events)
    expected = construir_quadro_rotacao(normalized, CONFIG)
    for name in utility_names():
        assert current[name] == pytest.approx(
            float(expected.loc[bars.index[385], name]),
            abs=1e-9, rel=1e-9,
        )


def test_labels_need_complete_future_horizon() -> None:
    bars = series()
    with pytest.raises(ValueError, match="indisponível"):
        economic_utility_at_decision(
            bars, (), decision_index=len(bars)-10,
        )
