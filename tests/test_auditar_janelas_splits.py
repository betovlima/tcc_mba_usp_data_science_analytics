"""Testes v1.24.3: split realista influencia labels ao cruzar evento, nao RAW."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auditar_janelas_splits_spyder import measure_event_boundary


def synthetic_split_series() -> tuple[pd.DataFrame, dict]:
    idx = pd.bdate_range("2018-01-02", periods=610, tz="UTC")
    n = len(idx)
    t = np.arange(n, dtype=float)
    close = 100. + 0.035 * t + np.sin(t * .12) * 2
    volume = 80000. + np.cos(t * .17) * 5000
    pos = 345
    # Evento realista: preco cai pela proporcao e quantidade de negocios sobe.
    close[pos:] /= 10
    volume[pos:] *= 10
    raw = pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.023,
        "low": close * .978,
        "close": close,
        "volume": volume,
    }, index=idx)
    event = {
        "id": "split-synthetic-2019",
        "symbol": "AAA",
        "action_type": "forward_split",
        "ex_date": idx[pos].date().isoformat(),
        "old_rate": 1,
        "new_rate": 10,
    }
    return raw, event


def test_boundary_event_changes_forward_labels_and_post_features_not_raw() -> None:
    raw, event = synthetic_split_series()
    original = raw.copy(deep=True)
    finding = measure_event_boundary("AAA", raw, event)
    assert finding["split_within_observed_bars"]
    assert finding["raw_open_gap_pct"] < -80
    assert abs(finding["custody_neutral_open_gap_pct"]) < 5
    assert finding["pre_event_feature_rows"] > 0
    assert finding["pre_event_feature_changed_cells"] == 0
    assert finding["pre_event_label_changed_cells"] > 0
    assert finding["post_event_feature_changed_cells"] > 0
    assert finding["post_event_label_changed_cells"] >= 0
    assert not finding["raw_modified"]
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


def test_event_after_last_bar_skips_without_false_change() -> None:
    raw, event = synthetic_split_series()
    original = raw.copy(deep=True)
    event["ex_date"] = "2033-01-01"
    finding = measure_event_boundary("AAA", raw, event)
    assert not finding["split_within_observed_bars"]
    assert finding["pre_event_feature_rows"] == 0
    assert finding["post_event_label_changed_cells"] == 0
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


def test_event_before_first_bar_skips_without_false_change() -> None:
    raw, event = synthetic_split_series()
    event["ex_date"] = "2017-01-01"
    finding = measure_event_boundary("AAA", raw, event)
    assert not finding["split_within_observed_bars"]


def test_reverse_split_synthetic_conserves_gap_unit() -> None:
    raw, event = synthetic_split_series()
    raw = raw.copy()
    k = 345
    raw.loc[raw.index[k]:, ["open", "high", "low", "close"]] *= 100
    # Stock performs a further 1-for-10 reverse split at same date,
    # so after/before ratio is circa 10 rather than 0.1
    event["action_type"] = "reverse_split"
    event["old_rate"] = 10
    event["new_rate"] = 1
    finding = measure_event_boundary("AAA", raw, event)
    assert finding["split_within_observed_bars"]
    assert finding["raw_open_gap_pct"] > 800
    assert abs(finding["custody_neutral_open_gap_pct"]) < 5


def test_negative_split_rate_rejected(tmp_path) -> None:
    raw, event = synthetic_split_series()
    event["new_rate"] = 0
    with pytest.raises((RuntimeError, ValueError, ZeroDivisionError)):
        measure_event_boundary("AAA", raw, event)
