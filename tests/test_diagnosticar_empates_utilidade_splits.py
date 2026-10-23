"""Regressoes de diagnostico de empate de float64 em labels de split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diagnosticar_empates_utilidade_splits_spyder import compare_positive_share


def test_one_ulp_close_can_change_strict_greater_than_after_legacy_division() -> None:
    """Exemplo numerico construtivo, nao cotacao real YANG ou CAPR."""
    entry = np.float64(0.1234567)
    one_ulp_above = np.nextafter(entry, np.inf)
    assert one_ulp_above > entry
    # O fator 3:1 faz um preço um ULP maior virar um empate no float64.
    assert np.float64(one_ulp_above/3.) == np.float64(entry/3.)
    closes = np.full(60, entry, dtype=np.float64)
    closes[0] = one_ulp_above  # proximo passo e queda, nao outra alta
    legacy_closes = closes / 3.
    dates = pd.bdate_range("2020-01-02", periods=60, tz="UTC")
    summary, rows = compare_positive_share(
        entry, closes, float(entry/3.), legacy_closes,
        days=dates, symbol="SYNTHETIC", decision_day="2019-12-31",
    )
    assert summary["positive_share_mismatch_count"] == 1
    assert summary["positive_steps_mismatch_count"] == 0
    assert summary["economic_close_gt_entry_count"] == 1
    assert summary["legacy_close_gt_entry_count"] == 0
    assert summary["persistence_delta"] == pytest.approx(1./120.)
    assert len(rows) == 1
    assert rows[0]["economic_abs_diff_in_entry_ulps"] == pytest.approx(1.)
    assert rows[0]["legacy_abs_diff_in_entry_ulps"] == pytest.approx(0.)
    assert rows[0]["comparison_type"] == "close_gt_entry"


def test_identical_economic_and_legacy_prices_produce_no_mismatch() -> None:
    price = 13.
    values = np.linspace(12., 14., num=60)
    days = pd.bdate_range("2020-01-02", periods=60, tz="UTC")
    summary, differences = compare_positive_share(
        price, values, price, values.copy(),
        days=days, symbol="SYNTHETIC", decision_day="2019-12-31",
    )
    assert summary["positive_share_mismatch_count"] == 0
    assert summary["positive_steps_mismatch_count"] == 0
    assert summary["persistence_delta"] == pytest.approx(0.)
    assert differences == []


def test_refuse_comparison_when_horizon_is_not_sixty_sessions() -> None:
    with pytest.raises(ValueError, match="60"):
        compare_positive_share(
            10., np.ones(59), 10., np.ones(59),
            days=pd.bdate_range("2020-01-02", periods=59, tz="UTC"),
            symbol="SYNTHETIC", decision_day="2019-12-31",
        )
