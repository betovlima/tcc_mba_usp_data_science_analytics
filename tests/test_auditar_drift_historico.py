"""Auditoria RAW e eventos: testes com series pequenas, sem rede ou backtest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import auditar_drift_historico_spyder as audit


def bars(*prices: float) -> pd.DataFrame:
    dates = pd.date_range("2020-07-27", periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000.0] * len(prices),
    }, index=dates)


def test_append_only_does_not_count_as_historical_revision() -> None:
    old = bars(10.0, 11.0)
    new = bars(10.0, 11.0, 12.0)
    result, details = audit.compare_bar_frames(
        "MGM", old, new, "2020-07-28",
    )
    assert result["changed_material_rows"] == 0
    assert result["changed_exact_rows"] == 0
    assert result["new_only_rows"] == 0  # dia adicionado esta apos o cutoff
    assert not details


def test_2020_price_revision_and_first_divergence_is_reported() -> None:
    old = bars(10.0, 11.0)
    new = bars(10.0, 12.0)
    summary, changes = audit.compare_bar_frames(
        "CXW", old, new, "2026-10-06",
    )
    assert summary["changed_material_rows"] == 1
    assert summary["first_material_change"].startswith("2020-07-28")
    assert summary["material_change_through_first_divergence"] == 1
    assert "close" in changes[0]["fields"]
    assert changes[0]["close_old"] == 11.0
    assert changes[0]["close_fresh"] == 12.0


def test_tiny_float_rounding_is_separated_from_material_change() -> None:
    old = bars(10.0)
    new = bars(10.0 + 1e-12)
    summary, details = audit.compare_bar_frames(
        "MGM", old, new, "2026-10-06",
    )
    assert summary["changed_exact_rows"] == 1
    assert summary["changed_material_rows"] == 0
    assert not details


def test_split_adjustment_can_change_old_rows_without_raw_change() -> None:
    same_raw = bars(10.0, 11.0)
    old_actions: list[dict] = []
    new_actions = [
        {"action_type": "forward_split", "ex_date": "2020-07-29",
         "old_rate": 1.0, "new_rate": 2.0}
    ]
    raw_summary, _ = audit.compare_bar_frames(
        "MGM", same_raw, same_raw, "2020-07-28",
    )
    old_adj, _ = audit.split_normalize(same_raw, old_actions)
    new_adj, _ = audit.split_normalize(same_raw, new_actions)
    adj_summary, details = audit.compare_bar_frames(
        "MGM", old_adj, new_adj, "2020-07-28", kind="split_adjusted",
    )
    assert raw_summary["changed_material_rows"] == 0
    assert adj_summary["changed_material_rows"] == 2
    assert len(details) == 2


def test_corporate_action_id_changes_are_not_treated_as_economic_change() -> None:
    old = [{"action_type": "cash_dividend", "symbol": "STAG",
            "ex_date": "2020-07-28", "rate": 0.1, "id": "old-id"}]
    new = [{"action_type": "cash_dividend", "symbol": "STAG",
            "ex_date": "2020-07-28", "rate": 0.1, "id": "new-id"}]
    summary, details = audit.compare_action_records("STAG", old, new)
    assert summary["action_added_before_cutoff"] == 0
    assert summary["action_removed_before_cutoff"] == 0
    assert not details


def test_event_after_cutoff_is_reported_but_not_counted_as_old_change() -> None:
    future_split = {"action_type": "reverse_split", "symbol": "CRT",
                    "ex_date": "2026-10-10", "old_rate": 10.0, "new_rate": 1.0}
    old = []
    summary, details = audit.compare_action_records(
        "CRT", old, [future_split], "2026-10-06",
    )
    assert summary["action_added_before_cutoff"] == 0
    assert summary["action_new_after_cutoff"] == 1
    assert summary["new_post_cutoff_splits"] == 1
    assert details == []


def test_missing_original_candidate_cache_cannot_be_filled_silently(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Cache historico obrigatorio"):
        audit.original_extra_paths(tmp_path, "CRT")
