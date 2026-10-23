"""v1.24.8 — prova de comparação racional a partir do texto RAW original."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from auditar_precisao_decimal_splits_spyder import (
    compare_exact_decimal_raw,
    file_sha256,
    raw_quote_literal,
    read_csv_literal,
)


def reverse_split() -> list[dict[str, str]]:
    return [{
        "action_type": "reverse_split", "ex_date": "2024-11-04",
        "old_rate": "20", "new_rate": "1",
    }]


def compare(entry: str, close: str, *, actions=None) -> dict:
    return compare_exact_decimal_raw(
        entry_open_literal=entry,
        future_close_literal=close,
        entry_day="2024-10-17",
        future_day="2024-11-19",
        corporate_actions=reverse_split() if actions is None else actions,
    )


def test_yang_displayed_quotes_are_equal_using_exact_decimals() -> None:
    result = compare("3.92", "78.40")
    assert result["exact_decimal_equal"]
    assert not result["exact_decimal_close_gt_entry"]
    assert result["quantity_multiplier_fraction"] == "1/20"
    assert result["exact_difference_numerator"] == "0"
    assert result["events_effectively_crossed"] == 1


def test_retains_tiny_actual_source_decimal_difference_without_rounding() -> None:
    result = compare("3.92", "78.400000000000001")
    assert not result["exact_decimal_equal"]
    assert result["exact_decimal_close_gt_entry"]
    assert result["exact_difference_numerator"] != "0"


def test_fractional_split_does_not_create_periodic_decimal_roundoff() -> None:
    result = compare("1", "3", actions=[{
        "action_type": "reverse_split",
        "ex_date": "2024-11-04",
        "old_rate": "3", "new_rate": "1",
    }])
    assert result["exact_decimal_equal"]
    assert result["quantity_multiplier_fraction"] == "1/3"


def test_event_effective_on_entry_is_not_applied_again() -> None:
    result = compare("3.92", "3.92", actions=[{
        "action_type": "reverse_split", "ex_date": "2024-10-17",
        "old_rate": "20", "new_rate": "1",
    }])
    assert result["exact_decimal_equal"]
    assert result["events_effectively_crossed"] == 0


def test_cash_dividend_is_ignored_as_quantity_factor() -> None:
    result = compare("3.92", "3.92", actions=[{
        "action_type": "cash_dividend", "ex_date": "2024-11-04",
        "old_rate": "20", "new_rate": "1",
    }])
    assert result["quantity_multiplier_fraction"] == "1/1"
    assert result["exact_decimal_equal"]


def test_wrong_or_zero_split_rate_fails_closed() -> None:
    with pytest.raises(ValueError, match="positivo"):
        compare("3.92", "78.40", actions=[{
            "action_type": "reverse_split", "ex_date": "2024-11-04",
            "old_rate": "20", "new_rate": "0",
        }])


def test_literal_csv_preserves_text_and_hash(tmp_path: Path) -> None:
    path = tmp_path/"bars.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "open", "close"])
        writer.writeheader()
        writer.writerow({
            "timestamp": "2024-10-17T04:00:00Z",
            "open": "3.9200000000000000", "close": "3.92",
        })
    before = file_sha256(path)
    rows = read_csv_literal(path)
    assert raw_quote_literal(rows, "2024-10-17", "open") == "3.9200000000000000"
    assert file_sha256(path) == before
    with pytest.raises(RuntimeError, match="duplicada"):
        raw_quote_literal(rows + rows, "2024-10-17", "open")
