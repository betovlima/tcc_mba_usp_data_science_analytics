"""Regressões v1.24.9 — float64 em %.17g vs decimal round-trip mínimo."""
from __future__ import annotations

import pytest

from auditar_serializacao_float64_spyder import probe_float_serialization


@pytest.mark.parametrize(
    "entry,exit_price,multiplier,expected_entry,expected_exit,delta",
    [
        (
            "0.32000000000000001", "3.2000000000000002",
            "1/10", "0.32", "3.2", "1/100000000000000000",
        ),
        (
            "0.30199999999999999", "3.02",
            "1/10", "0.302", "3.02", "1/100000000000000000",
        ),
        (
            "3.9199999999999999", "78.400000000000006",
            "1/20", "3.92", "78.4", "1/2500000000000000",
        ),
    ],
)
def test_literal_raw_serialization_is_17g_and_shortest_repr_ties(
    entry: str, exit_price: str, multiplier: str,
    expected_entry: str, expected_exit: str, delta: str,
) -> None:
    item = probe_float_serialization(entry, exit_price, multiplier)
    assert item["literal_matches_17g"]
    assert item["shortest_repr_same_float64"]
    assert item["stored_literal_greater"]
    assert item["exact_binary_greater"]
    assert item["shortest_repr_equal"]
    assert item["entry_shortest_round_trip_repr"] == expected_entry
    assert item["close_shortest_round_trip_repr"] == expected_exit
    assert item["stored_literal_delta_fraction"] == delta
    assert item["shortest_repr_delta_fraction"] == "0/1"


def test_not_all_small_differences_are_roundtrip_ties() -> None:
    item = probe_float_serialization("1.0000000000000002", "3.0", "1/3")
    assert item["literal_matches_17g"]
    assert not item["shortest_repr_equal"]
    assert not item["stored_literal_greater"]
    assert item["exact_binary_greater"] is False


def test_absent_split_equal_decimal_raw_remains_equal() -> None:
    item = probe_float_serialization("1.5", "1.5", "1/1")
    assert item["stored_literal_delta_fraction"] == "0/1"
    assert item["shortest_repr_equal"]
    assert not item["stored_literal_greater"]


def test_zero_price_is_invalid() -> None:
    with pytest.raises(ValueError, match="positivos"):
        probe_float_serialization("0.0", "1.0", "1/1")
