"""Teste fatorial somente com CSVs locais sintéticos; sem Alpaca ou modelos."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pesquisar_atribuicao_ohlc_volume_spyder import restore_selected_fields


def write_csv(path: Path, changes: bool = False, outside: bool = False) -> None:
    table = pd.DataFrame({
        "timestamp": [
            "2018-05-01T04:00:00Z", "2018-05-02T04:00:00Z",
            "2018-05-03T04:00:00Z", "2018-05-04T04:00:00Z",
        ],
        "open": [10.0, 11.0, 12.0, 13.0],
        "high": [10.5, 11.5, 12.5, 13.5],
        "low": [9.5, 10.5, 11.5, 12.5],
        "close": [10.2, 11.2, 12.2, 13.2],
        "volume": [1000, 2000, 3000, 4000],
        "trade_count": [1, 2, 3, 4],
    })
    if changes:
        table.loc[0, "volume"] = 10
        table.loc[1, "high"] = 23.0
        table.loc[1, "volume"] = 20
        table.loc[2, "close"] = 24.0
        table.loc[2, "volume"] = 30
    if outside:
        table.loc[3, "volume"] = 40
    table.to_csv(path, index=False)


def test_ohlc_and_volume_interventions_are_separate_and_sources_immutable(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old.csv"
    recent = tmp_path / "recent.csv"
    vol = tmp_path / "vol.csv"
    prices = tmp_path / "prices.csv"
    write_csv(old)
    write_csv(recent, changes=True)
    old_bytes = old.read_bytes()
    recent_bytes = recent.read_bytes()

    restored_volume = restore_selected_fields(
        old, recent, vol, symbol="NVDA", restore_fields=("volume",)
    )
    restored_prices = restore_selected_fields(
        old, recent, prices, symbol="NVDA",
        restore_fields=("open", "high", "low", "close"),
    )

    historical = pd.read_csv(old)
    current = pd.read_csv(recent)
    with_vol = pd.read_csv(vol)
    with_prices = pd.read_csv(prices)
    assert len(restored_volume) == 3
    assert len(restored_prices) == 2
    assert with_vol["volume"].equals(historical["volume"])
    for field in ("open", "high", "low", "close"):
        assert with_vol[field].equals(current[field])
        assert with_prices[field].equals(historical[field])
    assert with_prices["volume"].equals(current["volume"])
    assert with_vol["trade_count"].equals(current["trade_count"])
    assert with_prices["trade_count"].equals(current["trade_count"])
    assert old.read_bytes() == old_bytes
    assert recent.read_bytes() == recent_bytes


def test_fatorial_rejects_future_or_unauthorized_dates(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    recent = tmp_path / "recent.csv"
    clone = tmp_path / "clone.csv"
    write_csv(old)
    write_csv(recent, changes=True, outside=True)
    with pytest.raises(RuntimeError, match="fora de maio"):
        restore_selected_fields(
            old, recent, clone, symbol="NVDA", restore_fields=("volume",)
        )
    assert not clone.exists()


def test_fatorial_rejects_unsupported_fields(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    recent = tmp_path / "recent.csv"
    write_csv(old)
    write_csv(recent, changes=True)
    with pytest.raises(ValueError, match="campos permitidos"):
        restore_selected_fields(
            old, recent, tmp_path / "new.csv",
            symbol="NVDA", restore_fields=("trade_count",),
        )


def test_no_difference_keeps_binary_identical(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    recent = tmp_path / "recent.csv"
    out = tmp_path / "new.csv"
    write_csv(old)
    write_csv(recent)
    rows = restore_selected_fields(
        old, recent, out, symbol="CXW", restore_fields=("volume",)
    )
    assert rows == []
    assert out.read_bytes() == recent.read_bytes()
