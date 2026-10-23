"""Testes de contrato da intervencao de sensibilidade (somente dados temporarios)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from auditar_drift_historico_spyder import compare_bar_frames
from reproducao.preparacao import load_raw_bar_file
from pesquisar_causa_drift_2018_spyder import replace_verified_2018_rows


def make_csv(path: Path, *, altered: bool = False, outside: bool = False) -> None:
    frame = pd.DataFrame({
        "timestamp": [
            "2018-05-01T04:00:00+00:00",
            "2018-05-02T04:00:00+00:00",
            "2018-05-03T04:00:00+00:00",
            "2018-05-04T04:00:00+00:00",
        ],
        "open": [10.0, 11.0, 12.0, 13.0],
        "high": [10.2, 11.2, 12.2, 13.2],
        "low": [9.8, 10.8, 11.8, 12.8],
        "close": [10.1, 11.1, 12.1, 13.1],
        "volume": [1000, 2000, 3000, 4000],
        "trade_count": [5, 6, 7, 8],
        "vwap": [10.05, 11.05, 12.05, 13.05],
    })
    if altered:
        frame.loc[1, "volume"] = 20
        frame.loc[2, "high"] = 15.0
    if outside:
        frame.loc[3, "volume"] = 40
    frame.to_csv(path, index=False)


def test_restore_two_days_preserves_untouched_rows_and_sources(
    tmp_path: Path,
) -> None:
    old = tmp_path / "historical.csv"
    fresh = tmp_path / "fresh.csv"
    clone = tmp_path / "clone.csv"
    make_csv(old)
    make_csv(fresh, altered=True)
    old_bytes, new_bytes = old.read_bytes(), fresh.read_bytes()

    changes = replace_verified_2018_rows(
        old, fresh, clone, symbol="NVDA", cutoff="2026-10-06",
    )
    assert len(changes) == 2
    frozen = load_raw_bar_file(old, float_precision="round_trip")
    new_clone = load_raw_bar_file(clone, float_precision="round_trip")
    summary, material = compare_bar_frames("NVDA", frozen, new_clone)
    assert summary["changed_material_rows"] == 0
    assert material == []
    # As colunas auxiliares intactas e linhas fora da intervencao preservadas.
    a = pd.read_csv(fresh)
    b = pd.read_csv(clone)
    assert b["trade_count"].tolist() == a["trade_count"].tolist()
    assert b["vwap"].tolist() == a["vwap"].tolist()
    assert b.iloc[3].to_dict() == a.iloc[3].to_dict()
    assert old.read_bytes() == old_bytes
    assert fresh.read_bytes() == new_bytes


def test_rejects_difference_outside_three_2018_sessions(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    fresh = tmp_path / "fresh.csv"
    clone = tmp_path / "clone.csv"
    make_csv(old)
    make_csv(fresh, outside=True)
    with pytest.raises(RuntimeError, match="divergencia inesperada"):
        replace_verified_2018_rows(old, fresh, clone, symbol="NVDA")
    assert not clone.exists()


def test_no_changes_clone_matches_original_binary(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    fresh = tmp_path / "fresh.csv"
    clone = tmp_path / "clone.csv"
    make_csv(old)
    make_csv(fresh)
    assert not replace_verified_2018_rows(old, fresh, clone, symbol="CXW")
    assert clone.read_bytes() == fresh.read_bytes()


def test_different_calendar_is_not_patched(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    fresh = tmp_path / "fresh.csv"
    clone = tmp_path / "clone.csv"
    make_csv(old)
    make_csv(fresh, altered=True)
    frame = pd.read_csv(fresh).iloc[1:].copy()
    frame.to_csv(fresh, index=False)
    with pytest.raises(RuntimeError, match="calendario"):
        replace_verified_2018_rows(old, fresh, clone, symbol="NVDA")
