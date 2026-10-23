"""Guardas do executor frozen U75 RAW v1.25.0."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from validar_backtest_economico_raw_spyder import (
    CUTOFF, SNAPSHOT_SHA256, VERSION, _snapshot_file_hashes,
)


def test_final_replay_spec_is_frozen_not_optimize_again() -> None:
    assert VERSION == "1.25.0"
    assert CUTOFF == "2026-10-06"
    assert SNAPSHOT_SHA256 == (
        "595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc"
    )


def test_all_raw_and_events_file_hashes_are_recorded(tmp_path: Path) -> None:
    bars = tmp_path / "raw"
    actions = tmp_path / "actions"
    bars.mkdir()
    actions.mkdir()
    for symbol in ("AAA", "BBB"):
        (bars / f"{symbol}.csv").write_text(
            f"timestamp,open,close\n2020-01-02,100,100\n{symbol}\n",
            encoding="utf-8",
        )
        (actions / f"{symbol}.csv").write_text(
            "action_type,ex_date,old_rate,new_rate\n", encoding="utf-8"
        )
    source = SimpleNamespace(raw_bars=bars, corporate_actions=actions)
    hashes = _snapshot_file_hashes(source, ("AAA", "BBB"))
    assert len(hashes) == 4
    assert hashes == _snapshot_file_hashes(source, ("AAA", "BBB"))
    (bars / "AAA.csv").write_text(
        "timestamp,open,close\n2020-01-02,101,101\n",
        encoding="utf-8",
    )
    new_hashes = _snapshot_file_hashes(source, ("AAA", "BBB"))
    assert new_hashes["bars/AAA.csv"] != hashes["bars/AAA.csv"]
    assert new_hashes["bars/BBB.csv"] == hashes["bars/BBB.csv"]
