"""Smoke tests sem credenciais Alpaca para o runner experimental TCC."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import executar_atualizacao_series_spyder as fresh


def test_rejects_future_or_unfinished_current_session() -> None:
    before_close = datetime(2026, 10, 22, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="consolidacao"):
        fresh.assert_completed_session("2026-10-22", now=before_close)
    with pytest.raises(ValueError, match="futuro"):
        fresh.assert_completed_session("2026-10-23", now=before_close)
    fresh.assert_completed_session(
        "2026-10-22",
        now=datetime(2026, 10, 22, 17, 5, tzinfo=ZoneInfo("America/New_York")),
    )


def test_rejects_incomplete_aapl_bar(tmp_path: Path) -> None:
    paths = fresh.SnapshotPaths.from_root(tmp_path / "snapshot")
    paths.ensure()
    pd.DataFrame({"timestamp": ["2026-10-21T04:00:00+00:00"]}).to_csv(
        paths.raw_bars / "AAPL.csv", index=False,
    )
    with pytest.raises(RuntimeError, match="AAPL nao tem barra fechada"):
        fresh.verify_reference_session(paths, "2026-10-22")


def test_fresh_job_does_not_overwrite_official_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifica a orquestracao inteira com stubs, sem requests nem treino pesado.
    (tmp_path / "evidencias").mkdir()
    reference = json.loads(
        (fresh.ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    (tmp_path / "evidencias" / "manifest_universo_congelado.json").write_text(
        json.dumps(reference), encoding="utf-8",
    )
    original_marker = tmp_path / "dados" / "u67" / "historico_preservado.txt"
    original_marker.parent.mkdir(parents=True)
    original_marker.write_text("imutable", encoding="utf-8")

    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "EXPERIMENT_DATA_ROOT", tmp_path / "dados" / ".u67_fresh_runs")
    monkeypatch.setattr(fresh, "EXPERIMENT_OUTPUT_ROOT", tmp_path / "output" / "fresh")
    monkeypatch.setattr(fresh, "assert_completed_session", lambda cutoff: None)
    monkeypatch.setattr(fresh, "load_alpaca_credentials", lambda root: object())

    downloaded: list[tuple[Path, tuple[str, ...]]] = []

    def fake_bars(credentials: object, paths: fresh.SnapshotPaths, *,
                  assets: tuple[str, ...], bar_snapshot_as_of_end: str,
                  analysis_end_date: str) -> dict[str, Path]:
        paths.ensure()
        downloaded.append((paths.root, assets))
        saved = {}
        for symbol in assets:
            target = paths.raw_bars / f"{symbol}.csv"
            target.write_text(
                "timestamp,open,high,low,close,volume\n"
                "2026-10-22T04:00:00+00:00,1,1,1,1,100\n",
                encoding="utf-8",
            )
            saved[symbol] = target
        return saved

    def fake_actions(credentials: object, paths: fresh.SnapshotPaths, *,
                     assets: tuple[str, ...], query_end: str) -> dict[str, Path]:
        saved = {}
        for symbol in assets:
            target = paths.corporate_actions / f"{symbol}.csv"
            target.write_text("action_type\n", encoding="utf-8")
            saved[symbol] = target
        return saved

    monkeypatch.setattr(fresh, "download_raw_bars", fake_bars)
    monkeypatch.setattr(fresh, "download_corporate_actions", fake_actions)
    monkeypatch.setattr(fresh, "build_snapshot_manifest", lambda *a, **k: {
        "snapshot_sha256": "test-fresh-snapshot",
    })
    monkeypatch.setattr(fresh, "validate_snapshot", lambda paths: {
        "snapshot_sha256": "test-fresh-snapshot",
    })
    replay = Mock(return_value={"checkpoint": {"observed_capital": 12345.67}})
    monkeypatch.setattr(fresh, "reproduzir_universo_congelado", replay)

    result = fresh.main(cutoff="2026-10-22")
    assert result["checkpoint"]["observed_capital"] == 12345.67
    assert [len(assets) for _, assets in downloaded] == [67, 10]
    assert all(".u67_fresh_runs" in str(path) for path, _ in downloaded)
    assert original_marker.read_text(encoding="utf-8") == "imutable"
    kwargs = replay.call_args.kwargs
    assert kwargs["experimental"] is not None
    experimental = kwargs["experimental"]
    assert Path(experimental["base_root"]) != tmp_path / "dados" / "u67"
    contract = json.loads(Path(experimental["manifest_path"]).read_text(encoding="utf-8"))
    assert contract["analysis_end_date"] == "2026-10-22"
    assert contract["expected_ending_capital"] is None
    assert len(contract["candidate_file_hashes"]) == 10
