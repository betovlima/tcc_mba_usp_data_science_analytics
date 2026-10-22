"""Testes sem download para a comparacao de datas no universo fixo TCC."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
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


def test_cannot_compare_cutoff_after_snapshot() -> None:
    with pytest.raises(ValueError, match="ultrapassa"):
        fresh.validate_cutoffs("2026-10-06", fresh.EVALUATION_CUTOFFS)
    fresh.validate_cutoffs("2026-10-22", fresh.EVALUATION_CUTOFFS)


def test_sip_end_never_requests_future_midnight() -> None:
    from reproducao.dados import _delayed_sip_end

    now = pd.Timestamp("2026-10-22T22:00:00Z")
    requested = _delayed_sip_end(
        "2026-10-22", now_utc=now, delay_minutes=20,
    )
    assert requested == pd.Timestamp("2026-10-22T21:40:00Z")
    assert requested < now - pd.Timedelta(minutes=15)

    # Um snapshot ja passado nao e truncado desnecessariamente.
    historical = _delayed_sip_end(
        "2026-10-06", now_utc=now, delay_minutes=20,
    )
    assert historical == pd.Timestamp("2026-10-07T00:00:00Z")
    with pytest.raises(ValueError, match="15 minutos"):
        _delayed_sip_end("2026-10-22", now_utc=now, delay_minutes=10)


def test_rejects_incomplete_aapl_bar(tmp_path: Path) -> None:
    paths = fresh.SnapshotPaths.from_root(tmp_path / "snapshot")
    paths.ensure()
    pd.DataFrame({"timestamp": ["2026-10-21T04:00:00+00:00"]}).to_csv(
        paths.raw_bars / "AAPL.csv", index=False,
    )
    with pytest.raises(RuntimeError, match="AAPL nao tem barra fechada"):
        fresh.verify_reference_session(paths, "2026-10-22")


def test_download_once_77_assets_and_two_cutoffs_preserve_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    calls: list[tuple[str, Path, tuple[str, ...]]] = []

    def fake_bars(credentials: object, paths: fresh.SnapshotPaths, *,
                  assets: tuple[str, ...], bar_snapshot_as_of_end: str,
                  analysis_end_date: str,
                  sip_delay_guard_minutes: int = 0) -> dict[str, Path]:
        paths.ensure()
        calls.append(("bars", paths.root, assets))
        assert sip_delay_guard_minutes == 20
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
        calls.append(("actions", paths.root, assets))
        saved = {}
        for symbol in assets:
            target = paths.corporate_actions / f"{symbol}.csv"
            target.write_text("action_type\n", encoding="utf-8")
            saved[symbol] = target
        return saved

    def fake_manifest(paths: fresh.SnapshotPaths, *args: object, **kwargs: object) -> dict:
        identity = {
            "snapshot_sha256": "test-single-snapshot",
            "assets": list(kwargs["assets"]),
            "bars": {"bar_snapshot_as_of_end": kwargs["bar_snapshot_as_of_end"]},
        }
        paths.manifest.write_text(json.dumps(identity), encoding="utf-8")
        return identity

    monkeypatch.setattr(fresh, "download_raw_bars", fake_bars)
    monkeypatch.setattr(fresh, "download_corporate_actions", fake_actions)
    monkeypatch.setattr(fresh, "build_snapshot_manifest", fake_manifest)
    monkeypatch.setattr(
        fresh, "validate_snapshot",
        lambda paths: json.loads(paths.manifest.read_text(encoding="utf-8")),
    )

    replays: list[dict] = []

    def fake_replay(*, root: Path, output: Path, version: str,
                    execution_schema: str, experimental: dict) -> dict:
        output.mkdir(parents=True)
        (output / "predictions.csv").write_text("timestamp,asset\n", encoding="utf-8")
        saved_contract = json.loads(
            Path(experimental["manifest_path"]).read_text(encoding="utf-8")
        )
        replays.append({"experimental": experimental, "contract": saved_contract})
        return {
            "checkpoint": {"observed_capital": 10000.0 + len(replays)},
            "universe": {"effective_count": 75, "excluded_assets": [
                {"symbol": "DOC"}, {"symbol": "CLMT"},
            ]},
            "metrics": {"folds": []},
            "calendar": {"decision_sessions": 1560},
        }

    monkeypatch.setattr(fresh, "reproduzir_universo_congelado", fake_replay)
    summary = fresh.main(cutoff="2026-10-22")
    assert [name for name, _, _ in calls] == ["bars", "actions"]
    assert all(len(assets) == 77 for _, _, assets in calls)
    assert calls[0][1] == calls[1][1]
    assert len(replays) == 2
    assert [item["contract"]["analysis_end_date"] for item in replays] == [
        "2026-10-06", "2026-10-21",
    ]
    assert replays[0]["experimental"]["snapshot_root"] == replays[1]["experimental"]["snapshot_root"]
    assert original_marker.read_text(encoding="utf-8") == "imutable"
    assert summary["same_snapshot_both_cutoffs"] is True
    assert summary["effective_asset_count"] == 75
    archives = list((tmp_path / "output" / "fresh").rglob("pacote_comparacao_cutoffs.zip"))
    assert len(archives) == 1
    import zipfile
    with zipfile.ZipFile(archives[0]) as archive:
        assert "comparacao_cutoffs.json" in archive.namelist()
        assert "snapshot_77_manifest.json" in archive.namelist()
        assert "cutoff_2026-10-06/predictions.csv" in archive.namelist()
        assert "cutoff_2026-10-21/predictions.csv" in archive.namelist()

    # Retomar a mesma coleta validada: nenhum novo download da Alpaca.
    resumed = fresh.main(cutoff="2026-10-22")
    assert [name for name, _, _ in calls] == ["bars", "actions"]
    assert len(replays) == 4
    assert resumed["snapshot_sha256"] == summary["snapshot_sha256"]
    assert original_marker.read_text(encoding="utf-8") == "imutable"


def test_experimental_checkpoint_without_expected_capital() -> None:
    from reproducao.experimento import checkpoint_delta

    assert checkpoint_delta(34567.89, None) is None
    assert checkpoint_delta(34567.89, 10000.0) == pytest.approx(24567.89)
