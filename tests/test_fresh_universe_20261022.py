"""Contrato de universo: somente 75 ativos e treinamento conjunto."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

import pandas as pd
import pytest

import executar_atualizacao_series_spyder as fresh
from reproducao.universo_fixo75 import universo_75


def test_universe_75_never_contains_excluded_symbols() -> None:
    historical = json.loads(
        (fresh.ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(historical, ())
    assert len(assets) == len(set(assets)) == 75
    assert not (set(assets) & {"DOC", "CLMT"})
    with pytest.raises(RuntimeError, match="Lista de ativos"):
        universo_75(historical, ("NVDA",))


def test_sip_cutoff_and_session_closure() -> None:
    from reproducao.dados import _delayed_sip_end

    now = pd.Timestamp("2026-10-22T22:00:00Z")
    requested = _delayed_sip_end(
        "2026-10-22", now_utc=now, delay_minutes=20
    )
    assert requested == pd.Timestamp("2026-10-22T21:40:00Z")
    before = datetime(2026, 10, 22, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="consolidacao"):
        fresh.assert_completed_session("2026-10-22", now=before)
    with pytest.raises(ValueError, match="futuro"):
        fresh.assert_completed_session("2026-10-23", now=before)
    fresh.assert_completed_session(
        "2026-10-22",
        now=datetime(2026, 10, 22, 17, 5, tzinfo=ZoneInfo("America/New_York")),
    )
    with pytest.raises(ValueError, match="ultrapassa"):
        fresh.validate_cutoffs("2026-10-06", fresh.EVALUATION_CUTOFFS)


def test_rejects_incomplete_session(tmp_path: Path) -> None:
    paths = fresh.SnapshotPaths.from_root(tmp_path / "snapshot")
    paths.ensure()
    pd.DataFrame({"timestamp": ["2026-10-21T04:00:00Z"]}).to_csv(
        paths.raw_bars / "AAPL.csv", index=False,
    )
    with pytest.raises(RuntimeError, match="AAPL nao tem barra"):
        fresh.verify_reference_session(paths, "2026-10-22")


def test_only_75_downloaded_and_two_cutoffs_use_same_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "evidencias").mkdir()
    real = json.loads(
        (fresh.ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    (tmp_path / "evidencias" / "manifest_universo_congelado.json").write_text(
        json.dumps(real), encoding="utf-8"
    )
    original_marker = tmp_path / "dados" / "u67" / "preservado.txt"
    original_marker.parent.mkdir(parents=True)
    original_marker.write_text("sem mudar")
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "EXPERIMENT_DATA_ROOT", tmp_path / "dados" / ".u75_fresh_runs")
    monkeypatch.setattr(fresh, "LEGACY_77_DATA_ROOT", tmp_path / "dados" / ".u67_fresh_runs")
    monkeypatch.setattr(fresh, "EXPERIMENT_OUTPUT_ROOT", tmp_path / "output" / "fresh75")
    monkeypatch.setattr(fresh, "assert_completed_session", lambda cutoff: None)
    monkeypatch.setattr(fresh, "load_alpaca_credentials", lambda root: object())

    calls = []
    def fake_bars(credentials, paths, *, assets, bar_snapshot_as_of_end,
                  analysis_end_date, sip_delay_guard_minutes):
        paths.ensure()
        calls.append(("bars", tuple(assets)))
        assert sip_delay_guard_minutes == 20
        result = {}
        for symbol in assets:
            target = paths.raw_bars / f"{symbol}.csv"
            target.write_text(
                "timestamp,open,high,low,close,volume\n"
                "2026-10-22T04:00:00Z,1,1,1,1,100\n"
            )
            result[symbol] = target
        return result

    def fake_actions(credentials, paths, *, assets, query_end):
        calls.append(("actions", tuple(assets)))
        result = {}
        for symbol in assets:
            target = paths.corporate_actions / f"{symbol}.csv"
            target.write_text("action_type\n")
            result[symbol] = target
        return result

    def fake_manifest(paths, raw_files, actions, *, assets, bar_snapshot_as_of_end,
                      analysis_end_date, snapshot_name):
        payload = {
            "snapshot_sha256": "sha-single-75",
            "assets": list(assets),
            "bars": {"bar_snapshot_as_of_end": bar_snapshot_as_of_end},
        }
        paths.manifest.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(fresh, "download_raw_bars", fake_bars)
    monkeypatch.setattr(fresh, "download_corporate_actions", fake_actions)
    monkeypatch.setattr(fresh, "build_snapshot_manifest", fake_manifest)
    monkeypatch.setattr(
        fresh, "validate_snapshot",
        lambda paths: json.loads(paths.manifest.read_text()),
    )

    simulations = []
    def fake_replay(*, root, output, snapshot_root, analysis_end_date,
                    version, execution_schema):
        output.mkdir(parents=True, exist_ok=True)
        (output / "predictions.csv").write_text("asset\n")
        simulations.append((snapshot_root, analysis_end_date))
        return {
            "checkpoint": {"observed_capital": 10000.0},
            "universe": {"requested_count": 75, "effective_count": 75},
            "metrics": {"folds": []},
            "calendar": {"decision_sessions": 1560},
        }

    monkeypatch.setattr(fresh, "reproduzir_universo_fixo75", fake_replay)
    first = fresh.main(cutoff="2026-10-22")
    assert [name for name, _ in calls] == ["bars", "actions"]
    assert all(len(assets) == 75 and not {"DOC", "CLMT"} & set(assets) for _, assets in calls)
    assert [cutoff for _, cutoff in simulations] == ["2026-10-06", "2026-10-21"]
    assert simulations[0][0] == simulations[1][0]
    assert first["requested_count"] == 75
    assert first["effective_asset_count"] == 75
    assert original_marker.read_text() == "sem mudar"
    packages = list((tmp_path / "output").rglob("pacote_comparacao_cutoffs_75.zip"))
    assert len(packages) == 1
    with zipfile.ZipFile(packages[0]) as archive:
        assert "snapshot_75_manifest.json" in archive.namelist()
        assert "comparacao_cutoffs_75.json" in archive.namelist()
    second = fresh.main(cutoff="2026-10-22")
    assert len(calls) == 2  # Os dados foram reaproveitados; sem nova coleta.
    assert second["snapshot_sha256"] == first["snapshot_sha256"]
    assert len(simulations) == 4
