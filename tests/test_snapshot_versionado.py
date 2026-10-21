"""Snapshot: CSVs acessíveis sem comparação por checksum."""
from pathlib import Path

import pandas as pd
import pytest

from reproducao.dados import SnapshotPaths, read_snapshot_manifest


def test_snapshot_preserva_metadados_e_csvs_sem_checksums():
    root = Path(__file__).resolve().parents[1]
    paths = SnapshotPaths.referencia(root)
    if not paths.manifest.is_file():
        pytest.skip("Snapshot da nova campanha ainda não foi coletado e versionado.")
    manifest = read_snapshot_manifest(paths)
    assert manifest["source"] == "alpaca"
    assert "file_hashes" not in manifest
    assert "snapshot_sha256" not in manifest
    bar_file = paths.raw_bars / "AAPL.csv"
    assert bar_file.is_file()
    assert not pd.read_csv(bar_file).empty
