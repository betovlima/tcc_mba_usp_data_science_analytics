"""Regressoes para gravacao de evidencias em execucoes longas."""
from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from pesquisa.saida_resiliente_motivos_u75 import (
    check_required_files,
    write_with_parent,
)


def test_output_directory_recreated_if_deleted_after_run_started(tmp_path):
    output = tmp_path / "timestamp" / "results"
    output.mkdir(parents=True)
    (output / "old.csv").write_text("original", encoding="utf-8")
    shutil.rmtree(output)
    target = output / "control_paridade.json"
    write_with_parent(target, lambda p: p.write_text('{"ok": true}', encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == '{"ok": true}'


def test_retry_when_directory_disappears_during_first_write(tmp_path):
    target = tmp_path / "timestamp" / "control_paridade.json"
    calls = []

    def writer(path: Path):
        calls.append(path)
        if len(calls) == 1:
            shutil.rmtree(path.parent)
            raise FileNotFoundError("output directory temporarily vanished")
        path.write_text("recovered", encoding="utf-8")

    write_with_parent(target, writer)
    assert len(calls) == 2
    assert target.read_text(encoding="utf-8") == "recovered"


def test_missing_earlier_evidence_aborts_instead_of_reporting_completed(tmp_path):
    output = tmp_path / "timestamp"
    output.mkdir()
    (output / "control_paridade.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot_identidade.csv"):
        check_required_files(output, (
            "snapshot_identidade.csv",
            "control_paridade.json",
        ))


def test_present_evidence_passes(tmp_path):
    output = tmp_path / "timestamp"
    output.mkdir()
    for name in ("snapshot_identidade.csv", "control_paridade.json"):
        (output / name).write_text("ok", encoding="utf-8")
    assert check_required_files(output, (
        "snapshot_identidade.csv",
        "control_paridade.json",
    )) is None


def test_persistent_directory_removal_is_not_hidden(tmp_path):
    target = tmp_path / "timestamp" / "output.json"
    tries = []

    def never_succeeds(path: Path):
        tries.append(1)
        shutil.rmtree(path.parent)
        raise FileNotFoundError("removed again")

    with pytest.raises(FileNotFoundError):
        write_with_parent(target, never_succeeds)
    assert len(tries) == 2
