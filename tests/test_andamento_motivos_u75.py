"""Testes de diagnóstico persistente para execução longa do Control U75."""
from __future__ import annotations

import json
import shutil

import pytest

from pesquisa.andamento_motivos_u75 import record_progress, last_progress


def test_progress_records_snapshot_and_long_training_stage(tmp_path):
    output = tmp_path / "run"
    record_progress(output, "snapshot_validado", details={"ativos": 75})
    record_progress(output, "calibracao_inicio", details={"fold": 1})
    assert last_progress(output)["stage"] == "calibracao_inicio"
    records = [json.loads(x) for x in (
        output / "andamento.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["details"]["ativos"] == 75
    assert records[1]["details"]["fold"] == 1
    assert (output / "andamento_atual.json").is_file()


def test_progress_remains_running_after_unexpected_termination(tmp_path):
    output = tmp_path / "run"
    record_progress(output, "treinamento_final_inicio", details={"fold": 2})
    # Nenhum status fictício de conclusão em caso de encerramento do kernel.
    assert last_progress(output)["status"] == "running"
    assert last_progress(output)["stage"] == "treinamento_final_inicio"


def test_explicit_failure_keeps_previous_stage(tmp_path):
    output = tmp_path / "run"
    record_progress(output, "calibracao_inicio", details={"fold": 3})
    previous = last_progress(output)["stage"]
    record_progress(output, "falha_da_execucao", status="failed",
                    details={"etapa_anterior": previous})
    assert last_progress(output)["details"]["etapa_anterior"] == "calibracao_inicio"
    assert last_progress(output)["status"] == "failed"


def test_uses_new_folder_if_deleted_between_steps(tmp_path):
    output = tmp_path / "run"
    record_progress(output, "snapshot_validado")
    shutil.rmtree(output)
    record_progress(output, "calibracao_inicio")
    assert last_progress(output)["stage"] == "calibracao_inicio"


def test_missing_or_corrupted_progress_is_handled(tmp_path):
    output = tmp_path / "run"
    assert last_progress(output) is None
    output.mkdir()
    (output / "andamento.jsonl").write_text(
        '{"stage": "snapshot_validado"}\nnao-e-json\n', encoding="utf-8",
    )
    assert last_progress(output)["stage"] == "snapshot_validado"


def test_invalid_status_rejected(tmp_path):
    with pytest.raises(ValueError):
        record_progress(tmp_path / "run", "snapshot_validado",
                        status="success_without_proof")
