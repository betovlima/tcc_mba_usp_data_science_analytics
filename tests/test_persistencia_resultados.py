"""Testes de recuperacao da pasta e preservacao do pacote de evidencia."""
from __future__ import annotations

import importlib
import json
import shutil
from zipfile import ZipFile

import pandas as pd
import pytest

from pesquisa.persistencia_resultados import (
    garantir_pasta_resultados,
    salvar_csv,
    salvar_json,
    salvar_texto,
)


def test_artifact_writers_recreate_deleted_run_directory(tmp_path):
    output = tmp_path / "resultados" / "execucao_1"
    output.mkdir(parents=True)
    shutil.rmtree(output)  # Falha relatada: pasta desaparece durante o treino.
    assert not output.exists()
    frame = pd.DataFrame({"asset": ["CAPR"], "equity": [10000.0]})
    csv = salvar_csv(frame, output / "control_predictions.csv", index=False)
    assert csv.is_file()
    assert pd.read_csv(csv).loc[0, "asset"] == "CAPR"
    shutil.rmtree(output)
    path = salvar_json(
        output / "resultado.json",
        {"status": "completed", "asset_count": 75},
        lambda dest, value: dest.write_text(json.dumps(value), encoding="utf-8"),
    )
    assert json.loads(path.read_text(encoding="utf-8"))["asset_count"] == 75
    shutil.rmtree(output)
    text = salvar_texto(output / "resumo_para_analise.md", "Exame concluido")
    assert text.read_text(encoding="utf-8") == "Exame concluido"
    assert garantir_pasta_resultados(output) == output


@pytest.mark.parametrize(
    ("module_name", "executor_name"),
    [
        ("pesquisar_optimal_switching_cash_spyder", "_executar_experimento"),
        ("pesquisar_vantagem_cash_contrafactual_spyder", "_run"),
    ],
)
def test_exception_keeps_original_error_and_zips_when_directory_disappears(
    tmp_path, monkeypatch, module_name, executor_name,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "out")

    def fail_after_long_training(output):
        assert output.is_dir()
        shutil.rmtree(output)
        raise RuntimeError("Falha cientifica original da simulacao")

    monkeypatch.setattr(module, executor_name, fail_after_long_training)
    with pytest.raises(RuntimeError, match="Falha cientifica original"):
        module.main()

    zips = list((tmp_path / "out").glob("pacote_optimal_switching_cash_*.zip"))
    assert len(zips) == 1
    with ZipFile(zips[0]) as archive:
        assert archive.testzip() is None
        status = json.loads(archive.read("erro_execucao.json"))
        assert status["status"] == "failed"
        assert status["error_type"] == "RuntimeError"
        assert "Falha cientifica original" in status["message"]
        manifest = json.loads(archive.read("manifesto_pacote.json"))
        assert manifest["status"] == "failed"


@pytest.mark.parametrize(
    ("module_name", "executor_name"),
    [
        ("pesquisar_optimal_switching_cash_spyder", "_executar_experimento"),
        ("pesquisar_vantagem_cash_contrafactual_spyder", "_run"),
    ],
)
def test_success_zip_survives_directory_removed_during_execution(
    tmp_path, monkeypatch, module_name, executor_name,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "out")

    def complete_after_missing_directory(output):
        shutil.rmtree(output)
        salvar_csv(
            pd.DataFrame({"equity": [10000.0]}),
            output / "control_predictions.csv",
            index=False,
        )
        salvar_json(
            output / "resultado.json",
            {"status": "completed", "asset_count": 75},
            lambda path, item: path.write_text(json.dumps(item), encoding="utf-8"),
        )
        return {"status": "completed"}

    monkeypatch.setattr(module, executor_name, complete_after_missing_directory)
    result = module.main()
    assert result["status"] == "completed"
    zips = list((tmp_path / "out").glob("pacote_optimal_switching_cash_*.zip"))
    assert len(zips) == 1
    with ZipFile(zips[0]) as archive:
        assert archive.testzip() is None
        assert {"control_predictions.csv", "resultado.json", "manifesto_pacote.json"} <= set(archive.namelist())
