"""Controle importado Strategy #30: identidade, tempo e prova dos 75 ativos."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import pytest

from pesquisa.control_operacional_strategy30 import (
    carregar_contrato, validar_control_strategy30,
)


def _make_fixture(tmp_path):
    contract = carregar_contrato()
    files = contract["required_archive_entries"]
    metric_name = next(k for k in files if k.endswith("_metrics.json"))
    prediction_name = next(k for k in files if k.endswith("_predictions.csv"))
    trades_name = next(k for k in files if k.endswith("_trades.csv"))
    package = tmp_path / "strategy30.zip"
    manifest = {
        "status": "completed",
        "strategy_profile_id": contract["strategy_profile_id"],
        "strategy_configuration_hash": contract["strategy_configuration_hash"],
    }
    experiment = {"job_status": "completed", "eligible_asset_count": 75}
    metrics = {
        "assets": contract["assets"],
        "strategy_configuration_sha256": contract["strategy_configuration_sha256"],
        "market_data_signature_sha256": contract["market_data_signature_sha256"],
        "strategy_ending_capital": 250.0, "initial_capital": 10000,
        "strategy_sharpe": 2.3, "strategy_maximum_drawdown": -0.4,
        "total_transaction_fees": 2.0, "cash_days": 0,
        "capital_rotations": 1,
    }
    predictions = pd.DataFrame([
        {
            "timestamp": "2026-10-27T04:00:00+00:00",
            "walk_forward_fold": 1, "strategy_equity": 175.0,
            "selected_asset": "AAPL",
        },
        {
            "timestamp": "2026-10-28T04:00:00+00:00",
            "walk_forward_fold": 2, "strategy_equity": 250.0,
            "selected_asset": "MSFT",
        },
    ])
    archive_entries = {
        "strategy_manifest.json": json.dumps(manifest).encode(),
        "experiment_manifest.json": json.dumps(experiment).encode(),
        metric_name: json.dumps(metrics).encode(),
        prediction_name: predictions.to_csv(index=False).encode(),
        trades_name: pd.DataFrame({
            "asset": ["AAPL"], "action": ["BUY"], "total_fee": [2.0],
        }).to_csv(index=False).encode(),
    }
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as z:
        for name, content in archive_entries.items():
            z.writestr(name, content)
    fake = {
        **contract,
        "source_file_sha256": sha256(package.read_bytes()).hexdigest(),
        "required_archive_entries": {
            name: sha256(content).hexdigest()
            for name, content in archive_entries.items()
        },
        "expected_rows": 2,
        "first_session": "2026-10-27", "last_session": "2026-10-28",
        "expected_fold_counts": {"1": 1, "2": 1},
        "ending_capital": 250,
        "sharpe": 2.3,
        "max_drawdown": -0.4,
        "total_transaction_fees": 2,
        "capital_rotations": 1,
    }
    return package, fake, archive_entries


def test_current_control_contract_has_75_and_eighty_million():
    contract = carregar_contrato()
    assert contract["effective_asset_count"] == 75
    assert len(contract["assets"]) == 75
    assert len(set(contract["assets"])) == 75
    assert contract["ending_capital"] == pytest.approx(80685454.78647688)
    assert contract["last_session"] == "2026-10-28"
    assert contract["expected_fold_counts"] == {"1": 504, "2": 504, "3": 568}
    assert contract["archival_reference"]["status"] == "archived_historical_reference_not_current_control"


def test_validates_frozen_reference_with_all_75_symbols(tmp_path):
    package, fake, _ = _make_fixture(tmp_path)
    result, predictions, trades = validar_control_strategy30(
        package, contrato=fake,
    )
    assert result["status"] == "validated"
    assert result["effective_asset_count"] == 75
    assert result["model_retrained_here"] is False
    assert result["sessions"] == 2
    assert len(predictions) == 2
    assert len(trades) == 1


def test_refuses_modified_source_even_if_financial_numbers_match(tmp_path):
    package, fake, _ = _make_fixture(tmp_path)
    with ZipFile(package, "a", compression=ZIP_DEFLATED) as z:
        z.writestr("injected_metadata.txt", "extra")
    with pytest.raises(RuntimeError, match="Hash do ZIP divergente"):
        validar_control_strategy30(package, contrato=fake)


def test_refuses_wrong_period_without_replacing_checkpoint(tmp_path):
    package, fake, _ = _make_fixture(tmp_path)
    fake["first_session"] = "2026-10-26"
    with pytest.raises(RuntimeError, match="Periodo do Control divergente"):
        validar_control_strategy30(package, contrato=fake)


def test_refuses_wrong_financial_checkpoint(tmp_path):
    package, fake, _ = _make_fixture(tmp_path)
    fake["ending_capital"] = 587165941.1330212
    with pytest.raises(RuntimeError, match="Capital final do CSV diverge"):
        validar_control_strategy30(package, contrato=fake)


def test_refuses_assets_outside_canonical_seventy_five(tmp_path):
    package, fake, _ = _make_fixture(tmp_path)
    fake["assets"] = fake["assets"][:-1] + ["ZZZZ"]
    with pytest.raises(RuntimeError, match="universo"):
        validar_control_strategy30(package, contrato=fake)


def test_output_script_keeps_failure_zip_if_source_missing(tmp_path, monkeypatch):
    import pesquisar_controle_strategy30_spyder as runner
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(runner, "localizar_zip_original", lambda: tmp_path / "missing.zip")
    with pytest.raises(FileNotFoundError):
        runner.main()
    packages = list((tmp_path / "out").glob("pacote_optimal_switching_cash_*.zip"))
    assert len(packages) == 1
    with ZipFile(packages[0]) as z:
        assert z.testzip() is None
        error = json.loads(z.read("erro_execucao.json"))
        assert error["status"] == "failed"
