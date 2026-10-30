"""Contratos da reprodução independente Strategy #30."""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path

import pandas as pd
import pytest

from engine.configuracao import CONFIG
from pesquisa.control_operacional_strategy30 import carregar_contrato
from pesquisa.paridade_strategy30 import (
    compare_control_trajectories, config_from_frozen_manifest,
    load_snapshot_strategy30,
)


def _tiny_traces():
    dates = pd.to_datetime(["2020-07-22", "2020-07-23"], utc=True)
    return pd.DataFrame({
        "timestamp": dates,
        "selected_asset": ["NBR", "CXW"],
        "walk_forward_fold": [1, 1],
        "strategy_equity": [10485.56, 11250.0],
    })


def test_perfect_session_parity():
    ref = _tiny_traces()
    contract = {"expected_rows": 2}
    audit, frame = compare_control_trajectories(ref, ref.copy(), contract=contract)
    assert audit["fully_reproduced"] is True
    assert audit["mismatch_sessions"] == 0
    assert len(frame) == 2


def test_one_decision_difference_rejects_parity_even_if_final_capital_same():
    reference = _tiny_traces()
    reproduced = reference.copy()
    reproduced.loc[0, "selected_asset"] = "MSFT"
    audit, _ = compare_control_trajectories(
        reference, reproduced, contract={"expected_rows": 2},
    )
    assert audit["fully_reproduced"] is False
    assert audit["mismatch_sessions"] == 1
    assert audit["first_mismatch_date"].startswith("2020-07-22")


def test_one_cent_equity_discrepancy_blocks_parity():
    reference = _tiny_traces()
    reproduced = reference.copy()
    reproduced.loc[1, "strategy_equity"] += 0.02
    audit, _ = compare_control_trajectories(
        reference, reproduced, contract={"expected_rows": 2},
    )
    assert audit["fully_reproduced"] is False
    assert audit["mismatch_sessions"] == 1


def test_date_shift_blocks_parity():
    reference = _tiny_traces()
    reproduced = reference.copy()
    reproduced.loc[0, "timestamp"] = pd.Timestamp("2020-07-24", tz="UTC")
    audit, _ = compare_control_trajectories(
        reference, reproduced, contract={"expected_rows": 2},
    )
    assert not audit["fully_reproduced"]
    assert audit["mismatch_sessions"] >= 1


def test_strategy_manifest_not_a_numeric_checkpoint_override():
    contract = carregar_contrato()
    minimal = {
        "strategy_profile_id": contract["strategy_profile_id"],
        "strategy_configuration_hash": contract["strategy_configuration_hash"],
        "effective_configuration": {
            "assets": list(contract["assets"]),
            "analysis_end_date": contract["last_session"],
            "research_market_data_protocol": "raw_total_causal_v1",
            "research_model_family": "lightgbm_utility",
            "rotation_xgb_repetitions": 1,
            "rotation_seed_step": 1000,
            "rotation_switch_margin_candidates": [0.0, 0.0025, 0.005, 0.01],
            "research_model_settings": {
                **CONFIG.research_model_settings,
                "soft_horizon_consensus": {"enabled": False},
            },
        },
    }
    config = config_from_frozen_manifest(minimal, contract)
    assert len(config.assets) == 75
    assert config.analysis_end_date == "2026-10-28"
    assert config.strategy_mode == "COMPOUND_ROTATION_SWING_LIGHTGBM"
    assert config.rotation_model_repetitions == 1
    with pytest.raises(RuntimeError, match="Hash da configuracao|Hash da configuração"):
        config_from_frozen_manifest(
            {**minimal, "strategy_configuration_hash": "altered"},
            contract,
        )
    bad = json.loads(json.dumps(minimal))
    bad["effective_configuration"]["analysis_end_date"] = "2026-10-06"
    with pytest.raises(RuntimeError, match="corte temporal"):
        config_from_frozen_manifest(bad, contract)


def test_snapshot_requires_all_75_assets_without_exclusion(tmp_path):
    contract = carregar_contrato()
    manifest = {"market_data_signatures": {
        asset: {"rows": 1} for asset in contract["assets"]
    }}
    with pytest.raises(RuntimeError, match="incompleto"):
        load_snapshot_strategy30(
            tmp_path, contract=contract, experiment_manifest=manifest,
        )


@pytest.mark.parametrize("filename,blob_sha", [
    ("rotacao.py", "7277ba0d49723f15333348a5bd4aa000eca4face"),
    ("modelo_lightgbm.py", "116b5edd9eaeed59e071c7f1e741563a05005443"),
    ("diagnosticos.py", "3cf8910eb659b26fe88f45bdfbda5e25a94408c3"),
    ("execucao.py", "6cb7798e821ed0b6414262fb762ec20e67b58263"),
])
def test_frozen_reference_engine_matches_source_blob(filename, blob_sha):
    folder = Path(__file__).resolve().parents[1] / "pesquisa" / "referencia_strategy30"
    data = (folder / filename).read_bytes()
    git_blob = f"blob {len(data)}".encode("ascii") + bytes([0]) + data
    # A fonte cientifica original tem hash Git; alterações exigem nova versão.
    assert sha1(git_blob).hexdigest() == blob_sha


def test_replay_runner_retains_original_error_in_failure_zip(tmp_path, monkeypatch):
    from zipfile import ZipFile
    import pesquisar_paridade_strategy30_spyder as runner
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "out")
    def fail_before_model(_):
        raise RuntimeError("Snapshot de referencia indisponivel")
    monkeypatch.setattr(runner, "_run", fail_before_model)
    with pytest.raises(RuntimeError, match="Snapshot de referencia"):
        runner.main()
    files = list((tmp_path / "out").glob("*.zip"))
    assert len(files) == 1
    with ZipFile(files[0]) as archive:
        assert archive.testzip() is None
        error = json.loads(archive.read("erro_execucao.json"))
        assert error["status"] == "failed"
