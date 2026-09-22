from pathlib import Path

from reproducao.dados import SnapshotPaths
from engine.config import (
    ANALYSIS_END_DATE,
    ASSETS,
    BAR_SNAPSHOT_AS_OF_END,
    CONFIG,
    EXPERIMENT_VERSION,
    REFERENCE_ASSETS,
    SOFT_HORIZON_CONSENSUS_PENALTY,
    build_control_config,
    build_soft_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_reproduction_has_no_database_dependency() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pymongo" not in requirements
    assert "sqlalchemy" not in requirements

    official_files = [
        ROOT / "reproduzir_experimento_spyder.py",
        ROOT / "reproducao" / "dados.py",
        ROOT / "reproducao" / "preparacao.py",
        ROOT / "reproducao" / "experimento.py",
        ROOT / "engine" / "config.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in official_files)
    assert "pymongo" not in joined
    assert "mongodb" not in joined


def test_frozen_universe_and_dates_match_cpu_reference() -> None:
    assert len(ASSETS) == 56
    assert len(REFERENCE_ASSETS) == 25
    assert "DOC" in ASSETS
    assert "CLMT" in ASSETS
    assert ANALYSIS_END_DATE == "2026-09-17"
    assert BAR_SNAPSHOT_AS_OF_END == "2026-09-17"
    assert CONFIG.strategy_mode == "COMPOUND_ROTATION_SWING_LIGHTGBM"
    assert CONFIG.rotation_model_repetitions == 1
    assert CONFIG.rotation_target_horizons == (5, 10, 20, 40, 60)
    assert CONFIG.rotation_target_horizon_weights == (
        0.10,
        0.15,
        0.20,
        0.30,
        0.25,
    )


def test_control_and_soft_share_same_lightgbm() -> None:
    control = build_control_config(CONFIG)
    soft = build_soft_config(CONFIG)

    assert control.research_model_settings["lightgbm"] == soft.research_model_settings["lightgbm"]
    assert control.research_model_settings["soft_horizon_consensus"] == {
        "enabled": False
    }
    assert soft.research_model_settings["soft_horizon_consensus"] == {
        "enabled": True,
        "penalty_strength": SOFT_HORIZON_CONSENSUS_PENALTY,
    }


def test_lightgbm_parameters_are_frozen() -> None:
    settings = CONFIG.research_model_settings["lightgbm"]
    assert settings["n_estimators"] == 329
    assert settings["learning_rate"] == 0.020731
    assert settings["max_depth"] == 3
    assert settings["num_leaves"] == 6
    assert settings["min_child_samples"] == 18
    assert settings["colsample_bytree"] == 0.88067
    assert settings["reg_alpha"] == 0.050837
    assert settings["reg_lambda"] == 3.596305
    assert settings["n_jobs"] == -1
    assert settings["random_state"] == 42


def test_spyder_workflow_is_explicitly_sectioned() -> None:
    source = (ROOT / "reproduzir_experimento_spyder.py").read_text(encoding="utf-8")
    assert source.count("# %%") >= 11
    assert "# %% 7 - CONTROL" in source
    assert "# %% 8 - SOFT HORIZON CONSENSUS" in source
    assert "run_variant(" in source


def test_snapshot_layout_is_csv_per_asset() -> None:
    paths = SnapshotPaths.under(ROOT)
    assert paths.raw_bars.name == "raw_bars"
    assert paths.corporate_actions.name == "corporate_actions"
    assert paths.manifest.name == "manifest.json"


def test_engine_contains_soft_horizon_consensus_policy() -> None:
    source = (ROOT / "engine" / "lightgbm.py").read_text(
        encoding="utf-8"
    )
    assert "def _soft_horizon_consensus_policy(" in source
    assert "weighted_rank_margin_modifier" in source
    assert "SOFT_CONSENSUS_BLOCK_MARGINAL_SWITCH" in source


def test_official_runtime_has_no_historical_references() -> None:
    assert EXPERIMENT_VERSION == "1.1.0-dev"
    forbidden = (
        "series_historicas",
        "tiingo",
        "pymongo",
        "mongodb",
        "caro",
    )
    runtime_files = [
        ROOT / "reproduzir_experimento_spyder.py",
        *sorted((ROOT / "engine").glob("*.py")),
        *sorted((ROOT / "reproducao").glob("*.py")),
    ]
    for path in runtime_files:
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in source, f"{token} found in {path.relative_to(ROOT)}"


def test_runtime_has_no_retired_model_tokens() -> None:
    forbidden = ("x" + "gb", "x" + "gboost")
    runtime_files = [
        ROOT / "reproduzir_experimento_spyder.py",
        *sorted((ROOT / "engine").glob("*.py")),
        *sorted((ROOT / "reproducao").glob("*.py")),
    ]
    for path in runtime_files:
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in source, f"{token} found in {path.relative_to(ROOT)}"


def test_snapshot_does_not_persist_derived_normalized_bars() -> None:
    data_source = (ROOT / "reproducao" / "dados.py").read_text(encoding="utf-8")
    preparation_source = (ROOT / "reproducao" / "preparacao.py").read_text(
        encoding="utf-8"
    )
    assert "normalized_bars" not in data_source
    assert "normalized_bars" not in preparation_source
    assert "write_normalized_csv" not in preparation_source


def test_engine_contains_only_current_modules() -> None:
    expected = {
        "__init__.py",
        "config.py",
        "diagnostics.py",
        "execution.py",
        "lightgbm.py",
        "rotation.py",
    }
    actual = {
        path.name
        for path in (ROOT / "engine").glob("*.py")
    }
    assert actual == expected


def test_engine_has_no_retired_strategy_modes() -> None:
    forbidden = (
        "risk_off",
        "selective_opportunity",
        "opportunity_cash_gate",
        "absolute_utility",
        "optimized_allocation",
        "concentrated_allocation",
        "compound_risk_overlay",
        "iqn",
        "horizon_voting",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sorted((ROOT / "engine").glob("*.py"))
    )
    for token in forbidden:
        assert token not in source, token
