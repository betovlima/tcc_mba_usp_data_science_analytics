import ast
from pathlib import Path

import pandas as pd

from reproducao.dados import SnapshotPaths
from engine.rotacao import _executar_compra, _selecionar_ativo_fonte_calendario
from engine.configuracao import (
    ANALYSIS_END_DATE,
    ASSETS,
    BAR_SNAPSHOT_AS_OF_END,
    CONFIG,
    EXPERIMENT_VERSION,
    SOFT_HORIZON_CONSENSUS_PENALTY,
    construir_configuracao_controle,
    construir_configuracao_soft,
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
        ROOT / "engine" / "configuracao.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in official_files)
    assert "pymongo" not in joined
    assert "mongodb" not in joined


def test_frozen_universe_and_dates_match_cpu_reference() -> None:
    assert len(ASSETS) == 56
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



def test_asset_universe_has_no_manual_reference_or_candidate_split() -> None:
    config_source = (ROOT / "engine" / "configuracao.py").read_text(encoding="utf-8")
    runtime_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "engine" / "rotacao.py",
            ROOT / "engine" / "modelo_lightgbm.py",
            ROOT / "reproducao" / "experimento.py",
        ]
    )
    forbidden = (
        "REFERENCE_ASSETS",
        "CANDIDATE_ASSETS",
        "calendar_anchor_assets",
        "research_reference_assets",
        "research_candidate_assets",
    )
    for token in forbidden:
        assert token not in config_source
        assert token not in runtime_source


def test_calendar_source_is_derived_from_longest_valid_asset_history() -> None:
    short_index = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    long_index = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    frames = {
        "SHORT": pd.DataFrame(index=short_index),
        "LONG": pd.DataFrame(index=long_index),
    }

    assert _selecionar_ativo_fonte_calendario(frames) == "LONG"


def test_execution_helpers_use_portuguese_names() -> None:
    execution_source = (ROOT / "engine" / "execucao.py").read_text(
        encoding="utf-8"
    )
    experiment_source = (ROOT / "reproducao" / "experimento.py").read_text(
        encoding="utf-8"
    )

    for expected in (
        "arredondar_taxa_para_centavo",
        "calcular_taxas_referencia",
        "aplicar_deslizamento",
    ):
        assert expected in execution_source
        assert expected in experiment_source or expected == "arredondar_taxa_para_centavo"

    for retired in (
        "round_fee_to_cent",
        "calculate_reference_fees",
        "apply_slippage",
    ):
        assert retired not in execution_source
        assert retired not in experiment_source

def test_control_and_soft_share_same_lightgbm() -> None:
    control = construir_configuracao_controle(CONFIG)
    soft = construir_configuracao_soft(CONFIG)

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
    research = SnapshotPaths.research(ROOT)
    temporary = SnapshotPaths.temporary(ROOT)

    assert research.root == ROOT / "dados" / "pesquisa_v1"
    assert research.raw_bars.name == "raw_bars"
    assert research.corporate_actions.name == "corporate_actions"
    assert research.manifest.name == "manifest.json"

    assert temporary.root == (
        ROOT / "dados" / "temporario" / "reproducao_v1"
    )


def test_engine_contains_soft_horizon_consensus_policy() -> None:
    source = (ROOT / "engine" / "modelo_lightgbm.py").read_text(
        encoding="utf-8"
    )
    assert "def _politica_consenso_horizontes_soft(" in source
    assert "weighted_rank_margin_modifier" in source
    assert "SOFT_CONSENSUS_BLOCK_MARGINAL_SWITCH" in source


def test_official_runtime_has_no_historical_references() -> None:
    assert EXPERIMENT_VERSION == "1.2.0-dev.5"
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
        "configuracao.py",
        "diagnosticos.py",
        "execucao.py",
        "modelo_lightgbm.py",
        "rotacao.py",
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


def test_fractional_execution_is_fixed_experiment_semantics() -> None:
    quantity, execution_price, fees = _executar_compra(
        100.0,
        30.0,
        CONFIG,
        lambda side, qty, price, config: {"total_fee": 0.0},
        lambda price, side, config: price,
    )
    assert execution_price == 30.0
    assert fees["total_fee"] == 0.0
    assert abs(quantity - (100.0 / 30.0)) < 1e-12


def test_runtime_config_attribute_contract() -> None:
    config_attributes = set(CONFIG.__dataclass_fields__)
    config_attributes.add("copiar_modelo")
    runtime_files = [
        *sorted((ROOT / "engine").glob("*.py")),
        *sorted((ROOT / "reproducao").glob("*.py")),
    ]
    runtime_files = [
        path for path in runtime_files
        if path.name != "configuracao.py"
    ]

    referenced: set[str] = set()
    config_names = {"config", "rep_config"}

    for path in runtime_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in config_names
            ):
                referenced.add(node.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in config_names
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                referenced.add(node.args[1].value)

    missing = sorted(referenced - config_attributes)
    assert not missing, (
        "runtime config attributes missing from StandaloneBacktestConfig: "
        f"{missing}"
    )


def test_research_data_is_versioned_and_temporary_data_is_ignored() -> None:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "dados/temporario/" in rules
    assert "dados/reproducao_v1/" in rules
    assert "!dados/pesquisa_v1/raw_bars/*.csv" in rules
    assert "!dados/pesquisa_v1/corporate_actions/*.csv" in rules
    assert "!dados/pesquisa_v1/manifest.json" in rules


def test_spyder_data_modes_protect_frozen_research_snapshot() -> None:
    source = (ROOT / "reproduzir_experimento_spyder.py").read_text(
        encoding="utf-8"
    )
    assert "USAR_DADOS_PESQUISA_CONGELADOS = False" in source
    assert "CAMINHOS_TEMPORARIOS" in source
    assert "modo=download-temporario-forcado" in source
    assert "modo=download-temporario-reutilizavel" in source
    assert "modo=pesquisa-versionada" in source
    assert "CAMINHOS = CAMINHOS_TEMPORARIOS" in source
    assert "CAMINHOS = CAMINHOS_PESQUISA" in source
    assert "modo=atualizar-pesquisa-versionada" not in source
    assert "replace=FORCAR_DOWNLOAD" in source
