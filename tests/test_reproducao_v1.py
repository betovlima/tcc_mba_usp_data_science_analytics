import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from reproducao.dados import SnapshotPaths
from reproducao.preparacao import load_raw_bar_file, structural_identity_issue
from reproducao.graficos import (
    calcular_retornos_mensais,
    construir_rotacoes,
    gerar_analises_backtest,
)
from engine.rotacao import (
    _datas_decisao_analise,
    _executar_compra,
    _selecionar_ativo_fonte_calendario,
)
from engine.modelo_lightgbm import selecionar_switch_margin
from engine.configuracao import (
    ANALYSIS_END_DATE,
    ASSETS,
    BAR_SNAPSHOT_AS_OF_END,
    CONFIG,
    EXPERIMENT_VERSION,
    U67_REQUESTED_ASSETS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_round_trip_csv_parser_preserves_float64_for_reproduction(tmp_path) -> None:
    original = 950.4636963259353
    path = tmp_path / "AAA.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-02T05:00:00+00:00",
                "open": original,
                "high": original,
                "low": original,
                "close": original,
                "volume": original,
            }
        ]
    ).to_csv(path, index=False, float_format="%.17g")

    frame = load_raw_bar_file(
        path,
        float_precision="round_trip",
    )

    assert float(frame.iloc[0]["open"]) == original
    assert float(frame.iloc[0]["high"]) == original
    assert float(frame.iloc[0]["low"]) == original
    assert float(frame.iloc[0]["close"]) == original
    assert float(frame.iloc[0]["volume"]) == original



def test_all_runtime_modules_import_successfully() -> None:
    modules = (
        "engine.configuracao",
        "engine.diagnosticos",
        "engine.execucao",
        "engine.rotacao",
        "engine.modelo_lightgbm",
        "reproducao.artefatos",
        "reproducao.dados",
        "reproducao.preparacao",
        "reproducao.experimento",
        "reproducao.graficos",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None

def test_official_reproduction_has_no_database_dependency() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pymongo" not in requirements
    assert "sqlalchemy" not in requirements

    official_files = [
        ROOT / "reproduzir_experimento.py",
        ROOT / "reproducao" / "dados.py",
        ROOT / "reproducao" / "preparacao.py",
        ROOT / "reproducao" / "experimento.py",
        ROOT / "engine" / "configuracao.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in official_files)
    assert "pymongo" not in joined
    assert "mongodb" not in joined


def test_frozen_universe_and_dates_match_cpu_reference() -> None:
    assert len(ASSETS) == 67
    assert U67_REQUESTED_ASSETS == ASSETS == CONFIG.assets
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



def test_clmt_cusip_transition_accepts_float_parsed_cusip() -> None:
    issue = structural_identity_issue(
        "CLMT",
        [
            {
                "action_type": "name_change",
                "old_symbol": "CLMT",
                "new_symbol": "CLMT",
                "old_cusip": 131476103.0,
                "new_cusip": 131428104.0,
                "process_date": "2024-07-11",
            }
        ],
    )

    assert issue is not None
    assert issue["reason"] == "structural_identity_change"
    assert issue["symbol"] == "CLMT"


def test_clmt_cusip_transition_is_structurally_excluded() -> None:
    issue = structural_identity_issue(
        "CLMT",
        [
            {
                "action_type": "name_change",
                "old_symbol": "CLMT",
                "new_symbol": "CLMT",
                "old_cusip": "131476103",
                "new_cusip": "131428104",
                "process_date": "2024-07-11",
            }
        ],
    )

    assert issue is not None
    assert issue["reason"] == "structural_identity_change"
    assert issue["old_cusip"] == "131476103"
    assert issue["new_cusip"] == "131428104"


def test_asset_universe_has_no_manual_reference_or_candidate_split() -> None:
    config_source = (ROOT / "engine" / "configuracao.py").read_text(encoding="utf-8")
    runtime_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "engine" / "rotacao.py",
            ROOT / "engine" / "modelo_lightgbm.py",
            ROOT / "reproducao" / "experimento.py",
            ROOT / "reproduzir_experimento.py",
        ]
    )
    forbidden = (
        "REFERENCE_ASSETS",
        "CANDIDATE_ASSETS",
        "calendar_anchor_assets",
        "research_reference_assets",
        "research_candidate_assets",
        "u56_eligible",
        "U56_FIXED",
        "benchmark_override",
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

    for expected in (
        "arredondar_taxa_para_centavo",
        "calcular_taxas_referencia",
        "aplicar_deslizamento",
    ):
        assert expected in execution_source

    for retired in (
        "round_fee_to_cent",
        "calculate_reference_fees",
        "apply_slippage",
    ):
        assert retired not in execution_source


def test_switch_margin_selects_best_calibration_score() -> None:
    selection = selecionar_switch_margin(
        [(0.0, 1.0), (0.0025, 2.0), (0.005, 1.5)]
    )
    assert selection == {
        "selected_candidate_margin": 0.0025,
        "selected_calibration_score": 2.0,
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


def test_official_runner_delegates_to_exact_replay() -> None:
    source = (ROOT / "reproduzir_experimento.py").read_text(encoding="utf-8")
    assert "67" not in source
    assert "reproduzir_universo_congelado(" in source
    assert "REPRODUCTION_VERSION = \"1.23.0\"" in source
    assert "EXECUTION_SCHEMA = \"universe-reproduction-v3\"" in source
    assert "pesquisar_expansao_u67_spyder" not in source
    assert "download_raw_bars" not in source
    assert "download_corporate_actions" not in source


def test_snapshot_layout_is_csv_per_asset() -> None:
    snapshot = SnapshotPaths.u67(ROOT)

    assert snapshot.root == ROOT / "dados" / "u67"
    assert snapshot.raw_bars.name == "raw_bars"
    assert snapshot.corporate_actions.name == "corporate_actions"
    assert snapshot.manifest.name == "manifest.json"




def test_analysis_window_can_end_on_current_temporary_session() -> None:
    common_dates = pd.to_datetime(
        [
            "2026-09-15 04:00:00+00:00",
            "2026-09-16 04:00:00+00:00",
            "2026-09-17 04:00:00+00:00",
            "2026-09-18 04:00:00+00:00",
            "2026-09-21 04:00:00+00:00",
            "2026-09-22 04:00:00+00:00",
        ],
        utc=True,
    )
    folds = [
        {
            "test_start_index": 1,
            "test_end_index": len(common_dates),
        }
    ]
    config = CONFIG.copiar_modelo(
        update={
            "analysis_start_date": "2026-09-15",
            "analysis_end_date": "2026-09-22",
        }
    )

    decision_dates = _datas_decisao_analise(
        common_dates,
        folds,
        config,
    )

    assert decision_dates[-1].date().isoformat() == "2026-09-22"


def test_analysis_end_date_is_inclusive_for_nyse_utc_timestamp() -> None:
    common_dates = pd.to_datetime(
        [
            "2026-09-21 04:00:00+00:00",
            "2026-09-22 04:00:00+00:00",
            "2026-09-23 04:00:00+00:00",
        ],
        utc=True,
    )
    folds = [
        {
            "test_start_index": 1,
            "test_end_index": len(common_dates),
        }
    ]
    config = CONFIG.copiar_modelo(
        update={
            "analysis_start_date": "2026-09-21",
            "analysis_end_date": "2026-09-22",
        }
    )

    dates = _datas_decisao_analise(common_dates, folds, config)

    assert dates[-1] == pd.Timestamp("2026-09-22 04:00:00+00:00")
    assert pd.Timestamp("2026-09-23 04:00:00+00:00") not in dates


def test_official_reproduction_uses_exact_input_manifest() -> None:
    source = (ROOT / "reproduzir_experimento.py").read_text(encoding="utf-8")
    helper = (ROOT / "reproducao" / "experimento.py").read_text(encoding="utf-8")
    assert "67" not in source
    assert "manifest_universo_congelado.json" in helper
    assert "_train_baseline_states(" in helper
    assert "_train_candidate_caches(" in helper
    assert "_evaluate_scenario(" in helper
    assert "digest(path)" in helper
    assert "download_raw_bars(" not in helper
    assert "download_corporate_actions(" not in helper


def test_u67_snapshot_preparation_is_separate_from_reproduction() -> None:
    source = (ROOT / "preparar_snapshot_u67.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "download_raw_bars" in source
    assert "download_corporate_actions" in source
    assert "load_alpaca_credentials" in source
    assert "snapshot_end_date = \"2026-10-06\"" in source
    assert "dados" in source and ".u67_build" in source

def test_capital_rotations_follow_backtest_analytics_semantics() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2020-01-02T00:00:00+00:00",
                "action": "BUY",
                "asset": "AAA",
                "rotation_id": "r1",
                "rotation_from_asset": "CASH",
                "rotation_to_asset": "AAA",
                "total_fee": 1.0,
                "execution_price": 10.0,
            },
            {
                "timestamp": "2020-01-10T00:00:00+00:00",
                "action": "SELL",
                "asset": "AAA",
                "rotation_id": "r2",
                "rotation_from_asset": "AAA",
                "rotation_to_asset": "BBB",
                "total_fee": 2.0,
                "execution_price": 12.0,
                "realized_pnl": 200.0,
                "position_return": 0.20,
                "holding_bars": 8,
            },
            {
                "timestamp": "2020-01-10T00:00:00+00:00",
                "action": "BUY",
                "asset": "BBB",
                "rotation_id": "r2",
                "rotation_from_asset": "AAA",
                "rotation_to_asset": "BBB",
                "total_fee": 1.0,
                "execution_price": 20.0,
            },
            {
                "timestamp": "2020-01-31T00:00:00+00:00",
                "action": "FINAL_SELL",
                "asset": "BBB",
                "total_fee": 1.0,
                "realized_pnl": 100.0,
            },
        ]
    )

    rotations = construir_rotacoes(trades)

    assert len(rotations) == 2
    assert rotations.iloc[0]["from_asset"] == "CASH"
    assert rotations.iloc[0]["to_asset"] == "AAA"
    assert rotations.iloc[1]["from_asset"] == "AAA"
    assert rotations.iloc[1]["to_asset"] == "BBB"
    assert rotations.iloc[1]["transaction_fees"] == 3.0


def test_monthly_returns_match_backtest_analytics_reference() -> None:
    index = pd.to_datetime(
        [
            "2020-01-02T00:00:00Z",
            "2020-01-31T00:00:00Z",
            "2020-02-28T00:00:00Z",
            "2020-03-31T00:00:00Z",
        ],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "strategy_equity": [100.0, 110.0, 121.0, 108.9],
            "buy_hold_equity": [100.0, 105.0, 107.1, 117.81],
        },
        index=index,
    )

    monthly = calcular_retornos_mensais(predictions)

    assert list(monthly["month"]) == ["2020-02", "2020-03"]
    assert abs(monthly.iloc[0]["simulation_return"] - 0.10) < 1e-12
    assert abs(monthly.iloc[0]["reference_return"] - 0.02) < 1e-12
    assert abs(monthly.iloc[1]["simulation_return"] + 0.10) < 1e-12
    assert abs(monthly.iloc[1]["reference_return"] - 0.10) < 1e-12


def test_active_paths_no_longer_use_v1_suffix() -> None:
    active_files = [
        ROOT / "reproduzir_experimento.py",
        ROOT / "reproducao" / "dados.py",
        ROOT / ".gitignore",
        ROOT / "README.md",
        ROOT / "MANIFEST.md",
    ]
    for path in active_files:
        source = path.read_text(encoding="utf-8")
        assert "pesquisa_v1" not in source
        assert "reproducao_v1" not in source


def test_backtest_analytics_output_contract() -> None:
    source = (ROOT / "reproducao" / "graficos.py").read_text(
        encoding="utf-8"
    )
    expected = (
        "monthly_realized_pnl_",
        "monthly_realized_pnl_heatmap_",
        "monthly_returns_",
        "monthly_return_heatmap_",
        "capital_rotations_",
        "capital_rotations_monthly_",
        "capital_rotations_transition_matrix_",
        "capital_rotations_heatmap_",
        "backtest_analytics.xlsx",
    )
    for token in expected:
        assert token in source

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "matplotlib" in requirements
    assert "openpyxl" in requirements


def test_backtest_analytics_generation_creates_expected_files(tmp_path) -> None:
    index = pd.to_datetime(
        [
            "2020-01-31T00:00:00Z",
            "2020-02-28T00:00:00Z",
            "2020-03-31T00:00:00Z",
        ],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "strategy_equity": [100.0, 110.0, 121.0],
            "buy_hold_equity": [100.0, 105.0, 110.25],
        },
        index=index,
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2020-01-31T00:00:00+00:00",
                "action": "BUY",
                "asset": "AAA",
                "rotation_id": "r1",
                "rotation_from_asset": "CASH",
                "rotation_to_asset": "AAA",
                "total_fee": 1.0,
                "execution_price": 10.0,
            },
            {
                "timestamp": "2020-02-28T00:00:00+00:00",
                "action": "SELL",
                "asset": "AAA",
                "rotation_id": "r2",
                "rotation_from_asset": "AAA",
                "rotation_to_asset": "BBB",
                "total_fee": 1.0,
                "execution_price": 12.0,
                "realized_pnl": 200.0,
                "position_return": 0.20,
                "holding_bars": 20,
            },
            {
                "timestamp": "2020-02-28T00:00:00+00:00",
                "action": "BUY",
                "asset": "BBB",
                "rotation_id": "r2",
                "rotation_from_asset": "AAA",
                "rotation_to_asset": "BBB",
                "total_fee": 1.0,
                "execution_price": 20.0,
            },
            {
                "timestamp": "2020-03-31T00:00:00+00:00",
                "action": "FINAL_SELL",
                "asset": "BBB",
                "total_fee": 1.0,
                "realized_pnl": 100.0,
                "position_return": 0.05,
                "holding_bars": 22,
            },
        ]
    )
    result = SimpleNamespace(predictions=predictions, trades=trades)

    generated = gerar_analises_backtest(
        tmp_path,
        manifest={"snapshot_sha256": "test-snapshot"},
        result=result,
    )

    graph_dir = tmp_path / "graficos"
    assert generated["graficos_dir"] == graph_dir
    assert (graph_dir / "backtest_analytics.xlsx").exists()
    assert (graph_dir / "capital_rotations_control.csv").exists()
    assert (graph_dir / "capital_rotations_heatmap_control.png").exists()
    assert (graph_dir / "capital_rotations_heatmap_control.svg").exists()
    assert (graph_dir / "monthly_realized_pnl_heatmap_control.png").exists()
    assert (graph_dir / "monthly_return_heatmap_control_simulation.png").exists()
    assert (graph_dir / "monthly_return_heatmap_control_excess.svg").exists()

def test_official_runtime_has_no_historical_references() -> None:
    assert EXPERIMENT_VERSION == "1.22.1"
    forbidden = (
        "series_historicas",
        "tiingo",
        "pymongo",
        "mongodb",
        "caro",
    )
    runtime_files = [
        ROOT / "reproduzir_experimento.py",
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
        ROOT / "reproduzir_experimento.py",
        *sorted((ROOT / "engine").glob("*.py")),
        *sorted((ROOT / "reproducao").glob("*.py")),
    ]
    for path in runtime_files:
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in source, f"{token} found in {path.relative_to(ROOT)}"


def test_data_preparation_has_no_legacy_reference_file_dependency() -> None:
    source = (ROOT / "reproducao" / "preparacao.py").read_text(
        encoding="utf-8"
    )
    assert "reference_10_8_74_raw_snapshot_diagnostics.json" not in source
    assert "REFERENCE_FILE" not in source
    assert "comparar_snapshot_referencia" not in source


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


def test_u67_data_is_versioned_and_build_area_is_ignored() -> None:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "dados/.u67_build/" in rules
    assert "!dados/u67/raw_bars/*.csv" in rules
    assert "!dados/u67/corporate_actions/*.csv" in rules
    assert "!dados/u67/manifest.json" in rules
    assert "!dados/pesquisa/" not in rules


def test_official_reproduction_uses_selected_universe_contract() -> None:
    import json

    source = (ROOT / "reproduzir_experimento.py").read_text(encoding="utf-8")
    contract = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assert "67" not in source
    assert "76_927_051" not in source
    assert contract["analysis_end_date"] == "2026-10-06"
    assert contract["snapshot_sha256"] == (
        "e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575"
    )
    assert contract["expected_ending_capital"] == 587_165_941.1330212
    assert contract["selected_assets"] == [
        "CRT", "NBR", "CAPR", "PNBK", "PENN",
        "JBL", "STAG", "MCHB", "LAD", "TOWN",
    ]
    assert len(contract["candidate_file_hashes"]) == 10
    assert all(
        len(digests[field]) == 64
        for digests in contract["candidate_file_hashes"].values()
        for field in ("raw_sha256", "corporate_actions_sha256")
    )


def test_project_has_no_external_system_reference_token() -> None:
    forbidden_tokens = ("m" + "ct", "market " + "cycle trader")
    ignored_parts = {".git", "__pycache__", ".pytest_cache", "output"}

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".yml",
            ".yaml",
            ".txt",
        } and path.name != ".gitignore":
            continue

        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8", errors="ignore").lower()
        for forbidden in forbidden_tokens:
            assert forbidden not in str(relative).lower(), relative
            assert forbidden not in source, relative

def test_official_runner_silences_only_pandas4_warning() -> None:
    source = (ROOT / "reproduzir_experimento.py").read_text(
        encoding="utf-8"
    )
    assert 'getattr(pd.errors, "Pandas4Warning", None)' in source
    assert 'warnings.filterwarnings(' in source
    assert '"ignore"' in source
    assert 'warnings.filterwarnings("ignore")' not in source


def test_rotation_advantage_reuses_one_verified_report_zip(tmp_path) -> None:
    """Tests the actual ZIP publisher without running a costly TCC backtest."""
    import hashlib
    import json
    import math
    import shutil
    from datetime import datetime
    from zipfile import ZIP_DEFLATED, ZipFile

    import numpy as np
    import pytest

    script = ROOT / "pesquisar_permanencia_minima_spyder.py"
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = {
        "_hash_bytes", "_json_safe", "_write_json",
        "_pacote", "_publicar_pasta_vantagem",
    }
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    assert {node.name for node in definitions} == selected
    env = {
        "hashlib": hashlib, "json": json, "math": math,
        "shutil": shutil, "datetime": datetime,
        "np": np, "pd": pd, "Path": Path,
        "ZipFile": ZipFile, "ZIP_DEFLATED": ZIP_DEFLATED,
        "DESTINO": tmp_path / "pesquisa_vantagem_rotacao",
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(script), "exec"), env)
    stage = tmp_path / ".pesquisa_vantagem_rotacao_em_execucao"
    source_package = tmp_path / "pesquisa_permanencia_minima.zip"
    source_package.write_bytes(b"frozen-input")

    for content in ("primeiro", "segundo"):
        stage.mkdir()
        (stage / "comparacao_resultados.csv").write_text(content, encoding="utf-8")
        archive = env["_pacote"](stage, source_package)
        env["_publicar_pasta_vantagem"](stage)
        assert archive == tmp_path / "pesquisa_vantagem_rotacao.zip"
        assert archive.is_file()
        assert (tmp_path / "pesquisa_vantagem_rotacao" / "comparacao_resultados.csv").read_text(encoding="utf-8") == content
        with ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            assert zipped.read("pesquisa_vantagem_rotacao/comparacao_resultados.csv") == content.encode("utf-8")
            checksums = json.loads(zipped.read("pesquisa_vantagem_rotacao/arquivos_sha256.json"))
            assert checksums["comparacao_resultados.csv"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
            assert zipped.read("dados_origem/pesquisa_permanencia_minima.zip") == b"frozen-input"
    assert not list(tmp_path.glob("pesquisa_vantagem_rotacao_v*.zip"))

    # A failed new package must leave the last complete ZIP in place.
    previous = archive.read_bytes()
    stage.mkdir()
    (stage / "comparacao_resultados.csv").write_text("incompleto", encoding="utf-8")
    source_package.unlink()
    with pytest.raises(FileNotFoundError):
        env["_pacote"](stage, source_package)
    assert archive.read_bytes() == previous
    assert (tmp_path / "pesquisa_vantagem_rotacao" / "comparacao_resultados.csv").read_text(encoding="utf-8") == "segundo"
    assert "pesquisa_vantagem_rotacao_v2" not in source
    assert "pesquisa_vantagem_rotacao_v3" not in source


def test_deterioracao_replay_respeita_cash() -> None:
    """O modo antigo de replay nao pode forcar recompra quando a protecao quer CASH."""
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    b = SimpleNamespace(
        dates=pd.date_range("2026-01-02", periods=3, freq="B"),
        exec_dates=pd.date_range("2026-01-05", periods=3, freq="B"),
        fold=np.array([1, 2, 2]),
        symbols=["AAA"],
        scores=np.array([[0.01], [0.02], [0.03], [0.04]]),
        base_actions=np.array([-1, -1, -1]),
        op=np.array([[10.0], [10.0], [10.0], [10.0]]),
        cl=np.array([[10.0], [10.0], [10.0], [10.0]]),
        volume=np.full((4, 1), 100.0),
        config={"initial_capital": 10000.0},
    )
    positions = []
    def keep_cash(_, i, current, holding, capital, quantity):
        positions.append((i, current, capital))
        return -1, []
    curve, decisions, trades, metrics = script.simular(
        b, decision_callback=keep_cash, permitir_cash=True,
    )
    assert positions == [(1, -1, 10000.0), (2, -1, 10000.0)]
    assert np.allclose(curve, 10000.0)
    assert (decisions["posicao_depois"] == "CASH").all()
    assert trades.empty
    assert metrics["rotacoes"] == 0


def test_deterioracao_control_usa_posicao_real() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    b = SimpleNamespace(
        fold=np.array([2, 2]),
        scores=np.array([[0.020, 0.080], [0.010, 0.0001]]),
        margens={2: 0.005},
        config={
            "rotation_cash_threshold": 0.0,
            "rotation_min_expected_edge": 0.001,
            "rotation_switch_margin": 0.0005,
            "rotation_min_holding_days": 2,
        },
    )
    # Minimo de permanencia so impede troca antes de 2 sessoes.
    assert script._d_control(b, 0, 0, 1) == 0
    assert script._d_control(b, 0, 0, 2) == 1
    assert script._d_control(b, 0, -1, 0) == 1
    assert script._d_control(b, 1, -1, 0) == 0


def test_deterioracao_features_nao_usam_preco_futuro() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    n, m = 80, 2
    close = np.column_stack((
        np.linspace(10, 20, n + 1), np.linspace(20, 10, n + 1),
    ))
    b = SimpleNamespace(
        dates=pd.date_range("2026-01-01", periods=n, freq="B"),
        fold=np.array([1] * 40 + [2] * 40),
        symbols=["AAA", "BBB"],
        scores=np.column_stack((
            np.full(n + 1, 0.02), np.full(n + 1, 0.01),
        )),
        op=close.copy(), cl=close.copy(),
        ret5=np.column_stack((np.full(n + 1, 0.01), np.full(n + 1, -0.01))),
        vol20=np.full((n + 1, m), 0.01),
        adv20=np.full((n + 1, m), 100_000.0),
    )
    before_x, before_y, _ = script._d_features_labels(b)
    b.cl[55:, 0] *= 0.5
    after_x, after_y, _ = script._d_features_labels(b)
    np.testing.assert_array_equal(before_x[40], after_x[40])
    assert before_y[40, 0] != after_y[40, 0]
    assert script.D_PURGE >= script.D_HORIZONTE


def test_sensibilidade_margens_recalcula_fold_1_somente_quando_pedido() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    assert script.MODO == "permanencia_inteligente"
    assert script.MARGENS_ALTERNATIVAS == {1: 0.01, 2: 0.0, 3: 0.01}
    b = SimpleNamespace(
        dates=pd.date_range("2026-01-02", periods=3, freq="B"),
        exec_dates=pd.date_range("2026-01-05", periods=3, freq="B"),
        fold=np.array([1, 1, 1]),
        symbols=["AAA"],
        scores=np.array([[0.01], [0.02], [0.03], [0.04]]),
        base_actions=np.array([-1, -1, -1]),
        op=np.full((4, 1), 10.0), cl=np.full((4, 1), 10.0),
        volume=np.full((4, 1), 100.0),
        config={"initial_capital": 10000.0},
    )
    calls = []

    def keep_cash(_, i, current, holding, capital, quantity):
        calls.append(int(i))
        return -1, []

    original, _, _, _ = script.simular(
        b, decision_callback=keep_cash, permitir_cash=True,
    )
    assert calls == []
    alternativa, _, _, _ = script.simular(
        b, decision_callback=keep_cash,
        permitir_cash=True, aplicar_todos_folds=True,
    )
    assert calls == [0, 1, 2]
    np.testing.assert_array_equal(original, alternativa)
    np.testing.assert_array_equal(original, np.full(3, 10000.0))


def test_sensibilidade_margens_nao_altera_pesquisa_de_deterioracao() -> None:
    script = (ROOT / "pesquisar_permanencia_minima_spyder.py").read_text(
        encoding="utf-8",
    )
    assert 'MODO = "permanencia_inteligente"' in script
    assert 'elif MODO == "deterioracao"' in script
    assert 'permitir_cash=True, aplicar_todos_folds=True' in script
    assert "OOS" not in script.split("# %% 12. Sensibilidade")[1].split(
        'if __name__ == "__main__":'
    )[0]


def test_analise_permanencia_usa_indicadores_anteriores_ao_retorno() -> None:
    """A primeira observacao apos a decisao usa patrimonio anterior como base."""
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    dates = pd.date_range("2026-01-01", periods=9, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates.astype(str),
        "decision_date": (dates - pd.Timedelta(days=1)).astype(str),
        "current_asset": ["CASH"] + ["AAA"] * 8,
        "selected_asset": ["AAA"] * 9,
        "strategy_equity": [100, 90, 95, 102, 80, 82, 110, 120, 130],
        "walk_forward_fold": [1] * 9,
        "holding_days_at_decision": list(range(9)),
        "position_drawdown_from_peak": [0, 0, -.12, -.2, -.25, -.1, 0, 0, 0],
        "position_return_since_entry": [0] * 9,
        "score_change_from_entry": [0, 0, -.2, -.2, .1, .2, 0, 0, 0],
        "current_score": [0.1] * 9,
        "raw_best_score": [0.1] * 9,
        "decision_reason": ["HOLD_CURRENT_BEST"] * 9,
        "universe_breadth_5": [0.5] * 9,
        "consecutive_days_current_not_top1": [0] * 9,
    })
    output, episodes, stats = script._a_calcular(df)
    assert len(episodes) == 1
    assert output.loc[2, "retorno_observado_1d"] == 95 / 90 - 1
    assert output.loc[2, "retorno_observado_3d"] == 80 / 90 - 1
    assert output.loc[0, "posicao_mantida"] == False
    assert output["retorno_observado_5d"].notna().sum() > 0
    assert set(stats["horizonte"]) == {1, 3, 5}
    assert np.isfinite(episodes["retorno_capital"].iloc[0])


def test_analise_permanencia_censura_horizonte_quando_ha_rotacao() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    dates = pd.date_range("2026-01-01", periods=7, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates.astype(str),
        "decision_date": (dates - pd.Timedelta(days=1)).astype(str),
        "current_asset": ["CASH", "AAA", "AAA", "AAA", "AAA", "BBB", "BBB"],
        "selected_asset": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        "strategy_equity": [100, 105, 110, 115, 120, 130, 135],
        "walk_forward_fold": [1] * 7,
        "holding_days_at_decision": [0, 1, 2, 3, 4, 1, 2],
        "position_drawdown_from_peak": [0] * 7,
        "position_return_since_entry": [0] * 7,
        "score_change_from_entry": [0] * 7,
        "current_score": [0.1] * 7,
        "raw_best_score": [0.1] * 7,
        "decision_reason": ["HOLD_CURRENT_BEST"] * 7,
        "universe_breadth_5": [0.5] * 7,
        "consecutive_days_current_not_top1": [0] * 7,
    })
    output, episodes, _ = script._a_calcular(df)
    assert len(episodes) == 2
    assert np.isnan(output.loc[3, "retorno_observado_3d"])
    assert output.loc[3, "retorno_observado_1d"] == 115 / 110 - 1
    assert np.isnan(output.loc[4, "retorno_observado_1d"])
    assert output.loc[6, "retorno_observado_1d"] == 135 / 130 - 1


def test_alertas_cauda_ficam_agrupados_por_episodio() -> None:
    """Duas ocorrencias no mesmo episodio nao equivalem a dois eventos."""
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    historical = pd.DataFrame({
        "posicao_mantida": [True, True, True, True, False],
        "retorno_observado_1d": [-0.12, +0.02, -0.03, -0.06, None],
        "episodio_id": [1, 1, 2, 3, 3],
        "selected_asset": ["AAA", "AAA", "BBB", "CCC", "CCC"],
        "walk_forward_fold": [1, 1, 2, 3, 3],
        "data_decisao": ["2026-01-02", "2026-01-05", "2026-01-06",
                         "2026-01-07", "2026-01-08"],
        "position_drawdown_from_peak": [-0.20, -0.15, -0.30, -0.05, -0.20],
        "score_change_from_entry": [-0.01, -0.02, +0.03, -0.03, -0.03],
        "consecutive_days_current_not_top1": [0, 0, 1, 2, 3],
        "universe_breadth_5": [0.5, 0.5, 0.2, 0.1, 0.1],
        "position_return_since_entry": [0.01, 0.02, -0.11, -0.01, -0.05],
    })
    episodes = pd.DataFrame({
        "episodio_id": [1, 2, 3],
        "retorno_capital": [0.2, -0.1, 0.05],
        "drawdown_maximo": [-0.2, -0.3, -0.06],
    })
    by_fold, first_alerts = script._a_auditar_cobertura_de_perdas(
        historical, episodes,
    )
    f = by_fold.loc[
        by_fold["fold"].eq("todos")
        & by_fold["sinal"].eq("drawdown_10_score_caiu")
        & by_fold["limite_perda_diaria"].eq(-0.10)
    ].iloc[0]
    assert f["observacoes_validas"] == 4
    assert f["quedas_observadas"] == 1
    assert f["alertas_emitidos"] == 2
    assert f["quedas_antecipadas_por_alerta"] == 1
    assert f["episodios_distintos_com_alerta"] == 1
    assert f["cobertura_quedas"] == 1.0
    detail = first_alerts.loc[
        first_alerts["sinal"].eq("drawdown_10_score_caiu")
    ]
    assert len(detail) == 1
    assert detail.iloc[0]["primeira_data_decisao"] == "2026-01-02"
    assert np.isclose(
        detail.iloc[0]["retorno_observado_1d_primeiro_alerta"], -0.12,
    )
    assert detail.iloc[0]["resultado_episodio_futuro_nao_e_feature"]


def test_modelo_cauda_preserva_treino_anterior_e_limiar_de_treino() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(101)
    signals = rng.normal(0, 1, size=n)
    losses = ((np.arange(n) % 11) == 0)
    daily = pd.DataFrame({
        "posicao_mantida": np.ones(n, dtype=bool),
        "retorno_observado_1d": np.where(losses, -0.08, 0.01),
        "walk_forward_fold": np.repeat([1, 2, 3], 200),
        "data_decisao": dates.astype(str),
        "data_execucao": (dates + pd.Timedelta(days=1)).astype(str),
        "episodio_id": np.arange(n),
        "selected_asset": ["AAA"] * n,
        "holding_days_at_decision": np.full(n, 2),
        "score_change_from_entry": signals,
        "position_drawdown_from_peak": -np.abs(signals * 0.01),
        "position_return_since_entry": signals * 0.02,
        "best_vs_second_gap": np.abs(signals * 0.005),
        "spy_return_5": signals * 0.01,
        "universe_breadth_5": np.clip(0.5 + signals * 0.1, 0, 1),
    })
    predictions, metrics = script._a_experimento_risco_cauda_oos(daily)
    assert len(metrics) == 2
    assert metrics["fold_teste"].tolist() == [2, 3]
    assert len(predictions) == 400
    assert predictions["probabilidade_modelo_nao_calibrada"].between(0, 1).all()
    assert metrics["observacoes_treino"].gt(100).all()
    assert metrics["perdas_5pct_treino"].ge(8).all()
    assert metrics["perdas_5pct_teste"].ge(2).all()
    assert metrics["limiar_alerta_percentil_90_treino"].between(0, 1).all()
    assert metrics["cobertura_quedas"].between(0, 1).all()
    assert metrics["auc_roc_treino"].between(0, 1).all()
    assert metrics["auc_roc_oos_gap_treino_menos_teste"].notna().all()
    assert metrics["brier_oos_nao_calibrado"].ge(0).all()
    assert metrics["episodios_com_perda_teste"].ge(2).all()
    for _, row in metrics.iterrows():
        assert pd.Timestamp(row["ultima_execucao_treino"]) < (
            pd.Timestamp(row["primeira_decisao_teste"])
            - pd.Timedelta(days=30)
        )


def test_impacto_alertas_exposicao_so_altera_dias_sinalizados() -> None:
    """O capital de origem deve reproduzir exatamente o exportado."""
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    diario = pd.DataFrame({
        "data_decisao": [
            "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
        ],
        "data_execucao": [
            "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        ],
        "strategy_equity": [1100.0, 990.0, 1188.0, 1306.8],
        "walk_forward_fold": [1, 2, 2, 3],
    })
    previsoes = pd.DataFrame({
        "data_decisao": ["2026-01-05", "2026-01-06"],
        "data_execucao": ["2026-01-06", "2026-01-07"],
        "alerta_limiar_treino": [True, True],
        "episodio_id": [11, 11],
        "selected_asset": ["AAA", "AAA"],
        "probabilidade_modelo_nao_calibrada": [0.8, 0.9],
    })
    curva, alertas, resumo = script._a_impacto_alertas_exposicao(
        diario, previsoes, capital_inicial=1000.0,
    )
    np.testing.assert_allclose(
        curva["capital_control_recalculado"],
        diario["strategy_equity"], atol=1e-8,
    )
    assert len(alertas) == 2
    assert set(alertas["data_execucao"]) == {"2026-01-06", "2026-01-07"}
    np.testing.assert_allclose(
        curva["retorno_exposicao_shadow"],
        [.10, .0, .0, .10], atol=1e-9,
    )
    assert np.isclose(curva["capital_exposicao_shadow"].iloc[-1], 1210)
    total = resumo.loc[resumo["fold"].eq("todos")].iloc[0]
    assert total["sessoes_com_alerta"] == 2
    assert total["sessoes_positivas_alertadas"] == 1
    assert total["sessoes_negativas_alertadas"] == 1
    assert np.isclose(total["capital_control_fim_periodo"], 1306.8)
    assert np.isclose(total["capital_shadow_fim_periodo"], 1210.0)


def test_impacto_alertas_recusa_data_de_decisao_incorreta() -> None:
    import importlib
    import pytest

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    diario = pd.DataFrame({
        "data_decisao": ["2026-01-01", "2026-01-02"],
        "data_execucao": ["2026-01-02", "2026-01-05"],
        "strategy_equity": [1050.0, 1060.0],
        "walk_forward_fold": [1, 2],
    })
    previsoes = pd.DataFrame({
        "data_decisao": ["2026-01-03"],
        "data_execucao": ["2026-01-05"],
        "alerta_limiar_treino": [True],
        "episodio_id": [1],
        "selected_asset": ["CAPR"],
        "probabilidade_modelo_nao_calibrada": [0.8],
    })
    with pytest.raises(ValueError, match="incongruente"):
        script._a_impacto_alertas_exposicao(
            diario, previsoes, capital_inicial=1000.0,
        )


def test_primeiro_alerta_por_episodio_ignora_repeticoes_sem_olhar_futuro() -> None:
    """Um primeiro alerta por episodio, mesmo com novos sinais dias depois."""
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    diario = pd.DataFrame({
        "data_decisao": ["2026-01-02", "2026-01-05", "2026-01-06",
                         "2026-01-07", "2026-01-08"],
        "data_execucao": ["2026-01-05", "2026-01-06", "2026-01-07",
                          "2026-01-08", "2026-01-09"],
        "strategy_equity": [1100.0, 990.0, 1188.0, 1306.8, 1176.12],
        "walk_forward_fold": [1, 2, 2, 3, 3],
    })
    previsoes = pd.DataFrame({
        "data_decisao": ["2026-01-05", "2026-01-06", "2026-01-08"],
        "data_execucao": ["2026-01-06", "2026-01-07", "2026-01-09"],
        "alerta_limiar_treino": [True, True, True],
        "episodio_id": [11, 11, 12],
        "selected_asset": ["AAA", "AAA", "BBB"],
        "probabilidade_modelo_nao_calibrada": [0.8, 0.9, 0.75],
    })
    all_curves, all_days, _ = script._a_impacto_alertas_exposicao(
        diario, previsoes, capital_inicial=1000.0,
    )
    first_curves, first_days, metrics = script._a_impacto_alertas_exposicao(
        diario, previsoes, capital_inicial=1000.0,
        primeiro_por_episodio=True,
    )
    assert len(all_days) == 3
    assert len(first_days) == 2
    assert first_days["episodio_id"].tolist() == [11, 12]
    assert first_days["data_execucao"].tolist() == [
        "2026-01-06", "2026-01-09",
    ]
    np.testing.assert_allclose(
        first_curves["retorno_exposicao_shadow"],
        [.10, .0, .20, .10, .0], atol=1e-9,
    )
    assert np.isclose(
        first_curves["capital_exposicao_shadow"].iloc[-1], 1452.0,
    )
    assert not np.isclose(
        all_curves["capital_exposicao_shadow"].iloc[-1], 1452.0,
    )
    assert metrics.loc[metrics["fold"].eq("todos"),
                       "sessoes_com_alerta"].iloc[0] == 2


def test_primeiro_alerta_por_episodio_nao_usa_resultado_futuro() -> None:
    source = (ROOT / "pesquisar_permanencia_minima_spyder.py").read_text(
        encoding="utf-8"
    )
    assert "primeiro_por_episodio=True" in source
    assert '.drop_duplicates(subset=["episodio_id"], keep="first")' in source
    assert "concentracao_alertas_por_episodio.csv" in source


def test_sensibilidade_primeiro_alerta_mostra_inversao_por_um_episodio() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    events = pd.DataFrame({
        "episodio_id": [1, 2, 3],
        "data_decisao": ["2026-01-02", "2026-01-03", "2026-01-04"],
        "selected_asset": ["AAA", "BBB", "CCC"],
        "walk_forward_fold": [2, 2, 3],
        "retorno_control": [
            np.exp(0.3) - 1, np.exp(-0.1) - 1, np.exp(-0.05) - 1,
        ],
        "contribuicao_log_alerta": [0.3, -0.1, -0.05],
    })
    influence, summary = script._a_sensibilidade_primeiro_alerta(
        events, capital_control=10000.0,
    )
    assert len(influence) == 3
    assert summary["episodios_distintos"] == 3
    assert np.isclose(summary["capital_shadow_primeiros_alertas"],
                      10000 * np.exp(-0.15))
    assert summary["num_remocoes_que_invertem_conclusao"] == 1
    assert influence.iloc[0]["episodio_id"] == 1
    assert influence.iloc[0]["capital_apos_remover_um_alerta"] > 10000


def test_sensibilidade_nao_permite_alertas_duplicados_por_episodio() -> None:
    import importlib
    import pytest

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    events = pd.DataFrame({
        "episodio_id": [1, 1],
        "data_decisao": ["2026-01-02", "2026-01-03"],
        "selected_asset": ["AAA", "AAA"],
        "walk_forward_fold": [2, 2],
        "retorno_control": [0.01, 0.02],
        "contribuicao_log_alerta": [0.01, 0.02],
    })
    with pytest.raises(ValueError, match="Mais de um primeiro alerta"):
        script._a_sensibilidade_primeiro_alerta(
            events, capital_control=10000,
        )


def test_ic_auc_bootstrap_reconhece_episodios_e_eventos_raros() -> None:
    import importlib
    import numpy as np

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    target = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    score = np.array([0.1, 0.8, 0.2, 0.7, 0.1, 0.2, 0.9, 0.2])
    ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    low, high, n, positive = script._a_ic_auc_por_episodio(
        target, score, ids, n_amostras=200, seed=11,
    )
    assert (n, positive) == (4, 3)
    assert 0.0 <= low <= high <= 1.0
    # Um unico episodio positivo nao fornece intervalo estavel.
    small = script._a_ic_auc_por_episodio(
        np.array([1, 0, 0]), np.array([0.8, 0.1, 0.2]),
        np.array([1, 2, 3]), n_amostras=200,
    )
    assert small == (None, None, 3, 1)


def test_economia_shadow_reproduz_control_e_respeita_embargo() -> None:
    import importlib
    import numpy as np

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    dates = pd.date_range("2020-01-01", periods=480, freq="B", tz="UTC")
    daily = pd.DataFrame({
        "timestamp": dates.astype(str),
        "decision_date": (dates - pd.Timedelta(days=1)).astype(str),
        "walk_forward_fold": np.repeat([1, 2, 3], 160),
        "selected_asset": ["AAA"] * 480,
        "current_asset": ["AAA"] * 480,
    })
    t = np.arange(480)
    rng = np.random.default_rng(47)
    logr = np.where(t < 160, .001, -.001) + rng.normal(0, .003, 480)
    daily["strategy_equity"] = 10000 * np.exp(np.cumsum(logr))
    for k in module.ECO_FEATURES:
        daily[k] = np.sin(t/20)
    results, folds = module._eco_cash_diario(daily, initial=10000)
    assert folds["fold"].tolist() == [2, 3]
    assert results["log_retorno_esperado_control"].iloc[:160].isna().all()
    assert results["acao_shadow"].iloc[:160].eq("HOLD").all()
    assert results["acao_shadow"].isin(["HOLD", "ROTATE", "CASH"]).all()
    np.testing.assert_allclose(
        results["capital_control_reproduzido"],
        daily["strategy_equity"], rtol=1e-12, atol=1e-7,
    )
    assert (folds["decisoes_cash"] <= folds["decisoes_testadas"]).all()
    diagnostic, by_year, concentration = module._eco_diagnostico_previsao_diaria(
        results,
    )
    assert diagnostic["fold"].tolist() == [2, 3]
    assert diagnostic["r2_treino"].notna().all()
    assert diagnostic["r2_avaliacao_vs_media_do_treino"].notna().all()
    assert diagnostic["paridade_maxima_das_previsoes"].lt(1e-10).all()
    assert by_year["pregoes"].sum() == 320
    assert by_year["dias_cash"].sum() == folds["decisoes_cash"].sum()
    assert concentration["dias_cash"].sum() == folds["decisoes_cash"].sum()
    for _, row in folds.iterrows():
        assert pd.Timestamp(row["ultima_execucao_treino"]) < (
            pd.Timestamp(row["inicio_teste"]) - pd.Timedelta(days=30)
        )


def test_economia_rotacoes_pareadas_nao_inventa_trajetoria() -> None:
    import importlib
    import numpy as np

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    rng = np.random.default_rng(12)
    dates = pd.date_range("2020-01-01", periods=420, freq="B", tz="UTC")
    records = []
    for i, dt in enumerate(dates[::3]):
        fold = 1 if i < 50 else 2 if i < 95 else 3
        prev = float(rng.normal(.01, .02))
        chosen = float(rng.normal(.012, .025))
        feature = {key: float(np.sin(i/15)) for key in module.ECO_FEATURES}
        records.append({
            "timestamp": str(dt),
            "decision_timestamp": str(dt - pd.Timedelta(days=1)),
            "action": "BUY", "asset": "AAA",
            "rotation_from_asset": "BBB", "rotation_to_asset": "AAA",
            "walk_forward_fold": fold, "chosen_market_return": chosen,
            "counterfactual_previous_asset_return": prev,
            "rotation_value_added": chosen-prev,
            "subsequent_holding_days": 2, **feature,
        })
        records.append({
            "timestamp": str(dt + pd.Timedelta(days=2)),
            "entry_timestamp": str(dt), "action": "SELL", "asset": "AAA",
        })
    original = pd.DataFrame(records)
    pairs, folds = module._eco_rotacoes_pareadas(original)
    assert len(pairs) == 140
    assert folds["intervalos_avaliados"].tolist() == [45, 45]
    assert folds["fold"].tolist() == [2, 3]
    assert pairs.loc[
        pairs["walk_forward_fold"].eq(1), "acao_sugerida"
    ].eq("NAO_AVALIADO").all()
    assert pairs["log_vantagem_acao_sugerida_vs_rotate"].notna().sum() == 90
    assert folds["sugeriu_rotate"].sum() + folds["sugeriu_hold"].sum() + (
        folds["sugeriu_cash"].sum()
    ) == 90
    for _, row in folds.iterrows():
        assert pd.Timestamp(row["ultima_saida_com_rotulo_treino"]) < (
            pd.Timestamp(row["inicio_decisoes_teste"])
            - pd.Timedelta(days=30)
        )


def test_economia_rejeita_falsa_vantagem_em_rotacao() -> None:
    import importlib
    import pytest

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    frame = pd.DataFrame({
        "action": ["BUY"] * 101,
        "counterfactual_previous_asset_return": [.01] * 101,
        "chosen_market_return": [.02] * 101,
        "rotation_value_added": [999] * 101,
        "entry_timestamp": [None] * 101,
        "asset": ["AAA"] * 101,
        "timestamp": pd.date_range("2020-01-01", periods=101).astype(str),
        "decision_timestamp": pd.date_range("2019-12-31", periods=101).astype(str),
    })
    with pytest.raises(ValueError):
        module._eco_rotacoes_pareadas(frame)


def test_economia_diagnostico_sem_cash_nao_inventa_casos() -> None:
    import importlib
    import numpy as np

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    dates = pd.date_range("2020-01-01", periods=480, freq="B", tz="UTC")
    daily = pd.DataFrame({
        "timestamp": dates.astype(str),
        "decision_date": (dates - pd.Timedelta(days=1)).astype(str),
        "walk_forward_fold": np.repeat([1, 2, 3], 160),
        "selected_asset": ["AAA"] * 480,
        "current_asset": ["AAA"] * 480,
    })
    # Retornos positivos: o modelo tem motivo para manter exposicao.
    values = np.arange(480, dtype=float)
    daily["strategy_equity"] = (
        10000.0 * np.exp(np.cumsum(0.01 + 0.0001 * np.sin(values)))
    )
    for k in module.ECO_FEATURES:
        daily[k] = np.sin(values / 20)
    results, _ = module._eco_cash_diario(daily, initial=10000)
    result, annual, concentration = module._eco_diagnostico_previsao_diaria(
        results,
    )
    assert result["fold"].tolist() == [2, 3]
    assert result["dias_cash"].sum() == 0
    assert concentration.empty
    assert annual["dias_cash"].sum() == 0


def test_economia_diagnostico_detecta_previsoes_alteradas() -> None:
    import importlib
    import numpy as np
    import pytest

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    dates = pd.date_range("2020-01-01", periods=480, freq="B", tz="UTC")
    daily = pd.DataFrame({
        "timestamp": dates.astype(str),
        "decision_date": (dates - pd.Timedelta(days=1)).astype(str),
        "walk_forward_fold": np.repeat([1, 2, 3], 160),
        "selected_asset": ["AAA"] * 480,
        "current_asset": ["AAA"] * 480,
    })
    values = np.arange(480, dtype=float)
    daily["strategy_equity"] = (
        10000.0 * np.exp(np.cumsum(0.001 + 0.0001 * np.cos(values)))
    )
    for k in module.ECO_FEATURES:
        daily[k] = np.sin(values / 20)
    report, _ = module._eco_cash_diario(daily, initial=10000)
    report.loc[report["walk_forward_fold"].eq(2),
               "log_retorno_esperado_control"] = -100
    with pytest.raises(ValueError, match="diverge"):
        module._eco_diagnostico_previsao_diaria(report)


def test_permanencia_inteligente_decide_pelo_estado_e_nao_pelo_futuro() -> None:
    import importlib
    import types
    import numpy as np

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    u = np.array([
        [.6, .4], [.6, .4], [.6, .4],
        [.5, .66], [.5, .53], [.55, .57],
    ], dtype=float)
    b = types.SimpleNamespace(
        dates=pd.date_range("2025-01-01", periods=6, tz="UTC"),
        fold=np.array([2] * 6),
        symbols=["AAA", "BBB"], scores=u.copy(),
        config={"rotation_switch_margin": 0.01,
                "rotation_cash_threshold": 0.01,
                "rotation_min_expected_edge": 0.0,
                "rotation_min_holding_days": 2},
        margens={2: 0.01, 3: 0.01},
    )
    # Um dia de holding: Control bloqueia, excecao valida libera rotacao.
    early, reason, _ = module._pi_decisao(
        b, 3, pos=0, holding=1, politica="saida_antecipada",
    )
    assert early == 1
    assert reason == "SAIDA_ANTE_DOS_DOIS_PREGOES"
    # Perturbar fortemente qualquer dado FUTURO nao altera a decisao em i=3.
    b.scores[4:, :] = [[1.0, 0.0], [0.0, 1.0]]
    early_again, reason_again, _ = module._pi_decisao(
        b, 3, pos=0, holding=1, politica="saida_antecipada",
    )
    assert (early, reason) == (early_again, reason_again)
    # Sem excecao, continua cumprindo a permanencia de 2 pregoes.
    confirmed, _, _ = module._pi_decisao(
        b, 3, pos=0, holding=1, politica="confirmacao",
    )
    assert confirmed == 0


def test_permanencia_inteligente_confirma_gap_antes_de_rotacionar() -> None:
    import importlib
    import types
    import numpy as np

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    u = np.array([
        [.6, .4], [.55, .44], [.49, .48], [.5, .49], [.51, .53],
        [.6, .4],
    ], dtype=float)
    b = types.SimpleNamespace(
        dates=pd.date_range("2025-01-01", periods=6, tz="UTC"),
        fold=np.array([2] * 6),
        symbols=["AAA", "BBB"], scores=u,
        config={"rotation_switch_margin": 0.01,
                "rotation_cash_threshold": 0.01,
                "rotation_min_expected_edge": 0.0,
                "rotation_min_holding_days": 2},
        margens={2: 0.01, 3: 0.01},
    )
    # A vantagem apareceu so hoje: espera, mesmo com dois dias completos.
    decision, reason, _ = module._pi_decisao(
        b, 4, pos=0, holding=2, politica="confirmacao",
    )
    assert decision == 0
    assert reason == "ESPERAR_CONFIRMACAO_DA_VANTAGEM"
    # Se o ganho do concorrente for amplo, troca imediatamente aos dois dias.
    b.scores[4, 1] = 0.65
    decision, reason, _ = module._pi_decisao(
        b, 4, pos=0, holding=2, politica="confirmacao",
    )
    assert decision == 1
    assert reason == "CONTROL"
    # Nao extrapolar score de fold anterior para confirmar sinal atual.
    b.scores[4, 1] = 0.53
    b.fold[2] = 1
    decision, reason, _ = module._pi_decisao(
        b, 4, pos=0, holding=2, politica="confirmacao",
    )
    assert decision == 1
    assert reason == "CONTROL"


def test_permanencia_inteligente_paridade_diagnostico_replay(tmp_path) -> None:
    import importlib
    from zipfile import ZipFile

    module = importlib.import_module("pesquisar_permanencia_minima_spyder")
    columns = {
        "decision_date": ["2025-01-01", "2025-01-02"],
        "timestamp": ["2025-01-02", "2025-01-03"],
        "walk_forward_fold": [1, 2],
        "holding_days_at_decision": [1, 1],
        "min_hold_guard_applied": [True, True],
        "raw_best_asset": ["BBB", "AAA"],
        "current_asset": ["AAA", "AAA"],
        "raw_best_score": [0.75, 0.7],
        "best_vs_current_gap": [0.20, 0.00],
        "effective_switch_margin": [0.01, 0.01],
        "score_change_from_entry": [-0.1, 0.1],
        "rotation_cash_threshold": [0.0, 0.0],
        "strategy_equity": [10000.0, 10500.0],
        "decision_is_rotation": [False, False],
    }
    replay = tmp_path / "replay.zip"
    with ZipFile(replay, "w") as z:
        z.writestr("strategy_predictions.csv", pd.DataFrame(columns).to_csv(
            index=False,
        ))
    decisions, summary = module._pi_diagnostico_replay_atual(replay)
    assert len(decisions) == 2
    assert summary["travas_primeiro_pregao"] == 2
    assert summary["travas_com_outro_ativo_no_topo"] == 1
    assert summary["possiveis_excecoes_primeiro_pregao"] == 1
    assert summary["referencia_capital_final_observado"] == 10500.0


def _pi_fixture_zip_base() -> bytes:
    import io
    import json
    from zipfile import ZIP_DEFLATED, ZipFile

    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as z:
        prefix = "pesquisa_permanencia_minima/"
        z.writestr(prefix + "manifesto_experimento.json",
                   json.dumps({"status": "completed", "folds": 3}))
        z.writestr(prefix + "decisoes_2_pregoes.csv", "data\n2026-01-01\n")
        z.writestr(prefix + "patrimonio_diario_2_pregoes.csv",
                   "timestamp\n2026-01-02\n")
        z.writestr(prefix + "utilidades_oos_todos_ativos.csv",
                   "data_decisao,ativo,utilidade\n")
        z.writestr(prefix + "margens_controle_por_fold.csv", "fold,margem\n")
        for i in range(75):
            z.writestr(prefix + f"cotacoes_ohlcv/T{i:02d}.csv",
                       "timestamp,open,close,volume\n")
    return stream.getvalue()


def test_pi_recupera_base_completa_de_zip_anterior(tmp_path, monkeypatch) -> None:
    import hashlib
    import importlib
    from zipfile import ZipFile

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    expected = _pi_fixture_zip_base()
    output = tmp_path / "output"
    output.mkdir()
    previous = tmp_path / "pesquisa_vantagem_rotacao(3).zip"
    with ZipFile(previous, "w") as z:
        z.writestr("dados_origem/pesquisa_permanencia_minima.zip", expected)
    monkeypatch.setattr(script, "_pi_pastas_de_busca",
                        lambda: [output, tmp_path])
    monkeypatch.setattr(
        script, "ARQUIVO_ZIP", output / "pesquisa_permanencia_minima.zip"
    )
    found = script._pi_localizar_base()
    assert found == output / "pesquisa_permanencia_minima.zip"
    assert hashlib.sha256(found.read_bytes()).hexdigest() == (
        hashlib.sha256(expected).hexdigest()
    )
    assert script._pi_validar_zip_base(found)
    # Execucao posterior reutiliza a fonte validada, sem reextrair.
    assert script._pi_localizar_base() == found


def test_pi_ignora_pacote_embutido_corrompido(tmp_path, monkeypatch) -> None:
    import importlib
    from zipfile import ZipFile

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    invalid = tmp_path / "pesquisa_vantagem_rotacao.zip"
    with ZipFile(invalid, "w") as z:
        z.writestr("dados_origem/pesquisa_permanencia_minima.zip",
                   b"isso nao e um ZIP")
    monkeypatch.setattr(script, "_pi_pastas_de_busca", lambda: [tmp_path])
    monkeypatch.setattr(script, "ARQUIVO_ZIP",
                        tmp_path / "pesquisa_permanencia_minima.zip")
    assert script._pi_localizar_base() is None
    assert not (tmp_path / "pesquisa_permanencia_minima.zip").exists()
    assert not (tmp_path / ".pesquisa_permanencia_minima_recuperacao.tmp").exists()


def test_pi_somente_diagnostico_nao_inventa_backtest(tmp_path, monkeypatch) -> None:
    import importlib
    import json
    from zipfile import ZipFile

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    output = tmp_path / "output"
    output.mkdir()
    replay = tmp_path / "replay_atual.zip"
    replay.write_bytes(b"teste")
    monkeypatch.setattr(script, "_a_localizar_replay", lambda: replay)
    frame = pd.DataFrame({
        "current_asset": ["AAA"],
        "raw_best_asset": ["BBB"],
        "candidato_saida_antecipada": [True],
        "walk_forward_fold": [2],
        "outro_ativo_melhor": [True],
        "retorno_control_no_pregao": [-0.02],
    })
    monkeypatch.setattr(
        script, "_pi_diagnostico_replay_atual",
        lambda _: (frame, {
            "sessoes": 1, "travas_primeiro_pregao": 1,
            "possiveis_excecoes_primeiro_pregao": 1,
        }),
    )
    monkeypatch.setattr(script, "DESTINO_TEMP", output / ".temporario")
    monkeypatch.setattr(script, "DESTINO", output / "pesquisa_vantagem_rotacao")
    script._pi_somente_diagnostico_atual()
    archive = output / "pesquisa_vantagem_rotacao.zip"
    with ZipFile(archive) as z:
        meta = json.loads(z.read(
            "pesquisa_vantagem_rotacao/manifesto_experimento.json"
        ))
    assert meta["status"] == "completed_observational_diagnostic_only"
    assert meta["simulacao_contrafactual_executada"] is False
    assert meta["replay_atual_somente_diagnostico"][
        "possiveis_excecoes_primeiro_pregao"
    ] == 1
    assert "capital_alternativo" not in meta


def test_pi_recupera_pasta_completa_com_sha_verificado(
    tmp_path, monkeypatch,
) -> None:
    import hashlib
    import importlib
    import json
    from zipfile import ZipFile

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    base = tmp_path / "output" / "pesquisa_permanencia_minima"
    bars = base / "cotacoes_ohlcv"
    bars.mkdir(parents=True)
    contents = {
        "manifesto_experimento.json":
            json.dumps({"status": "completed", "folds": 3}),
        "decisoes_2_pregoes.csv": "data\n2025-01-01\n",
        "patrimonio_diario_2_pregoes.csv": "timestamp\n2025-01-02\n",
        "utilidades_oos_todos_ativos.csv": "data_decisao,ativo,utilidade\n",
        "margens_controle_por_fold.csv": "fold,margem\n",
    }
    contents.update({
        f"cotacoes_ohlcv/T{i:02d}.csv": "timestamp,open,close,volume\n"
        for i in range(75)
    })
    hashes = {}
    for name, value in contents.items():
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
        hashes[name] = {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
    (base / "arquivos_sha256.json").write_text(
        json.dumps({"algoritmo": "SHA-256", "arquivos": hashes}),
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "_pi_pastas_de_busca",
                        lambda: [tmp_path / "output"])
    recovered = tmp_path / "output" / "pesquisa_permanencia_minima.zip"
    monkeypatch.setattr(script, "ARQUIVO_ZIP", recovered)
    assert script._pi_localizar_base() == recovered
    assert script._pi_validar_zip_base(recovered)
    with ZipFile(recovered) as z:
        assert z.testzip() is None
        assert len([x for x in z.namelist() if "cotacoes_ohlcv/" in x]) == 75
    # Se qualquer preco foi alterado, nao usar a pasta para gerar nova fonte.
    recovered.unlink()
    (bars / "T00.csv").write_text("adulterado\n", encoding="utf-8")
    assert script._pi_localizar_base() is None
    assert not recovered.exists()


def test_pi_retorno_da_trava_e_observado_so_depois_da_decisao(tmp_path) -> None:
    import importlib
    from zipfile import ZipFile

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    data = pd.DataFrame({
        "timestamp": [
            "2025-01-02", "2025-01-03", "2025-01-06",
            "2025-01-07", "2025-01-08",
        ],
        "decision_date": [
            "2025-01-01", "2025-01-02", "2025-01-03",
            "2025-01-06", "2025-01-07",
        ],
        "walk_forward_fold": [1, 1, 2, 2, 3],
        "holding_days_at_decision": [0, 1, 1, 1, 1],
        "min_hold_guard_applied": [False, True, True, True, True],
        "current_asset": ["AAA", "AAA", "BBB", "CCC", "DDD"],
        "raw_best_asset": ["AAA", "BBB", "DDD", "CCC", "EEE"],
        "raw_best_score": [0.3, 0.5, 0.7, 0.5, 0.7],
        "best_vs_current_gap": [0, .08, .1, 0, .1],
        "effective_switch_margin": [.01] * 5,
        "score_change_from_entry": [0, -.05, -.1, .01, -.1],
        "rotation_cash_threshold": [0] * 5,
        "strategy_equity": [100.0, 110.0, 104.5, 125.4, 118.0],
        "decision_is_rotation": [False] * 5,
    })
    original = tmp_path / "replay.zip"
    with ZipFile(original, "w") as z:
        z.writestr("strategy_predictions.csv", data.to_csv(index=False))
    checks, summary = script._pi_diagnostico_replay_atual(original)
    assert summary["sessoes"] == 5
    assert summary["travas_primeiro_pregao"] == 4
    assert summary["possiveis_excecoes_primeiro_pregao"] == 3
    assert "retorno_control_no_pregao" in checks
    result = checks["retorno_control_no_pregao"].round(8).tolist()
    assert result == [
        0.1, -0.05, 0.2, round(118/125.4-1, 8),
    ]
    stat = pd.DataFrame(summary["retornos_observados_da_trava"])
    flagged = stat.loc[
        stat["fold"].eq("todos") & stat["grupo"].eq("excecao_proposta")
    ].iloc[0]
    assert flagged["observacoes"] == 3
    assert flagged["pregoes_com_perda_5pct"] == 2
    assert flagged["pregoes_positivos"] == 1
    assert flagged["pregoes_negativos"] == 2
    assert round(flagged["media_retorno_control"], 4) == round(
        (.1 - .05 + (118/125.4-1)) / 3, 4,
    )


def test_pi_rejeita_replay_fora_de_ordem_antes_de_retornos(tmp_path) -> None:
    import importlib
    from zipfile import ZipFile
    import pytest

    script = importlib.import_module("pesquisar_permanencia_minima_spyder")
    df = pd.DataFrame({
        "timestamp": ["2025-01-03", "2025-01-02"],
        "strategy_equity": [100., 101.],
        "decision_date": ["2025-01-02", "2025-01-01"],
        "walk_forward_fold": [1, 2],
        "holding_days_at_decision": [1, 1],
        "min_hold_guard_applied": [True, True],
        "current_asset": ["AAA", "BBB"],
        "raw_best_asset": ["AAA", "BBB"],
        "raw_best_score": [.3, .4],
        "best_vs_current_gap": [0, 0],
        "effective_switch_margin": [.01, .01],
        "score_change_from_entry": [0, 0],
        "rotation_cash_threshold": [0, 0],
        "decision_is_rotation": [False, False],
    })
    file = tmp_path / "replay.zip"
    with ZipFile(file, "w") as z:
        z.writestr("strategy_predictions.csv", df.to_csv(index=False))
    with pytest.raises(ValueError, match="ordenado e unico"):
        script._pi_diagnostico_replay_atual(file)
