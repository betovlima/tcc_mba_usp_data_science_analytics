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
    assert script.MODO == "analisar_permanencia"
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
    assert 'MODO = "analisar_permanencia"' in script
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
