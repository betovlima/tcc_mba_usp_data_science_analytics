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
)


ROOT = Path(__file__).resolve().parents[1]


def test_round_trip_csv_parser_preserves_float64_for_mct_parity(tmp_path) -> None:
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


def test_official_u67_workflow_is_explicitly_sectioned() -> None:
    source = (ROOT / "reproduzir_experimento.py").read_text(encoding="utf-8")
    assert source.count("# %%") >= 9
    assert "# %% 1 - Snapshot U67 congelado e versionado" in source
    assert "# %% 2 - Mesmo processamento estrutural observado no MCT" in source
    assert "# %% 3 - Mesmo calendario U56 elegivel" in source
    assert "# %% 5 - Treino, calibracao e politicas identicos ao MCT" in source
    assert "# %% 6 - Replay financeiro" in source
    config_source = (ROOT / "engine" / "configuracao.py").read_text(
        encoding="utf-8"
    )
    assert 'U59_ADDITIONS = ("COLB", "AMS", "FOXF")' in config_source
    assert "U67_ADDITIONS = (" in config_source
    assert "U67_EXPECTED_REQUESTED_COUNT = 67" in config_source
    assert "U67_EXPECTED_EFFECTIVE_COUNT = 65" in config_source
    assert 'U67_EXPECTED_EXCLUSIONS = frozenset({"CLMT", "DOC"})' in config_source
    assert "MCT_ENDING_CAPITAL = 76_927_051.38897176" in source
    assert "_simular_exato(" in source


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


def test_official_reproduction_uses_frozen_u67_snapshot() -> None:
    source = (ROOT / "reproduzir_experimento.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "snapshotpaths.u67(root)" in source
    assert "validate_snapshot(data)" in source
    assert "download_raw_bars" not in source
    assert "download_corporate_actions" not in source
    assert "load_alpaca_credentials" not in source
    assert 'csv_float_precision="round_trip"' in source
    assert 'mct_analysis_end_date = "2026-10-06"' in source


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
        ROOT / "dados" / "README.md",
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
    assert EXPERIMENT_VERSION == "1.22.0-dev.5"
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


def test_official_reproduction_uses_mct_u67_parity_contract() -> None:
    source = (ROOT / "reproduzir_experimento.py").read_text(
        encoding="utf-8"
    )
    config_source = (ROOT / "engine" / "configuracao.py").read_text(
        encoding="utf-8"
    )
    assert "DATA = SnapshotPaths.u67(ROOT)" in source
    assert 'U59_ADDITIONS = ("COLB", "AMS", "FOXF")' in config_source
    assert "U67_ADDITIONS = (" in config_source
    for asset in ("THO", "WDAY", "EXR", "XEL", "SBFG", "PAYX", "MUX", "SXC"):
        assert f'"{asset}"' in config_source
    assert "U67_EXPECTED_REQUESTED_COUNT = 67" in config_source
    assert "U67_EXPECTED_EFFECTIVE_COUNT = 65" in config_source
    assert 'U67_EXPECTED_EXCLUSIONS = frozenset({"CLMT", "DOC"})' in config_source
    assert 'MCT_JOB_ID = "20261007T095423-60e489c0"' in source
    assert "MCT_CAPITAL_AT_2026_09_17 = 78_782_538.31270888" in source
    assert "MCT_ENDING_CAPITAL = 76_927_051.38897176" in source
