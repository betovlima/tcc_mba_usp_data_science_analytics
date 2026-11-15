import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from reproducao.dados import SnapshotPaths
from reproducao.preparacao import (
    _validated_structural_lineage, load_raw_bar_file,
    prepare_model_frames, structural_identity_issue,
)
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


def test_clmt_cusip_transition_is_detected_and_audited() -> None:
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
    assert 'REPRODUCTION_VERSION = "1.23.0+universe77.test"' in source
    assert 'EXECUTION_SCHEMA = "universe-reproduction-v4"' in source
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
    # O manifesto preserva a historia, mas NAO valida o novo replay de 77.
    runner = (ROOT / "reproducao" / "experimento.py").read_text(encoding="utf-8")
    assert "len(requested) != 77" in runner
    assert '"checkpoint_validation": "awaiting_first_77_asset_baseline"' in runner
    assert "expected_capital = None" in runner
    assert runner.count("prepare_model_frames(") == 1
    assert "source_paths={symbol: candidate_data for symbol in selected}" in runner
    assert "_train_baseline_states(" not in runner
    assert "_train_candidate_caches(" not in runner
    assert "_evaluate_scenario(" not in runner
    assert "_ajustar_modelos_lightgbm(" in runner
    assert "_precalcular_utilidades_modelo(" in runner
    assert '"trained_count_each_fold": 77' in runner
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


def test_real_snapshot_prepares_all_source_assets_with_audited_doc_clmt() -> None:
    """Regression: never report success after silently excluding two securities."""
    frames, exclusions, diagnostics, audit = prepare_model_frames(
        SnapshotPaths.u67(ROOT),
        assets=U67_REQUESTED_ASSETS,
        csv_float_precision="round_trip",
    )
    assert tuple(frames) == U67_REQUESTED_ASSETS
    assert len(frames) == 67
    assert exclusions == []
    assert audit["eligible_assets"] == 67
    lineages = {row["symbol"]: row for row in audit["validated_structural_lineages"]}
    assert set(lineages) == {"DOC", "CLMT"}
    assert lineages["DOC"]["historical_execution_symbol"] == "PEAK"
    assert lineages["CLMT"]["conversion_ratio"] == "1:1"
    assert len(diagnostics) == 67
    for symbol in ("DOC", "CLMT"):
        assert len(frames[symbol]) > 2500


def test_historical_lineage_refuses_unverified_corporate_actions() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="DOC"):
        _validated_structural_lineage("DOC", [])
    with pytest.raises(RuntimeError, match="CLMT"):
        _validated_structural_lineage("CLMT", [])
    with pytest.raises(RuntimeError, match="transicao societaria"):
        _validated_structural_lineage("UNKNOWN", [])


def test_selected_universe_contract_never_reduces_to_fewer_assets() -> None:
    import json

    contract = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    requested = list(U67_REQUESTED_ASSETS) + contract["selected_assets"]
    assert len(requested) == 77
    assert len(set(requested)) == 77
    assert requested[46:48] == ["DOC", "CLMT"]
    assert requested[-10:] == contract["selected_assets"]


def test_single_preparation_pass_uses_both_frozen_input_directories(tmp_path) -> None:
    from reproducao.dados import SnapshotPaths

    base = SnapshotPaths.from_root(tmp_path / "source")
    additional = SnapshotPaths.from_root(tmp_path / "additional")
    base.ensure()
    additional.ensure()
    sessions = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    for symbol, source in (("AAA", base), ("BBB", additional)):
        pd.DataFrame({
            "timestamp": sessions,
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1000, 1100, 1200, 1300],
        }).to_csv(source.raw_bars / f"{symbol}.csv", index=False)
        pd.DataFrame(columns=["action_type", "symbol"]).to_csv(
            source.corporate_actions / f"{symbol}.csv", index=False
        )

    frames, exclusions, diagnostics, audit = prepare_model_frames(
        base,
        assets=("AAA", "BBB"),
        source_paths={"BBB": additional},
        csv_float_precision="round_trip",
    )
    assert list(frames) == ["AAA", "BBB"]
    assert exclusions == []
    assert len(diagnostics) == 2
    assert audit["eligible_assets"] == 2


def test_reproducao_integrada_usa_77_desde_o_primeiro_fold(tmp_path, monkeypatch) -> None:
    """Exercise the runner with sorted model order and an unsorted 77-asset contract.

    Simulates the expensive ML/replay calls while checking that the actual
    orchestration never builds a separate reference or joins model caches.
    """
    import hashlib
    import json
    import numpy as np

    import engine.modelo_lightgbm as model
    import engine.rotacao as rotation
    import reproducao.artefatos as artifacts
    import reproducao.dados as data
    import reproducao.experimento as experiment
    import reproducao.graficos as graphs

    additions = ("CRT", "NBR", "CAPR", "PNBK", "PENN", "JBL",
                 "STAG", "MCHB", "LAD", "TOWN")
    requested = (*U67_REQUESTED_ASSETS, *additions)
    assert len(requested) == 77
    root = tmp_path / "tcc"
    evidence_dir = root / "evidencias"
    evidence_dir.mkdir(parents=True)
    cache = root / "dados" / ".u67_expansion_cache" / "full_raw_20261006"
    for folder in ("raw_bars", "corporate_actions"):
        (cache / folder).mkdir(parents=True)
    hashes = {}
    for symbol in additions:
        raw = (cache / "raw_bars" / f"{symbol}.csv")
        actions = (cache / "corporate_actions" / f"{symbol}.csv")
        raw.write_text("frozen-" + symbol, encoding="utf-8")
        actions.write_text("actions-" + symbol, encoding="utf-8")
        hashes[symbol] = {
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "corporate_actions_sha256": hashlib.sha256(actions.read_bytes()).hexdigest(),
        }
    (evidence_dir / "manifest_universo_congelado.json").write_text(
        json.dumps({
            "selected_assets": list(additions),
            "candidate_file_hashes": hashes,
            "snapshot_sha256": "test_frozen_snapshot",
            "analysis_end_date": "2020-01-17",
        }), encoding="utf-8"
    )
    monkeypatch.setattr(
        data, "validate_snapshot",
        lambda path: {
            "snapshot_sha256": "test_frozen_snapshot",
            "assets": list(U67_REQUESTED_ASSETS),
        },
    )
    dates = pd.date_range("2020-01-02", periods=12, freq="B", tz="UTC")
    bars = pd.DataFrame({
        "open": [10.0] * len(dates),
        "high": [11.0] * len(dates),
        "low": [9.0] * len(dates),
        "close": [10.5] * len(dates),
        "volume": [1000.0] * len(dates),
    }, index=dates)
    prepared_calls = []

    def prepare(_paths, *, assets, source_paths, csv_float_precision):
        prepared_calls.append(tuple(assets))
        assert tuple(assets) == requested
        assert set(source_paths) == set(additions)
        return (
            {symbol: bars.copy() for symbol in assets},
            [],
            [],
            {"eligible_assets": 77, "excluded_assets": []},
        )

    import reproducao.preparacao as preparation
    monkeypatch.setattr(preparation, "prepare_model_frames", prepare)

    def context(raw, config):
        assert len(raw) == 77
        symbols = sorted(raw)
        folds = [
            {
                "fold_id": index,
                "train_end_index": 3,
                "calibration_start_index": 3,
                "calibration_end_index": 6,
                "final_fit_end_index": 8,
                "decision_dates": dates[8:12],
            }
            for index in (1, 2, 3)
        ]
        return raw, dates, "AAA", symbols, folds, dates[8:12], {}, {}

    monkeypatch.setattr(model, "_construir_contexto_execucao", context)
    fits = []

    def fit(frames, symbols, train_dates, config, *, phase, progress_callback):
        assert len(frames) == len(symbols) == 77
        assert symbols == sorted(requested)
        fits.append(phase)
        progress_callback(77, 77, "cpu")
        return {symbol: object() for symbol in symbols}

    monkeypatch.setattr(model, "_ajustar_modelos_lightgbm", fit)
    monkeypatch.setattr(
        model, "_precalcular_utilidades_modelo",
        lambda models, frames, symbols, timestamps, config: (
            {pd.Timestamp(t): np.array([0.0] + [0.5] * 77)
             for t in timestamps},
            {},
        ),
    )
    monkeypatch.setattr(
        model, "diagnosticos_fora_amostra",
        lambda models, frames, symbols, dates, cache, *, fold_id:
            {"fold_id": fold_id, "by_asset": {symbol: {} for symbol in symbols}},
    )
    monkeypatch.setattr(
        rotation, "_politica_utilidade",
        lambda *args, **kwargs: lambda timestamp, current, holding: (1, 0.5),
    )
    monkeypatch.setattr(rotation, "_crescimento_politica_simples",
                        lambda *args: 1.0)
    monkeypatch.setattr(
        rotation, "_simular_exato",
        lambda backend, policy, frames, symbols, decision_dates, *args, **kwargs:
            SimpleNamespace(
                metrics={"capital_rotations": 2},
                predictions=pd.DataFrame(
                    {"strategy_equity": [10000.0, 10500.0, 11000.0]},
                    index=decision_dates[1:],
                ),
                trades=pd.DataFrame({"asset": ["DOC"], "action": ["BUY"]}),
            ) if len(frames) == len(symbols) == 77 else
            (_ for _ in ()).throw(AssertionError("Replay without 77 assets")),
    )
    monkeypatch.setattr(
        experiment, "summarize_metrics",
        lambda result, folds, initial: {"ending_capital": 11000.0, "folds": []},
    )
    monkeypatch.setattr(graphs, "gerar_analises_backtest",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(artifacts, "sinal_sonoro_conclusao", lambda: None)

    result = experiment.reproduzir_universo_congelado(
        root=root,
        output=root / "output" / "reproducao_universo",
        version="universe77-test",
        execution_schema="universe-reproduction-v4",
    )
    assert prepared_calls == [requested]
    assert len(fits) == 6
    assert all("universe77_fold_" in phase for phase in fits)
    assert result["universe"]["effective_count"] == 77
    assert set(result["universe"]["trained_assets"]) == set(requested)
    assert len(result["calibration"]["fit_counts"]) == 6
    assert all(row["model_count"] == 77 for row in
               result["calibration"]["fit_counts"])
    assert (root / "output" / "reproducao_universo" /
            "pacote_reproducao_universo.zip").is_file()
