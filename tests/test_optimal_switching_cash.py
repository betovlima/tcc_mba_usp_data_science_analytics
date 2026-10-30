"""Contratos de causalidade e reentrada para a pesquisa Switching."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from pesquisa.optimal_switching_cash import (
    SwitchingModel,
    SwitchingSettings,
    build_overlay,
    fit_switching_model,
)
from engine.rotacao import _simular_exato
from engine.configuracao import CONFIG


def _bars(length=320, *, trend=0.001):
    dates = pd.date_range("2024-01-02", periods=length, freq="B", tz="UTC")
    increments = trend + 0.007 * np.sin(np.arange(length) / 5.0)
    close = 10 * np.exp(np.cumsum(increments))
    open_values = close * 1.001
    return pd.DataFrame({
        "open": open_values,
        "close": close,
        "high": np.maximum(close, open_values),
        "low": np.minimum(close, open_values),
        "volume": 1_000_000.0,
    }, index=dates)


def test_fit_cannot_observe_any_prices_after_fit_end():
    bars = _bars()
    settings = SwitchingSettings(minimum_observations=80)
    fit_end = bars.index[230]
    first = fit_switching_model("AAA", bars, fit_end=fit_end, settings=settings)
    changed = bars.copy()
    changed.loc[changed.index > fit_end, ["open", "close", "high", "low"]] *= 200
    second = fit_switching_model("AAA", changed, fit_end=fit_end, settings=settings)
    assert first.fit_end == second.fit_end
    assert first.boundaries == second.boundaries
    np.testing.assert_array_equal(first.transitions, second.transitions)
    np.testing.assert_array_equal(first.expected_returns, second.expected_returns)


def test_decision_regime_is_immune_to_next_session_prices():
    bars = _bars()
    settings = SwitchingSettings(minimum_observations=80)
    model = fit_switching_model("AAA", bars, fit_end=bars.index[225], settings=settings)
    decision = bars.index[250]
    before = model.state_at(bars, decision)
    changed = bars.copy()
    changed.loc[changed.index > decision, "close"] *= 100
    assert before == model.state_at(changed, decision)


def _dummy_model(rewards):
    return SwitchingModel(
        symbol="AAA", fit_end=pd.Timestamp("2024-01-01", tz="UTC"),
        boundaries=(-0.5, 0.5), transitions=np.eye(3),
        expected_returns=np.asarray(rewards, dtype=float),
        downside_losses=np.zeros(3), fit_observations=200, lookback=20,
    )


def test_bellman_sells_when_expected_return_negative():
    settings = SwitchingSettings(horizon=10)
    qc, qi = _dummy_model([-0.02] * 3).values(
        1, True, settings=settings, buy_cost=0.0, sell_cost=0.0,
    )
    assert qc > qi


def test_bellman_chooses_invest_when_expected_return_positive():
    settings = SwitchingSettings(horizon=10)
    qc, qi = _dummy_model([0.02] * 3).values(
        1, False, settings=settings, buy_cost=0.0, sell_cost=0.0,
    )
    assert qi > qc


def test_overlay_can_exit_and_reenter_without_changing_control():
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    rows = pd.DataFrame({"close": [10.0, 11.0]}, index=dates)
    registry = {1: {"AAA": SimpleNamespace(
        fit_end=dates[0], fit_observations=100,
        state_at=lambda _frame, _date: 1,
        values=lambda _state, invested, **kw: (
            (1.0, 0.0) if invested else (0.0, 1.0)
        ),
    )}}
    diagnostics = {}
    overlay = build_overlay(
        lambda date, current, holding: (1, 0.72),
        frames={"AAA": rows}, symbols=["AAA"],
        decision_to_fold={dates[0]: 1, dates[1]: 1},
        models_by_fold=registry,
        settings=SwitchingSettings(),
        config=CONFIG,
        diagnostics=diagnostics,
    )
    assert overlay(dates[0], 1, 3)[0] == 0
    assert overlay(dates[1], 0, 0)[0] == 1
    assert diagnostics[dates[0]]["switching_reason"] == "cash_preferred"
    assert diagnostics[dates[1]]["switching_reason"] == "exposure_approved"


def test_replay_records_real_cash_interval_and_new_buy_with_fees():
    dates = pd.date_range("2024-02-01", periods=5, freq="B", tz="UTC")
    prices = pd.DataFrame({
        "open": [10.0] * 5, "close": [10.0] * 5,
        "high": [10.0] * 5, "low": [10.0] * 5,
    }, index=dates)
    def policy(date, position, holding):
        mapping = {dates[0]: 1, dates[1]: 0, dates[2]: 0, dates[3]: 1}
        return mapping[date], 1.0
    fees = lambda side, quantity, price, config: {
        "total_fee": 1.0, "commission_fee": 1.0, "sec_fee": 0.0,
        "taf_fee": 0.0, "cat_fee": 0.0,
    }
    config = CONFIG.copiar_modelo(update={"assets": ("AAA",), "initial_capital": 1000.0})
    simulation = _simular_exato(
        "test", policy, {"AAA": prices}, ["AAA"], dates, config,
        fees, lambda p, side, _config: p,
    )
    assert simulation.predictions["selected_asset"].tolist() == [
        "AAA", "CASH", "CASH", "AAA",
    ]
    assert "MOVE_TO_CASH" in simulation.trades["reason"].tolist()
    assert simulation.metrics["cash_days"] == 2
    assert simulation.metrics["total_transaction_fees"] >= 3.0


def test_risk_feature_is_no_leak_when_future_high_is_modified():
    bars = _bars()
    model = fit_switching_model(
        "AAA", bars, fit_end=bars.index[225],
        settings=SwitchingSettings(minimum_observations=80),
    )
    chosen = bars.index[240]
    changed = bars.copy()
    changed.loc[changed.index > chosen, "high"] *= 1000
    changed.loc[changed.index > chosen, "low"] /= 1000
    assert model.state_at(bars, chosen) == model.state_at(changed, chosen)

def test_loader_reads_exactly_75_assets_in_one_pass(tmp_path, monkeypatch):
    """Nenhuma logica de universo intermediario ou exclusoes e executada."""
    import json
    from types import SimpleNamespace
    import pesquisar_optimal_switching_cash_spyder as research

    official = json.loads(research.UNIVERSE.read_text(encoding="utf-8"))
    assert official["effective_asset_count"] == 75
    assert len(official["assets"]) == 75
    assert len(set(official["assets"])) == 75
    source = tmp_path / "snapshot"
    raw = source / "raw_bars"
    actions = source / "corporate_actions"
    raw.mkdir(parents=True)
    actions.mkdir(parents=True)
    for symbol in official["assets"]:
        (raw / f"{symbol}.csv").write_text("bar", encoding="utf-8")
        (actions / f"{symbol}.csv").write_text("action", encoding="utf-8")
    contract = {
        "selected_assets": [],
        "snapshot_sha256": "snapshot-test",
        "candidate_file_hashes": {},
    }
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(research, "CONTRACT", cpath)
    monkeypatch.setattr(research, "SOURCE", SimpleNamespace(
        raw_bars=raw, corporate_actions=actions,
    ))
    monkeypatch.setattr(
        research, "validate_snapshot",
        lambda _paths: {"snapshot_sha256": "snapshot-test"},
    )
    seen = []
    monkeypatch.setattr(
        research, "load_raw_bar_file",
        lambda path, **kwargs: seen.append(path.stem) or pd.DataFrame(
            {"open": [10.0], "close": [10.0]}, index=[pd.Timestamp("2020-01-02", tz="UTC")],
        ),
    )
    monkeypatch.setattr(research, "load_actions_file", lambda _path: [])
    monkeypatch.setattr(research, "split_normalize", lambda frame, _actions: (frame, []))
    frames, _, _, _ = research.carregar_universo_oficial()
    assert len(frames) == 75
    assert sorted(seen) == official["assets"]


def test_loader_refuses_missing_asset_instead_of_excluding_it(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pesquisar_optimal_switching_cash_spyder as research

    official = json.loads(research.UNIVERSE.read_text(encoding="utf-8"))
    source = tmp_path / "snapshot"
    raw = source / "raw_bars"
    actions = source / "corporate_actions"
    raw.mkdir(parents=True)
    actions.mkdir(parents=True)
    # Apenas um ativo disponivel: a carga deve abortar antes de qualquer treino.
    symbol = official["assets"][0]
    (raw / f"{symbol}.csv").write_text("bar", encoding="utf-8")
    (actions / f"{symbol}.csv").write_text("action", encoding="utf-8")
    cpath = tmp_path / "contract.json"
    cpath.write_text(
        json.dumps({"selected_assets": [], "snapshot_sha256": "snapshot-test"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(research, "CONTRACT", cpath)
    monkeypatch.setattr(research, "SOURCE", SimpleNamespace(
        raw_bars=raw, corporate_actions=actions,
    ))
    monkeypatch.setattr(
        research, "validate_snapshot",
        lambda _paths: {"snapshot_sha256": "snapshot-test"},
    )
    with pytest.raises(RuntimeError, match="arquivos congelados ausentes"):
        research.carregar_universo_oficial()


def test_json_serialization_maps_nan_and_inf_to_null(tmp_path):
    import json
    import numpy as np
    import pesquisar_expansao_u67_spyder as pesquisa
    path = tmp_path / "resultado.json"
    pesquisa._write_json(path, {
        "capital": 123.0,
        "optional": np.nan,
        "nested": [{"sharpe": float("inf"), "maxdrawdown": -0.3}],
    })
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["capital"] == 123
    assert saved["optional"] is None
    assert saved["nested"][0]["sharpe"] is None
    assert saved["nested"][0]["maxdrawdown"] == -0.3


def test_analysis_zip_has_required_evidence_and_file_hashes(tmp_path):
    import hashlib
    import json
    from zipfile import ZipFile

    from pesquisa.pacote_analise_switching import gerar_zip_resultados

    folder = tmp_path / "run_001"
    folder.mkdir()
    (folder / "resultado.json").write_text(
        json.dumps({"status": "completed", "effective_asset_count": 75}),
        encoding="utf-8",
    )
    (folder / "switching_predictions.csv").write_text(
        "timestamp,selected_asset\n2026-10-01,CASH\n", encoding="utf-8",
    )
    (folder / "switching_decisions.csv").write_text(
        "date,reason\n2026-10-01,cash_preferred\n", encoding="utf-8",
    )
    (folder / "secreto.env").write_text("SECRET", encoding="utf-8")
    package = gerar_zip_resultados(folder, versao="v1-test", status="completed")
    assert package.is_file()
    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert "resultado.json" in names
        assert "switching_predictions.csv" in names
        assert "switching_decisions.csv" in names
        assert "manifesto_pacote.json" in names
        assert "secreto.env" not in names
        manifest = json.loads(archive.read("manifesto_pacote.json"))
        assert manifest["file_count"] == 3
        assert manifest["status"] == "completed"
        assert manifest["data_raw_included"] is False
        for info in manifest["files"]:
            content = archive.read(info["name"])
            assert info["sha256"] == hashlib.sha256(content).hexdigest()
        assert archive.testzip() is None
    with pytest.raises(FileExistsError):
        gerar_zip_resultados(folder, versao="v1-test", status="completed")


def test_analysis_zip_also_exists_for_failed_run(tmp_path):
    import json
    from zipfile import ZipFile

    from pesquisa.pacote_analise_switching import gerar_zip_resultados

    folder = tmp_path / "failed_001"
    folder.mkdir()
    (folder / "erro_execucao.json").write_text(
        json.dumps({"status": "failed", "error_type": "RuntimeError"}),
        encoding="utf-8",
    )
    package = gerar_zip_resultados(folder, versao="v1-test", status="failed")
    with ZipFile(package) as archive:
        assert "erro_execucao.json" in archive.namelist()
        manifest = json.loads(archive.read("manifesto_pacote.json"))
        assert manifest["status"] == "failed"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="Arquivo principal"):
        gerar_zip_resultados(empty, versao="v1-test", status="completed")


def test_processed_calendar_removes_indicators_warmup_before_folds():
    """Regressao: o calendario de OHLCV bruto gerava um fold adicional."""
    import pesquisar_optimal_switching_cash_spyder as research
    from engine.rotacao import (
        _construir_folds_walk_forward, _datas_decisao_analise,
    )
    base = _bars(length=2705)
    # Volume nao constante para validar os atributos volume_zscore.
    ticks = np.arange(len(base), dtype=float)
    base["volume"] = 1e6 + 1e5 * np.sin(ticks / 9.0) + 1e4 * np.cos(ticks / 11.0)
    dates = base.index
    # Sem calendario de fechamento real, esta amostra sintetica testa
    # contagem de sessoes, nao a data literal do checkpoint historico.
    prepared = research.construir_calendario_control_75({"AAPL": base}, CONFIG)
    assert len(dates) == 2705
    assert len(prepared) == 2506
    assert prepared[0] == dates[199]
    folds = _construir_folds_walk_forward(prepared, CONFIG)
    assert len(folds) == 3
    test_dates = _datas_decisao_analise(
        prepared, folds, CONFIG.copiar_modelo(update={"analysis_end_date": None}),
    )
    assert len(test_dates) - 1 == 1560
    assert len(_construir_folds_walk_forward(dates, CONFIG)) == 4


def test_chronology_contract_rejects_wrong_fold_count_before_training(tmp_path, monkeypatch):
    import json
    import pesquisar_optimal_switching_cash_spyder as research

    calendar = pd.date_range("2016-12-01", periods=2506, freq="B", tz="UTC")
    decisions = pd.date_range("2020-07-21", periods=1561, freq="B", tz="UTC")
    contract = {
        "prepared_sessions": 2506,
        "walk_forward_folds": 3,
        "out_of_sample_sessions": 1560,
        "first_execution_date": str(decisions[1].date()),
        "last_execution_date": str(decisions[-1].date()),
    }
    manifest = tmp_path / "calendar.json"
    manifest.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(research, "CALENDAR_CONTRACT", manifest)
    assert research.verificar_protocolo_cronologico(
        calendar, [{"fold_id": 1}, {"fold_id": 2}, {"fold_id": 3}], decisions,
    )["prepared_sessions"] == 2506
    with pytest.raises(RuntimeError, match="walk_forward_folds"):
        research.verificar_protocolo_cronologico(
            calendar, [{"fold_id": i} for i in range(4)], decisions,
        )
