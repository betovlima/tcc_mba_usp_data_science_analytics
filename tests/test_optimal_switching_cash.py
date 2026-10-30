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
