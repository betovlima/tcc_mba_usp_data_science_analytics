"""Testes de CASH condicionado ao estado, contra o Control Strategy #30."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pesquisa.cash_estados_strategy30 import (
    CounterfactualSettings,
    build_calibration_samples,
    fit_counterfactual_advantage,
    build_counterfactual_overlay,
)


def _fixture(days=140):
    dates = pd.date_range("2025-01-02", periods=days, freq="B", tz="UTC")
    opens = 99.0 * np.power(1.002, np.arange(days))
    closes = opens * (1 + 0.005 * np.sin(np.arange(days) / 10.0))
    frame = pd.DataFrame({
        "open": opens, "close": closes,
        "return_5": np.sin(np.arange(days) / 6.0) * 0.02,
        "return_20": np.sin(np.arange(days) / 20.0) * 0.04,
        "vol_20": np.ones(days) * 0.02,
        "distance_from_high_20": np.ones(days) * -0.04,
        "trend_efficiency_20": np.ones(days) * 0.5,
    }, index=dates)
    cfg = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0)
    settings = CounterfactualSettings(
        horizon=10, train_fraction=0.75, ridge_alpha=10.0,
        min_train_samples=45, min_validation_samples=12,
    )
    def policy(_date, _pos, _holding):
        return (1, 0.1)
    return dates, {"AAA": frame}, cfg, settings, policy


def test_cash_training_covers_cash_and_invested_histories():
    dates, frames, cfg, settings, policy = _fixture()
    training = dates[:126]
    sample = build_calibration_samples(
        policy, frames, ["AAA"], training, config=cfg,
        settings=settings, fold_id=1,
    )
    assert set(sample["current_asset"]) == {"CASH", "AAA"}
    assert int(sample["cash_position"].sum()) >= 20
    assert int((sample["cash_position"] == 0).sum()) > 30
    assert sample["label_last_date"].max() <= training[-1]
    assert not sample.duplicated(subset=[
        "decision_date", "current_asset", "holding_before", "control_asset",
    ]).any()
    assert {"cash_8_of_20", "control_shadow"} <= set(sample["scenario"])
    assert sample["decision_date"].nunique() <= len(training)


def test_cash_labels_ignore_modified_future_outside_calibration():
    dates, frames, cfg, settings, policy = _fixture()
    sample_a = build_calibration_samples(
        policy, frames, ["AAA"], dates[:126],
        config=cfg, settings=settings, fold_id=1,
    )
    modified = frames["AAA"].copy()
    modified.loc[dates[126]:, ["open", "close", "return_5", "return_20"]] *= 500
    sample_b = build_calibration_samples(
        policy, {"AAA": modified}, ["AAA"], dates[:126],
        config=cfg, settings=settings, fold_id=1,
    )
    pd.testing.assert_frame_equal(sample_a, sample_b)


def test_inner_validation_is_purged_and_never_uses_oos():
    dates, frames, cfg, settings, policy = _fixture()
    calibration = dates[:126]
    samples = build_calibration_samples(
        policy, frames, ["AAA"], calibration, config=cfg,
        settings=settings, fold_id=1,
    )
    fit, validation = fit_counterfactual_advantage(
        samples, calibration, settings=settings,
    )
    assert fit.training_rows >= 45
    assert fit.validation_rows >= 12
    assert fit.fit_last_label_date <= calibration[-1]
    assert fit.validation_first_date > calibration[0]
    assert validation["label_last_date"].max() <= calibration[-1]
    assert np.isfinite(validation["predicted_cash_minus_control_log_return"]).all()


def test_state_dependent_policy_exits_cash_then_reenters_next_session():
    dates, frames, _cfg, _settings, policy = _fixture(days=30)
    class StateModel:
        def predict(self, features):
            return 0.05 if features["cash_position"] < 0.5 else -0.05
    class Fit:
        def predict(self, features):
            return StateModel().predict(features)
    decisions = {}
    mapping = {date: 1 for date in dates}
    overlay = build_counterfactual_overlay(
        policy, fits_by_fold={1: Fit()}, decision_to_fold=mapping,
        frames=frames, symbols=["AAA"], diagnostics=decisions,
    )
    target1, _ = overlay(dates[3], 1, 3)
    target2, _ = overlay(dates[4], 0, 0)
    assert target1 == 0
    assert target2 == 1
    assert decisions[dates[3]]["reason"] == "predicted_cash_advantage"
    assert decisions[dates[4]]["reason"] == "continue_control"
    assert len(decisions) == 2


def test_calibration_rejects_too_short_horizon():
    dates, frames, cfg, settings, policy = _fixture(days=5)
    with pytest.raises(ValueError, match="insuficiente"):
        build_calibration_samples(
            policy, frames, ["AAA"], dates,
            config=cfg, settings=settings, fold_id=1,
        )


def test_spells_use_effective_position_and_only_next_session_reentry():
    from pesquisar_cash_estados_strategy30_spyder import _audit_cash_spells
    dates = pd.date_range("2026-09-01", periods=6, freq="B", tz="UTC")
    predictions = pd.DataFrame({
        "selected_asset": ["AAA", "CASH", "CASH", "BBB", "CASH", "BBB"],
    }, index=dates)
    report, intervals = _audit_cash_spells(predictions)
    assert report["cash_sessions"] == 3
    assert report["cash_intervals"] == 2
    assert report["entered_cash_from_asset"] == 2
    assert report["reentered_asset_from_cash"] == 2
    assert intervals["sessions"].tolist() == [2, 1]
    assert intervals["reentered"].tolist() == [True, True]


def test_failed_cash_runner_emits_zip_with_original_error(tmp_path, monkeypatch):
    import json
    from zipfile import ZipFile
    import pesquisar_cash_estados_strategy30_spyder as runner
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "output")
    def fail(_):
        raise RuntimeError("Control nao reproduzido; CASH nao autorizado")
    monkeypatch.setattr(runner, "_run", fail)
    with pytest.raises(RuntimeError, match="Control nao reproduzido"):
        runner.main()
    archives = list((tmp_path / "output").glob("pacote_optimal_switching_cash_*.zip"))
    assert len(archives) == 1
    with ZipFile(archives[0]) as package:
        assert package.testzip() is None
        error = json.loads(package.read("erro_execucao.json"))
        assert error["status"] == "failed"
        assert "Control nao reproduzido" in error["message"]
