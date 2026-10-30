"""Testes da vantagem contrafactual de CASH, separada do Control oficial."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from pesquisa.vantagem_cash_contrafactual import (
    CounterfactualSettings,
    FEATURE_COLUMNS,
    observable_features,
    counterfactual_cash_advantage,
    build_calibration_samples,
    fit_counterfactual_advantage,
    build_counterfactual_overlay,
)


def _example_bars(n=145, daily=0.004):
    dates = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = 10 * np.exp(np.arange(n) * daily)
    frame = pd.DataFrame({
        "open": close,
        "close": close,
        "return_5": np.linspace(-0.08, 0.08, n),
        "return_20": np.linspace(-0.12, 0.12, n),
        "vol_20": 0.015 + 0.001 * np.cos(np.arange(n) / 11),
        "distance_from_high_20": -0.02 + 0.005 * np.sin(np.arange(n) / 13),
        "trend_efficiency_20": 0.3 + 0.05 * np.cos(np.arange(n) / 5),
    }, index=dates)
    return frame


def _control(t, position, holding):
    return 1, 0.5


def test_cash_versus_control_label_respects_future_price_path():
    config = CONFIG.copiar_modelo(update={"slippage_bps": 0.0, "commission_rate": 0.0})
    settings = CounterfactualSettings()
    growing = _example_bars()
    falling = _example_bars(daily=-0.004)
    out_growing = counterfactual_cash_advantage(
        _control, {"AAA": growing}, ["AAA"], growing.index,
        start=40, current_position=0, holding_days=0,
        config=config, settings=settings,
    )
    out_falling = counterfactual_cash_advantage(
        _control, {"AAA": falling}, ["AAA"], falling.index,
        start=40, current_position=0, holding_days=0,
        config=config, settings=settings,
    )
    assert out_growing[0] < 0  # atrasar uma alta perde dinheiro
    assert out_falling[0] > 0  # atrasar uma queda preserva dinheiro
    assert out_growing[3] == 1  # apenas primeira etapa forcada em CASH


def test_counterfactual_labels_cannot_cross_calibration_end():
    bars = _example_bars()
    dates = bars.index[:126]
    settings = CounterfactualSettings()
    samples = build_calibration_samples(
        _control, {"AAA": bars}, ["AAA"], dates, config=CONFIG,
        settings=settings, fold_id=1,
    )
    assert len(samples) == len(dates) - settings.horizon
    assert pd.Timestamp(samples["label_last_date"].max()) <= dates[-1]
    assert samples["fold_id"].nunique() == 1
    with pytest.raises(ValueError, match="ultrapassaria"):
        counterfactual_cash_advantage(
            _control, {"AAA": bars}, ["AAA"], dates,
            start=122, current_position=1, holding_days=4,
            config=CONFIG, settings=settings,
        )


def test_ridge_uses_inner_purged_chronological_split():
    bars = _example_bars()
    dates = bars.index[:126]
    settings = CounterfactualSettings()
    samples = build_calibration_samples(
        _control, {"AAA": bars}, ["AAA"], dates,
        config=CONFIG, settings=settings, fold_id=2,
    )
    model, validation = fit_counterfactual_advantage(
        samples, dates, settings=settings,
    )
    assert model.training_rows >= settings.min_train_samples
    assert model.validation_rows >= settings.min_validation_samples
    assert pd.Timestamp(model.fit_last_label_date) <= dates[-1]
    assert model.validation_first_date > samples.loc[
        samples["label_last_date"] < model.validation_first_date,
        "label_last_date",
    ].max()
    assert len(validation) == model.validation_rows
    assert list(FEATURE_COLUMNS) == list(samples[list(FEATURE_COLUMNS)].columns)
    features = observable_features(
        bars, dates[-1], control_score=0.5,
        current_position=0, holding_days=0,
    )
    assert np.isfinite(model.predict(features))


def test_future_change_cannot_modify_observable_features_or_fitted_model():
    bars = _example_bars()
    date = bars.index[70]
    original = observable_features(
        bars, date, control_score=0.7,
        current_position=1, holding_days=5,
    )
    modified = bars.copy()
    modified.loc[modified.index > date, ["open", "close", "return_5", "return_20"]] *= 1000
    assert original == observable_features(
        modified, date, control_score=0.7,
        current_position=1, holding_days=5,
    )
    dates = bars.index[:126]
    settings = CounterfactualSettings()
    base = build_calibration_samples(
        _control, {"AAA": bars}, ["AAA"], dates,
        config=CONFIG, settings=settings, fold_id=1,
    )
    changed = bars.copy()
    changed.loc[changed.index >= dates[-1] + pd.Timedelta(days=1), "close"] *= 1000
    alt = build_calibration_samples(
        _control, {"AAA": changed}, ["AAA"], dates,
        config=CONFIG, settings=settings, fold_id=1,
    )
    pd.testing.assert_frame_equal(base, alt)


def test_policy_can_choose_cash_then_reenter_from_fitted_predictions():
    from pesquisa.vantagem_cash_contrafactual import CounterfactualFit

    dates = pd.date_range("2024-02-01", periods=2, tz="UTC")
    frames = {"AAA": pd.DataFrame({
        **{k: [0.1, 0.1] for k in FEATURE_COLUMNS if k not in (
            "control_score", "cash_position", "holding_days",
        )},
        "close": [10.0, 11.0],
    }, index=dates)}
    class ToyModel:
        def predict(self, array):
            return [0.02 if float(array.iloc[0]["cash_position"]) == 0 else -0.02]
    fit = CounterfactualFit(
        model=ToyModel(), training_rows=80, validation_rows=20,
        calibration_rows=110, fit_last_label_date=dates[0],
        validation_first_date=dates[0], validation_mae=0.01,
        validation_direction_accuracy=0.5,
    )
    diagnostics = {}
    policy = build_counterfactual_overlay(
        _control,
        fits_by_fold={1: fit},
        decision_to_fold={dt: 1 for dt in dates},
        frames=frames,
        symbols=["AAA"],
        diagnostics=diagnostics,
    )
    assert policy(dates[0], 1, 2)[0] == 0
    assert policy(dates[1], 0, 0)[0] == 1
    assert diagnostics[dates[0]]["reason"] == "predicted_cash_advantage"
    assert diagnostics[dates[1]]["reason"] == "continue_control"
