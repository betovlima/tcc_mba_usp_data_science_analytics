"""Unit tests: no lookahead, old forecast memory and horizon confidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pesquisar_persistencia_horizontes_spyder import (
    HorizonPersistencePolicy,
    uncertainty_from_matured_calibration,
    safe_json,
)


def _policy(
    *,
    aggregate,
    horizon,
    uncertainty=0.1,
    multiplier=1.0,
    dates=None,
):
    dates = dates or pd.date_range("2026-09-01", periods=len(aggregate), freq="B", tz="UTC")
    aggregate_cache = {
        pd.Timestamp(day): np.array(row, dtype=float)
        for day, row in zip(dates, aggregate)
    }
    horizon_cache = {
        pd.Timestamp(day): np.array(row, dtype=float)
        for day, row in zip(dates, horizon)
    }
    audit = []
    policy = HorizonPersistencePolicy(
        symbols=["MAN", "TSLA"],
        dates=dates,
        aggregate=aggregate_cache,
        per_horizon={5: horizon_cache},
        uncertainties={5: uncertainty},
        weights={5: 1.0},
        margin=0.001,
        cash_threshold=0.0,
        entry_threshold=0.001,
        confidence_multiplier=multiplier,
        audit=audit,
        fold_id=1,
    )
    return dates, policy, audit


def test_swaps_after_single_session_when_horizon_evidence_is_strong():
    days, policy, audit = _policy(
        aggregate=[[0, 0.3, 0.2], [0, 0.2, 0.7]],
        horizon=[[0, 0.3, 0.2], [0, 0.1, 0.8]],
        multiplier=1.0,
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 1)[0] == 2
    assert audit[-1]["reason"] == "ROTATE_HORIZON_CONFIDENT"
    assert audit[-1]["forecast_age_sessions"] == 1
    assert audit[-1]["H5_entry_forecast"] == 0.3


def test_rejects_weak_new_leader_despite_no_holding_minimum():
    days, policy, audit = _policy(
        aggregate=[[0, 0.3, 0.2], [0, 0.30, 0.34]],
        horizon=[[0, 0.3, 0.2], [0, 0.30, 0.34]],
        uncertainty=0.20,
        multiplier=1.0,
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 1)[0] == 1
    assert audit[-1]["reason"] == "HORIZON_CONFIDENCE_NOT_MET"


def test_cash_exit_not_blocked_by_horizon_memory():
    days, policy, audit = _policy(
        aggregate=[[0, 0.3, 0.2], [0, -0.2, -0.1]],
        horizon=[[0, 0.3, 0.2], [0, -0.2, -0.1]],
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 1)[0] == 0
    assert audit[-1]["reason"] == "CASH_THRESHOLD"


def test_entry_forecast_is_never_rewritten_by_later_predictions():
    days, policy, audit = _policy(
        aggregate=[[0, 0.3, 0.2], [0, 0.3, 0.1], [0, 0.2, 0.25]],
        horizon=[[0, 0.5, 0.2], [0, 0.1, 0.0], [0, 0.2, 0.3]],
        uncertainty=5.0,
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 1)[0] == 1
    assert policy(days[2], 1, 2)[0] == 1
    assert audit[-1]["H5_entry_forecast"] == 0.5
    assert audit[-1]["H5_current_today"] == 0.2


def test_uncertainty_ignores_labels_that_mature_after_training_cutoff():
    dates = pd.date_range("2026-01-01", periods=60, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {"forward_horizon_utility_5": np.concatenate([
            np.zeros(48), np.full(12, 100.0),
        ])},
        index=dates,
    )
    predictions = {
        pd.Timestamp(date): np.array([0.0, 0.0])
        for date in dates
    }
    error, n = uncertainty_from_matured_calibration(
        h=5,
        timestamps=dates,
        full_dates=dates,
        latest_label_date_index=50,
        symbols=["MAN"],
        frames={"MAN": frame},
        predictions=predictions,
    )
    assert n == 45
    assert error == 0.0


def test_insufficient_matured_validation_fails_instead_of_using_future():
    dates = pd.date_range("2026-01-01", periods=15, freq="B", tz="UTC")
    frame = pd.DataFrame({"forward_horizon_utility_10": np.zeros(15)}, index=dates)
    predictions = {
        pd.Timestamp(date): np.array([0.0, 0.0])
        for date in dates
    }
    with pytest.raises(RuntimeError, match="insuficientes residuos maturados"):
        uncertainty_from_matured_calibration(
            h=10,
            timestamps=dates,
            full_dates=dates,
            latest_label_date_index=15,
            symbols=["MAN"],
            frames={"MAN": frame},
            predictions=predictions,
        )


def test_nonfinite_metrics_are_serializable_without_fake_values():
    assert safe_json({"risk": np.float64(float("nan"))}) == {"risk": None}
    assert safe_json({"risk": float("inf")}) == {"risk": None}
