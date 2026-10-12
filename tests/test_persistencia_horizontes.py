"""Unit tests: no lookahead, old forecast memory and horizon confidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pesquisar_persistencia_horizontes_spyder import (
    HorizonPersistencePolicy,
    uncertainty_from_matured_calibration,
    safe_json,
    select_purged_horizon_training_dates,
    EarlyHorizonOverridePolicy,
    calibrate_pair_gap_error,
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



def test_holding_position_inherited_across_fold_marks_forecast_proxy():
    days, policy, audit = _policy(
        aggregate=[[0, 0.3, 0.8]],
        horizon=[[0, 0.3, 0.9]],
        multiplier=0.0,
    )
    policy(days[0], 1, 4)
    assert audit[-1]["entry_forecast_origin"] == "fold_boundary_proxy"


def test_uncertainty_requires_multiple_matured_days_not_only_many_assets():
    dates = pd.date_range("2026-01-01", periods=25, freq="B", tz="UTC")
    symbols = [f"A{i:02d}" for i in range(75)]
    frames = {
        s: pd.DataFrame({"forward_horizon_utility_5": np.zeros(len(dates))}, index=dates)
        for s in symbols
    }
    predictions = {
        pd.Timestamp(d): np.zeros(len(symbols) + 1)
        for d in dates
    }
    with pytest.raises(RuntimeError, match="insuficientes residuos maturados"):
        uncertainty_from_matured_calibration(
            h=5,
            timestamps=dates[:1],
            full_dates=dates,
            latest_label_date_index=20,
            symbols=symbols,
            frames=frames,
            predictions=predictions,
        )


def test_official_sixty_session_purge_preserves_all_700_training_rows():
    calendar = pd.date_range("2020-01-01", periods=1600, freq="B", tz="UTC")
    for horizon in (5, 10, 20, 40, 60):
        calibration_training = select_purged_horizon_training_dates(
            calendar=calendar,
            training_end_index=700,
            evaluation_start_index=760,
            horizon=horizon,
            min_training_rows=700,
        )
        assert len(calibration_training) == 700
        assert calibration_training[-1] == calendar[699]
        # Even H60 matures by the final purged session, before validation.
        assert calendar.get_loc(calibration_training[-1]) + horizon < 760

        final_training = select_purged_horizon_training_dates(
            calendar=calendar,
            training_end_index=886,
            evaluation_start_index=946,
            horizon=horizon,
            min_training_rows=700,
        )
        assert len(final_training) == 886


def test_horizon_training_must_never_leak_into_calibration():
    calendar = pd.date_range("2020-01-01", periods=1000, freq="B", tz="UTC")
    with pytest.raises(RuntimeError, match="CausalHorizonPurgeInsufficient"):
        select_purged_horizon_training_dates(
            calendar=calendar,
            training_end_index=700,
            evaluation_start_index=750,
            horizon=60,
            min_training_rows=700,
        )


def test_horizon_training_must_not_hide_insufficient_model_rows():
    calendar = pd.date_range("2020-01-01", periods=1000, freq="B", tz="UTC")
    with pytest.raises(RuntimeError, match="HorizonTrainingRowsInsufficient"):
        select_purged_horizon_training_dates(
            calendar=calendar,
            training_end_index=699,
            evaluation_start_index=760,
            horizon=5,
            min_training_rows=700,
        )



def _early_override_case(
    *,
    aggregate,
    horizon,
    control_actions,
    multiplier=0.5,
    sigma=0.01,
):
    dates = pd.date_range(
        "2026-09-01", periods=len(aggregate), freq="B", tz="UTC"
    )
    aggregate_cache = {
        pd.Timestamp(day): np.asarray(row, dtype=float)
        for day, row in zip(dates, aggregate)
    }
    horizon_cache = {
        pd.Timestamp(day): np.asarray(row, dtype=float)
        for day, row in zip(dates, horizon)
    }
    actions = iter(control_actions)
    audit = []

    def controlled(day, current, holding):
        del day, holding
        action = next(actions)
        return action, float(aggregate_cache[pd.Timestamp(dates[0])][action])

    policy = EarlyHorizonOverridePolicy(
        symbols=["MAN", "TSLA"],
        dates=dates,
        aggregate=aggregate_cache,
        per_horizon={5: horizon_cache},
        pair_error_p75={5: sigma},
        weights={5: 1.0},
        control_policy=controlled,
        margin=0.001,
        minimum_holding=2,
        confidence_multiplier=multiplier,
        audit=audit,
        fold_id=1,
    )
    return dates, policy, audit


def test_early_override_changes_only_control_minimum_hold_decision():
    days, policy, audit = _early_override_case(
        aggregate=[[0, 0.2, 0.1], [0, 0.15, 0.8]],
        horizon=[[0, 0.2, 0.1], [0, 0.10, 0.90]],
        control_actions=[1, 1],
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 1)[0] == 2
    assert audit[-1]["control_target"] == "MAN"
    assert audit[-1]["final_target"] == "TSLA"
    assert audit[-1]["early_override"]
    assert audit[-1]["reason"] == "EARLY_OVERRIDE_HORIZON_PAIRS"


def test_override_does_not_block_control_rotation_after_minimum_hold():
    days, policy, audit = _early_override_case(
        aggregate=[[0, 0.2, 0.1], [0, 0.1, 0.8]],
        horizon=[[0, 0.2, 0.1], [0, -10, -10]],
        control_actions=[1, 2],
        multiplier=1000.0,
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 2)[0] == 2
    assert audit[-1]["reason"] == "CONTROL_HOLDING_ALREADY_ELIGIBLE"
    assert not audit[-1]["early_override"]


def test_override_does_not_fabricate_entry_forecast_when_fold_inherits_position():
    days, policy, audit = _early_override_case(
        aggregate=[[0, 0.1, 0.8]],
        horizon=[[0, 0.1, 0.8]],
        control_actions=[1],
        multiplier=0.0,
    )
    assert policy(days[0], 1, 1)[0] == 1
    assert audit[-1]["reason"] == "FOLD_INHERITED_OR_NO_ANCHOR"
    assert not audit[-1]["entry_forecast_observed"]


def test_override_keeps_cash_exit_from_control():
    days, policy, audit = _early_override_case(
        aggregate=[[0, 0.2, 0.1], [0, -0.1, -0.2]],
        horizon=[[0, 0.2, 0.1], [0, 0.3, 0.8]],
        control_actions=[1, 0],
        multiplier=0.0,
    )
    assert policy(days[0], 0, 0)[0] == 1
    assert policy(days[1], 1, 1)[0] == 0
    assert audit[-1]["final_target"] == "CASH"
    assert not audit[-1]["early_override"]


def test_pair_gap_error_uses_only_fully_matured_days():
    days = pd.date_range("2026-01-01", periods=90, freq="B", tz="UTC")
    labels_a = np.zeros(90)
    labels_b = np.zeros(90)
    # H5 labels after index 64 must be ignored by cutoff 70.
    labels_b[65:] = 1000.0
    frames = {
        "MAN": pd.DataFrame({"forward_horizon_utility_5": labels_a}, index=days),
        "TSLA": pd.DataFrame({"forward_horizon_utility_5": labels_b}, index=days),
    }
    predictions = {pd.Timestamp(d): np.array([0.0, 0.2, 0.1]) for d in days}
    aggregate = {pd.Timestamp(d): np.array([0.0, 0.2, 0.1]) for d in days}
    q, n, pairs = calibrate_pair_gap_error(
        horizon=5,
        days=days[:70],
        calendar=days,
        last_matured_exclusive=70,
        symbols=["MAN", "TSLA"],
        frames=frames,
        aggregate=aggregate,
        predictions=predictions,
    )
    assert n == 65
    assert pairs == 65
    assert abs(q - 0.1) < 1e-12


def test_pair_gap_error_does_not_count_many_pairs_as_many_days():
    days = pd.date_range("2026-01-01", periods=90, freq="B", tz="UTC")
    symbols = [f"S{i:02d}" for i in range(75)]
    frames = {
        s: pd.DataFrame({"forward_horizon_utility_5": np.zeros(90)}, index=days)
        for s in symbols
    }
    prediction = np.array([0.0, *np.linspace(0.1, 0.2, len(symbols))])
    predictions = {pd.Timestamp(d): prediction for d in days}
    with pytest.raises(RuntimeError, match="evidência histórica insuficiente"):
        calibrate_pair_gap_error(
            horizon=5,
            days=days[:2],
            calendar=days,
            last_matured_exclusive=30,
            symbols=symbols,
            frames=frames,
            aggregate=predictions,
            predictions=predictions,
        )
