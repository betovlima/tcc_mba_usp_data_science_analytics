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


def test_v104_calibrated_holding_selects_only_given_historical_scores():
    from pesquisar_persistencia_horizontes_spyder import select_calibrated_holding

    assert select_calibrated_holding({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.0}) == 3
    assert select_calibrated_holding({1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}) == 2
    assert select_calibrated_holding({1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0}) == 1
    with pytest.raises(ValueError, match="todos os holdings"):
        select_calibrated_holding({1: 0.1, 2: 0.2, 3: 0.3})
    with pytest.raises(ValueError, match="nao finitos"):
        select_calibrated_holding({1: 0.1, 2: float("nan"), 3: 0.3, 4: 0.4})


def test_v104_holding_changes_only_min_hold_gate_with_frozen_utilities():
    from types import SimpleNamespace
    from engine.rotacao import _politica_utilidade

    day = pd.Timestamp("2026-10-01", tz="UTC")
    cache = {day: np.asarray([0.0, 0.20, 0.80])}
    for holding in (1, 2, 3, 4):
        cfg = SimpleNamespace(
            rotation_cash_threshold=0.0,
            rotation_min_expected_edge=0.001,
            rotation_switch_margin=0.0005,
            rotation_min_holding_days=holding,
        )
        policy = _politica_utilidade(
            {}, {}, ["MAN", "TSLA"], cfg, 0.0005, utility_cache=cache
        )
        assert policy(day, 1, 1)[0] == (2 if holding == 1 else 1)
        assert policy(day, 1, holding)[0] == 2


def test_v104_point_in_time_features_do_not_export_forward_labels():
    from pesquisar_persistencia_horizontes_spyder import causal_hold_guard_features

    day = pd.Timestamp("2026-10-01", tz="UTC")
    frame = pd.DataFrame({
        "vol_20": [0.03], "vol_60": [0.06], "return_5": [0.10],
        "trend_efficiency_20": [0.40], "distance_from_high_20": [-0.02],
        "forward_horizon_utility_5": [99999.0],
    }, index=pd.DatetimeIndex([day]))
    audit = {day: {
        "min_hold_guard_applied": True,
        "decision_fold_id": 1,
        "current_asset": "MAN",
        "raw_best_asset": "TSLA",
        "holding_days_at_decision": 1,
        "best_vs_current_gap": 0.11,
    }}
    table = causal_hold_guard_features(audit, {"MAN": frame})
    assert len(table) == 1
    assert table.iloc[0]["vol_20"] == 0.03
    assert not any("forward" in col for col in table.columns)

    audit[day]["min_hold_guard_applied"] = False
    assert causal_hold_guard_features(audit, {"MAN": frame}).empty


def _v105_fixture(
    *,
    incumbent_values=(0.01, 0.02, 0.03, 0.04),
    challenger_values=(0.04, 0.05, 0.08, 0.10),
    other_values=(0.02, 0.025, 0.02, 0.03),
    control_action=1,
    aggregate=None,
):
    from pesquisar_persistencia_horizontes_spyder import (
        RelativeHorizonGainPolicy,
    )

    day = pd.Timestamp("2026-10-01", tz="UTC")
    mat = {
        h: {day: np.array([0.0, incumbent_values[i], challenger_values[i], other_values[i]])}
        for i, h in enumerate((5, 10, 20, 40))
    }
    agg = {day: np.array(
        aggregate if aggregate is not None else [0.0, 0.3, 0.8, 0.2]
    )}

    def baseline(timestamp, current, holding):
        assert timestamp == day
        del current, holding
        return control_action, float(agg[day][control_action])

    audit = []
    ranks = []
    p = RelativeHorizonGainPolicy(
        symbols=["AAPL", "NVDA", "MSFT"],
        aggregate=agg,
        forecasts=mat,
        control_policy=baseline,
        minimum_holding=2,
        margin=0.001,
        require_each_horizon=False,
        fold_id=1,
        audit=audit,
        ranking_audit=ranks,
    )
    return day, p, audit, ranks


def test_v105_candidate_relatively_superior_both_buckets():
    day, policy, audit, ranking = _v105_fixture()
    assert policy(day, 1, 2)[0] == 2
    assert audit[-1]["changes_control_decision"]
    assert audit[-1]["reason"] == "RELATIVE_HORIZON_SWITCH"
    assert ranking[0]["candidate"] == "NVDA"
    assert ranking[0]["candidate_expected_H5_pct"] == 0.04
    assert ranking[0]["delta_H40_pct_points"] == pytest.approx(0.06)


def test_v105_short_term_only_does_not_justify_rotation():
    day, policy, audit, _ = _v105_fixture(
        challenger_values=(0.04, 0.06, 0.02, 0.03),
    )
    assert policy(day, 1, 2)[0] == 1
    assert audit[-1]["reason"] == "NO_SUPERIOR_HORIZON_CANDIDATE"


def test_v105_min_holding_and_cash_control_are_preserved():
    day, policy, audit, _ = _v105_fixture()
    assert policy(day, 1, 1)[0] == 1
    assert audit[-1]["reason"] == "MIN_HOLD_PRESERVED"

    day, policy, audit, _ = _v105_fixture(control_action=0)
    assert policy(day, 1, 2)[0] == 0
    assert audit[-1]["reason"] == "CONTROL_CASH_EXIT_PRESERVED"


def test_v105_strict_variant_requires_each_horizon_and_margin():
    from pesquisar_persistencia_horizontes_spyder import (
        relative_horizon_candidates,
    )
    day, policy, _, _ = _v105_fixture(
        challenger_values=(0.0, 0.09, 0.04, 0.20)
    )
    accepted, _ = relative_horizon_candidates(
        date=day,
        incumbent=1,
        control_target=1,
        symbols=policy.symbols,
        aggregate=policy.aggregate[day],
        forecasts=policy.forecasts,
        min_group_advantage=0.005,
        require_each_horizon=False,
        effective_margin=0.001,
    )
    assert accepted and accepted[0]["candidate"] == "NVDA"
    strict, _ = relative_horizon_candidates(
        date=day,
        incumbent=1,
        control_target=1,
        symbols=policy.symbols,
        aggregate=policy.aggregate[day],
        forecasts=policy.forecasts,
        min_group_advantage=0.005,
        require_each_horizon=True,
        effective_margin=0.001,
    )
    assert strict == []


def test_v105_compares_with_existing_control_target_not_just_incumbent():
    day, policy, audit, _ = _v105_fixture(
        control_action=3,
        other_values=(0.12, 0.14, 0.15, 0.16),
    )
    assert policy(day, 1, 2)[0] == 3
    assert not audit[-1]["changes_control_decision"]


def test_v105_future_labels_are_training_only():
    from pesquisar_persistencia_horizontes_spyder import (
        add_horizon_percentage_training_targets,
    )
    day = pd.Timestamp("2026-10-01", tz="UTC")
    frames = {"AAPL": pd.DataFrame({
        f"forward_horizon_net_log_return_{h}": [np.log1p(h / 1000)]
        for h in (5, 10, 20, 40)
    }, index=[day])}
    add_horizon_percentage_training_targets(frames)
    assert frames["AAPL"].loc[day, "forward_horizon_net_return_pct_20"] == pytest.approx(0.02)


def test_v106_capture_prediction_does_not_read_labels():
    from pesquisar_persistencia_horizontes_spyder import select_horizon_candidates

    day = pd.Timestamp("2025-05-01", tz="UTC")
    forecast = {
        h: {day: np.array([0.0, 0.01, 0.07, 0.02])}
        for h in (2, 3, 5, 10)
    }
    top, selected = select_horizon_candidates(
        day=day, incumbent="AAPL",
        symbols=["AAPL", "NVDA", "MSFT"],
        forecast=forecast, horizons=(2, 3, 5, 10),
        extra_switch_cost=0.005,
    )
    assert selected["best_candidate"] == "NVDA"
    assert top[0]["candidate"] == "NVDA"
    assert selected["H2_pred_increment_pct_points"] == pytest.approx(5.5)
    # Selected only from the forecast arrays: no realized labels exist here.


def test_v106_audit_selection_independent_of_capture():
    from pesquisar_persistencia_horizontes_spyder import select_horizon_candidates

    day = pd.Timestamp("2025-05-01", tz="UTC")
    forecasts = {
        h: {day: np.array([0.0, 0.01, 0.09, 0.02])}
        for h in (2, 3, 5, 10)
    }
    forecasts.update({
        h: {day: np.array([0.0, 0.01, 0.02, 0.16])}
        for h in (1, 15, 20, 30, 40, 60)
    })
    capture, capture_selected = select_horizon_candidates(
        day=day, incumbent="AAPL", symbols=["AAPL", "NVDA", "MSFT"],
        forecast=forecasts, horizons=(2, 3, 5, 10),
        extra_switch_cost=0.005,
    )
    broad, broad_selected = select_horizon_candidates(
        day=day, incumbent="AAPL", symbols=["AAPL", "NVDA", "MSFT"],
        forecast=forecasts,
        horizons=(1, 2, 3, 5, 10, 15, 20, 30, 40, 60),
        extra_switch_cost=0.005,
    )
    assert capture_selected["best_candidate"] == "NVDA"
    assert broad_selected["best_candidate"] == "MSFT"
    assert capture[0]["candidate"] == "NVDA"
    assert broad[0]["candidate"] == "MSFT"


def test_v106_new_horizon_targets_use_fixed_calendar_not_sparse_shifts():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import build_horizon_return_labels

    calendar = pd.date_range("2025-01-02", periods=16, freq="B", tz="UTC")
    frame = pd.DataFrame({
        "open": np.arange(100.0, 116.0),
        "close": np.arange(100.0, 116.0),
    }, index=calendar)
    frames = {"A": frame.copy()}
    cfg = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0)
    build_horizon_return_labels(frames, calendar, cfg, (2, 3, 15))
    assert frames["A"].loc[calendar[0], "forward_horizon_net_return_pct_2"] == pytest.approx(
        102.0 / 101.0 - 1
    )
    assert pd.isna(frames["A"].loc[calendar[1], "forward_horizon_net_return_pct_15"])

    # Missing middle session must NOT transform H2 into an H3 bar.
    sparse = frame.drop(index=calendar[2])
    frames = {"A": sparse}
    build_horizon_return_labels(frames, calendar, cfg, (2,))
    assert pd.isna(
        frames["A"].loc[calendar[0], "forward_horizon_net_return_pct_2"]
    )


def test_v106_maturity_and_realized_outcome_are_strictly_post_selection():
    from pesquisar_persistencia_horizontes_spyder import (
        attach_fixed_horizon_outcomes,
        fixed_horizon_value,
        summarize_selection_quality,
    )

    day = pd.Timestamp("2025-05-01", tz="UTC")
    frame_a = pd.DataFrame({
        "forward_horizon_net_return_pct_2": [0.02],
        "forward_horizon_net_return_pct_3": [0.04],
    }, index=[day])
    frame_b = pd.DataFrame({
        "forward_horizon_net_return_pct_2": [0.08],
        "forward_horizon_net_return_pct_3": [0.07],
    }, index=[day])
    frames = {"A": frame_a, "B": frame_b}
    maturities = {1: {day: 2}}
    assert fixed_horizon_value(
        day=day, horizon=3, symbol="B",
        frames=frames, maturity=maturities[1],
    ) is None
    selected = [{
        "decision_date": day.isoformat(),
        "fold_id": 1,
        "incumbent": "A",
        "best_candidate": "B",
        "H2_pred_increment_pct_points": 5.0,
        "H3_pred_increment_pct_points": 4.0,
    }]
    _, outputs = attach_fixed_horizon_outcomes(
        [], selected, horizons=(2, 3), folds_maturity=maturities,
        frames=frames, extra_switch_cost=0.005,
    )
    assert outputs[0]["H2_realized_increment_pp"] == pytest.approx(5.5)
    assert outputs[0]["H3_realized_increment_pp"] is None
    quality = summarize_selection_quality(outputs, (2, 3))
    two = next(r for r in quality if r["fold_id"] == 1 and r["horizon"] == 2)
    assert two["hit_rate"] == 1.0
    assert two["prediction_mae_pp"] == pytest.approx(0.5)
    three = next(r for r in quality if r["fold_id"] == 1 and r["horizon"] == 3)
    assert three["matured_evaluable"] == 0


def test_v106_longest_forecast_requires_a_causal_training_purge():
    from pesquisar_persistencia_horizontes_spyder import (
        ALL_RESEARCH_HORIZONS,
        CAPTURE_HORIZONS,
        EXTENDED_AUDIT_HORIZONS,
    )
    assert CAPTURE_HORIZONS == (2, 3, 5, 10)
    assert 60 in EXTENDED_AUDIT_HORIZONS
    assert max(ALL_RESEARCH_HORIZONS) <= 60


def test_v106_capture_requires_advantage_in_short_and_medium():
    from pesquisar_persistencia_horizontes_spyder import select_horizon_candidates

    day = pd.Timestamp("2025-05-01", tz="UTC")
    data = {
        2: np.array([0.0, 0.01, -0.03]),
        3: np.array([0.0, 0.01, -0.03]),
        5: np.array([0.0, 0.01, 0.16]),
        10: np.array([0.0, 0.01, 0.16]),
    }
    forecasts = {h: {day: v} for h, v in data.items()}
    ranked, capture = select_horizon_candidates(
        day=day, incumbent="AAPL", symbols=["AAPL", "NVDA"],
        forecast=forecasts, horizons=(2, 3, 5, 10),
        extra_switch_cost=0.005,
        capture_require_short_medium_advantage=True,
    )
    assert ranked[0]["candidate"] == "NVDA"
    assert capture["best_candidate"] is None
    assert capture["eligible_candidate_count"] == 0

    _, audit = select_horizon_candidates(
        day=day, incumbent="AAPL", symbols=["AAPL", "NVDA"],
        forecast=forecasts, horizons=(2, 3, 5, 10),
        extra_switch_cost=0.005,
        capture_require_short_medium_advantage=False,
    )
    assert audit["best_candidate"] == "NVDA"
