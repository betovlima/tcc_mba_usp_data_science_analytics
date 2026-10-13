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


def test_v107_pair_ranking_compares_control_target_not_only_current():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import rank_direct_pair_gap

    day = pd.Timestamp("2025-05-06", tz="UTC")
    frames = {
        asset: pd.DataFrame({"close": [100.0]}, index=[day])
        for asset in ("AAPL", "MSFT", "NVDA")
    }
    # AAPL current=2%, MSFT Control target=10%, NVDA candidate=8%.
    preds = {
        h: {day: np.asarray([0.0, 0.02, 0.10, 0.08])}
        for h in (2, 3, 5, 10)
    }
    cfg = SimpleNamespace(
        slippage_bps=0.0, commission_rate=0.0, cat_fee_per_share=0.0,
        sec_fee_rate=0.0, taf_fee_per_share=0.0,
    )
    options = rank_direct_pair_gap(
        day=day, incumbent="AAPL", control_target="MSFT",
        symbols=["AAPL", "MSFT", "NVDA"],
        forecasts=preds, frames=frames, config=cfg,
    )
    assert options[0]["candidate"] == "NVDA"
    assert options[0]["raw_incumbent_short_gap"] > 0
    assert options[0]["raw_pair_short_gap"] < 0
    assert not options[0]["raw_qualifies"]


def test_v107_cost_proxy_uses_same_day_prices_and_nonzero_fees():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        estimated_incremental_pair_cost,
    )

    day = pd.Timestamp("2025-05-06", tz="UTC")
    frames = {
        "AAPL": pd.DataFrame({"close": [100.0]}, index=[day]),
        "NVDA": pd.DataFrame({"close": [30.0]}, index=[day]),
        "MSFT": pd.DataFrame({"close": [200.0]}, index=[day]),
    }
    cfg = SimpleNamespace(
        slippage_bps=0.0,
        commission_rate=0.0,
        sec_fee_rate=0.0000206,
        taf_fee_per_share=0.000195,
        cat_fee_per_share=0.000003,
    )
    switching = estimated_incremental_pair_cost(
        day=day, incumbent="AAPL", control_target="AAPL",
        candidate="NVDA", frames=frames, config=cfg,
    )
    assert switching > 0.0000206
    replacing = estimated_incremental_pair_cost(
        day=day, incumbent="AAPL", control_target="MSFT",
        candidate="NVDA", frames=frames, config=cfg,
    )
    assert replacing > 0
    assert replacing < switching


def test_v107_pair_calibration_has_fully_purged_split_and_ignores_validation():
    from pesquisar_persistencia_horizontes_spyder import (
        train_pairwise_winner_calibration,
    )

    calendar = pd.date_range("2025-01-01", periods=140, freq="B", tz="UTC")
    values = np.full(140, 0.02)
    values[91:] = 5.0
    frames = {
        "AAPL": pd.DataFrame({
            f"forward_horizon_net_return_pct_{h}": np.zeros(140)
            for h in (2, 3, 5, 10)
        }, index=calendar),
        "NVDA": pd.DataFrame({
            f"forward_horizon_net_return_pct_{h}": values
            for h in (2, 3, 5, 10)
        }, index=calendar),
    }
    samples = [{
        "decision_date": d.isoformat(),
        "control_target": "AAPL",
        "candidate": "NVDA",
        "incremental_cost_fraction": 0.0,
        **{f"H{h}_raw_pair_gap_fraction": 0.05 for h in (2, 3, 5, 10)},
    } for d in calendar[:116]]
    params = train_pairwise_winner_calibration(
        selected=samples, frames=frames, calendar=calendar,
        split_date=calendar[90], cutoff_date=calendar[135],
    )
    for h in (2, 3, 5, 10):
        row = params["by_horizon"][str(h)]
        assert row["slope"] == pytest.approx(0.4)
        assert row["abs_residual_p75"] < 1e-10
        assert row["training_unique_days"] <= 90 - h
    assert params["by_horizon"]["10"]["training_unique_days"] == 80


def test_v107_confident_pair_requires_robust_calibrated_gap():
    from pesquisar_persistencia_horizontes_spyder import calibrated_pair_choice

    raw = {
        "candidate": "NVDA",
        "raw_qualifies": True,
        **{f"H{h}_raw_pair_gap_fraction": 0.04 for h in (2, 3, 5, 10)},
        **{f"H{h}_raw_incumbent_gap_fraction": 0.04 for h in (2, 3, 5, 10)},
    }
    data = {"by_horizon": {
        str(h): {"slope": 0.0, "abs_residual_p75": 0.015}
        for h in (2, 3, 5, 10)
    }}
    winner, ranked = calibrated_pair_choice([raw], data)
    assert winner is None
    assert len(ranked) == 1
    data = {"by_horizon": {
        str(h): {"slope": 0.8, "abs_residual_p75": 0.0001}
        for h in (2, 3, 5, 10)
    }}
    winner, ranked = calibrated_pair_choice([raw], data)
    assert winner is not None
    assert ranked[0]["calibrated_short_gap"] == pytest.approx(0.032)


def test_v107_fold_boundary_and_evaluation_maturity_do_not_leak():
    from pesquisar_persistencia_horizontes_spyder import (
        assess_pairwise_fixed_outcomes,
    )

    dates = pd.date_range("2025-01-01", periods=15, freq="B", tz="UTC")
    frames = {
        "AAPL": pd.DataFrame({
            f"forward_horizon_net_return_pct_{h}": np.zeros(15)
            for h in (2, 3, 5, 10)
        }, index=dates),
        "NVDA": pd.DataFrame({
            f"forward_horizon_net_return_pct_{h}": np.full(15, 0.2)
            for h in (2, 3, 5, 10)
        }, index=dates),
    }
    pairs = [{
        "decision_date": dates[9].isoformat(),
        "control_target": "AAPL",
        "candidate": "NVDA",
        "incremental_cost_fraction": 0.002,
        **{f"H{h}_raw_pair_gap_fraction": 0.02 for h in (2, 3, 5, 10)},
    }]
    observed = assess_pairwise_fixed_outcomes(
        pairs, frames=frames, calendar=dates,
        horizon_end_exclusive=dates[11],
    )
    assert observed[0]["H2_realized_gap_fraction"] is None
    assert observed[0]["H3_realized_gap_fraction"] is None
    observed = assess_pairwise_fixed_outcomes(
        pairs, frames=frames, calendar=dates,
        horizon_end_exclusive=dates[-1] + pd.Timedelta(days=7),
    )
    assert observed[0]["H2_realized_gap_fraction"] == pytest.approx(0.198)


def test_v107_expanded_audit_has_candidate_specific_friction():
    from pesquisar_persistencia_horizontes_spyder import select_horizon_candidates

    day = pd.Timestamp("2025-05-06", tz="UTC")
    forecasts = {h: {day: np.array([0.0, 0.03, 0.05])} for h in (1, 2)}
    ranks, chosen = select_horizon_candidates(
        day=day, incumbent="AAPL", symbols=["AAPL", "NVDA"],
        forecast=forecasts, horizons=(1, 2), extra_switch_cost=0.0,
        candidate_cost_fn=lambda symbol: 0.01,
    )
    assert chosen["best_candidate"] == "NVDA"
    assert chosen["extra_switch_cost_fraction"] == 0.01
    assert ranks[0]["H1_pred_increment_pct_points"] == pytest.approx(1.0)


def test_v108_independent_horizon_labels_mature_near_end():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        build_horizon_return_labels,
    )

    dates = pd.date_range("2026-01-01", periods=90, freq="B", tz="UTC")
    frame = pd.DataFrame({
        "open": np.full(90, 100.0),
        "close": 100.0 + np.arange(90, dtype=float),
        "forward_horizon_net_log_return_5": np.full(90, np.nan),
    }, index=dates)
    frames = {"AAPL": frame}
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0)
    build_horizon_return_labels(frames, dates, config, (2, 5, 60))
    assert frames["AAPL"].loc[dates[-3], "forward_horizon_net_return_pct_2"] == pytest.approx(
        (100.0 + 89.0) / 100.0 - 1.0
    )
    assert pd.isna(
        frames["AAPL"].loc[dates[-3], "forward_horizon_net_return_pct_5"]
    )
    assert pd.isna(
        frames["AAPL"].loc[dates[-2], "forward_horizon_net_return_pct_2"]
    )
    assert np.isfinite(
        frames["AAPL"].loc[dates[0], "forward_horizon_net_return_pct_60"]
    )


def test_v108_pair_block_is_oriented_and_uses_only_present_features():
    from pesquisar_persistencia_horizontes_spyder import (
        PAIR_TRAINING_FEATURES,
        direct_pair_feature_block,
        pair_feature_columns,
    )

    n = len(PAIR_TRAINING_FEATURES)
    a = np.full((2, n), 0.12, dtype=float)
    b = np.full((2, n), 0.04, dtype=float)
    forward = direct_pair_feature_block(a, b)
    reverse = direct_pair_feature_block(b, a)
    assert forward.shape == (2, 3*n)
    assert len(pair_feature_columns()) == 3*n
    assert np.allclose(forward[:, 2*n:], 0.08)
    assert np.allclose(reverse[:, 2*n:], -0.08)


def test_v108_pair_training_enforces_maturity_and_balanced_directions():
    from pesquisar_persistencia_horizontes_spyder import (
        PAIR_TRAINING_FEATURES,
        train_direct_pair_dataset,
    )

    n = len(PAIR_TRAINING_FEATURES)
    features = np.ones((35, 4, n), dtype=np.float32)
    target = np.tile(np.array([0.02, 0.1, -0.04, 0.2]), (35, 1))
    x, y, metadata = train_direct_pair_dataset(
        feature_cube=features, target_cube=target,
        training_date_indices=np.arange(20),
        horizon=5, evaluation_start_index=25,
        reference_count=2,
    )
    assert metadata["training_unique_days"] == 20
    assert metadata["examples"] == len(y)
    assert len(x) == len(y)
    assert abs(float(np.sum(y))) < 1e-4
    with pytest.raises(RuntimeError, match="crosses evaluation boundary"):
        train_direct_pair_dataset(
            feature_cube=features, target_cube=target,
            training_date_indices=np.arange(21),
            horizon=5, evaluation_start_index=25,
            reference_count=2,
        )


def test_v108_direct_pair_predictions_compare_with_control_and_incumbent():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        PAIR_TRAINING_FEATURES,
        predict_direct_pair_gaps,
    )

    day = pd.Timestamp("2026-01-01", tz="UTC")
    n = len(PAIR_TRAINING_FEATURES)
    features = np.ones((3, n), dtype=np.float32)
    prices = {"AAPL": 100.0, "MSFT": 200.0, "NVDA": 30.0}
    frames = {
        symbol: pd.DataFrame({"close": [price]}, index=[day])
        for symbol, price in prices.items()
    }
    cfg = SimpleNamespace(
        slippage_bps=0.0, commission_rate=0.0,
        cat_fee_per_share=0.0, sec_fee_rate=0.0,
        taf_fee_per_share=0.0,
    )

    class FixedPairModel:
        def __init__(self, value):
            self.value = value

        def predict(self, matrix):
            assert len(matrix.columns) == n * 3
            return np.full(len(matrix), self.value)

    models = {h: FixedPairModel(0.05) for h in (2, 3, 5, 10)}
    rows = predict_direct_pair_gaps(
        models, features, symbols=["AAPL", "MSFT", "NVDA"],
        incumbent="AAPL", control_target="MSFT", day=day,
        frames=frames, config=cfg,
    )
    assert len(rows) == 1
    assert rows[0]["candidate"] == "NVDA"
    assert rows[0]["H2_raw_pair_gap_fraction"] > 0


def test_v108_winner_pair_calibration_excludes_unmatured_future():
    from pesquisar_persistencia_horizontes_spyder import (
        calibrate_direct_pair_confidence,
        choose_confident_direct_pair,
    )

    dates = pd.date_range("2026-01-01", periods=125, freq="B", tz="UTC")
    frame_b = pd.DataFrame({
        f"forward_horizon_net_return_pct_{h}": np.zeros(len(dates))
        for h in (2, 3, 5, 10)
    }, index=dates)
    frame_a = pd.DataFrame({
        f"forward_horizon_net_return_pct_{h}": np.array(
            [0.03]*80 + [200.0]*(len(dates)-80)
        )
        for h in (2, 3, 5, 10)
    }, index=dates)
    rows = [{
        "decision_date": day.isoformat(),
        "incumbent": "AAPL",
        "control_target": "AAPL",
        "candidate": "NVDA",
        "incremental_cost_fraction": 0.0,
        "raw_qualifies": True,
        **{f"H{h}_raw_pair_gap_fraction": 0.03 for h in (2, 3, 5, 10)},
        **{f"H{h}_raw_incumbent_gap_fraction": 0.03 for h in (2, 3, 5, 10)},
    } for day in dates[:104]]
    params = calibrate_direct_pair_confidence(
        rows, frames={"AAPL": frame_b, "NVDA": frame_a},
        calendar=dates, split_date=dates[80],
    )
    assert params["by_horizon"]["10"]["training_unique_days"] == 70
    assert params["by_horizon"]["10"]["abs_residual_p75"] == pytest.approx(0.0)
    winner, rows = choose_confident_direct_pair(rows[:1], params)
    assert winner is not None
    assert winner["candidate"] == "NVDA"


def test_v109_all_eight_methods_use_same_universe_and_frozen_inputs():
    from pesquisar_persistencia_horizontes_spyder import (
        RANKING_METHODS, rank_with_all_methods,
    )

    symbols = ["AAPL", "AMD", "MSFT", "NVDA"]
    preds = {
        2: np.array([0.01, 0.06, 0.02, 0.04]),
        3: np.array([0.01, 0.05, 0.03, 0.04]),
        5: np.array([0.01, 0.04, 0.03, 0.06]),
        10: np.array([0.01, 0.03, 0.02, 0.07]),
    }
    errors = {
        h: {
            "center": np.zeros(4),
            "scale": np.full(4, 0.04),
            "effective_days": 30,
        }
        for h in (2, 3, 5, 10)
    }
    kwargs = {
        "forecasts": preds, "symbols": symbols,
        "reference_symbol": "AAPL",
        "incremental_cost": np.array([0.0, 0.001, 0.001, 0.001]),
        "volatility": np.array([0.03, 0.07, 0.015, 0.1]),
        "historical_error": errors,
        "ltr_scores": np.array([0.1, 0.6, 0.4, 0.8]),
    }
    scores1 = rank_with_all_methods(**kwargs)
    scores2 = rank_with_all_methods(**kwargs)
    assert set(scores1) == set(RANKING_METHODS)
    assert scores1 == scores2
    for ordered in scores1.values():
        assert [index for index, _ in sorted(ordered)] == list(range(4))
        assert len(ordered) == 4
    assert scores1["predicted_return"][0][0] == 1
    assert scores1["learning_to_rank"][0][0] == 3


def test_v109_pareto_fronts_are_nondominated_then_recursive():
    from pesquisar_persistencia_horizontes_spyder import nondominated_layers

    values = np.array([
        [2., 2.], [1., 1.], [0., 0.], [2., 1.],
    ])
    layers = nondominated_layers(values)
    assert layers.tolist() == [0, 2, 3, 1]
    with pytest.raises(ValueError, match="finite"):
        nondominated_layers(np.array([[0.0, np.nan]]))


def test_v109_rank_quality_does_not_replace_missing_top1_using_future():
    from pesquisar_persistencia_horizontes_spyder import rank_quality_metrics

    ranking = [(2, 9.0), (1, 6.0), (0, 0.0), (3, -1.0)]
    out = rank_quality_metrics(
        ranking=ranking,
        realized=np.array([0., 0.04, np.nan, -0.02]),
        reference_index=0, costs=np.zeros(4),
    )
    assert out["ndcg_at_5"] is None
    assert out["top1_gap"] is None

    out2 = rank_quality_metrics(
        ranking=ranking,
        realized=np.array([0.0, 0.04, 0.07, -0.02]),
        reference_index=0, costs=np.zeros(4),
    )
    assert out2["top1_gap"] == pytest.approx(0.07)
    assert out2["top1_win"] == 1
    assert 0 <= out2["ndcg_at_5"] <= 1.0
    assert out2["top1_regret"] == pytest.approx(0.0)

    # Mesmo com Top-1 conhecido, nao substituir o segundo colocado
    # por outro ativo se faltar o desfecho do ranking original.
    missing_second = rank_quality_metrics(
        ranking=ranking,
        realized=np.array([0.0, np.nan, 0.07, -0.02]),
        reference_index=0, costs=np.zeros(4),
    )
    assert missing_second["top1_gap"] == pytest.approx(0.07)
    assert missing_second["top5_gap"] is None
    assert missing_second["top5_original_complete"] is False


def test_v109_probabilistic_error_calibration_matures_before_split():
    from pesquisar_persistencia_horizontes_spyder import (
        calibrate_asset_horizon_errors,
    )

    dates = pd.date_range("2025-01-01", periods=85, freq="B", tz="UTC")
    frames = {
        sym: pd.DataFrame({
            f"forward_horizon_net_return_pct_{h}": np.concatenate([
                np.full(55, 0.02 + i * 0.01),
                np.full(30, 500.0),
            ])
            for h in (2, 3, 5, 10)
        }, index=dates)
        for i, sym in enumerate(("AAPL", "NVDA"))
    }
    pred = {
        h: {
            pd.Timestamp(day): np.array([0.0, 0.0, 0.0])
            for day in dates[10:70]
        } for h in (2, 3, 5, 10)
    }
    cal = calibrate_asset_horizon_errors(
        forecasts=pred, frames=frames,
        symbols=["AAPL", "NVDA"],
        calibration_days=dates[10:70],
        calendar=dates,
        split_date=dates[55],
    )
    for h in (2, 3, 5, 10):
        assert cal[h]["effective_days"] == 45 - h
        assert cal[h]["center"][0] == pytest.approx(0.02)
        assert cal[h]["center"][1] == pytest.approx(0.03)
        assert cal[h]["pooled_error_std"] < 0.02


def test_v109_lambdarank_labels_must_be_mature_before_test():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        PAIR_TRAINING_FEATURES, fit_ranker_h05,
    )

    data = np.ones((50, 6, len(PAIR_TRAINING_FEATURES)), dtype=float)
    labels = np.tile(np.linspace(-0.1, 0.1, 6), (50, 1))
    cfg = SimpleNamespace(
        random_state=123,
        rotation_minimum_training_rows=10,
    )
    with pytest.raises(RuntimeError, match="chronological cutoff"):
        fit_ranker_h05(
            feature_cube=data, targets=labels,
            training_indices=np.arange(25),
            evaluation_start_index=28,
            config=cfg, fold_id=1,
        )


def test_v110_independent_math_rotation_respects_holding_and_switch_edge():
    from pesquisar_persistencia_horizontes_spyder import (
        AUTONOMOUS_POLICY_METHODS, mathematical_rotation_choice,
    )

    symbols = ["AAPL", "MSFT", "NVDA", "TSLA"]
    forecasts = np.array([
        [0.01, 0.01, 0.01, 0.01],
        [0.02, 0.03, 0.04, 0.04],
        [0.06, 0.08, 0.11, 0.19],
        [0.00, 0.03, 0.08, 0.20],
    ])
    opts = {
        "symbols": symbols,
        "predicted_returns": forecasts,
        "volatility": np.array([0.02, 0.02, 0.03, 0.05]),
        "switch_costs": np.array([0.0, 0.005, 0.005, 0.005]),
        "current_position": 1,
    }
    for method in AUTONOMOUS_POLICY_METHODS:
        pos, score, diag = mathematical_rotation_choice(
            method=method, holding_sessions=1, **opts,
        )
        assert pos == 1
        assert diag["reason"] == "MINIMUM_HOLD"
        assert diag["uses_control_action_as_feature"] is False
        assert score == 0.0
    target, _, why = mathematical_rotation_choice(
        method="net_gain", holding_sessions=2, **opts,
    )
    assert target == 3
    assert why["reason"] == "MATH_ROTATION"
    assert why["positive_horizons"] >= 3
    assert "control" not in why


def test_v110_net_gain_can_hold_and_exit_cash_without_control():
    from pesquisar_persistencia_horizontes_spyder import (
        mathematical_rotation_choice,
    )

    opts = dict(
        method="net_gain", symbols=["AAPL", "NVDA", "MSFT"],
        volatility=np.array([0.01, 0.02, 0.04]),
        switch_costs=np.array([0.0, 0.1, 0.1]),
        current_position=1, holding_sessions=3,
    )
    held = np.array([
        [0.02, 0.02, 0.02, 0.02],
        [0.03, 0.03, 0.03, 0.03],
        [0.02, 0.03, 0.04, 0.05],
    ])
    target, _, diag = mathematical_rotation_choice(
        predicted_returns=held, **opts,
    )
    assert target == 1
    assert diag["reason"] == "RANKED_INCUMBENT"

    negative = np.array([
        [-0.03, -0.03, -0.03, -0.03],
        [-0.07, -0.07, -0.07, -0.07],
        [-0.05, -0.05, -0.05, -0.05],
    ])
    target, _, diag = mathematical_rotation_choice(
        predicted_returns=negative, **opts,
    )
    assert target == 0
    assert diag["reason"] == "NEGATIVE_POSITION_EXPECTATION"


def test_v110_mathematical_rotation_switch_cost_uses_only_today():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        point_in_time_switch_cost,
    )

    dates = pd.date_range("2026-01-01", periods=5, freq="B", tz="UTC")
    frames = {
        sym: pd.DataFrame({"close": np.full(5, close)},
                          index=dates)
        for sym, close in [("AAPL", 100.0), ("NVDA", 50.0)]
    }
    config = SimpleNamespace(
        slippage_bps=4.0, commission_rate=0.0001,
        cat_fee_per_share=0.00003, sec_fee_rate=0.00002,
        taf_fee_per_share=0.0001,
    )
    before = point_in_time_switch_cost(
        day=dates[1], current_asset="AAPL", candidate="NVDA",
        frames=frames, config=config,
    )
    assert before > 0
    assert point_in_time_switch_cost(
        day=dates[1], current_asset="AAPL", candidate="AAPL",
        frames=frames, config=config,
    ) == 0
    frames["NVDA"].at[dates[2], "close"] = 1e10
    after = point_in_time_switch_cost(
        day=dates[1], current_asset="AAPL", candidate="NVDA",
        frames=frames, config=config,
    )
    assert before == after


def test_v110_realized_rotation_precision_only_uses_matured_fold_labels():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        autonomous_decision_quality,
    )

    dates = pd.date_range("2026-01-01", periods=12, freq="B", tz="UTC")
    frames = {
        sym: pd.DataFrame({
            "close": np.full(len(dates), 100.0),
            **{
                f"forward_horizon_net_return_pct_{h}":
                    np.full(len(dates), ret)
                for h in (2, 3, 5, 10)
            },
        }, index=dates)
        for sym, ret in (("AAPL", 0.01), ("NVDA", 0.04))
    }
    audit = [
        {
            "method": "net_gain",
            "fold_id": 1,
            "decision_date": day.isoformat(),
            "current_asset": "AAPL", "action_asset": "NVDA",
            "reason": "MATH_ROTATION",
        } for day in (dates[1], dates[6])
    ]
    cfg = SimpleNamespace(
        slippage_bps=0.0, commission_rate=0.0,
        cat_fee_per_share=0.0, sec_fee_rate=0.0,
        taf_fee_per_share=0.0,
    )
    quality, summary = autonomous_decision_quality(
        audit=audit, frames=frames, calendar=dates,
        folds=[{"fold_id": 1, "test_end_index": 8}],
        config=cfg, methods=("net_gain",),
    )
    scored_h2 = [r for r in quality if r["horizon"] == 2]
    assert len(scored_h2) == 1
    assert scored_h2[0]["decision_date"] == dates[1].isoformat()
    assert scored_h2[0]["realized_switch_vs_hold"] == pytest.approx(0.03)
    stats = next(row for row in summary if row["horizon"] == 2)
    assert stats["rotations_proposed"] == 2
    assert stats["matured_switches"] == 1
    assert stats["switch_precision"] == pytest.approx(1.0)


def test_v110_beep_distinguishes_success_error(monkeypatch):
    import sys
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import audible_completion

    emitted = []
    monkeypatch.setitem(
        sys.modules, "winsound",
        SimpleNamespace(Beep=lambda hz, duration: emitted.append((hz, duration))),
    )
    assert audible_completion(succeeded=True) == "windows_success_two_tones"
    assert emitted == [(880, 160), (1046, 220)]
    emitted.clear()
    assert audible_completion(succeeded=False) == "windows_error_two_tones"
    assert emitted == [(440, 420), (330, 470)]


def test_v111_ablation_factorial_config_preserves_control_config():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        utility_ablation_variant_config,
    )

    class Config(SimpleNamespace):
        def copiar_modelo(self, update):
            return Config(**{**vars(self), **update})

    official = Config(
        rotation_min_holding_days=2, rotation_switch_margin=0.0005
    )
    zero, zmargin = utility_ablation_variant_config(
        official, minimum_hold=0, margin_mode="zero",
        calibrated_margin=0.01,
    )
    base, bmargin = utility_ablation_variant_config(
        official, minimum_hold=0, margin_mode="base",
        calibrated_margin=0.01,
    )
    calibrated, cmargin = utility_ablation_variant_config(
        official, minimum_hold=2, margin_mode="calibrated",
        calibrated_margin=0.01,
    )
    assert (zero.rotation_min_holding_days, zero.rotation_switch_margin, zmargin) == (0, 0, 0)
    assert (base.rotation_min_holding_days, base.rotation_switch_margin, bmargin) == (0, 0.0005, 0.0005)
    assert (calibrated.rotation_min_holding_days, cmargin) == (2, 0.01)
    assert official.rotation_min_holding_days == 2
    assert official.rotation_switch_margin == 0.0005
    with pytest.raises(ValueError, match="Unsupported ablation"):
        utility_ablation_variant_config(
            official, minimum_hold=3, margin_mode="zero",
            calibrated_margin=0.01,
        )


def test_v111_same_frozen_utility_hold_and_margin_causes_direct_decision_change():
    from types import SimpleNamespace
    from pesquisar_persistencia_horizontes_spyder import (
        prepare_utility_ablation_policies,
    )

    class Config(SimpleNamespace):
        def copiar_modelo(self, update):
            return Config(**{**vars(self), **update})

    d = pd.Timestamp("2026-04-15", tz="UTC")
    cfg = Config(
        rotation_min_holding_days=2,
        rotation_switch_margin=0.0005,
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.0,
    )
    features = {d: np.array([0.0, 0.01, 0.03])}
    fold_cache = {1: {"margin": 0.025, "oos": features}}
    def build(hold, margin):
        return prepare_utility_ablation_policies(
            config=cfg, frames={}, symbols=["AAPL", "NVDA"],
            fold_caches=fold_cache, hold=hold, margin_mode=margin,
            decision_diagnostics=None,
        )[1]

    h2 = build(2, "base")
    h0 = build(0, "base")
    large = build(0, "calibrated")
    assert h2(d, 1, 1)[0] == 1
    assert h0(d, 1, 1)[0] == 2
    assert h2(d, 1, 2)[0] == 2
    assert large(d, 1, 2)[0] == 1
    # "Control" forneceu apenas o array de utilidades, nunca a acao.


def test_v111_paired_counterfactual_keeps_control_state_constant():
    from pesquisar_persistencia_horizontes_spyder import (
        one_step_utility_ablation,
    )

    date = pd.Timestamp("2026-01-05", tz="UTC")
    baseline = {date: {
        "current_asset": "AAPL",
        "holding_days_at_decision": 1,
        "final_action_asset": "AAPL",
    }}
    def stay(t, position, holding):
        assert (t, position, holding) == (date, 1, 1)
        return 1, 0.1
    def change(t, position, holding):
        assert (t, position, holding) == (date, 1, 1)
        return 2, 0.2
    observed = one_step_utility_ablation(
        baseline_decisions=baseline,
        policies={"control_like": {1: stay}, "no_hold": {1: change}},
        symbols=["AAPL", "NVDA"], date_to_fold={date: 1},
    )
    assert len(observed) == 2
    assert observed[0]["differs_from_control"] is False
    assert observed[1]["differs_from_control"] is True
    assert observed[0]["same_state_as_official_control"] is True
    assert observed[1]["holding_sessions"] == 1


def test_v111_zip_contains_success_sound_and_complete_stdout(tmp_path, monkeypatch):
    import zipfile
    import pesquisar_persistencia_horizontes_spyder as module

    monkeypatch.setattr(module, "OUT", tmp_path)
    (tmp_path / "experiment.json").write_text('{"status":"completed"}')
    (tmp_path / "execution_full.log").write_text(
        "[experiment] COMPLETED\n[audio] SUCCESS windows_success_two_tones\n"
    )
    zipped = module.finalize_experiment_package()
    assert zipped.name == "pacote_persistencia_horizontes_u77_v111.zip"
    with zipfile.ZipFile(zipped) as archive:
        assert archive.testzip() is None
        assert "experiment.json" in archive.namelist()
        assert "[audio] SUCCESS" in archive.read(
            "execution_full.log"
        ).decode("utf-8")
        assert zipped.name not in archive.namelist()
