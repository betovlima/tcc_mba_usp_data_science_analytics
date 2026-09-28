"""[TCC-DFL:FIX-005] Testes do auditor de suporte, sem treino ou internet."""
from __future__ import annotations

import pandas as pd
import pytest

from reproducao.diagnostico_estado import diagnosticar_cobertura_estados


def _frames():
    dates = pd.date_range("2020-01-01", periods=9, freq="B", tz="UTC")
    records = []
    # 2 datas de treino, 2 estados; 1 label de validacao excluido.
    for day, outcome, asset, hold in (
        (dates[0], dates[2], "A", 0),
        (dates[1], dates[3], "A", 1),
        (dates[3], dates[5], "B", 2),
    ):
        for choice in (0, 1):
            records.append({
                "fold_id": 1,
                "state_source": "OOF_CONTROL",
                "decision_date": day,
                "outcome_end": outcome,
                "incumbent": asset,
                "holding_days": hold,
                "candidate_position": choice,
                "control_position": 0,
                "log_advantage": 0 if choice == 0 else 0.01,
            })
    original = pd.DataFrame(records)
    calibration = pd.DataFrame({
        "fold_id": [1],
        "validation_start": [dates[4]],
    })
    control = pd.DataFrame({
        "walk_forward_fold": [1, 1, 1],
        "decision_date": dates[6:9],
        "current_asset": ["A", "A", "B"],
        "holding_days_at_decision": [1, 5, 2],
    })
    regression = pd.DataFrame({
        "decision_fold_id": [1, 1, 1],
        "decision_date": dates[6:9],
        "current_asset": ["A", "A", "B"],
        "research_holding_days_at_decision": [1, 5, 2],
    })
    return original, calibration, control, regression


def test_partial_state_coverage_excludes_unfinished_labels():
    labels, calibration, control, regression = _frames()
    result = diagnosticar_cobertura_estados(
        labels, calibration, control, regression,
    )
    fold = result.folds.loc[result.folds.policy == "CONTROL"].iloc[0]
    # Row at dates[1] ends dates[3] < dates[4], row dates[3]
    # ends dates[5] and cannot enter training.
    assert fold.training_rows == 4
    assert fold.training_unique_dates == 2
    assert fold.training_unique_state_groups == 2
    assert fold.training_unique_incumbents == 1
    assert fold.oos_unseen_incumbent_decisions == 1
    assert fold.oos_unseen_pair_decisions == 2
    assert fold.oos_seen_asset_unseen_hold_decisions == 1
    assert fold.oos_seen_learned_pair_decisions == 0
    assert result.checks["no_model_training"]
    assert len(result.sessoes) == 6


def test_missing_learner_hold_is_not_filled_from_control():
    labels, calibration, control, regression = _frames()
    regression = regression.drop(columns=["research_holding_days_at_decision"])
    with pytest.raises(ValueError, match="Regressao"):
        diagnosticar_cobertura_estados(labels, calibration, control, regression)


def test_different_test_calendars_fail_closed():
    labels, calibration, control, regression = _frames()
    regression.loc[1, "decision_date"] = pd.Timestamp("2020-02-01", tz="UTC")
    with pytest.raises(AssertionError, match="calendarios OOS"):
        diagnosticar_cobertura_estados(labels, calibration, control, regression)


def test_baseline_advantage_must_be_zero():
    labels, calibration, control, regression = _frames()
    labels.loc[labels.candidate_position == 0, "log_advantage"] = 0.02
    with pytest.raises(AssertionError, match="vantagem zero"):
        diagnosticar_cobertura_estados(labels, calibration, control, regression)
