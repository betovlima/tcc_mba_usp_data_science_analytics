"""[TCC-DFL:FIX-005] Testes do auditor de suporte, sem treino ou internet."""
from __future__ import annotations

import pandas as pd
import pytest

from reproducao.diagnostico_estado import (
    diagnosticar_cobertura_estados, recuperar_holding_regressao,
)


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


def _replay_case():
    dates = pd.date_range("2025-01-02", periods=5, freq="B", tz="UTC")
    predictions = pd.DataFrame({
        "timestamp": dates[1:],
        "decision_date": dates[:-1],
        "previous_asset": ["CASH", "A", "A", "B"],
        "selected_asset": ["A", "A", "B", "B"],
    }).set_index("timestamp")
    decisions = pd.DataFrame({
        "decision_date": dates[:-1],
        "current_asset": ["CASH", "A", "A", "B"],
        "decision_fold_id": [1, 1, 1, 1],
    })
    return predictions, decisions


def test_recover_holding_from_real_replay_when_spyder_import_is_stale():
    predictions, decisions = _replay_case()
    restored = recuperar_holding_regressao(decisions, predictions)
    assert restored["research_holding_days_at_decision"].tolist() == [0, 1, 2, 1]
    assert restored["current_asset"].tolist() == ["CASH", "A", "A", "B"]


def test_replay_holding_verifies_original_diagnostic_when_present():
    predictions, decisions = _replay_case()
    decisions["research_holding_days_at_decision"] = [0, 1, 2, 1]
    restored = recuperar_holding_regressao(decisions, predictions)
    assert restored["research_holding_days_at_decision"].tolist() == [0, 1, 2, 1]
    decisions.loc[2, "research_holding_days_at_decision"] = 3
    with pytest.raises(AssertionError, match="Holding original diverge"):
        recuperar_holding_regressao(decisions, predictions)


def test_replay_refuses_wrong_state_and_wrong_dates():
    predictions, decisions = _replay_case()
    decisions.loc[0, "current_asset"] = "A"
    with pytest.raises(AssertionError, match="Ativo incumbente"):
        recuperar_holding_regressao(decisions, predictions)
    _, decisions = _replay_case()
    decisions.loc[0, "decision_date"] = pd.Timestamp("2025-02-01", tz="UTC")
    with pytest.raises(AssertionError, match="datas distintas"):
        recuperar_holding_regressao(decisions, predictions)


def test_replay_refuses_discontinuous_positions_and_partial_holding():
    predictions, decisions = _replay_case()
    broken = predictions.copy()
    broken.loc[broken.index[1], "previous_asset"] = "B"
    with pytest.raises(AssertionError, match="descontinuo"):
        recuperar_holding_regressao(decisions, broken)
    decisions["research_holding_days_at_decision"] = [0, 1, None, 1]
    with pytest.raises(ValueError, match="parcialmente ausente"):
        recuperar_holding_regressao(decisions, predictions)


def test_spyder_entrypoint_explicitly_reload_research_modules():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "pesquisar_decision_focused_spyder.py"
    ).read_text(encoding="utf-8")
    assert "importlib.reload(_decision_focused_v2)" in source
    assert "importlib.reload(_diagnostico_estado)" in source
    assert "recuperar_holding_regressao(" in source
    assert "pesquisa.regression_result.predictions," in source
