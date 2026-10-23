"""CI v1.24.5: labels t+60 só são admissíveis depois de amadurecer."""
from __future__ import annotations

import pandas as pd
import pytest

from auditar_maturidade_labels_spyder import audit_maturity_for_phase
from engine.configuracao import CONFIG
from engine.rotacao import _construir_folds_walk_forward


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=240, tz="UTC")


def test_mature_label_last_training_row_before_evaluation() -> None:
    dates = _calendar()
    result = audit_maturity_for_phase(
        "AAA", dates, dates[:61],
        horizon=60,
        evaluation_start=dates[121],
        fold_id=1, phase="final_model_fit",
    )
    assert result["used_training_rows"] == 61
    assert result["labels_not_mature_before_evaluation"] == 0
    assert result["last_label_maturity"] == dates[120].isoformat()


def test_label_that_matures_on_evaluation_start_fails_audit() -> None:
    dates = _calendar()
    result = audit_maturity_for_phase(
        "AAA", dates, dates[:62], horizon=60,
        evaluation_start=dates[121], fold_id=1,
        phase="calibration_model_fit",
    )
    assert result["labels_not_mature_before_evaluation"] == 1
    assert result["first_late_sample"] == dates[61].isoformat()


def test_symbol_calendar_gaps_can_delay_real_label_maturity() -> None:
    market = _calendar()
    # Ativo perde uma sessão depois de amostra: maturidade de t+60
    # do ativo fica um pregão MAIS TARDE do que no calendário de mercado.
    asset = market.delete(50)
    expected_start = market[121]
    sample = pd.DatetimeIndex([market[60]])
    result = audit_maturity_for_phase(
        "AAA", asset, sample, horizon=60,
        evaluation_start=expected_start,
        fold_id=1, phase="final_model_fit",
    )
    # A leitura de t+60 usa 60 sessões efetivas DO ATIVO.
    pos = asset.get_loc(market[60])
    assert result["last_label_maturity"] == asset[pos+60].isoformat()
    assert result["labels_not_mature_before_evaluation"] == 0


def test_unknown_or_unmatured_label_is_rejected() -> None:
    dates = _calendar()
    with pytest.raises(RuntimeError, match="sem barra RAW"):
        audit_maturity_for_phase(
            "AAA", dates, pd.DatetimeIndex(["2018-01-01"], tz="UTC"),
            horizon=60, evaluation_start=dates[121],
            fold_id=1, phase="final_model_fit",
        )
    with pytest.raises(RuntimeError, match="incompleto"):
        audit_maturity_for_phase(
            "AAA", dates, dates[-10:],
            horizon=60, evaluation_start=dates[-1],
            fold_id=1, phase="final_model_fit",
        )


def test_fold_purge_exceeds_maximum_forward_label_horizon() -> None:
    dates = pd.bdate_range("2016-01-04", periods=2650, tz="UTC")
    folds = _construir_folds_walk_forward(dates, CONFIG)
    assert folds
    horizon = max(CONFIG.rotation_target_horizons)
    assert horizon == 60
    for fold in folds:
        assert fold["train_end_index"]-1+horizon < fold["calibration_start_index"]
        assert fold["final_fit_end_index"]-1+horizon < fold["test_start_index"]
        assert fold["calibration_end_index"]-1+horizon < fold["test_start_index"]
