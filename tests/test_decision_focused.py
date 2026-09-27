"""Testes unitarios da pesquisa; dispensam Alpaca e o snapshot completo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from reproducao.decision_focused import (
    ATRIBUTOS_ATIVO, FEATURE_COLUMNS, _atributos, _candidatos,
    _escolher_acao, _gerar_rotulos, _pares_regret, _replay_emparelhado,
)


def _mercado(n: int = 35):
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for symbol, scale in (("AAA", 1.02), ("BBB", 0.99)):
        close = np.array([100.0 * scale**i for i in range(n)])
        frame = pd.DataFrame({"open": close, "close": close}, index=dates)
        for feature in ATRIBUTOS_ATIVO:
            frame[feature] = 0.0
        frames[symbol] = frame
    return frames, ["AAA", "BBB"], dates


def _policy(now, position, holding):
    return 1, 0.5


def test_same_forced_action_has_identical_paired_replay():
    frames, symbols, dates = _mercado()
    values = [
        _replay_emparelhado(_policy, frames, symbols, dates, 2, 10, 1, 3, 1, CONFIG)
        for _ in range(2)
    ]
    assert values[0] == values[1]


def test_forced_investment_changes_wealth_with_same_continuation():
    frames, symbols, dates = _mercado()
    cash = _replay_emparelhado(
        lambda *_: (0, 0.0), frames, symbols, dates,
        0, 10, 0, 0, 0, CONFIG,
    )
    invested = _replay_emparelhado(
        lambda *_: (0, 0.0), frames, symbols, dates,
        0, 10, 0, 0, 1, CONFIG,
    )
    assert cash == pytest.approx(0.0)
    assert invested > cash


def test_candidates_preserve_control_cash_and_incumbent():
    utilities = np.array([0.0, 0.30, 0.25, 0.20])
    found = _candidatos(utilities, 2, 1, 4, CONFIG, top=1)
    assert found == [1, 2, 0]
    assert _candidatos(utilities, 2, 1, 1, CONFIG) == [2]


def test_features_use_only_current_observation():
    frames, symbols, dates = _mercado()
    util = np.array([0.0, 0.3, 0.2])
    row = _atributos(frames, symbols, dates[3], 1, 1, 2, 2, util, CONFIG)
    assert set(row) == set(FEATURE_COLUMNS)
    assert row["candidate_utility"] == pytest.approx(0.2)
    assert row["edge_vs_control"] == pytest.approx(-0.1)


def test_labels_end_inside_calibration_and_control_is_zero():
    frames, symbols, dates = _mercado()
    cache = {stamp: np.array([0.0, 0.3, 0.2]) for stamp in dates}
    data = _gerar_rotulos(
        frames, symbols, dates, _policy, cache, CONFIG, fold_id=1, horizonte=5,
    )
    assert not data.empty
    assert data["outcome_end"].max() <= dates[-1]
    assert data.loc[
        data["candidate_position"] == data["control_position"],
        "log_advantage",
    ].abs().max() == pytest.approx(0.0)
    assert data.groupby("decision_date").size().min() >= 2


def test_regret_pairwise_training_set_is_symmetric():
    frame = pd.DataFrame({
        "fold_id": [1, 1],
        "decision_date": [pd.Timestamp("2020-01-01", tz="UTC")] * 2,
        "log_advantage": [0.0, 0.1],
        **{col: [0.0, 1.0] for col in FEATURE_COLUMNS},
    })
    features, labels, weights = _pares_regret(frame)
    assert len(features) == 2
    assert list(labels) == [0, 1]
    assert weights[0] == weights[1]
    np.testing.assert_allclose(features.iloc[0], -features.iloc[1])


def test_dfl_selection_uses_pairwise_probability_without_target():
    class Model:
        def predict_proba(self, values):
            z = values["candidate_utility"].to_numpy(dtype=float)
            p = 1 / (1 + np.exp(-z))
            return np.column_stack((1 - p, p))

    x = pd.DataFrame({
        "candidate_position": [1, 2],
        **{col: [0.0, 0.0] for col in FEATURE_COLUMNS},
    })
    x["candidate_utility"] = [0.0, 1.0]
    choice, _ = _escolher_acao("DFL", Model(), x, control=1)
    assert choice == 2


def test_invalid_horizon_must_be_rejected_before_training():
    from reproducao.decision_focused import executar_pesquisa
    with pytest.raises(ValueError, match="purge"):
        executar_pesquisa({}, CONFIG, horizonte=61)
