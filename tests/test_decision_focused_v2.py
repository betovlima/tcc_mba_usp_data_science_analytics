"""Testes independentes de rede para o protocolo v2 do TCC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from reproducao.decision_focused import ATRIBUTOS_ATIVO, FEATURE_COLUMNS, _gerar_rotulos
from reproducao.decision_focused_v2 import (
    RESEARCH_VERSION, SegmentoOOF, _aumentar_estados, _calibrar_guardas,
    _criar_politica_v2, _divisao_temporal, _estados_da_politica_aprendida,
    _gerar_oof, _parametros_internos,
)


def _frames(n=100):
    dates = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for symbol in ["A", "B"]:
        values = np.arange(1, n + 1, dtype=float)
        frame = pd.DataFrame({
            "open": values + 10, "close": values + 10,
        }, index=dates)
        for feature in ATRIBUTOS_ATIVO:
            frame[feature] = 0.0
        frames[symbol] = frame
    return frames, ["A", "B"], dates


def _sample_labels(n=100, horizon=5):
    dates = pd.date_range("2020-01-01", periods=n + horizon, freq="B", tz="UTC")
    rows = []
    for i in range(n):
        for action in [0, 1]:
            rows.append({
                "fold_id": 1, "decision_date": dates[i],
                "outcome_end": dates[i + horizon],
                "candidate_position": action, "control_position": 0,
                "log_advantage": 0.0 if action == 0 else (-0.05 if i % 2 else 0.02),
                "state_source": "OOF_CONTROL",
                **{k: (float(action) if k == "candidate_utility" else 0.0)
                   for k in FEATURE_COLUMNS},
            })
    return pd.DataFrame(rows)


class _Regression:
    def predict(self, x):
        return x["candidate_utility"].to_numpy(dtype=float)


class _Pairwise:
    def predict_proba(self, x):
        z = x["candidate_utility"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack((1.0 - p, p))


def test_version_and_inner_minimum_is_not_official_minimum():
    assert RESEARCH_VERSION == "1.3.0-dev.3"
    assert CONFIG.rotation_minimum_training_rows == 700
    assert _parametros_internos(CONFIG).rotation_minimum_training_rows == 250
    assert CONFIG.rotation_minimum_training_rows == 700


def test_temporal_split_purges_overlapping_outcomes():
    fit, validation, start = _divisao_temporal(_sample_labels())
    assert len(fit) > 0
    assert validation["decision_date"].nunique() >= 20
    assert fit["outcome_end"].max() < start
    assert fit["decision_date"].max() < validation["decision_date"].min()
    assert validation["decision_date"].min() == start


def test_missing_temporal_history_fails_closed():
    with pytest.raises(ValueError, match="OOF insuficiente"):
        _divisao_temporal(_sample_labels(n=30))


def test_regret_calibration_uses_only_held_out_rows():
    _, validation, _ = _divisao_temporal(_sample_labels())
    guard = _calibrar_guardas(_Regression(), _Pairwise(), _Regression(), validation)
    assert guard["validation_decision_dates"] == validation["decision_date"].nunique()
    assert guard["regression_optimism_margin"] > 0
    assert guard["dfl_unsafe_probability_threshold"] >= 0.5
    assert guard["softmax_unsafe_margin_threshold"] >= 0.0
    assert guard["calibration_role"] == "prior_temporal_oof_only"


@pytest.mark.parametrize("variant", ["REGRESSION", "DFL", "SOFTMAX_DFL"])
def test_guard_reverts_to_control_and_cache_is_fold_local(variant):
    frames, symbols, dates = _frames()
    diagnostics = {}
    model = _Pairwise() if variant == "DFL" else _Regression()
    day = dates[10]
    cache = {day: np.array([0.0, 0.10, 0.80])}
    closed = _criar_politica_v2(
        fold_id=1, variant=variant, model=model,
        baseline=lambda *_: (1, 0.1), utility_cache=cache,
        diagnostics=diagnostics,
        calibration={
            "regression_optimism_margin": 1.0,
            "dfl_unsafe_probability_threshold": 1.0,
            "softmax_unsafe_margin_threshold": float("inf"),
        },
        frames=frames, symbols=symbols, config=CONFIG,
    )
    action, _ = closed(day, 1, 3)
    assert action == 1
    assert not diagnostics[day]["research_guard_accepted"]
    with pytest.raises(KeyError, match="fold=1"):
        closed(dates[11], 1, 3)
    opened = _criar_politica_v2(
        fold_id=2, variant=variant, model=model,
        baseline=lambda *_: (1, 0.1), utility_cache=cache,
        diagnostics={},
        calibration={
            "regression_optimism_margin": 0.1,
            "dfl_unsafe_probability_threshold": 0.55,
            "softmax_unsafe_margin_threshold": 0.1,
        },
        frames=frames, symbols=symbols, config=CONFIG,
    )
    assert opened(day, 1, 3)[0] == 2


def test_learned_state_rollout_changes_state_without_future_labels():
    frames, symbols, dates = _frames()
    segment = SegmentoOOF(
        fold_id=1, train_end=dates[0], dates=dates[:10],
        policy=lambda *_: (1, 0.1),
        utility_cache={d: np.array([0.0, 0.10, 0.80]) for d in dates[:10]},
    )
    states = _estados_da_politica_aprendida(
        _Regression(), segment, frames, symbols, CONFIG,
    )
    assert len(states) == len(segment.dates) - 1
    assert states[0][0] == 0
    assert any(pos == 2 for pos, _, _ in states[2:])
    labels = _gerar_rotulos(
        frames, symbols, segment.dates, segment.policy,
        segment.utility_cache, CONFIG, 1, 3,
        states_override=states, state_source="OOF_LEARNED_STATE",
    )
    assert (labels["state_source"] == "OOF_LEARNED_STATE").all()
    assert labels["outcome_end"].max() <= segment.dates[-1]


def test_on_policy_augmentation_never_touches_calibration():
    frames, symbols, dates = _frames()
    base = pd.DataFrame([
        {"decision_date": dates[i], "outcome_end": dates[i+3],
         "log_advantage": float((i % 3) - 1) / 100,
         "fold_id": 1, "candidate_position": action,
         "control_position": 1, "state_source": "OOF_CONTROL",
         **{k: float(action) if k == "candidate_utility" else 0.0
            for k in FEATURE_COLUMNS}}
        for i in range(80) for action in (0, 1, 2)
    ])
    # O teste e uma garantia de corte do conjunto interno, nao do OOS real.
    segment = SegmentoOOF(
        1, dates[0], dates[:83], lambda *_: (1, 0.1),
        {d: np.array([0.0, 0.10, 0.80]) for d in dates[:83]},
    )
    augmented, audit = _aumentar_estados(
        base, [segment], frames, symbols, CONFIG, 3,
    )
    assert len(augmented) >= len(base)
    extras = augmented.loc[augmented["state_source"] == "OOF_LEARNED_STATE"]
    assert extras["outcome_end"].max() <= base["decision_date"].max()
    assert audit["augmented_rows"] > 0


def test_paired_labels_refuse_state_length_mismatch():
    frames, symbols, dates = _frames(n=50)
    cache = {d: np.array([0.0, 0.3, 0.2]) for d in dates}
    with pytest.raises(ValueError, match="Estados"):
        _gerar_rotulos(
            frames, symbols, dates, lambda *_: (1, 0.3), cache,
            CONFIG, 1, 3, states_override=[(0, 0, 1)],
        )


def test_empty_outer_training_history_is_rejected():
    frames, symbols, dates = _frames()
    with pytest.raises(ValueError, match="nenhuma janela OOF"):
        _gerar_oof(
            frames, symbols, dates,
            {"fold_id": 1, "final_fit_end_index": len(dates),
             "test_start": dates[-1]},
            CONFIG, 20,
        )


def test_softmax_regret_gradient_matches_numerical_direction():
    from reproducao.dfl_softmax import gradiente_regret_softmax

    rewards = np.asarray([0.0, 0.04, -0.03], dtype=float)
    pred = np.asarray([0.1, 0.0, -0.1], dtype=float)
    grad, hess = gradiente_regret_softmax(
        pred, rewards, [(0, 3)], scale=1.0,
    )
    assert (hess > 0).all()
    assert grad[1] < 0  # Elevar score da melhor acao reduz perda.
    def loss(z):
        p = np.exp(z - max(z))
        p /= p.sum()
        return float(np.max(rewards) - np.dot(p, rewards))
    eps = 1e-5
    numeric = np.array([
        (loss(pred + eps * np.eye(3)[i]) -
         loss(pred - eps * np.eye(3)[i])) / (2 * eps)
        for i in range(3)
    ])
    np.testing.assert_allclose(grad, numeric, atol=1e-7)
    assert abs(grad.sum()) < 1e-10


def test_softmax_model_learns_ranking_without_label_at_inference():
    from reproducao.dfl_softmax import escolher_softmax, treinar_softmax_dfl

    rows = []
    dates = pd.date_range("2020-01-01", periods=35, freq="B", tz="UTC")
    for i, d in enumerate(dates):
        for candidate in (0, 1, 2):
            advantage = [0.0, 0.05, -0.05][candidate]
            rows.append({
                "fold_id": 1, "decision_date": d,
                "candidate_position": candidate,
                "control_position": 0,
                "log_advantage": advantage,
                **{k: float(candidate == 1) if k == "candidate_utility" else 0.0
                   for k in FEATURE_COLUMNS},
            })
    rows_df = pd.DataFrame(rows)
    model = treinar_softmax_dfl(rows_df)
    choice, delta = escolher_softmax(model, rows_df.iloc[:3], 0)
    assert choice == 1
    assert delta > 0
