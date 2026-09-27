"""[TCC-DFL:FIX-004] Atribuicao por replays exatos e sem novo treinamento."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import reproducao.auditoria_intervencoes as module


def _synthetic_case(monkeypatch):
    dates = pd.date_range("2023-01-02", periods=15, freq="B", tz="UTC")
    actions = {dates[2], dates[6], dates[7]}
    symbols = ["A", "B"]
    baseline = lambda *_: (1, 0.5)
    learned = lambda day, *_: (2, 0.7) if day in actions else (1, 0.5)

    def fake_sim(backend, policy, frames, symbols, decision_dates, config,
                 fee_calculator, slippage, **kwargs):
        capital, current, holding, rotations = 100.0, 0, 0, 0
        rows = []
        for now, execution in zip(decision_dates[:-1], decision_dates[1:]):
            position, _ = policy(now, current, holding)
            if current > 0 and position != current:
                rotations += 1
            capital *= (1.02 if position == 1 else 0.97)
            holding = holding + 1 if position == current else 1
            current = position
            rows.append({
                "timestamp": execution,
                "strategy_equity": capital,
                "selected_asset": "A" if position == 1 else "B",
                "trade_action": "HOLD" if position == 1 else "BUY",
                "walk_forward_fold": 1,
            })
        frame = pd.DataFrame(rows).set_index("timestamp")
        frame.index.name = "timestamp"
        return SimpleNamespace(
            predictions=frame,
            trades=pd.DataFrame(columns=["timestamp", "action"]),
            metrics={
                "strategy_ending_capital": capital,
                "total_transaction_fees": 0.0,
                "capital_rotations": rotations,
                "cash_days": 0,
            },
        )

    original_diags = {}
    holding, position = 0, 0
    for date in dates[:-1]:
        selected, _ = learned(date, position, holding)
        original_diags[date] = {
            "research_changed_base_action": date in actions,
            "final_action_asset": "A" if selected == 1 else "B",
            "research_base_action": "A",
            "current_asset": "CASH" if position == 0 else symbols[position - 1],
            "decision_fold_id": 1,
            "research_model_score": 0.8 if date in actions else 0.0,
        }
        holding = holding + 1 if selected == position else 1
        position = selected
    monkeypatch.setattr(module, "_simular_exato", fake_sim)
    original = fake_sim("reg", learned, {}, symbols, dates, None, None, None)
    control = fake_sim("control", baseline, {}, symbols, dates, None, None, None)
    return dates, symbols, baseline, learned, original_diags, original, control


def test_consecutive_sessions_are_one_episode_even_across_weekend(monkeypatch):
    dates, symbols, base, learned, diags, original, control = _synthetic_case(
        monkeypatch,
    )
    episodes = module._episodios(dates, diags)
    assert len(episodes) == 2
    assert [len(item["dates"]) for item in episodes] == [1, 2]
    assert episodes[0]["start"] == dates[2]
    assert episodes[1]["start"] == dates[6]


def test_cumulative_bridge_exactly_reconciles_original_path(monkeypatch):
    dates, symbols, base, learned, diags, original, control = _synthetic_case(
        monkeypatch,
    )
    audit = module.auditar_intervencoes(
        frames={}, symbols=symbols, decision_dates=dates, config=None,
        baseline_policy=base, learned_policy=learned,
        original_decisions=diags, original_regression=original,
        official_control=control, horizon=3,
    )
    assert audit.checks["episodes"] == 2
    assert audit.checks["override_sessions"] == 3
    assert audit.checks["baseline_vs_official_control_max_abs"] == 0
    assert audit.checks["final_bridge_vs_regression_max_abs"] == 0
    assert audit.checks["marginal_reconciliation_abs_error"] < 1e-8
    assert audit.episodios["delta_final_usd"].sum() == pytest.approx(
        original.metrics["strategy_ending_capital"] -
        control.metrics["strategy_ending_capital"]
    )
    assert set(audit.curvas["scenario"]) == {"BASELINE", "EP001", "EP002"}
    assert audit.episodios["horizon_sessions"].tolist() == [3, 3]


def test_bridge_refuses_wrong_original_asset_or_state(monkeypatch):
    dates, symbols, base, learned, diags, original, control = _synthetic_case(
        monkeypatch,
    )
    diags[dates[2]]["final_action_asset"] = "A"
    with pytest.raises(AssertionError, match="Override inconsistente"):
        module.auditar_intervencoes(
            frames={}, symbols=symbols, decision_dates=dates, config=None,
            baseline_policy=base, learned_policy=learned,
            original_decisions=diags, original_regression=original, horizon=3,
        )


def test_equity_match_alone_not_sufficient(monkeypatch):
    dates, symbols, base, learned, diags, original, control = _synthetic_case(
        monkeypatch,
    )
    copied = SimpleNamespace(predictions=control.predictions.copy())
    copied.predictions.iloc[0, copied.predictions.columns.get_loc(
        "selected_asset"
    )] = "B"
    with pytest.raises(AssertionError, match="ativos selecionados"):
        module._validar_equivalencia(control, copied, context="test")


def test_no_overrides_cannot_produce_fake_attribution(monkeypatch):
    dates, symbols, base, learned, diags, original, control = _synthetic_case(
        monkeypatch,
    )
    for entry in diags.values():
        entry["research_changed_base_action"] = False
    with pytest.raises(ValueError, match="nenhum episodio"):
        module.auditar_intervencoes(
            frames={}, symbols=symbols, decision_dates=dates, config=None,
            baseline_policy=base, learned_policy=learned,
            original_decisions=diags, original_regression=original, horizon=3,
        )
