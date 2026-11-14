"""Tests for causal first-day exit research, without historical ZIP or Alpaca."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pesquisar_permanencia_minima_spyder as research


def fixture_base(*, fold: int = 2, gap: float = 0.07, fall: float = -0.08):
    scores = np.array([
        [0.50, 0.40],
        [0.50 + fall, 0.50 + fall + gap],
        [0.50 + fall, 0.50 + fall + gap],
    ])
    dates = pd.date_range("2026-10-19", periods=3, freq="B", tz="UTC")
    config = {
        "rotation_min_holding_days": 2,
        "rotation_switch_margin": 0.0005,
        "rotation_cash_threshold": 0.0,
        "rotation_min_expected_edge": 0.001,
        "commission_rate": 0.0,
        "sec_fee_rate": 0.0000206,
        "taf_fee_per_share": 0.000195,
        "taf_fee_cap": 9.79,
        "cat_fee_per_share": 0.000003,
        "slippage_bps": 0.0,
        "initial_capital": 1000.0,
    }
    base = SimpleNamespace(
        dates=dates,
        exec_dates=dates + pd.offsets.BDay(1),
        fold=np.array([fold, fold, fold]),
        symbols=["AAA", "BBB"],
        scores=scores,
        cl=np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]),
        op=np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]),
        config=config,
        margens={1: 0.005, 2: 0.005, 3: 0.005},
    )
    forecast = np.array([
        [np.nan, np.nan],
        [0.002, 0.025],
        [0.002, 0.025],
    ])
    return base, forecast


def decide(base, forecast, policy, *, holding=1):
    return research._pa_decidir(
        base, 1, 0, holding, 1000.0, 100.0, forecast,
        politica=policy,
    )


def test_urgent_first_day_rotation_when_evidence_and_forecast_agree():
    b, forecast = fixture_base()
    target, row, _ = decide(b, forecast, "saida_urgente_com_vantagem_liquida")
    assert target == 1
    assert row["saida_antecipada"] is True
    assert row["idade_posicao"] == 1
    assert row["decisao_control_no_mesmo_estado"] == "AAA"
    assert row["vantagem_liquida_prevista_1d"] > 0


def test_negative_net_forecast_vetoes_economic_but_not_utility_control():
    b, forecast = fixture_base()
    forecast[1, :] = [0.03, 0.01]
    econ, row, _ = decide(b, forecast, "saida_urgente_com_vantagem_liquida")
    utility, _, _ = decide(b, forecast, "saida_urgente_utilidade")
    assert econ == 0
    assert row["motivo"] == "VANTAGEM_ECONOMICA_LIQUIDA_NAO_POSITIVA"
    assert utility == 1


def test_relaxed_policy_requires_decline_and_economic_advantage():
    b, forecast = fixture_base(gap=0.02, fall=-0.01)
    urgent, _, _ = decide(b, forecast, "saida_urgente_com_vantagem_liquida")
    relaxed, row, _ = decide(b, forecast, "saida_economica_com_deterioracao")
    assert urgent == 0
    assert relaxed == 1
    assert row["gap_utilidade"] >= row["margem_calibrada"]


def test_no_decline_does_not_release_minimum_hold_even_if_model_is_positive():
    b, forecast = fixture_base(fall=0.01)
    for policy in research.PA_POLITICAS:
        target, row, _ = decide(b, forecast, policy)
        assert target == 0
        assert row["saida_antecipada"] is False


def test_fold_one_and_mismatch_fold_preserve_control():
    b, forecast = fixture_base(fold=1)
    for policy in research.PA_POLITICAS:
        target, _, _ = decide(b, forecast, policy)
        assert target == 0
    b.fold[0] = 1
    b.fold[1] = 2
    target, row, _ = decide(
        b, forecast, "saida_economica_com_deterioracao",
    )
    assert target == 0
    assert row["motivo"] == "FRONTEIRA_DE_FOLD"


def test_after_two_sessions_control_may_rotate_normally():
    b, forecast = fixture_base()
    target, row, _ = decide(
        b, forecast, "saida_urgente_com_vantagem_liquida", holding=2,
    )
    assert target == 1
    assert row["saida_antecipada"] is False


def test_future_execution_prices_cannot_influence_first_day_decision():
    b, forecast = fixture_base()
    first = decide(b, forecast, "saida_urgente_com_vantagem_liquida")
    b.op[2, :] = [1_000_000.0, 0.01]
    b.cl[2, :] = [1_000_000.0, 0.01]
    second = decide(b, forecast, "saida_urgente_com_vantagem_liquida")
    assert first == second


def test_invalid_forecast_must_not_trigger_economic_exit():
    b, forecast = fixture_base()
    forecast[1, :] = np.nan
    for policy in research.PA_POLITICAS[1:]:
        target, row, _ = decide(b, forecast, policy)
        assert target == 0
        assert row["saida_antecipada"] is False


def test_annual_compounding_uses_previous_calendar_year_close():
    b, _ = fixture_base()
    b.exec_dates = pd.DatetimeIndex(pd.to_datetime([
        "2025-12-30", "2025-12-31", "2026-01-02",
    ], utc=True))
    row = research._pa_desempenho_anual(
        b, {"policy": np.array([1100.0, 1210.0, 1270.5])},
    )
    assert row["ano"].tolist() == [2025, 2026]
    assert row["capital_inicial"].tolist() == [1000.0, 1210.0]
    assert row["capital_final"].tolist() == [1210.0, 1270.5]
    assert np.isclose(float(row.iloc[0]["retorno_ano_pct"]), 21)


def test_research_uses_existing_single_file_and_same_zip_not_live():
    source = (Path(__file__).resolve().parents[1]
              / "pesquisar_permanencia_minima_spyder.py").read_text(
                  encoding="utf-8")
    assert 'MODO = "saida_antecipada_economica"' in source
    assert 'DESTINO = ROOT / "output" / "pesquisa_vantagem_rotacao"' in source
    assert "_pi_localizar_base()" in source
    assert '"codigo_executado.py"' in source
    assert "reproducao_control_desvio_maximo_usd" in source
    assert "execute_shared_winner_for_oauth_account" not in source
