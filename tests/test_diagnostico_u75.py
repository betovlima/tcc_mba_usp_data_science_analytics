"""Testes para diagnostico observavel do U75 e mapa mensal de ativos."""
from __future__ import annotations

import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from pesquisa.diagnostico_u75 import (
    StressSettings, calcular_indicadores_u75, juntar_decisoes,
    extrair_episodios, posicoes_mensais, serializar_mapa_html,
    construir_politica_protecao_diaria,
)
from pesquisa.pacote_visual_u75 import gerar_zip_visual


def _universe(*, one_rising=False, days=110):
    dates = pd.date_range("2025-01-02", periods=days, freq="B", tz="UTC")
    frames = {}
    symbols = [f"S{i:02d}" for i in range(75)]
    for i, symbol in enumerate(symbols):
        increasing = i == 0 and one_rising
        factor = 1.003 if increasing else 0.997
        prices = 100.0 * factor ** np.arange(days)
        frames[symbol] = pd.DataFrame({
            "close": prices, "open": prices,
        }, index=dates)
    return frames, symbols, dates


def test_global_decline_and_no_technical_candidate():
    frames, symbols, dates = _universe()
    daily, per_asset = calcular_indicadores_u75(
        frames, symbols, dates[-20:],
    )
    assert len(daily) == 20
    assert len(per_asset) == 75 * 20
    assert daily["valid_assets"].eq(75).all()
    assert daily["positive_share_5"].eq(0).all()
    assert daily["positive_share_20"].eq(0).all()
    assert daily["below_ma50_share"].eq(1).all()
    assert daily["technical_candidates_count"].eq(0).all()
    assert daily["stress_without_candidate"].all()


def test_one_resilient_asset_prevents_no_candidate_flag():
    frames, symbols, dates = _universe(one_rising=True)
    daily, _ = calcular_indicadores_u75(
        frames, symbols, dates[-3:],
    )
    assert daily["universe_stress"].all()
    assert daily["technical_candidates_count"].eq(1).all()
    assert not daily["no_technical_candidate"].any()
    assert not daily["stress_without_candidate"].any()


def test_changing_future_prices_does_not_change_past_signal():
    frames, symbols, dates = _universe()
    observed = dates[80:90]
    before, _ = calcular_indicadores_u75(frames, symbols, observed)
    altered = {symbol: x.copy() for symbol, x in frames.items()}
    for symbol in symbols:
        altered[symbol].loc[dates[90]:, "close"] *= 1000
    after, _ = calcular_indicadores_u75(altered, symbols, observed)
    pd.testing.assert_frame_equal(before, after)


def test_coverage_unknown_is_not_interpreted_as_stress():
    frames, symbols, dates = _universe()
    for symbol in symbols[60:]:
        frames[symbol] = frames[symbol].iloc[:5].copy()
    daily, _ = calcular_indicadores_u75(
        frames, symbols, dates[-2:],
    )
    assert daily["valid_assets"].eq(60).all()
    assert not daily["coverage_ok"].any()
    assert not daily["universe_stress"].any()
    assert not daily["stress_without_candidate"].any()


def test_join_uses_previous_close_for_actual_next_session_position():
    indicators = pd.DataFrame({
        "decision_date": pd.to_datetime(["2025-03-03"], utc=True),
        "valid_assets": [75], "universe_stress": [True],
        "stress_without_candidate": [True],
        "coverage_ok": [True],
    })
    predictions = pd.DataFrame({
        "decision_date": pd.to_datetime(["2025-03-03"], utc=True),
        "timestamp": pd.to_datetime(["2025-03-04"], utc=True),
        "selected_asset": ["AAA"],
        "strategy_equity": [10050.0],
    })
    joined = juntar_decisoes(indicators, predictions)
    assert bool(joined.iloc[0]["invested_during_stress"])
    assert bool(joined.iloc[0]["invested_without_candidate"])
    assert joined.iloc[0]["timestamp"] > joined.iloc[0]["decision_date"]


def test_stress_episode_horizons_do_not_overlap():
    dt = pd.date_range("2025-01-02", periods=40, freq="B", tz="UTC")
    flags = np.zeros(40, dtype=bool)
    flags[2:5] = True
    flags[9:12] = True
    flags[25:29] = True
    joined = pd.DataFrame({
        "decision_date": dt, "invested_without_candidate": flags,
    })
    episodes = extrair_episodios(
        joined, settings=StressSettings(audit_sessions=10),
    )
    assert episodes["start_index"].tolist() == [2, 9, 25]
    assert episodes["audit_eligible"].tolist() == [True, False, True]


def test_cash_reenters_after_one_session_when_universe_recovers():
    dates = pd.date_range("2025-01-02", periods=18, freq="B", tz="UTC")
    signals = pd.DataFrame({
        "decision_date": dates,
        "stress_without_candidate": [False, False, True, False, True, True, False]
                                    + [False] * 11,
    })
    states = []
    def base(_dt, position, holding):
        states.append((position, holding))
        return (1 if position == 0 else 2, 0.5)
    wrapper = construir_politica_protecao_diaria(base, signals)
    assert wrapper(dates[2], 1, 3)[0] == 0
    # A condicao some no fechamento seguinte; recompra na execucao t+1.
    assert wrapper(dates[3], 0, 0)[0] == 1
    assert wrapper(dates[4], 1, 1)[0] == 0
    assert wrapper(dates[5], 0, 0)[0] == 0
    assert wrapper(dates[6], 0, 0)[0] == 1
    assert states == [(0, 0), (0, 0)]


def test_cash_can_last_longer_than_five_sessions_without_maximum():
    dates = pd.date_range("2025-01-02", periods=20, freq="B", tz="UTC")
    signals = pd.DataFrame({
        "decision_date": dates,
        "stress_without_candidate": [False] * 2 + [True] * 9 + [False] * 9,
    })
    def base(_dt, position, _holding):
        return (1 if position == 0 else 2, 0.5)
    wrapper = construir_politica_protecao_diaria(base, signals)
    for i in range(2, 11):
        assert wrapper(dates[i], 0, 0)[0] == 0
    assert wrapper(dates[11], 0, 0)[0] == 1


def test_signal_covers_entire_universe_not_current_asset():
    dates = pd.date_range("2025-01-02", periods=3, freq="B", tz="UTC")
    signals = pd.DataFrame({
        "decision_date": dates,
        "stress_without_candidate": [False, True, False],
    })
    def weak_base(_dt, _position, _holding):
        return (2, -2.5)
    wrapper = construir_politica_protecao_diaria(weak_base, signals)
    assert wrapper(dates[0], 7, 11)[0] == 2
    assert wrapper(dates[1], 7, 11)[0] == 0
    assert wrapper(dates[2], 0, 0)[0] == 2


def test_episode_window_limits_attribution_not_cash_duration():
    dates = pd.date_range("2025-01-02", periods=15, freq="B", tz="UTC")
    signals = pd.DataFrame({
        "decision_date": dates,
        "stress_without_candidate": [False] * 2 + [True] * 10 + [False] * 3,
    })
    def base(_dt, position, _holding):
        return (1 if position == 0 else 2, .5)
    wrapper = construir_politica_protecao_diaria(
        base, signals, active_dates=dates[2:5],
    )
    for i in (2, 3, 4):
        assert wrapper(dates[i], 0, 0)[0] == 0
    # Em um replay local, a janela so delimita a comparacao financeira.
    assert wrapper(dates[5], 0, 0)[0] == 1


def test_daily_signal_rejects_missing_or_duplicate_dates():
    dates = pd.date_range("2025-01-02", periods=3, freq="B", tz="UTC")
    data = pd.DataFrame({
        "decision_date": dates,
        "stress_without_candidate": [False, True, False],
    })
    wrapper = construir_politica_protecao_diaria(
        lambda _d, _p, _h: (1, 0.1), data,
    )
    with pytest.raises(KeyError, match="Sem avaliacao"):
        wrapper(dates[-1] + pd.Timedelta(days=3), 0, 0)
    with pytest.raises(ValueError, match="duplicadas"):
        construir_politica_protecao_diaria(
            lambda _d, _p, _h: (1, .1),
            pd.concat([data, data.iloc[[0]]], ignore_index=True),
        )


def test_monthly_heatmap_counts_effective_positions_and_click_details():
    dates = pd.to_datetime([
        "2025-01-29T05:00:00+00:00",
        "2025-01-30T05:00:00+00:00",
        "2025-02-03T05:00:00+00:00",
        "2025-02-04T05:00:00+00:00",
    ], utc=True)
    decisions = dates - pd.Timedelta(days=1)
    p = pd.DataFrame({
        "timestamp": dates,
        "decision_date": decisions,
        "selected_asset": ["AAA", "BBB", "BBB", "CASH"],
        "strategy_equity": [10000, 11000, 12000, 12000],
    })
    t = pd.DataFrame({
        "timestamp": [dates[0], dates[1], dates[2]],
        "action": ["BUY", "SELL", "BUY"],
        "asset": ["AAA", "AAA", "BBB"],
    })
    indicators = pd.DataFrame({
        "decision_date": decisions,
        "universe_stress": [False, True, True, True],
        "stress_without_candidate": [False, False, True, True],
    })
    months, holdings = posicoes_mensais(p, t, indicators)
    assert len(months) == 2
    assert months["sessions"].sum() == 4
    assert holdings["sessions"].sum() == 4
    feb = months.loc[months["month"].eq("2025-02")].iloc[0]
    assert feb["cash_sessions"] == 1
    assert feb["stress_sessions"] == 2
    assert feb["stress_no_candidate_sessions"] == 2
    html = serializar_mapa_html(months, holdings, title="Mapa U75")
    assert "Mapa U75" in html
    assert "addEventListener" in html
    assert "BBB" in html and "CASH" in html


def test_package_includes_png_and_clickable_html_and_hashes(tmp_path):
    folder = tmp_path / "20261031T000000000000Z"
    folder.mkdir()
    (folder / "resultado.json").write_text(json.dumps({"status": "completed"}))
    (folder / "mapa_mensal_interativo.html").write_text("<html>mapa</html>")
    (folder / "mapa_mensal_ativos.png").write_bytes(b"fake-png-for-packager")
    zip_path = gerar_zip_visual(folder, versao="test", status="completed")
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
        files = json.loads(z.read("manifesto_pacote.json"))["files"]
        names = {x["name"] for x in files}
        assert "mapa_mensal_interativo.html" in names
        assert "mapa_mensal_ativos.png" in names
        assert "resultado.json" in names


def test_failure_package_is_generated_with_diagnostic(tmp_path, monkeypatch):
    import pesquisar_diagnostico_u75_spyder as run
    monkeypatch.setattr(run, "OUTPUT", tmp_path / "diagnostico")
    def fail(_output):
        raise RuntimeError("Control divergente")
    monkeypatch.setattr(run, "_run", fail)
    with pytest.raises(RuntimeError, match="Control divergente"):
        run.main()
    packages = list((tmp_path / "diagnostico").glob("pacote_diagnostico_u75_*.zip"))
    assert len(packages) == 1
    with ZipFile(packages[0]) as z:
        error = json.loads(z.read("erro_execucao.json"))
        assert "Control divergente" in error["message"]
