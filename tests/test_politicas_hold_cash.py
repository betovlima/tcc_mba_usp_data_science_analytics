"""Testes da segunda camada economica; sem corretora ou operacoes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _import(filename, alias):
    p = ROOT / filename
    spec = importlib.util.spec_from_file_location(alias, p)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


core = _import("pesquisar_segunda_camada_intradiaria_spyder.py", "core_tcc_policy")
pol = _import("pesquisar_politicas_intradiarias.py", "policy_tcc")


def make_grid(days=4):
    rng = np.random.default_rng(1001)
    rows = []
    for session in pd.bdate_range("2026-03-02", periods=days):
        t0 = pd.Timestamp(str(session.date()) + "T09:30:00",
                          tz="America/New_York").tz_convert("UTC")
        current = 100.0
        for i in range(78):
            nxt = current * np.exp(rng.normal(0, .003))
            rows.append({"timestamp": t0 + pd.Timedelta(minutes=5*i),
                         "symbol": "SYN", "open": current,
                         "high": max(current, nxt)*1.001,
                         "low": min(current, nxt)*.999,
                         "close": nxt, "volume": 1000000})
            current = nxt
    bars = core.preparar(pd.DataFrame(rows))
    frames = core.features(bars)
    positions = pd.DataFrame(
        {"session": sorted(frames.session.unique()), "symbol": "SYN"})
    return frames, positions


def test_replay_respeita_trava_e_venda_causal_no_proximo_candle():
    frame, positions = make_grid()
    audit, extreme, sessions, ages = pol._auditar(frame, positions)
    assert len(extreme) == 0
    assert set(audit.estado) == {"utilizavel"}
    assert [ages[d] for d in positions.session] == [1, 2, 3, 4]
    day = positions.session.iloc[2]
    group = sessions[(day, "SYN")]
    checkpoint = group[group.horario].iloc[0].timestamp
    signal = {(day, "SYN"): {checkpoint: .1}}
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    summary, daily, events = pol._replay(
        positions, sessions, ages, state, signal, .4, 10000, save_events=True)
    assert daily.acao.tolist()[:3] == ["HOLD", "HOLD", "CASH"]
    assert summary["saidas_antecipadas"] == 1
    assert len(events) == 1
    assert pd.Timestamp(events.iloc[0].execucao_teorica_utc) == checkpoint + pd.Timedelta(minutes=5)
    assert events.iloc[0].dias_mesmo_ativo == 3
    assert all(daily.acao.iloc[i] == "HOLD" for i in (0, 1))
    assert summary["control_intradiario_final"] > 0


def test_stress_liquidez_nao_finge_execucao_de_capital_milionario():
    frame, positions = make_grid()
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    initial = 68160546.38689804
    res, d, _ = pol._replay(positions, sessions, ages, state, {},
                            .4, initial)
    assert res["nao_executaveis"] == len(positions)
    assert np.isclose(res["politica_final"], initial)
    assert np.isclose(res["control_intradiario_final"], initial)
    assert (d.acao == "SEM_EXECUCAO").all()


def test_retorno_extremo_auditado_sem_filtrar_futuro_do_replay():
    frame, positions = make_grid()
    day = positions.session.iloc[2]
    altered = frame.copy()
    stamp = altered.loc[(altered.session == day) & altered.horario, "timestamp"].iloc[0]
    altered.loc[(altered.session == day) & (altered.timestamp == stamp),
                "ret_futuro_30m"] = .337673
    audit, extreme, _, _ = pol._auditar(altered, positions)
    entry = audit[audit.session == day].iloc[0]
    assert bool(entry.revisao_extremo)
    assert entry.estado == "utilizavel"
    assert len(extreme) == 1
    assert extreme.iloc[0].status == "revisao_manual_nao_usar_como_prova_de_execucao"


def test_calibracao_nao_usa_labels_da_amostra_de_teste():
    frame, _ = make_grid(days=24)
    days = sorted(frame.session.unique())
    cal = frame[frame.session.isin(days[:10])]
    sel = frame[frame.session.isin(days[10:16])]
    test = frame[frame.session.isin(days[16:])]
    s_cal = core.rolling_scores(cal)
    s_sel = core.rolling_scores(sel)
    s_test = core.rolling_scores(test)
    pack, _ = pol._calibrar(cal, sel, test, s_cal, s_sel, s_test)
    assert pack is not None
    other = test.copy()
    valid = other.alvo_alta.notna()
    other.loc[valid, "alvo_alta"] = 1 - other.loc[valid, "alvo_alta"]
    pack_changed, _ = pol._calibrar(cal, sel, other, s_cal, s_sel, s_test)
    pd.testing.assert_series_equal(pack[0], pack_changed[0])
    pd.testing.assert_series_equal(pack[1], pack_changed[1])


def test_replay_geral_gera_os_arquivos_e_preserva_control_original():
    frame, positions = make_grid(days=100)
    funcs = {"rolling": core.rolling_scores, "bocpd": core.bocpd,
             "hmm": core.hmm, "lgb": core.lgb_scores, "fqi": core.fqi,
             "split": core.split_cronologico}
    with patch.object(pol, "METHODS", ("multirrolling",)):
        files, meta = pol.executar(frame, positions, funcs)
    assert meta["control_original_comparavel"] is False
    assert meta["replay_original_587m_reproduzido"] is False
    assert "replay_patrimonio_intradiario.csv" in files
    assert "decisoes_hold_cash.csv" in files
    assert "previsoes_calibradas_teste.csv" in files
    assert meta["periodo_teste"][0] is not None
    import io
    comparison = pd.read_csv(io.BytesIO(files["comparacao_politicas_intradiarias.csv"]))
    assert len(comparison) == 2
    assert set(comparison.cenario_capital) == {"micro_10000",
                                               "stress_control_68160546"}
    metrics = pd.read_csv(io.BytesIO(files["comparacao_modelos_calibrados.csv"]))
    assert metrics.metodo.tolist() == ["multirrolling"]
    assert metrics.status.iloc[0] == "ok"
    assert metrics.amostras_teste.iloc[0] > 0
