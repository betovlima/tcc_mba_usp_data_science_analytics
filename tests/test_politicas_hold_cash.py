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
    orders = events[events.quantidade_executada > 0]
    assert len(events) >= 2  # decisao ao fechamento + fill posterior
    assert len(orders) == 1
    assert pd.Timestamp(orders.iloc[0].execucao_teorica_utc) == checkpoint + pd.Timedelta(minutes=5)
    assert orders.iloc[0].dias_mesmo_ativo == 3
    assert summary["rotacoes_sem_cotacao_ticker_antigo"] == 0
    assert daily.control_quantidade.iloc[0] > 0
    assert all(daily.acao.iloc[i] == "HOLD" for i in (0, 1))
    assert summary["control_intradiario_final"] > 0


def test_reentrada_reinicia_a_trava_de_dois_pregoes():
    frame, positions = make_grid(days=6)
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    sinais = {}
    for session in positions.session.iloc[2:]:
        checkpoint = sessions[(session, "SYN")].loc[
            lambda x: x.horario].iloc[0].timestamp
        sinais[(session, "SYN")] = {checkpoint: .1}
    report, daily, events = pol._replay(
        positions, sessions, ages, state, sinais, .4, 10000, save_events=True)
    # Se a primeira venda foi no terceiro dia, a recompra no quarto
    # nao pode ser liquidada nesse mesmo quarto ou quinto dia.
    sinais_executados = events[
        events.observacao == "sinal_fechado_aguardando_candle_seguinte"]
    assert sinais_executados.session.tolist() == [
        positions.session.iloc[2], positions.session.iloc[5]]
    assert report["tentativas_saida"] == 2
    assert report["saidas_bloqueadas"] == 0
    # Dias na posicao efetiva, nao dias de selecao da Winner.
    assert daily.dias_posicao_politica.tolist()[:6] == [1, 2, 3, 1, 2, 3]
    assert all(events[events.observacao.str.startswith("sinal")].dias_posicao_politica == 3)


def test_saida_total_nao_registra_bloqueios_falsos_depois():
    frame, positions = make_grid(days=4)
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    date = positions.session.iloc[2]
    checkpoint = sessions[(date, "SYN")].loc[
        lambda x: x.horario].iloc[0].timestamp
    summary, daily, events = pol._replay(
        positions, sessions, ages, state,
        {(date, "SYN"): {checkpoint: .1}},
        .4, 10000, save_events=True)
    assert summary["tentativas_saida"] == 1
    assert summary["saidas_bloqueadas"] == 0
    assert (daily.saidas_bloqueadas_volume_ou_preco == 0).all()
    assert summary["saidas_antecipadas"] == 1


def test_stress_liquidez_nao_finge_execucao_de_capital_milionario():
    frame, positions = make_grid()
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    initial = 68160546.38689804
    res, d, _ = pol._replay(positions, sessions, ages, state, {},
                            .4, initial)
    # Volume de 5Min limita QUANTIDADE, nao impede todas as compras.
    assert res["nao_executaveis"] == 0
    assert res["compras_parciais_ref"] > 0
    assert (d.control_quantidade > 0).all()
    assert res["control_original_comparavel"] is False
    assert (d.acao == "HOLD").all()
    assert res["caixa_ref_final"] >= 0


def test_overnight_e_posicao_carregada_sem_recompra_diaria():
    frame, positions = make_grid(days=4)
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    result, daily, _ = pol._replay(
        positions, sessions, ages, state, {}, .35, 10000)
    assert (daily.control_quantidade > 0).all()
    assert daily.control_quantidade.iloc[1] == daily.control_quantidade.iloc[0]
    assert daily.control_taxas_dia.iloc[1] == 0
    assert daily.controle_inicio_dia.iloc[1] == daily.control_capital.iloc[0]
    assert result["taxas_ref_usd"] > 0


def test_saida_parcial_continua_sem_recompra_no_mesmo_dia():
    frame, positions = make_grid(days=4)
    # A barra que fecha em 10h30 tem pouco volume. O primeiro
    # fill na abertura seguinte deve usar o volume conhecido as 10h25.
    date = positions.session.iloc[2]
    frame = frame.copy()
    clock = frame.timestamp.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    after = (frame.session == date) & (clock >= "10:25")
    frame.loc[after, "volume"] = 100.0
    audit, _, sessions, ages = pol._auditar(frame, positions)
    checkpoint = sessions[(date, "SYN")].loc[
        lambda x: x.horario].iloc[0].timestamp
    signal = {(date, "SYN"): {checkpoint: .1}}
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    result, daily, events = pol._replay(
        positions, sessions, ages, state, signal, .4, 10000, save_events=True)
    row = daily[daily.session == date].iloc[0]
    assert row.acao == "SAIDA_PARCIAL"
    assert row.pendente_liquidacao
    assert row.politica_quantidade > 0
    assert result["saidas_parciais_ao_fim_do_pregao"] > 0
    sells = events[(events.session == date) & (events.quantidade_executada > 0)]
    assert len(sells) > 0
    assert sells.quantidade_executada.max() <= 1.000001
    assert row.politica_em_caixa > 0
    assert daily[daily.session == positions.session.iloc[3]].iloc[0].politica_quantidade < row.politica_quantidade


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
