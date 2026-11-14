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


def test_primeiro_pregao_pode_sair_com_sinal_causal():
    frame, positions = make_grid(days=4)
    audit, extreme, sessions, ages = pol._auditar(frame, positions)
    assert len(extreme) == 0
    assert set(audit.estado) == {"utilizavel"}
    day = positions.session.iloc[0]
    group = sessions[(day, "SYN")]
    checkpoint = group[group.horario].iloc[0].timestamp
    signal = {(day, "SYN"): {checkpoint: .1}}
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    summary, daily, events = pol._replay(
        positions, sessions, ages, state, signal, .4, 10000, save_events=True)

    # Sem nenhuma restricao de 2 pregoes: venda no primeiro dia.
    assert daily.loc[0, "acao"] == "CASH"
    assert summary["primeiro_pregao_elegivel_saida"] == 1
    assert summary["permanencia_minima_imposta"] is False
    assert summary["tentativas_saida"] == 1
    assert summary["saidas_antecipadas"] == 1
    signals = events[events.observacao == "sinal_fechado_aguardando_candle_seguinte"]
    orders = events[
        events.observacao == "fill_proxy_1pct_volume_candle_anterior_sem_orderbook"]
    assert len(signals) == 1
    assert len(orders) == 1
    assert signals.iloc[0].dias_posicao_politica == 1
    assert pd.Timestamp(orders.iloc[0].execucao_teorica_utc) == checkpoint + pd.Timedelta(minutes=5)
    assert orders.iloc[0].dias_mesmo_ativo == 1

    # O fechamento obrigatorio recompra integralmente o ativo escolhido.
    assert daily.loc[0, "alocacao_politica_100pct"]
    assert daily.loc[0, "quantidade_comprada_fechamento_politica"] > 0
    assert daily.loc[0, "politica_em_caixa"] <= 1e-7
    assert not daily.loc[0, "pendente_liquidacao"]
    assert summary["rotacoes_sem_cotacao_ticker_antigo"] == 0


def test_saida_disponivel_no_primeiro_e_segundo_pregoes_apos_recompra():
    frame, positions = make_grid(days=6)
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    sinais = {}
    for session in positions.session:
        checkpoint = sessions[(session, "SYN")].loc[
            lambda x: x.horario].iloc[0].timestamp
        sinais[(session, "SYN")] = {checkpoint: .1}
    report, daily, events = pol._replay(
        positions, sessions, ages, state, sinais, .4, 10000, save_events=True)
    signals = events[
        events.observacao == "sinal_fechado_aguardando_candle_seguinte"]

    # A recompra no fechamento NAO torna o pregao seguinte inelegivel.
    assert signals.session.tolist() == positions.session.tolist()
    assert report["tentativas_saida"] == len(positions)
    assert report["saidas_bloqueadas"] == 0
    assert (daily.acao == "CASH").all()
    assert daily.alocacao_politica_100pct.all()
    assert (daily.politica_em_caixa <= 1e-7).all()
    assert (signals.dias_posicao_politica == 1).all()
    assert report["permanencia_minima_imposta"] is False


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


def test_baseline_sem_ml_sai_as_1025_no_primeiro_pregao():
    frame, positions = make_grid(days=4)
    signal = pol._baseline_saida_fixa(frame)
    assert len(signal) == 4
    assert all(len(v) == 1 for v in signal.values())
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    summary, daily, events = pol._replay(
        positions, sessions, ages, state, signal, .50, 10000,
        save_events=True)
    assert summary["tentativas_saida"] == 4
    assert summary["saidas_antecipadas"] == 4
    assert daily.alocacao_politica_100pct.all()
    assert (daily.politica_em_caixa <= 1e-7).all()
    signals = events[
        events.observacao == "sinal_fechado_aguardando_candle_seguinte"]
    assert signals.session.nunique() == 4
    assert set(pd.to_datetime(signals.checkpoint_utc, utc=True)
               .dt.tz_convert("America/New_York").dt.strftime("%H:%M")) == {"10:25"}


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
    assert res["caixa_ref_final"] == 0
    assert res["caixa_politica_final"] == 0
    # Com volume suficiente, toda a compra pode ocorrer intradiariamente;
    # se necessario, a recompra integral sera feita ao fechamento.
    assert d.alocacao_politica_100pct.all()
    assert not d.fechamento_executavel_verificado.any()


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


def test_saida_parcial_intra_e_recompra_integral_ao_fechamento():
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
    assert not row.pendente_liquidacao
    assert row.politica_quantidade > 0
    assert result["saidas_parciais_ao_fim_do_pregao"] > 0
    sells = events[(events.session == date) &
                   (events.observacao == "fill_proxy_1pct_volume_candle_anterior_sem_orderbook")]
    assert len(sells) > 0
    assert sells.quantidade_executada.max() <= 1.000001
    assert row.politica_em_caixa <= 1e-7
    assert row.alocacao_politica_100pct
    assert row.quantidade_comprada_fechamento_politica > 0
    assert (events.observacao == "compra_integral_fechamento_proxy_sem_fill_verificado").any()


def test_primeira_camada_troca_ativo_apenas_no_pregao_seguinte():
    frame, positions = make_grid(days=4)
    other = positions.session.iloc[2]
    frame.loc[frame.session >= other, "symbol"] = "NEW"
    positions.loc[positions.session >= other, "symbol"] = "NEW"
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    summary, daily, _ = pol._replay(positions, sessions, ages, state, {}, .4, 10000)
    assert daily.symbol.tolist() == ["SYN", "SYN", "NEW", "NEW"]
    assert daily.decisao_primeira_camada.tolist() == [
        "MANTER", "MANTER", "TROCAR", "MANTER"]
    assert summary["rotacoes_sem_cotacao_ticker_antigo"] == 1
    assert daily.alocacao_politica_100pct.all()
    assert (daily.politica_em_caixa <= 1e-7).all()
    assert daily.loc[2, "quantidade_comprada_fechamento_politica"] >= 0


def test_compra_de_fechamento_independente_do_volume_ou_montante():
    frame, positions = make_grid(days=4)
    frame.loc[:, "volume"] = 1.0
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    capital = 68160546.38689804
    summary, daily, events = pol._replay(
        positions, sessions, ages, state, {}, .4, capital, save_events=True)
    # Liquidez de apenas 1 acao/candle: mercado real nao permitiria
    # a compra, mas o cenario teorico precisa registrar a alocacao completa.
    assert (daily.politica_em_caixa <= 1e-7).all()
    assert daily.alocacao_politica_100pct.all()
    assert (daily.quantidade_comprada_fechamento_politica > 1000).any()
    assert summary["fills_proxy_fechamento_sem_volume"] > 0
    assert not summary["execucao_comprovada"]
    assert (events.observacao == "compra_integral_fechamento_proxy_sem_fill_verificado").any()


def test_sem_cotacao_nao_gera_preco_de_fechamento_sintetico():
    frame, positions = make_grid(days=2)
    first_day = positions.session.iloc[0]
    frame = frame.loc[frame.session != first_day]
    audit, _, sessions, ages = pol._auditar(frame, positions)
    state = {(x.session, x.symbol): x.estado for x in audit.itertuples(index=False)}
    summary, daily, _ = pol._replay(positions, sessions, ages, state, {}, .4, 10000)
    row = daily.iloc[0]
    assert row.rebalanceamento_fechamento == "sem_preco"
    assert pd.isna(row.preco_fechamento_usado)
    assert row.politica_quantidade == 0
    assert summary["fechamentos_sem_preco"] >= 1
    assert not row.alocacao_politica_100pct
    assert daily.iloc[1].alocacao_politica_100pct


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
    assert "sensibilidade_limiares_teste.csv" in files
    assert "selecao_limiares_validacao.csv" in files
    assert "referencia_saida_fixa_1025.csv" in files
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
    assert metrics.limiar_validado.iloc[0] in pol.THRESHOLDS
    assert "prob_minima_teste" in metrics.columns
    selection = pd.read_csv(io.BytesIO(files["selecao_limiares_validacao.csv"]))
    exploration = pd.read_csv(io.BytesIO(files["sensibilidade_limiares_teste.csv"]))
    fixed = pd.read_csv(io.BytesIO(files["referencia_saida_fixa_1025.csv"]))
    assert len(selection) == len(pol.THRESHOLDS)
    assert len(exploration) == len(pol.THRESHOLDS) * len(pol.CAPITALS)
    assert exploration.limiar_escolhido_na_validacao.sum() == len(pol.CAPITALS)
    assert fixed.metodo.tolist() == ["saida_fixa_1025_sem_ml"] * 2
    assert (fixed.tentativas_saida > 0).all()
    assert meta["primeiro_pregao_elegivel_saida"] == 1
    assert meta["permanencia_minima_imposta"] is False
