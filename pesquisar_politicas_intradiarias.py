"""Experimento complementar de politicas HOLD/CASH intradiarias.

Somente pesquisa. Sinais causais a partir de candles encerrados, ordem
hipotetica no candle seguinte, CASH ate o proximo pregao, sem rotacao
intradiaria para outro ativo. Nao chama corretora nem envia ordens.

O replay e uma APROXIMACAO intradiaria pareada, sem overnight; nao
reproduz os 587 milhoes do Control diario congelado. No replay, ambos
os lados compram na abertura e liquidam ao fim do mesmo pregao.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

METHODS = ("multirrolling", "bocpd", "hmm", "lightgbm",
           "pca_lightgbm", "fitted_q")
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
COST_BPS_PER_SIDE = 10.0
MAX_VOLUME_PARTICIPATION = .01
EXTREME_30M = .20
CAPITALS = (10000.0, 68160546.38689804)
ORIGINAL_CONTROL_FINAL = 587165941.1330212


def _number(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _metricas_serie(returns):
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return {"max_drawdown": None, "sharpe": None}
    equity = np.cumprod(np.r_[1.0, 1 + r])
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0
    return {"max_drawdown": float(dd.min()),
            "sharpe": float(np.mean(r) / std * np.sqrt(252)) if std > 0 else None}


def _scores(method, train, sample, funcs):
    if method == "multirrolling":
        return funcs["rolling"](sample)
    if method == "bocpd":
        v = train.r1.dropna().var()
        return funcs["bocpd"](sample, v)
    if method == "hmm":
        return funcs["hmm"](train, sample)
    if method == "lightgbm":
        return funcs["lgb"](train, sample, pca=False)
    if method == "pca_lightgbm":
        return funcs["lgb"](train, sample, pca=True)
    if method == "fitted_q":
        return funcs["fqi"](train, sample)
    raise ValueError("Metodo nao reconhecido")


def _log(registrar, etapa, info, nivel="INFO"):
    if registrar is not None:
        registrar(etapa, info, nivel)


def _auditar(frame, positions):
    """Grade completa: NaN representa falta de negocio, jamais preco zero."""
    by_day = {(session, sym): g.sort_values("timestamp").reset_index(drop=True)
              for (sym, session), g in frame.groupby(["symbol", "session"], sort=False)}
    ages = {}
    previous = None
    age = 0
    for pos in positions.sort_values("session").itertuples(index=False):
        age = age + 1 if pos.symbol == previous else 1
        previous = pos.symbol
        ages[pos.session] = age
    audits, extreme, sessions = [], [], {}
    for pos in positions.sort_values("session").itertuples(index=False):
        key = (pos.session, pos.symbol)
        g = by_day.get((pos.session, pos.symbol))
        audit = {"session": pos.session, "symbol": pos.symbol,
                 "dias_mesmo_ativo": ages[pos.session], "candles_observados": 0,
                 "intervalos_sem_candle": 78, "checkpoints_validos": 0,
                 "extremos_30m": 0, "max_retorno_30m_abs": None,
                 "revisao_extremo": False, "estado": "sem_candles"}
        if g is not None and not g.empty:
            known = g.close.notna()
            audit["candles_observados"] = int(known.sum())
            audit["intervalos_sem_candle"] = int(78 - known.sum())
            audit["checkpoints_validos"] = int(g.horario.fillna(False).sum())
            future = pd.to_numeric(g.ret_futuro_30m, errors="coerce")
            abnormal = g[future.abs() >= EXTREME_30M]
            audit["extremos_30m"] = len(abnormal)
            audit["revisao_extremo"] = bool(len(abnormal))
            audit["max_retorno_30m_abs"] = _number(future.abs().max())
            for x in abnormal.itertuples(index=False):
                extreme.append({"session": pos.session, "symbol": pos.symbol,
                                "timestamp_utc": x.timestamp.isoformat(),
                                "preco_fechamento": _number(x.close),
                                "retorno_30m": _number(x.ret_futuro_30m),
                                "status": "revisao_manual_nao_usar_como_prova_de_execucao"})
            if len(g) == 78 and known.iloc[0] and known.iloc[-1]:
                # Flag de auditoria EX-POST: nunca filtrar retorno futuro
                # da amostra principal, porque isso geraria lookahead.
                audit["estado"] = "utilizavel"
            else:
                audit["estado"] = "sem_abertura_ou_fechamento_observado"
            sessions[key] = g
        audits.append(audit)
    return pd.DataFrame(audits), pd.DataFrame(extreme, columns=[
        "session", "symbol", "timestamp_utc", "preco_fechamento",
        "retorno_30m", "status"]), sessions, ages


def _calibrar(cal, sel, tst, scores_cal, scores_sel, scores_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    def xy(data, score):
        signal = score.reindex(data.index)
        keep = data.horario & data.alvo_alta.notna() & signal.notna()
        return data.loc[keep], signal.loc[keep]
    a, sa = xy(cal, scores_cal)
    if len(a) < 40 or a.alvo_alta.nunique() < 2:
        return None, {"status": "calibracao_insuficiente",
                      "amostras_calibracao": len(a)}
    x = sa.to_numpy(dtype=float).reshape(-1, 1)
    model = LogisticRegression(random_state=20261114, max_iter=300)
    model.fit(x, a.alvo_alta.astype(int).to_numpy())
    base = float(a.alvo_alta.mean())
    def predict(score):
        result = pd.Series(np.nan, index=score.index, dtype=float)
        valid = score.notna()
        if valid.any():
            result.loc[valid] = model.predict_proba(
                score.loc[valid].to_numpy(dtype=float).reshape(-1, 1))[:, 1]
        return result
    p_sel, p_test = predict(scores_sel), predict(scores_test)
    b, sb = xy(tst, scores_test)
    yt = b.alvo_alta.astype(int)
    pt = p_test.loc[b.index]
    auc = (float(roc_auc_score(yt, pt))
           if len(b) >= 3 and yt.nunique() == 2 else None)
    brier = float(brier_score_loss(yt, pt)) if len(b) else None
    base_brier = float(np.mean((yt.to_numpy() - base)**2)) if len(b) else None
    meta = {"status": "ok", "amostras_calibracao": len(a),
            "frequencia_positiva_calibracao": base,
            "amostras_teste": len(b), "auc_teste": auc,
            "brier_calibrado_teste": brier,
            "brier_constante_historica_teste": base_brier,
            "coeficiente_calibracao": float(model.coef_[0, 0]),
            "intercepto_calibracao": float(model.intercept_[0])}
    return (p_sel, p_test), meta


def _replay(positions, sessions, ages, audit_status, prob, threshold,
            capital, save_events=False):
    """Replay intradiario proxy, igual para baseline e candidatos.

    Valor total somente e investido se o volume de 5min comportaria 100%
    da compra e da venda teorica com participacao maxima de 1% do volume.
    A capacidade da saida e observada ex-post, nunca usada como feature.
    """
    b_cap, p_cap = float(capital), float(capital)
    cost = COST_BPS_PER_SIDE / 10000
    daily, events = [], []
    for pos in positions.itertuples(index=False):
        key = (pos.session, pos.symbol)
        g = sessions.get(key)
        reason = audit_status.get(key, "sem_candles")
        age = ages.get(pos.session, 0)
        choice = "HOLD"
        attempts, blocks = 0, 0
        exit_price = None
        exit_time = None
        if g is not None and reason == "utilizavel":
            first, last = g.iloc[0], g.iloc[-1]
            entry, end = _number(first.open), _number(last.close)
            entry_vol, close_vol = _number(first.volume), _number(last.volume)
            if (entry is None or end is None or entry <= 0 or end <= 0
                or entry_vol is None or close_vol is None):
                reason = "ohlcv_execucao_invalido"
            else:
                # Capacidade deve considerar AMBOS os caminhos, pois o
                # patrimonio da politica pode divergir do Control.
                shares = max(b_cap, p_cap) / entry
                if (shares > MAX_VOLUME_PARTICIPATION * entry_vol
                    or max(b_cap, p_cap) / end >
                       MAX_VOLUME_PARTICIPATION * close_vol):
                    reason = "liquidez_insuficiente_abertura_ou_fechamento"
        if g is not None and reason == "utilizavel":
            signal = prob.get(key, {})
            if age >= 3:
                for i in range(77):
                    r = g.iloc[i]
                    if not bool(r.horario) or r.timestamp not in signal:
                        continue
                    probability = signal[r.timestamp]
                    if probability is None or not np.isfinite(probability):
                        continue
                    if probability >= threshold:
                        continue
                    attempts += 1
                    next_bar = g.iloc[i + 1]
                    px, vol = _number(next_bar.open), _number(next_bar.volume)
                    if px is None or vol is None or px <= 0:
                        blocks += 1
                        continue
                    shares_at_exit = p_cap / entry
                    if shares_at_exit > MAX_VOLUME_PARTICIPATION * vol:
                        blocks += 1
                        continue
                    exit_price = px
                    exit_time = next_bar.timestamp.isoformat()
                    choice = "CASH"
                    if save_events:
                        events.append({"session": pos.session, "symbol": pos.symbol,
                                       "checkpoint_utc": r.timestamp.isoformat(),
                                       "execucao_teorica_utc": exit_time,
                                       "probabilidade_calibrada": probability,
                                       "limiar": threshold,
                                       "preco_execucao_teorico": exit_price,
                                       "volume_5min": vol,
                                       "dias_mesmo_ativo": age,
                                       "observacao": "preco_open_5min_optimista_sem_orderbook"})
                    break
            reference_factor = end / entry * (1-cost)**2
            policy_factor = (exit_price / entry if exit_price is not None
                             else end / entry) * (1-cost)**2
        else:
            reference_factor = 1.0
            policy_factor = 1.0
            choice = "SEM_EXECUCAO"
        before_b, before_p = b_cap, p_cap
        b_cap *= reference_factor
        p_cap *= policy_factor
        daily.append({"session": pos.session, "symbol": pos.symbol,
                      "dias_mesmo_ativo": age, "estado_dados": reason,
                      "acao": choice, "tentativas_saida": attempts,
                      "saidas_bloqueadas_volume_ou_preco": blocks,
                      "retorno_control_intradiario": reference_factor-1,
                      "retorno_politica_intradiaria": policy_factor-1,
                      "control_capital": b_cap, "politica_capital": p_cap,
                      "controle_inicio_dia": before_b,
                      "politica_inicio_dia": before_p,
                      "preco_saida": exit_price, "saida_utc": exit_time})
    d = pd.DataFrame(daily)
    risk_b = _metricas_serie(d.retorno_control_intradiario)
    risk_p = _metricas_serie(d.retorno_politica_intradiaria)
    summary = {"capital_inicial": capital,
               "control_intradiario_final": b_cap, "politica_final": p_cap,
               "diferenca_usd": p_cap-b_cap,
               "diferenca_pct": (p_cap/b_cap-1)*100 if b_cap > 0 else None,
               "control_max_dd": risk_b["max_drawdown"],
               "politica_max_dd": risk_p["max_drawdown"],
               "control_sharpe": risk_b["sharpe"],
               "politica_sharpe": risk_p["sharpe"],
               "saidas_antecipadas": int((d.acao=="CASH").sum()),
               "nao_executaveis": int((d.acao=="SEM_EXECUCAO").sum()),
               "tentativas_saida": int(d.tentativas_saida.sum()),
               "saidas_bloqueadas": int(d.saidas_bloqueadas_volume_ou_preco.sum()),
               "dias_avaliados": len(d)}
    return summary, d, pd.DataFrame(events)


def _sigmap(data, probability):
    values = probability.reindex(data.index)
    d = data.loc[data.horario & values.notna(), ["session", "symbol", "timestamp"]].copy()
    d["prob"] = values.loc[d.index].values
    result = {}
    for (session, sym), g in d.groupby(["session", "symbol"], sort=False):
        result[(session, sym)] = dict(zip(g.timestamp, g.prob))
    return result


def _group_dates(d, days):
    return d[d.session.isin(days)].copy()


def executar(frame, positions, funcoes, registrar=None):
    """Produz artefatos em memoria para o ZIP principal, sem acesso externo."""
    _log(registrar, "politicas", "Auditando cobertura, anomalias e viabilidade de capital.")
    positions = positions[["session", "symbol"]].copy()
    positions["session"] = positions.session.astype(str)
    positions["symbol"] = positions.symbol.astype(str)
    positions = positions.sort_values("session").reset_index(drop=True)
    audit, extreme, sessions, ages = _auditar(frame, positions)
    status = {(x.session, x.symbol): x.estado
              for x in audit.itertuples(index=False)}
    tr, va, te, split = funcoes["split"](frame)
    days = sorted(va.session.unique())
    midpoint = len(days)//2
    cal, sel = _group_dates(va, days[:midpoint]), _group_dates(va, days[midpoint:])
    selected = positions[positions.session.isin(te.session.unique())]
    tune_positions = positions[positions.session.isin(sel.session.unique())]
    _log(registrar, "politicas", f"Calibracao={cal.session.nunique()} dias, "
         f"selecao={sel.session.nunique()} dias, teste={te.session.nunique()} dias.")
    comparisons, metrics, decisions, curves, calibrated = [], [], [], [], []
    for method in METHODS:
        _log(registrar, "politicas", f"{method}: calculando scores temporais.")
        val_scores = _scores(method, tr, va, funcoes)
        test_scores = _scores(method, tr, te, funcoes)
        if val_scores is None or test_scores is None:
            metrics.append({"metodo": method, "status": "modelo_indisponivel"})
            continue
        pack, calibration = _calibrar(cal, sel, te, val_scores, val_scores, test_scores)
        if pack is None:
            metrics.append({"metodo": method, **calibration})
            continue
        p_val, p_test = pack
        valid = te.horario & p_test.notna()
        p_rows = te.loc[valid, ["session", "symbol", "timestamp", "alvo_alta"]].copy()
        p_rows["metodo"] = method
        p_rows["probabilidade"] = p_test.loc[p_rows.index].to_numpy()
        calibrated.append(p_rows)
        tune_map = _sigmap(sel, p_val)
        test_map = _sigmap(te, p_test)
        tuning = []
        for threshold in THRESHOLDS:
            summ, _, _ = _replay(tune_positions, sessions, ages, status,
                                 tune_map, threshold, CAPITALS[0])
            final = max(summ["politica_final"], 1e-12)
            dd = summ["politica_max_dd"] or 0.0
            objective = math.log(final/CAPITALS[0]) + .5 * dd
            tuning.append((objective, threshold))
        # Seleciona na segunda metade da validacao, jamais no teste.
        best = sorted(tuning, key=lambda v: (-v[0], v[1]))[0][1]
        metrics.append({"metodo": method, **calibration,
                        "limiar_validado": best,
                        "calibracao_dias": cal.session.nunique(),
                        "selecao_dias": sel.session.nunique(),
                        "metodo_selecao": "log_capital_mais_0.5_maxDD_validacao"})
        for cap in CAPITALS:
            summary, curve, chosen = _replay(selected, sessions, ages, status,
                                             test_map, best, cap, save_events=True)
            summary.update({"metodo": method, "cenario_capital": (
                "micro_10000" if cap==CAPITALS[0] else "stress_control_68160546"),
                "limiar": best, "status": "proxy_nao_replay_exato_control"})
            comparisons.append(summary)
            curve["metodo"] = method
            curve["cenario_capital"] = summary["cenario_capital"]
            curves.append(curve)
            if not chosen.empty:
                chosen["metodo"] = method
                chosen["cenario_capital"] = summary["cenario_capital"]
                decisions.append(chosen)
        _log(registrar, "politicas", f"{method}: limiar={best:.2f}; "
             f"micro_final={comparisons[-2]['politica_final']:.2f}; "
             f"micro_base={comparisons[-2]['control_intradiario_final']:.2f}")
    _log(registrar, "politicas", f"Concluido. Sessoes sem cobertura="
         f"{int((audit.estado=='sem_candles').sum())}; "
         f"eventos extremos={len(extreme)}.")
    def csv(rows):
        return rows.to_csv(index=False, float_format="%.12g").encode("utf-8")
    out = {
        "auditoria_sessoes.csv": csv(audit),
        "auditoria_movimentos_extremos.csv": csv(extreme),
        "comparacao_modelos_calibrados.csv": csv(pd.DataFrame(metrics)),
        "comparacao_politicas_intradiarias.csv": csv(pd.DataFrame(comparisons)),
        "replay_patrimonio_intradiario.csv": csv(pd.concat(curves, ignore_index=True)
            if curves else pd.DataFrame()),
        "decisoes_hold_cash.csv": csv(pd.concat(decisions, ignore_index=True)
            if decisions else pd.DataFrame()),
        "previsoes_calibradas_teste.csv": csv(pd.concat(calibrated, ignore_index=True)
            if calibrated else pd.DataFrame()),
    }
    manifesto = {
        "tipo": "exploratorio_intradiario_pareado_sem_overnight",
        "status": "nao_equivalente_ao_control_diario_original",
        "testes": "somente pregoes fora da amostra (split do estudo)",
        "calibracao": "metade inicial da validacao; modelo aprendido no treino",
        "selecao_limiar": "metade final da validacao; jamais usa teste",
        "limiares_testados": list(THRESHOLDS),
        "custo_bps_por_lado": COST_BPS_PER_SIDE,
        "max_participacao_candle": MAX_VOLUME_PARTICIPATION,
        "aplicacao": "ativo da Winner congelado; nenhuma troca intradiaria",
        "trava": "saida antecipada apenas a partir do terceiro pregao consecutivo",
        "execucao": "open do candle imediatamente posterior ao sinal; preco idealizado",
        "liquidez": ("volume historico de 5min; capacidade verificada ex-post; "
                     "nao e simulacao de livro de ofertas ou garantia de fill"),
        "sem_negocio": "sessao sem abertura ou ultimo candle nao e operada por nenhum lado",
        "extremos": ("retorno 30min absoluto >= 20% sinalizado para revisao; "
                     "nao removido do replay principal para evitar filtro ex-post"),
        "capital_inicial_cenarios": list(CAPITALS),
        "control_original_patrimonio_final_usd": ORIGINAL_CONTROL_FINAL,
        "control_original_comparavel": False,
        "replay_original_587m_reproduzido": False,
        "dados_brutos_exportados": False,
        "nenhuma_ordem_real": True,
        "status_cobertura": {"sessoes_control": len(audit),
                            "sessoes_com_grade": int((audit.candles_observados>0).sum()),
                            "sessoes_com_falta_de_abertura_ou_fechamento": int(
                                (audit.estado=="sem_abertura_ou_fechamento_observado").sum()),
                            "sessoes_extremas": int(audit.revisao_extremo.sum())},
        "periodo_teste": [str(te.session.min()) if not te.empty else None,
                          str(te.session.max()) if not te.empty else None],
        "nota": ("NAO comparar diretamente o capital do replay intradiario "
                 "com os US$ 587m originais: o Control original inclui "
                 "overnight, decisoes diarias e execucao propria."),
    }
    out["manifesto_politicas_intradiarias.json"] = json.dumps(
        manifesto, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return out, manifesto
