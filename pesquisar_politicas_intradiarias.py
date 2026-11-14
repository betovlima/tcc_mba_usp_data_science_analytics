"""Experimento complementar de politicas HOLD/CASH intradiarias.

Somente pesquisa. A camada 2 pode sair para CASH durante o pregao;
no processamento de fechamento, RECOMPRA integralmente o ativo que
a primeira camada escolheu no inicio do dia, independente do capital.
No dia seguinte a camada 1 manda MANTER ou TROCAR. Nao envia ordens.

Fechamento a preco conhecido somente apos o encerramento e compra
integral sem limite de volume sao SUPOSICOES TEORICAS, nao fills reais.
O replay tem overnight, mas nao reproduz os US$ 587m do Control original.
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
    """Carteira continua com dois livros, cotas fracionarias e fills PARCIAIS.

    Mantem o ativo escolhido pela primeira camada INTEGRALMENTE em carteira
    ao final de cada pregao, mesmo se a camada 2 mandou CASH intradiario.
    Ao fechar, compra todo o saldo ao preco do ultimo close observado, sem
    limite de volume: fill IDEALIZADO, impossivel de confirmar ex-post.
    Esse processamento e a unica excecao a regra de nao recomprar no dia.
    No proximo pregao a camada 1 manda manter ou trocar, sempre soberana.

    Durante o dia, sinais usam candles encerrados e venda no OPEN do
    candle seguinte, limitada a 1% do volume do candle anterior.
    Nao existe trava de permanencia minima na segunda camada: ela pode
    vender desde o PRIMEIRO pregao, no primeiro checkpoint valido apos
    existir uma posicao. A idade da posicao fica apenas para auditoria.

    LIMITACAO MATERIAL: na ROTACAO entre ativos nao ha cotacao do ticker
    anterior no dia seguinte no cache corrente. A liquidacao e contabilizada
    pelo ultimo fechamento real do dia anterior, sem validar o fill. E uma
    hipotese otimista de contabilidade, NAO e um replay executavel completo.
    Cada rotacao assim estimada e registrada e invalida a certificacao de
    viabilidade do patrimonio final. Nunca usar resultado como Control real.
    """
    cost = COST_BPS_PER_SIDE / 10000
    books = {
        name: {"cash": float(capital), "shares": 0., "symbol": None,
               "last_price": None, "pending_exit": False, "dias_posicao": 0,
               "fees": 0., "buys": 0., "sells": 0.,
               "fills_parciais": 0, "rotacoes_estimadas": 0}
        for name in ("ref", "policy")
    }
    n_exits, n_blocked, n_attempts, n_partial_exits = 0, 0, 0, 0
    n_stale, n_sessions_no_quote, n_rotations = 0, 0, 0
    n_close_rebalances, n_close_without_price, n_proxy_close_fills = 0, 0, 0
    daily, events = [], []
    reference_return, policy_return = [], []

    def equity(book):
        price = book["last_price"]
        return book["cash"] + book["shares"] * (price if price is not None else 0.)

    def trade(book, side, price, volume):
        if price is None or volume is None or price <= 0 or volume <= 0:
            return 0., False
        capacity = MAX_VOLUME_PARTICIPATION * volume
        if side == "buy":
            quantity = min(book["cash"] / (price * (1 + cost)), capacity)
            quantity = max(0., quantity)
            book["cash"] -= quantity * price * (1 + cost)
            book["shares"] += quantity
            book["buys"] += int(quantity > 1e-12)
            incomplete = book["cash"] > max(1e-7, equity(book)*1e-6)
        else:
            quantity = min(book["shares"], capacity)
            quantity = max(0., quantity)
            book["cash"] += quantity * price * (1 - cost)
            book["shares"] -= quantity
            if book["shares"] < 1e-10:
                book["shares"] = 0.
            book["sells"] += int(quantity > 1e-12)
            incomplete = book["shares"] > 1e-10
        book["fees"] += quantity * price * cost
        if quantity > 0 and incomplete:
            book["fills_parciais"] += 1
        return quantity, incomplete

    for pos in positions.sort_values("session").itertuples(index=False):
        key = (pos.session, pos.symbol)
        session = sessions.get(key)
        status = audit_status.get(key, "sem_candles")
        age = ages.get(pos.session, 0)
        initial = {name: equity(book) for name, book in books.items()}
        fees_initial = {name: book["fees"] for name, book in books.items()}
        action, blocks, attempts, exit_time, exit_price = "HOLD", 0, 0, None, None
        close_rebalance = {"ref": 0., "policy": 0.}
        close_status = "sem_cotacao"
        close_price = None
        close_timestamp = None
        day_rotations = 0
        first_layer_action = "TROCAR" if books["policy"]["symbol"] not in (None, pos.symbol) else "MANTER"
        for name, book in books.items():
            if book["symbol"] != pos.symbol:
                if book["shares"] > 0 and book["last_price"] is not None:
                    # Sem barras do ativo antigo no novo dia, NAO ha fill
                    # verificavel. Liquidacao pelo fechamento anterior,
                    # identificada explicitamente como estimativa.
                    close = book["last_price"]
                    book["cash"] += book["shares"] * close * (1-cost)
                    book["fees"] += book["shares"] * close * cost
                    book["shares"] = 0.
                    book["rotacoes_estimadas"] += 1
                    day_rotations += 1
                book["symbol"] = pos.symbol
                book["pending_exit"] = False
                book["dias_posicao"] = 0
            elif book["shares"] > 1e-10:
                # A idade conta pregoes de posicao REAL mantida.
                book["dias_posicao"] += 1
            else:
                # Reentrada reinicia a idade auditada da posicao, sem bloquear sinais.
                book["dias_posicao"] = 0
                if book["pending_exit"]:
                    book["pending_exit"] = False
        if day_rotations:
            n_rotations += 1
        if session is None or session.empty:
            n_sessions_no_quote += 1
            n_close_without_price += 1
            close_status = "sem_preco"
            action = "SEM_COTACAO"
        else:
            g = session.sort_values("timestamp").reset_index(drop=True)
            exit_triggered_today = False
            for i, bar in enumerate(g.itertuples(index=False)):
                px = _number(bar.open)
                # O limite da ordem na abertura usa apenas volume de
                # candle ANTERIOR ja encerrado. Nunca usa volume futuro
                # do proprio candle que esta abrindo.
                volume = _number(g.iloc[i-1].volume) if i > 0 else None
                prior_close = _number(g.iloc[i-1].close) if i > 0 else None
                if px is None or volume is None or prior_close is None:
                    continue
                # Marca no ultimo OPEN conhecido antes de negociar,
                # sem carregar preco de outro ativo como se fosse atual.
                for book in books.values():
                    book["last_price"] = px
                for name, book in books.items():
                    if name == "policy" and book["pending_exit"]:
                        if book["shares"] <= 1e-10:
                            # Ja liquidada: nao transformar candles seguintes
                            # em falsas ordens recusadas por liquidez.
                            continue
                        quantity, incomplete = trade(book, "sell", px, volume)
                        if quantity > 0:
                            action = "CASH" if not incomplete else "SAIDA_PARCIAL"
                            if exit_time is None:
                                exit_time, exit_price = bar.timestamp.isoformat(), px
                            if save_events:
                                events.append({
                                    "session": pos.session, "symbol": pos.symbol,
                                    "checkpoint_utc": None if i == 0 else
                                        g.iloc[max(0, i-1)].timestamp.isoformat(),
                                    "execucao_teorica_utc": bar.timestamp.isoformat(),
                                    "probabilidade_calibrada": None,
                                    "limiar": threshold,
                                    "preco_execucao_teorico": px,
                                    "volume_5min": volume,
                                    "quantidade_executada": quantity,
                                    "execucao_parcial": bool(incomplete),
                                    "dias_mesmo_ativo": age,
                                    "dias_posicao_politica": books["policy"]["dias_posicao"],
                                    "observacao": "fill_proxy_1pct_volume_candle_anterior_sem_orderbook"})
                        else:
                            blocks += 1
                    elif book["cash"] > 1e-7:
                        nova_posicao = book["shares"] <= 1e-10
                        quantity, incomplete = trade(book, "buy", px, volume)
                        if name == "policy" and nova_posicao and quantity > 1e-10:
                            book["dias_posicao"] = 1
                # So le o score APOS o fechamento do candle corrente.
                # NENHUMA permanencia minima: o primeiro pregao ja e elegivel.
                if (not books["policy"]["pending_exit"]
                    and not exit_triggered_today and bool(bar.horario)
                    and i + 1 < len(g) and bar.timestamp in prob.get(key, {})
                    and books["policy"]["shares"] > 0):
                    signal = _number(prob[key][bar.timestamp])
                    if signal is not None and signal < threshold:
                        books["policy"]["pending_exit"] = True
                        exit_triggered_today = True
                        attempts += 1
                        if save_events:
                            events.append({
                                "session": pos.session, "symbol": pos.symbol,
                                "checkpoint_utc": bar.timestamp.isoformat(),
                                "execucao_teorica_utc": None,
                                "probabilidade_calibrada": signal,
                                "limiar": threshold, "preco_execucao_teorico": None,
                                "volume_5min": None,
                                "quantidade_executada": 0,
                                "execucao_parcial": False,
                                "dias_mesmo_ativo": age,
                                "dias_posicao_politica": books["policy"]["dias_posicao"],
                                "observacao": "sinal_fechado_aguardando_candle_seguinte"})
            # As ordens de saida nao completadas durante o dia continuam
            # parciais na auditoria mesmo depois da recompra de fechamento.
            if exit_triggered_today and books["policy"]["shares"] > 1e-10:
                n_partial_exits += 1
            # Fechamento: CADA LIVRO aloca 100% do caixa disponivel
            # no ativo selecionado pela camada 1 no inicio deste dia.
            # Um preco observado e necessario. Nao cria candles/precos.
            available_close = g.loc[g.close.notna(), ["timestamp", "close"]]
            if available_close.empty:
                n_close_without_price += 1
                close_status = "sem_preco"
                status = "sem_fechamento_observado"
            else:
                last = available_close.iloc[-1]
                actual_end = _number(last["close"])
                close_price = actual_end
                close_timestamp = (last["timestamp"] + pd.Timedelta(minutes=5)).isoformat()
                last_bar_present = bool(pd.notna(g.iloc[-1].close))
                close_status = ("preco_ultimo_candle" if last_bar_present
                                else "ultimo_preco_disponivel_proxy")
                if not last_bar_present:
                    n_stale += 1
                    status = "sem_candle_final_preco_proxy"
                if actual_end is not None:
                    for name, book in books.items():
                        book["last_price"] = actual_end
                        available_cash = max(0., book["cash"])
                        # Alocacao integral independente do valor da conta
                        # e de liquidez: simula fill no close apos conhece-lo.
                        quantity = available_cash / (actual_end*(1+cost))
                        if quantity > 0:
                            was_out = book["shares"] <= 1e-10
                            book["shares"] += quantity
                            book["cash"] = 0.
                            book["fees"] += quantity * actual_end * cost
                            book["buys"] += 1
                            close_rebalance[name] = quantity
                            n_proxy_close_fills += 1
                            if name == "policy" and was_out:
                                # Compra no fechamento zera a idade auditada;
                                # isso NAO interfere na elegibilidade de saida.
                                book["dias_posicao"] = 0
                            if name == "policy" and save_events:
                                events.append({
                                    "session": pos.session, "symbol": pos.symbol,
                                    "checkpoint_utc": None,
                                    "execucao_teorica_utc": close_timestamp,
                                    "probabilidade_calibrada": None,
                                    "limiar": threshold,
                                    "preco_execucao_teorico": actual_end,
                                    "volume_5min": None,
                                    "quantidade_executada": quantity,
                                    "execucao_parcial": False,
                                    "dias_mesmo_ativo": age,
                                    "dias_posicao_politica": book["dias_posicao"],
                                    "observacao": "compra_integral_fechamento_proxy_sem_fill_verificado"})
                        book["pending_exit"] = False
                    n_close_rebalances += 1
                else:
                    n_close_without_price += 1
                    close_status = "sem_preco"
            if action in ("CASH", "SAIDA_PARCIAL"):
                n_exits += 1
            elif books["policy"]["pending_exit"]:
                action = "SAIDA_PENDENTE"
        n_attempts += attempts
        n_blocked += blocks
        end_equity = {name: equity(book) for name, book in books.items()}
        ref_factor = end_equity["ref"] / initial["ref"] if initial["ref"] > 0 else 1.
        policy_factor = end_equity["policy"] / initial["policy"] if initial["policy"] > 0 else 1.
        reference_return.append(ref_factor-1)
        policy_return.append(policy_factor-1)
        daily.append({
            "session": pos.session, "symbol": pos.symbol,
            "dias_mesmo_ativo": age,
            "decisao_primeira_camada": first_layer_action,
            "dias_posicao_politica": books["policy"]["dias_posicao"],
            "estado_dados": status,
            "rebalanceamento_fechamento": close_status,
            "fechamento_executavel_verificado": False,
            "preco_fechamento_usado": close_price,
            "horario_fechamento_proxy_utc": close_timestamp,
            "quantidade_comprada_fechamento_ref": close_rebalance["ref"],
            "quantidade_comprada_fechamento_politica": close_rebalance["policy"],
            "alocacao_ref_100pct": (books["ref"]["cash"] <= 1e-7
                                   and books["ref"]["shares"] > 0),
            "alocacao_politica_100pct": (books["policy"]["cash"] <= 1e-7
                                       and books["policy"]["shares"] > 0),
            "acao": action, "tentativas_saida": attempts,
            "saidas_bloqueadas_volume_ou_preco": blocks,
            "retorno_control_intradiario": ref_factor-1,
            "retorno_politica_intradiaria": policy_factor-1,
            "control_capital": end_equity["ref"],
            "politica_capital": end_equity["policy"],
            "controle_inicio_dia": initial["ref"],
            "politica_inicio_dia": initial["policy"],
            "preco_saida": exit_price, "saida_utc": exit_time,
            "control_em_caixa": books["ref"]["cash"],
            "politica_em_caixa": books["policy"]["cash"],
            "control_quantidade": books["ref"]["shares"],
            "politica_quantidade": books["policy"]["shares"],
            "control_taxas_dia": books["ref"]["fees"]-fees_initial["ref"],
            "politica_taxas_dia": books["policy"]["fees"]-fees_initial["policy"],
            "rotacoes_estimadas_dia": day_rotations,
            "pendente_liquidacao": books["policy"]["pending_exit"]})
    d = pd.DataFrame(daily)
    risk_b, risk_p = _metricas_serie(reference_return), _metricas_serie(policy_return)
    b_cap = equity(books["ref"])
    p_cap = equity(books["policy"])
    summary = {
        "capital_inicial": capital,
        "control_intradiario_final": b_cap, "politica_final": p_cap,
        "diferenca_usd": p_cap-b_cap,
        "diferenca_pct": (p_cap/b_cap-1)*100 if b_cap > 0 else None,
        "control_max_dd": risk_b["max_drawdown"],
        "politica_max_dd": risk_p["max_drawdown"],
        "control_sharpe": risk_b["sharpe"],
        "politica_sharpe": risk_p["sharpe"],
        "saidas_antecipadas": n_exits,
        "nao_executaveis": n_sessions_no_quote,
        "tentativas_saida": n_attempts, "saidas_bloqueadas": n_blocked,
        "saidas_parciais_ao_fim_do_pregao": n_partial_exits,
        "compras_parciais_ref": books["ref"]["fills_parciais"],
        "compras_ou_vendas_parciais_politica": books["policy"]["fills_parciais"],
        "rotacoes_sem_cotacao_ticker_antigo": n_rotations,
        "sem_cotacao_fechamento": n_stale,
        "fechamentos_com_rebalanceamento": n_close_rebalances,
        "fechamentos_sem_preco": n_close_without_price,
        "fills_proxy_fechamento_sem_volume": n_proxy_close_fills,
        "cash_overnight_nao_desejado": int((
            (d.politica_em_caixa > 1e-7)
            & (~d.rebalanceamento_fechamento.isin(["sem_cotacao", "sem_preco"]))
        ).sum()),
        "taxas_ref_usd": books["ref"]["fees"],
        "taxas_politica_usd": books["policy"]["fees"],
        "posicao_ref_final": books["ref"]["shares"],
        "posicao_politica_final": books["policy"]["shares"],
        "caixa_ref_final": books["ref"]["cash"],
        "caixa_politica_final": books["policy"]["cash"],
        "dias_avaliados": len(d),
        "permanencia_minima_imposta": False,
        "primeiro_pregao_elegivel_saida": 1,
        "execucao_comprovada": False,
        "control_original_comparavel": False,
    }
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
        "tipo": "exploratorio_carteira_continua_recompra_total_no_fechamento",
        "status": "proxy_continuo_nao_equivalente_ao_control_diario_original",
        "testes": "somente pregoes fora da amostra (split do estudo)",
        "calibracao": "metade inicial da validacao; modelo aprendido no treino",
        "selecao_limiar": "metade final da validacao; jamais usa teste",
        "limiares_testados": list(THRESHOLDS),
        "custo_bps_por_lado": COST_BPS_PER_SIDE,
        "max_participacao_candle": MAX_VOLUME_PARTICIPATION,
        "aplicacao": "ativo da Winner congelado; nenhuma troca intradiaria",
        "trava": "nenhuma permanencia minima; saida intradiaria permitida no primeiro pregao",
        "permanencia_minima_imposta": False,
        "primeiro_pregao_elegivel_saida": 1,
        "observacao_idade": ("dias_posicao_politica e uma metrica de auditoria; "
                            "nao bloqueia a decisao de venda"),
        "execucao": ("open de candle posterior ao sinal; limite de volume "
                     "calculado pelo candle ANTERIOR ja encerrado"),
        "liquidez": ("entradas e saidas parciais em cada candle, ate 1% do volume da "
                     "barra anterior; sem livro de ofertas nem garantia de fill"),
        "sem_negocio": ("apenas candles observados originam ordens; se faltar cotacao "
                       "de fechamento, posicao e marcada a ultimo preco observado"),
        "extremos": ("retorno 30min absoluto >= 20% sinalizado para revisao; "
                     "nao removido do replay principal para evitar filtro ex-post"),
        "capital_inicial_cenarios": list(CAPITALS),
        "control_original_patrimonio_final_usd": ORIGINAL_CONTROL_FINAL,
        "control_original_comparavel": False,
        "posicao_overnight": ("todo patrimonio teoricamente alocado no ativo escolhido "
                              "pela camada 1 ao fim de cada pregao com cotacao"),
        "reentrada": ("a camada 2 nao recompra durante o pregao apos saida; "
                      "no fechamento a primeira camada determina recompra integral "
                      "do mesmo ativo, sem limite de capital ou volume"),
        "validacao_fill_fechamento": (
            "preco de fechamento so conhecido ao final; compra integral no mesmo preco "
            "e uma hipotese de pesquisa nao executavel apos o fechamento"),
        "camada1_dia_seguinte": "seguir a proxima selecao congelada: MANTER ou TROCAR",
        "rotacoes_entre_ativos": (
            "serie de preco antigo no novo pregao indisponivel; liquidacao "
            "integral teorica ao fechamento anterior, SEM checar fill/volume. "
            "Nao e execucao comprovada."),
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
        "nota": ("NAO comparar diretamente com US$ 587m originais. A carteira "
                 "e recomposta integralmente no fechamento pelo preco do ultimo "
                 "candle disponivel. O fill, especialmente para valores elevados, "
                 "nao e verificavel e pode ser impossivel no mercado real. "
                 "As rotacoes tambem permanecem estimativas."),
    }
    out["manifesto_politicas_intradiarias.json"] = json.dumps(
        manifesto, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return out, manifesto
