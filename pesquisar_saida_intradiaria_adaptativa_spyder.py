"""TCC - camada 2 HOLD/CASH adaptativa. Spyder F5, somente dados historicos.
Reutiliza a coleta e o snapshot da pesquisa anterior; nao altera Winner.
Execucao modelada na abertura seguinte (primeiro fill possivel 09:35 NY),
sem recompra no dia da saida, sem fill retroativo no fechamento.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import os
import traceback
import logging

import numpy as np
import pandas as pd
import pesquisar_segunda_camada_intradiaria_spyder as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pesquisa_saida_intradiaria_adaptativa.zip"
LOG = OUT.with_suffix(".log")
SEED = 20261123
CAPITAL = 10000.
COST = .001  # custo por lado, nao gatilho de stop
PARTICIPATION = .01  # proxy de capacidade usando volume ja conhecido
METHODS = ("kalman", "multirrolling", "bocpd", "lightgbm", "kalman_lightgbm")
RISKS = (0., .25, .5, 1.)
FEATURES = list(base.FEATURES)
KFEATURES = ["k_slope", "k_accel", "k_gap"]
LOGGER = logging.getLogger("intraday_adaptive")


def log(message):
    LOGGER.info(message)


def csv(d):
    return d.to_csv(index=False).encode("utf-8")


def pack(files):
    hashes = {k: hashlib.sha256(v).hexdigest() for k,v in files.items()}
    files["sha256.json"] = json.dumps(hashes, indent=2).encode()
    tmp = OUT.with_suffix(".zip.tmp")
    with ZipFile(tmp, "w", ZIP_DEFLATED) as z:
        for k,v in files.items():
            z.writestr(k,v)
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in hashes.items())
    os.replace(tmp, OUT)


def load_data():
    import exchange_calendars as xc
    pos, origin = base.preparar_posicoes_control()
    pos = pos[["session","symbol"]].sort_values("session").reset_index(drop=True)
    if pos.empty or pos.session.duplicated().any():
        raise ValueError("Calendario Control invalido")
    pos["previous"] = pos.symbol.shift()
    log(f"Control: {len(pos)} sessoes; {pos.symbol.nunique()} ativos")
    main_report = base.preencher_cache_control(pos)
    # O ticker anterior precisa de cotacao REAL no dia da rotacao.
    # Cache separado: os CSVs antigos foram filtrados pelas sessoes do target.
    swaps = pos[pos.previous.notna() & pos.previous.ne(pos.symbol)][["session","previous"]]
    swaps = swaps.rename(columns={"previous":"symbol"})
    original = base.INPUT
    try:
        base.INPUT = original / "rotacoes_adaptativas"
        rotation_report = base.preencher_cache_control(swaps)
    finally:
        base.INPUT = original
    paths = sorted((original/"cotacoes_5min").glob("*.csv"))
    paths += sorted((original/"rotacoes_adaptativas"/"cotacoes_5min").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("Sem candles reais; confira cache e leitor Alpaca local")
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    raw["symbol"] = raw.symbol.astype(str).str.strip().str.upper()
    raw = raw.drop_duplicates()
    if raw.duplicated(["symbol","timestamp"]).any():
        raise ValueError("Candles conflitantes nas fontes")
    bars = base.preparar(raw, informar_lacunas=True)
    if bars.empty:
        raise ValueError("Sem barras validas")
    cal = xc.get_calendar("XNYS", start=pos.session.min(), end=pos.session.max())
    closes = {}
    for day in pos.session:
        label = pd.Timestamp(day)
        if not cal.is_session(label):
            raise ValueError(f"Control contem dia sem sessao XNYS: {day}")
        closes[day] = cal.session_close(label)
    bars["end"] = bars.session.map(closes)
    bars = bars[bars.end.notna() & (bars.timestamp < bars.end)].copy()
    # Uma grade por ticker/sessao; nenhuma interpolacao.
    sessions = {(str(s),str(a)): g.sort_values("timestamp").reset_index(drop=True)
                for (s,a),g in bars.groupby(["session","symbol"],sort=False)}
    coverage = []
    for row in pos.itertuples(index=False):
        g = sessions.get((row.session,row.symbol))
        old = sessions.get((row.session,row.previous)) if pd.notna(row.previous) else None
        switch = pd.notna(row.previous) and row.previous!=row.symbol
        n = int(g.close.notna().sum()) if g is not None else 0
        coverage.append(dict(session=row.session,symbol=row.symbol,
            expected_bars=int((closes[row.session]-cal.session_open(pd.Timestamp(row.session))).total_seconds()/300),
            observed_bars=n,
            final_bar=bool(g is not None and pd.notna(g.close.iloc[-1])),
            rotation_quote=bool(not switch or (old is not None and old.open.notna().any()))))
    source = dict(origin=origin, target_download=main_report, rotation_download=rotation_report,
                  source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return pos, bars, sessions, pd.DataFrame(coverage), source


def kalman_features(g):
    # Estado local linear [log-preco, velocidade]; filtro forward apenas.
    # Razao Q/R predefinida; escala atual usa somente retornos observados.
    x = None
    P = np.eye(2)
    F = np.array([[1.,1.],[0.,1.]])
    H = np.array([1.,0.])
    prev = np.nan
    result = []
    c = np.log(g.close.astype(float))
    scale = c.diff().rolling(24,min_periods=6).std().shift(1)
    for price,vol in zip(c,scale):
        if not np.isfinite(price):
            x = None
            prev = np.nan
            result.append((np.nan,np.nan,np.nan))
            continue
        if x is None:
            x = np.array([price,0.])
            P = np.eye(2)*.001
        if not np.isfinite(vol) or vol<=0:
            result.append((np.nan,np.nan,np.nan))
            continue
        R = max(float(vol)**2,1e-10)
        pred = F @ x
        P = F @ P @ F.T + R*np.diag([.05,.005])
        gain = P @ H / (H @ P @ H + R)
        x = pred + gain*(price-H@pred)
        P = (np.eye(2)-np.outer(gain,H)) @ P
        slope = x[1]/np.sqrt(R)
        result.append((slope,slope-prev if np.isfinite(prev) else np.nan,
                       (price-x[0])/np.sqrt(R)))
        prev = slope
    return pd.DataFrame(result,index=g.index,columns=KFEATURES)


def frame_features(bars,pos):
    selected = bars.merge(pos[["session","symbol"]],on=["session","symbol"],how="inner")
    if selected.empty:
        raise ValueError("Nenhuma barra do Control disponivel")
    f = base.features(selected).reset_index(drop=True)
    k = pd.concat([kalman_features(g) for _,g in f.groupby(["symbol","session"],sort=False)])
    f = f.join(k)
    f["rolling"] = base.rolling_scores(f)
    # Label: retorno de MANTER desde o primeiro open posterior ate o close
    # regular da sessao. Ultima barra ausente => label ausente, jamais ffill.
    grouped = f.groupby(["symbol","session"],sort=False)
    next_open = grouped.open.shift(-1)
    final_close = grouped.close.transform(lambda s:s.iloc[-1])
    f["target"] = np.log(final_close/next_open)
    f["downside"] = (-f.target).clip(lower=0)
    f["horario"] &= grouped.timestamp.shift(-1).notna()
    return f


def split(pos):
    days = pos.session.tolist()
    a,b = int(.60*len(days)),int(.80*len(days))
    if a<20 or b-a<10 or len(days)-b<10:
        raise ValueError("Amostra insuficiente")
    # Uma sessao de embargo entre treino, validacao e teste.
    return days[:a-1], days[a:b-1], days[b:]


def make_scores(f, train_days, val_days, test_days):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    tr = f[f.session.isin(train_days)]
    variance = tr.r1.dropna().var()
    if not np.isfinite(variance) or variance<=0:
        raise ValueError("Variancia de treino invalida")
    # Filtro causal reiniciado por sessao; parametro estimado so no treino.
    f = f.copy()
    f["bocpd"] = base.bocpd(f, variance)
    tr = f[f.session.isin(train_days) & f.horario & f.target.notna()]
    ev = f[f.session.isin(val_days+test_days) & f.horario].copy()
    if len(tr)<100 or ev.empty:
        raise ValueError("Checkpoints reais insuficientes")
    cols = {"kalman":KFEATURES+["vol12","time_frac"],
            "multirrolling":["rolling","accel","vol12","time_frac"],
            "bocpd":["bocpd","vol12","time_frac"],
            "lightgbm":FEATURES,
            "kalman_lightgbm":FEATURES+KFEATURES+["rolling","bocpd"]}
    out = {}
    for name in METHODS:
        log(f"Treinando {name}: {len(tr)} checkpoints")
        x = tr[cols[name]].replace([np.inf,-np.inf],np.nan)
        z = ev[cols[name]].replace([np.inf,-np.inf],np.nan)
        preds = ev[["session","symbol","timestamp","target"]].copy()
        for label in ("target","downside"):
            model = (LGBMRegressor(n_estimators=100,num_leaves=7,max_depth=3,
                     learning_rate=.04,min_child_samples=35,random_state=SEED,
                     n_jobs=1,verbosity=-1) if "lightgbm" in name else
                     make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=20.)))
            model.fit(x,tr[label])
            preds["mean" if label=="target" else "risk"] = model.predict(z)
        preds["risk"] = preds.risk.clip(lower=0)
        # Nao autorizar ordem com estado de tendencia completamente ausente.
        essential = {"kalman":"k_slope","multirrolling":"rolling","bocpd":"bocpd",
                     "lightgbm":"r6","kalman_lightgbm":"k_slope"}[name]
        preds = preds[z[essential].notna()] if essential in z else preds
        out[name] = preds
    return out


def signal_map(preds, risk):
    if risk is None:
        return {}
    # Permanecer vs vender: vender custa COST; sem stop por % de preco.
    score = preds["mean"] - risk*preds["risk"] - np.log(1-COST)
    return {(r.session,r.symbol,r.timestamp):float(v)
            for r,v in zip(preds.itertuples(index=False),score) if np.isfinite(v)}


def replay(pos,sessions,signals,capital=CAPITAL,cost=COST):
    cash,qty,held,mark = float(capital),0.,None,None
    equity0 = float(capital)
    peak = equity0
    daily,events = [],[]
    missing_marks = 0
    for row in pos.itertuples(index=False):
        day,target = row.session,row.symbol
        tg = sessions.get((day,target))
        old = sessions.get((day,held)) if held is not None else None
        by_symbol = {}
        for sym,g in ((target,tg),(held,old)):
            if sym is not None and g is not None:
                by_symbol[sym] = g.set_index("timestamp")
        stamps = sorted(set().union(*(set(g.index) for g in by_symbol.values())))
        pending = False
        triggered = False
        fees = 0.
        proceeds = 0.
        exit_qty = 0.
        exit_count = 0
        blocked_rotation = False
        prev_equity = cash + qty*(mark or 0.)
        peak_before = peak
        def valid(v):
            return pd.notna(v) and np.isfinite(v) and v>0
        for t in stamps:
            ny = t.tz_convert("America/New_York")
            opening = ny.hour*60+ny.minute < 630
            def quote(sym):
                g = by_symbol.get(sym)
                prior = t-pd.Timedelta(minutes=5)
                if g is None or t not in g.index:
                    return None,0.
                p = g.loc[t,"open"]
                volume = g.loc[prior,"volume"] if prior in g.index else np.nan
                return (float(p) if valid(p) else None,
                        PARTICIPATION*float(volume) if valid(volume) else 0.)
            # Rotacao somente na janela de abertura, vender ticker antigo antes de comprar.
            if qty>1e-10 and held!=target and opening:
                px,capacity = quote(held)
                if px is not None:
                    mark = px
                    q = min(qty,capacity)
                    if q>0:
                        fee = q*px*cost
                        cash += q*px-fee; qty -= q; fees += fee
                        events.append(dict(session=day,symbol=held,timestamp=t,side="sell_rotation",qty=q,price=px,fee=fee))
                    if qty<=1e-10:
                        qty=0.;held=None;mark=None
            # Saida da camada 2: ordem posterior ao sinal; fills parciais.
            if pending and qty>1e-10 and held==target:
                px,capacity = quote(held)
                if px is not None:
                    mark=px
                    q=min(qty,capacity)
                    if q>0:
                        fee=q*px*cost
                        cash+=q*px-fee;qty-=q;fees+=fee
                        proceeds+=q*px-fee;exit_qty+=q;exit_count+=1
                        events.append(dict(session=day,symbol=held,timestamp=t,side="sell_protection",qty=q,price=px,fee=fee))
                    if qty<=1e-10:
                        qty=0.;held=None;mark=None
            # A camada 1 pode alocar apenas na abertura; nunca recomprar apos sinal.
            if opening and not triggered and held in (None,target) and cash>1e-8:
                px,capacity=quote(target)
                if px is not None:
                    q=min(cash/(px*(1+cost)),capacity)
                    if q>0:
                        fee=q*px*cost
                        cash-=q*px+fee;qty+=q;held=target;mark=px;fees+=fee
                        events.append(dict(session=day,symbol=target,timestamp=t,side="buy_winner",qty=q,price=px,fee=fee))
            # Barra atual encerrou. O score nunca executa no seu proprio open.
            if qty>1e-10 and held is not None:
                g=by_symbol.get(held)
                if g is not None and t in g.index and valid(g.loc[t,"close"]):
                    mark=float(g.loc[t,"close"])
                value=cash+qty*(mark or 0.)
                peak=max(peak,value)
                s=signals.get((day,held,t))
                if held==target and not triggered and s is not None and s<0:
                    triggered=True;pending=True
                    events.append(dict(session=day,symbol=held,timestamp=t+pd.Timedelta(minutes=5),
                                       side="signal_exit",qty=0.,price=np.nan,fee=0.,score=s))
        if qty>1e-10:
            g=by_symbol.get(held)
            if g is None or g.empty or not valid(g.close.iloc[-1]):
                missing_marks+=1
            blocked_rotation=held!=target
        equity=cash+qty*(mark or 0.)
        peak=max(peak,equity)
        last_close=float(tg.close.iloc[-1]) if tg is not None and valid(tg.close.iloc[-1]) else np.nan
        benefit=proceeds-exit_qty*last_close if exit_qty>0 and np.isfinite(last_close) else np.nan
        daily.append(dict(session=day,target=target,held=held,cash=cash,shares=qty,
            equity=equity,ret=equity/prev_equity-1,drawdown=equity/peak-1,
            fees=fees,signal=triggered,exit_fills=exit_count,exit_qty=exit_qty,
            exit_cash_advantage_vs_close=benefit,blocked_rotation=blocked_rotation,
            stale_mark=bool(qty>0 and (by_symbol.get(held) is None or
                      not valid(by_symbol[held].close.iloc[-1])))))
    d=pd.DataFrame(daily)
    r=d.ret
    final=float(d.equity.iloc[-1])
    dd=float((d.equity/pd.Series(np.maximum.accumulate(np.r_[equity0,d.equity])[1:],
             index=d.index)-1).min())
    stat=dict(capital_initial=capital,capital_final=final,return_pct=100*(final/capital-1),
              max_drawdown_daily=dd,sharpe=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else None,
              fees=float(d.fees.sum()),signals=int(d.signal.sum()),
              sessions_with_exit=int(d.exit_fills.gt(0).sum()),
              positive_exits=int(d.exit_cash_advantage_vs_close.gt(0).sum()),
              premature_exits=int(d.exit_cash_advantage_vs_close.lt(0).sum()),
              blocked_rotations=int(d.blocked_rotation.sum()),stale_marks=missing_marks,
              sessions=len(d))
    return stat,d,pd.DataFrame(events)


def smoke():
    # Sinal de 10:25 (disponivel 10:30) nao pode executar a 10:25.
    ts=pd.date_range("2026-07-28T13:30:00Z",periods=78,freq="5min")
    g=pd.DataFrame(dict(timestamp=ts,open=100.,close=100.,volume=100000.))
    g.loc[12:,["open","close"]]=90.
    p=pd.DataFrame(dict(session=["2026-07-28"],symbol=["DEMO"]))
    m={("2026-07-28","DEMO"):g}
    _,d,e=replay(p,m,{("2026-07-28","DEMO",ts[11]):-1.})
    sell=e[e.side=="sell_protection"]
    assert len(sell)>0 and sell.timestamp.min()>=ts[12]
    assert not (e[(e.timestamp>=ts[12])].side=="buy_winner").any()
    assert d.cash.iloc[-1]>0 and d.shares.iloc[-1]<1e-8
    a,_,_=replay(p,m,{})
    b,_,_=replay(p,m,{("2026-07-28","DEMO",ts[11]):1.})
    assert a==b
    # Prefix invariance: future modification cannot change Kalman history.
    full=g.copy();full.loc[40:,"close"]*=2
    np.testing.assert_allclose(kalman_features(g).iloc[:40],
                               kalman_features(full).iloc[:40],equal_nan=True)
    return "synthetic_passed_not_market_result"


def run():
    files={}
    smoke_result=smoke()
    log(f"Testes sinteticos: {smoke_result}")
    pos,bars,sessions,coverage,source=load_data()
    files["auditoria_cobertura.csv"]=csv(coverage)
    f=frame_features(bars,pos)
    tr,va,te=split(pos)
    log(f"Treino {len(tr)}; validacao {len(va)}; teste {len(te)}")
    pred=make_scores(f,tr,va,te)
    select=pos[pos.session.isin(va)]
    test=pos[pos.session.isin(te)]
    grid=[];summary=[];curves=[];events=[];forecasts=[]
    baseline,bd,be=replay(test,sessions,{})
    summary.append(dict(method="Control_intraday_replay",risk=None,**baseline))
    curves.append(bd.assign(method="Control_intraday_replay"))
    if not be.empty:events.append(be.assign(method="Control_intraday_replay"))
    selected={}
    for name,p in pred.items():
        v=p[p.session.isin(va)];t=p[p.session.isin(te)]
        candidates=[]
        for risk in (None,*RISKS):
            stat,_,_=replay(select,sessions,signal_map(v,risk))
            objective=np.log(stat["capital_final"]/CAPITAL)+.5*stat["max_drawdown_daily"]
            candidates.append((objective,risk))
            grid.append(dict(method=name,risk=risk,objective=objective,**stat))
        # None = sem saida, primeiro candidato; empate favorece nao operar.
        best=max(candidates,key=lambda x:x[0])[1]
        selected[name]=best
        stat,d,e=replay(test,sessions,signal_map(t,best))
        actual=t.target.notna()
        stat["prediction_mae"]=float((t.loc[actual,"mean"]-t.loc[actual,"target"]).abs().mean()) if actual.any() else None
        summary.append(dict(method=name,risk=best,**stat))
        curves.append(d.assign(method=name))
        if not e.empty:events.append(e.assign(method=name))
        forecasts.append(t.assign(method=name))
        log(f"{name}: validacao risco={best}; teste capital={stat['capital_final']:.2f}; saidas={stat['sessions_with_exit']}")
    result=pd.DataFrame(summary)
    result["delta_control_usd"]=result.capital_final-baseline["capital_final"]
    result["delta_control_pct"]=100*(result.capital_final/baseline["capital_final"]-1)
    # Sensibilidade de custos: politica congelada, jamais reotimizar no teste.
    stress=[]
    for bps in (5,10,25,50):
        for name in ("Control_intraday_replay",*METHODS):
            sig={} if name=="Control_intraday_replay" else signal_map(pred[name][pred[name].session.isin(te)],selected[name])
            st,_,_=replay(test,sessions,sig,cost=bps/10000)
            stress.append(dict(method=name,cost_bps=bps,**st))
    files.update({"comparacao_estrategias.csv":csv(result),
                  "selecao_validacao.csv":csv(pd.DataFrame(grid)),
                  "patrimonio_diario.csv":csv(pd.concat(curves,ignore_index=True)),
                  "ordens_simuladas.csv":csv(pd.concat(events,ignore_index=True) if events else pd.DataFrame()),
                  "previsoes_teste.csv":csv(pd.concat(forecasts,ignore_index=True)),
                  "sensibilidade_custos.csv":csv(pd.DataFrame(stress))})
    cov=coverage[coverage.session.isin(te)]
    status={"status":"completed_proxy" if cov.final_bar.all() and cov.rotation_quote.all()
            and not result.stale_marks.any() and not result.blocked_rotations.any() else "partial_data_or_execution_proxy",
            "smoke":smoke_result,"source":source,
            "split":{"train":tr,"validation":va,"test":te},
            "capital_initial":CAPITAL,"cost_per_side":COST,"volume_participation_proxy":PARTICIPATION,
            "selected_risk":selected,"first_layer":"frozen historical target assumed known before open; timestamps not certified",
            "cash_rule":"no reentry same day; next session opening Winner target; no close fill",
            "control":"same continuous replay, no exit overlay; NOT original daily Control wealth",
            "limitations":["research proxy, no real orders", "next-open prices and previous bar volume do not guarantee fills",
                           "no dividends/corporate actions in RAW intraday replay; gaps across splits may distort wealth",
                           "opening-only model; no independent closing Winner signal in position snapshot",
                           "historical target pre-open availability not independently verified",
                           "stale marks and failed rotations reported; do not promote partial results"],
            "test_sessions_missing_final":int((~cov.final_bar).sum()),
            "test_missing_rotation_quote":int((~cov.rotation_quote).sum())}
    files["manifesto.json"]=json.dumps(status,ensure_ascii=False,indent=2,default=str).encode()
    return files


if __name__=="__main__":
    OUT.parent.mkdir(parents=True,exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    for handler in (logging.FileHandler(LOG,mode="w",encoding="utf-8"),logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    try:
        files=run()
        log("Pesquisa concluida. Gravando ZIP.")
        files["execucao.log"]=LOG.read_bytes()
        files["codigo_executado.py"]=Path(__file__).read_bytes()
        pack(files)
        print("ZIP:",OUT)
    except Exception:
        LOGGER.exception("Falha na pesquisa; resultado anterior preservado.")
        raise
