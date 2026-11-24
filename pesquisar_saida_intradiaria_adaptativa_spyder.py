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
import io
import sys

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
METHODS = ("kalman", "multirrolling", "bocpd", "lightgbm", "kalman_lightgbm", "lightgbm_excursions")
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


# Arquivo exportado pelo MCT: coloque na raiz, ou informe --mct-zip CAMINHO.
# Nunca usar silenciosamente as posicoes congeladas antigas.
MCT_ZIP = ROOT / "market_cycle_trader_20261124T005959-1a1fa2f1.zip"
CACHE = ROOT / "dados_intraday" / "raw_sip_5min_full_month"


def load_reference():
    arg = sys.argv.index("--mct-zip") if "--mct-zip" in sys.argv else None
    path = Path(sys.argv[arg+1]).expanduser() if arg is not None else MCT_ZIP
    if not path.is_file():
        candidates = sorted(ROOT.glob("market_cycle_trader_*.zip"))
        if arg is None and len(candidates)==1:
            path = candidates[0]
        else:
            raise FileNotFoundError(f"Coloque o ZIP MCT na raiz do TCC: {path}")
    with ZipFile(path) as z:
        def read(suffix):
            names = [n for n in z.namelist() if n.endswith(suffix)]
            if len(names)!=1:
                raise ValueError(f"Esperado exatamente um arquivo {suffix}")
            return z.read(names[0])
        manifest = json.loads(read("strategy_manifest.json"))
        metrics = json.loads(read("_metrics.json"))
        p = pd.read_csv(io.BytesIO(read("_predictions.csv")))
        t = pd.read_csv(io.BytesIO(read("_trades.csv")))
    if manifest["status"]!="completed" or not metrics["portfolio_rotation"]:
        raise ValueError("Exportacao nao concluida ou nao e portfolio")
    def session(x):
        return pd.to_datetime(x,utc=True,errors="raise").dt.tz_convert("America/New_York").dt.date.astype(str)
    p["session"] = session(p.timestamp)
    p["decision_session"] = session(p.decision_date)
    p = p.sort_values("session").reset_index(drop=True)
    p["symbol"] = p.selected_asset.astype(str).str.upper()
    p["previous"] = p.symbol.shift()
    if p.session.duplicated().any() or not (p.decision_session<p.session).all():
        raise ValueError("Sessoes duplicadas ou decisao nao anterior ao pregao")
    if not p.symbol.isin(metrics["assets"]).all():
        raise ValueError("Este replay requer Control sem CASH no historico exportado")
    if not np.isfinite(p.strategy_equity).all() or (p.strategy_equity<=0).any():
        raise ValueError("Patrimonio Control invalido")
    initial = float(metrics["initial_capital"])
    if not np.isclose(p.strategy_equity.iloc[-1],metrics["strategy_ending_capital"],rtol=1e-10):
        raise ValueError("Curva oficial diverge das metricas")
    t["session"] = session(t.timestamp)
    buys = t[t.action.eq("BUY")].set_index("session")
    sells = t[t.action.eq("SELL")].set_index("session")
    final = t[t.action.eq("FINAL_SELL")].set_index("session")
    if buys.index.duplicated().any() or sells.index.duplicated().any() or final.index.duplicated().any():
        raise ValueError("Mais de uma operacao da mesma natureza no pregao")
    p["ref_qty"] = p.session.map(buys.quantity).ffill()
    p["ref_cash"] = p.session.map(buys.cash_after_trade).ffill()
    p["ref_close"] = (p.strategy_equity-p.ref_cash)/p.ref_qty
    p["ref_close"] = p.session.map(final.execution_price).combine_first(p.ref_close)
    p["ref_buy_open"] = p.session.map(buys.execution_price)
    p["ref_sell_open"] = p.session.map(sells.execution_price)
    p["ref_previous_equity"] = p.strategy_equity.shift(fill_value=initial)
    p["ref_factor"] = p.strategy_equity/p.ref_previous_equity
    p["ref_previous_close"] = p.ref_close.shift()
    if not np.isfinite(p.ref_close).all() or (p.ref_close<=0).any():
        raise ValueError("Nao foi possivel reconstruir marca diaria Control")
    source = {"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "job_id":manifest["job_id"],"strategy":manifest["strategy_profile_name"],
              "revision":manifest["strategy_profile_revision"],"assets":len(metrics["assets"]),
              "sessions":len(p),"initial":initial,"official_final":float(p.strategy_equity.iloc[-1]),
              "official_max_dd":metrics["strategy_maximum_drawdown"],
              "decision_precedes_session":True}
    log(f"Referencia MCT: {source}")
    return p, source


class SessionStore(dict):
    """Cache mensal INTEGRAL. Busca tardia para qualquer posicao residual.
    Nunca limita a coleta aos dias em que o ticker e o alvo Winner.
    """
    def __init__(self, calendar):
        super().__init__()
        self.calendar=calendar
        self.months=set()
        self.missing=set()
        self.hashes={}
        self.errors=[]
        self.reader=None

    def get(self,key,default=None):
        day,symbol=key
        if not symbol or pd.isna(symbol):
            return default
        if key in self:
            return super().get(key,default)
        month=day[:7]
        pair=(symbol,month)
        if pair not in self.months:
            self.months.add(pair)
            try:
                self.load_month(symbol,month)
            except Exception as exc:
                self.errors.append({"symbol":symbol,"month":month,"error":type(exc).__name__})
                log(f"Falha de dados {symbol}/{month}: {type(exc).__name__}")
        if key not in self:
            self.missing.add(key)
        return super().get(key,default)

    def load_month(self,symbol,month):
        CACHE.mkdir(parents=True,exist_ok=True)
        path=CACHE/f"{symbol}_{month}.csv"
        if not path.exists():
            if self.reader is None:
                self.reader=base._market_reader_local()
            download,creds=self.reader
            start=pd.Timestamp(month+"-01",tz="America/New_York")
            end=start+pd.offsets.MonthBegin(1)
            log(f"Alpaca RAW SIP 5Min: {symbol}/{month} (mes completo)")
            raw=download(**creds,symbol=symbol,timeframe="5Min",
                         start=start.tz_convert("UTC"),end=end.tz_convert("UTC"),
                         feed="sip",adjustment="raw")
            if raw is None or raw.empty:
                raise ValueError("Sem cotacoes")
            raw=raw.reset_index()
            raw["symbol"]=symbol
            raw=raw[["timestamp","symbol","open","high","low","close","volume"]]
            tmp=path.with_suffix(".tmp")
            raw.to_csv(tmp,index=False)
            os.replace(tmp,path)
        self.hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        raw=pd.read_csv(path)
        if raw.empty:
            raise ValueError("Cache mensal vazio")
        bars=base.preparar(raw)
        if bars.empty:
            return
        for day,g in bars.groupby("session",sort=False):
            label=pd.Timestamp(day)
            if not self.calendar.is_session(label):
                continue
            end=self.calendar.session_close(label)
            g=g[g.timestamp<end].sort_values("timestamp").reset_index(drop=True)
            if not g.empty:
                self[(day,symbol)]=g


def load_data():
    import exchange_calendars as xc
    pos, source = load_reference()
    # Margem de um mes para os arquivos mensais completos nas duas pontas.
    cal=xc.get_calendar("XNYS",start=pd.Timestamp(pos.session.min())-pd.Timedelta(days=35),
                        end=pd.Timestamp(pos.session.max())+pd.Timedelta(days=35))
    sessions=SessionStore(cal)
    coverage=[];parts=[]
    for row in pos.itertuples(index=False):
        label=pd.Timestamp(row.session)
        if not cal.is_session(label):
            raise ValueError(f"Data Control fora do calendario XNYS: {row.session}")
        g=sessions.get((row.session,row.symbol))
        old=sessions.get((row.session,row.previous)) if pd.notna(row.previous) and row.previous!=row.symbol else g
        if g is not None:parts.append(g)
        close=float(g.close.iloc[-1]) if g is not None and pd.notna(g.close.iloc[-1]) else np.nan
        coverage.append(dict(session=row.session,symbol=row.symbol,
          expected_bars=int((cal.session_close(label)-cal.session_open(label)).total_seconds()/300),
          observed_bars=int(g.close.notna().sum()) if g is not None else 0,
          final_bar=bool(np.isfinite(close)),
          reference_close=row.ref_close,observed_close=close,
          close_difference_pct=100*(close/row.ref_close-1) if np.isfinite(close) else np.nan,
          rotation_quote=bool(old is not None and old.open.notna().any())))
    if not parts:raise FileNotFoundError("Sem candles intradiarios; confira credenciais e cache")
    return pos,pd.concat(parts,ignore_index=True),sessions,pd.DataFrame(coverage),source


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
            x = np.array([price,0.])
            P = np.eye(2)*.001
            prev = np.nan
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
    f["peak_drawdown"] = f.close/grouped.high.cummax()-1
    f["efficiency"] = grouped.close.transform(lambda c:
        (c-c.shift(12)).abs()/c.diff().abs().rolling(12,min_periods=12).sum().replace(0,np.nan))
    f["mfe"] = np.nan
    f["mae"] = np.nan
    # Futuro apenas nos alvos, requer barras completas ate o fechamento.
    for _,g in f.groupby(["symbol","session"],sort=False):
        hi=g.high.to_numpy(float);lo=g.low.to_numpy(float);op=g.open.to_numpy(float)
        for i in range(len(g)-1):
            if np.isfinite(op[i+1]) and np.isfinite(hi[i+1:]).all() and np.isfinite(lo[i+1:]).all():
                f.loc[g.index[i],"mfe"]=max(0.,np.log(hi[i+1:].max()/op[i+1]))
                f.loc[g.index[i],"mae"]=max(0.,np.log(op[i+1]/lo[i+1:].min()))
    f["horario"] &= grouped.timestamp.shift(-1).notna()
    return f


def split(pos):
    f1=pos.loc[pos.walk_forward_fold.eq(1),"session"].tolist()
    f2=pos.loc[pos.walk_forward_fold.eq(2),"session"].tolist()
    f3=pos.loc[pos.walk_forward_fold.eq(3),"session"].tolist()
    if min(len(f1),len(f2),len(f3))<20:
        raise ValueError("Esperados os tres folds da exportacao MCT")
    mid=len(f2)//2
    return f1+f2[:mid-1], f2[mid:-1], f3


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
            "kalman_lightgbm":FEATURES+KFEATURES+["rolling","bocpd"],
            "lightgbm_excursions":FEATURES+KFEATURES+["rolling","bocpd","peak_drawdown","efficiency"]}
    out = {}
    for name in METHODS:
        log(f"Treinando {name}: {len(tr)} checkpoints")
        x = tr[cols[name]].replace([np.inf,-np.inf],np.nan)
        z = ev[cols[name]].replace([np.inf,-np.inf],np.nan)
        preds = ev[["session","symbol","timestamp","target"]].copy()
        labels=("target","mae","mfe") if name=="lightgbm_excursions" else ("target","downside")
        for label in labels:
            model = (LGBMRegressor(n_estimators=100,num_leaves=7,max_depth=3,
                     learning_rate=.04,min_child_samples=35,random_state=SEED,
                     n_jobs=1,verbosity=-1) if "lightgbm" in name else
                     make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=20.)))
            valid=tr[label].notna()
            if valid.sum()<100:raise ValueError(f"Amostra insuficiente {name}/{label}")
            model.fit(x.loc[valid],tr.loc[valid,label])
            preds[{"target":"mean","downside":"risk","mae":"risk","mfe":"expected_mfe"}[label]]=model.predict(z)
        preds["actual_mae"]=ev.mae
        preds["actual_mfe"]=ev.mfe
        preds["risk"] = preds.risk.clip(lower=0)
        # Nao autorizar ordem com estado de tendencia completamente ausente.
        essential = {"kalman":"k_slope","multirrolling":"rolling","bocpd":"bocpd",
                     "lightgbm":"r6","kalman_lightgbm":"k_slope","lightgbm_excursions":"k_slope"}[name]
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
    noisy=g.copy()
    rng=np.random.default_rng(SEED)
    noisy["close"]=100*np.exp(np.cumsum(rng.normal(0,.003,len(g))))
    full=noisy.copy();full.loc[40:,"close"]*=2
    past=kalman_features(noisy).iloc[:40]
    assert past.k_slope.notna().any()
    np.testing.assert_allclose(past,kalman_features(full).iloc[:40],equal_nan=True)
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
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6))
    for d in curves:
        ax.plot(pd.to_datetime(d.session),d.equity,label=d.method.iloc[0])
    ax.set_title("Patrimonio simulado fora da amostra (proxy intradiario)")
    ax.set_ylabel("USD");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    buffer=io.BytesIO();fig.savefig(buffer,format="png",dpi=150);plt.close(fig)
    files["comparacao_patrimonio.png"]=buffer.getvalue()
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
        import sys
        if "--self-test" in sys.argv:
            print(smoke())
            sys.exit(0)
        files=run()
        log("Pesquisa concluida. Gravando ZIP.")
        files["execucao.log"]=LOG.read_bytes()
        files["codigo_executado.py"]=Path(__file__).read_bytes()
        pack(files)
        print("ZIP:",OUT)
    except Exception:
        LOGGER.exception("Falha na pesquisa; resultado anterior preservado.")
        raise
