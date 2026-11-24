"""TCC - camada 2 HOLD/CASH adaptativa. Spyder F5, somente dados historicos.
Reutiliza cache SPLIT/SIP e exportacao oficial do MCT; nao altera Winner.
Uma venda de protecao e recompra programada no open da ultima barra.
Nao presume sinal Winner independente antes do fechamento.
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
METHODS = ("kalman", "multirrolling", "bocpd", "lightgbm", "kalman_lightgbm", "lightgbm_excursions", "cusum_adaptive", "quantile_lightgbm", "regime_recovery", "recovery_survival")
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
CACHE = ROOT / "dados_intraday" / "split_sip_5min_full_month"
FREQUENCIES = (5, 15, 30, 60)
PRICE_TOLERANCE = .03  # auditoria de escala, nao gatilho de negociacao


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
    def __init__(self, calendar, last_session):
        super().__init__()
        self.calendar=calendar
        self.last_session=str(last_session)
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
                # Registrar mensagem sem credenciais ou headers de autenticacao.
                import re
                msg=str(exc)
                msg=re.sub(r"(?i)(api[_ -]?key|secret|authorization|token)(\\s*[:=]\\s*)[^\\s,;]+",r"\\1\\2[redacted]",msg)
                if self.reader is not None:
                    for secret in self.reader[1].values():
                        if secret:msg=msg.replace(str(secret),"[redacted]")
                msg=msg[:500]
                self.errors.append({"symbol":symbol,"month":month,"error":type(exc).__name__,"message":msg})
                log(f"Falha de dados {symbol}/{month}: {type(exc).__name__}: {msg}")
        if key not in self:
            self.missing.add(key)
        return super().get(key,default)

    def load_month(self,symbol,month):
        CACHE.mkdir(parents=True,exist_ok=True)
        start=pd.Timestamp(month+"-01",tz="America/New_York").tz_convert("UTC")
        cutoff=self.calendar.session_close(pd.Timestamp(self.last_session))
        end=min((start.tz_convert("America/New_York")+pd.offsets.MonthBegin(1)).tz_convert("UTC"),cutoff)
        if end<=start:
            raise ValueError("Mes fora do recorte do backtest")
        suffix=f"_{self.last_session}" if month==self.last_session[:7] else ""
        path=CACHE/f"{symbol}_{month}{suffix}.csv"
        if not path.exists():
            if self.reader is None:
                self.reader=base._market_reader_local()
            download,creds=self.reader
            log(f"Alpaca SPLIT SIP 5Min: {symbol}/{month} ate {end}")
            import time
            for attempt in range(3):
                try:
                    raw=download(**creds,symbol=symbol,timeframe="5Min",
                                 start=start,end=end,feed="sip",adjustment="split")
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2**attempt)
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
            if not g.empty and (day,symbol) not in self:
                self[(day,symbol)]=g


def load_data():
    import exchange_calendars as xc
    pos, source = load_reference()
    # Margem de um mes para os arquivos mensais completos nas duas pontas.
    cal=xc.get_calendar("XNYS",start=pd.Timestamp(pos.session.min())-pd.Timedelta(days=35),
                        end=pd.Timestamp(pos.session.max())+pd.Timedelta(days=35))
    # Nao misturar caches RAW e SPLIT. O cache RAW anterior fica intacto.
    sessions=SessionStore(cal,pos.session.max())
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
          scale_ok=bool(np.isfinite(close) and abs(close/row.ref_close-1)<=PRICE_TOLERANCE),
          rotation_quote=bool(old is not None and old.open.notna().any())))
    if not parts:raise FileNotFoundError("Sem candles intradiarios; confira credenciais e cache")
    audit=pd.DataFrame(coverage)
    source["intraday_adjustment"]="split"
    source["price_scale_tolerance"]=PRICE_TOLERANCE
    sessions.scale_ok={(r.session,r.symbol):r.scale_ok for r in audit.itertuples(index=False)}
    return pos,pd.concat(parts,ignore_index=True),sessions,audit,source


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
    # Label: retorno de MANTER desde o primeiro open posterior ate a recompra programada.
    # Open da ultima barra ausente => label ausente, jamais ffill.
    grouped = f.groupby(["symbol","session"],sort=False)
    next_open = grouped.open.shift(-1)
    # Reentrada programada no open da ultima barra (15:55 NY em dia normal).
    # Alvo economico da permanencia ate a reentrada, NAO ate o ultimo close.
    reentry_open = grouped.open.transform(lambda s:s.iloc[-1])
    f["target"] = np.log(reentry_open/next_open)
    f["downside"] = (-f.target).clip(lower=0)
    f["peak_drawdown"] = f.close/grouped.high.cummax()-1
    f["efficiency"] = grouped.close.transform(lambda c:
        (c-c.shift(12)).abs()/c.diff().abs().rolling(12,min_periods=12).sum().replace(0,np.nan))
    f["mfe"] = np.nan
    f["mae"] = np.nan
    # Futuro apenas nos alvos, requer barras completas ate o fechamento.
    for _,g in f.groupby(["symbol","session"],sort=False):
        hi=g.high.to_numpy(float);lo=g.low.to_numpy(float);op=g.open.to_numpy(float)
        # Extremos ate o instante da recompra (open da ultima barra).
        hi=hi.copy();lo=lo.copy();hi[-1]=op[-1];lo[-1]=op[-1]
        future_hi=np.maximum.accumulate(hi[::-1])[::-1][1:]
        future_lo=np.minimum.accumulate(lo[::-1])[::-1][1:]
        with np.errstate(invalid="ignore",divide="ignore"):
            f.loc[g.index[:-1],"mfe"]=np.maximum(0.,np.log(future_hi/op[1:]))
            f.loc[g.index[:-1],"mae"]=np.maximum(0.,np.log(op[1:]/future_lo))
    # Cada barra e avaliada apenas apos encerrar. Cadencias escolhidas na validacao.
    ny=f.timestamp.dt.tz_convert("America/New_York")
    f["minutes_elapsed"]=ny.dt.hour*60+ny.dt.minute-570+5
    # base.preparar cria grade ate fechamento; load_data corta pelo calendario.
    f["remaining_minutes"]=(grouped.timestamp.transform("max")-f.timestamp).dt.total_seconds()/60
    recent=grouped.close.transform(lambda c:c.rolling(6,min_periods=6).count())
    f["horario"]=f.close.notna() & recent.eq(6) & f.remaining_minutes.gt(5)
    f["cusum_pos"]=np.nan;f["cusum_neg"]=np.nan
    f["recovery_steps"]=np.nan
    f["recovery_observed"]=np.nan
    f["regime"]=np.nan
    for _,g in f.groupby(["symbol","session"],sort=False):
        # Volatilidade anterior normaliza inovacoes; k=.5 em unidades de sigma.
        sigma=g.r1.rolling(24,min_periods=6).std().shift(1)
        z=g.r1/sigma.replace(0,np.nan)
        up=down=0.
        cp=[];cn=[]
        for v in z:
            if not np.isfinite(v):
                up=down=0.;cp.append(np.nan);cn.append(np.nan);continue
            up=max(0.,up+float(v)-.5)
            down=max(0.,down-float(v)-.5)
            cp.append(up);cn.append(down)
        f.loc[g.index,"cusum_pos"]=cp;f.loc[g.index,"cusum_neg"]=cn
        highs=g.high.to_numpy(float).copy();closes=g.close.to_numpy(float)
        highs[-1]=g.open.iloc[-1]
        peak=g.high.cummax().to_numpy(float)
        # Tempo ate recuperar maximo conhecido no momento da decisao.
        # Censura administrativa no fechamento; lacunas nao viram recuperacao.
        for i in range(len(g)-1):
            path=highs[i+1:]
            if not np.isfinite(peak[i]) or not np.isfinite(closes[i]) or not np.isfinite(path).all():
                continue
            if closes[i]>=peak[i]:
                continue  # nao esta em drawdown
            hit=np.flatnonzero(path>=peak[i])
            f.loc[g.index[i],"recovery_steps"]=int(hit[0]+1) if len(hit) else len(path)
            f.loc[g.index[i],"recovery_observed"]=int(bool(len(hit)))
        # Regime futuro: variacao de preco comparada ao ruido esperado.
        remain=len(g)-1-np.arange(len(g))
        band=g.vol12.to_numpy(float)*np.sqrt(remain)
        y=g.target.to_numpy(float)
        valid=np.isfinite(band)&np.isfinite(y)
        f.loc[g.index[valid],"regime"]=np.where(y[valid]>band[valid],2,np.where(y[valid]<-band[valid],0,1))
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
    from lightgbm import LGBMRegressor, LGBMClassifier
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
            "lightgbm_excursions":FEATURES+KFEATURES+["rolling","bocpd","peak_drawdown","efficiency"],
            "cusum_adaptive":["cusum_pos","cusum_neg","vol12","time_frac","peak_drawdown","efficiency"],
            "quantile_lightgbm":FEATURES+KFEATURES+["rolling","bocpd","peak_drawdown","efficiency","cusum_pos","cusum_neg"],
            "regime_recovery":FEATURES+KFEATURES+["rolling","bocpd","peak_drawdown","efficiency","cusum_pos","cusum_neg"],
            "recovery_survival":FEATURES+KFEATURES+["rolling","bocpd","peak_drawdown","efficiency","cusum_pos","cusum_neg"]}
    out = {}
    for name in METHODS:
        log(f"Treinando {name}: {len(tr)} checkpoints")
        x = tr[cols[name]].replace([np.inf,-np.inf],np.nan)
        z = ev[cols[name]].replace([np.inf,-np.inf],np.nan)
        preds = ev[["session","symbol","timestamp","target","minutes_elapsed"]].copy()
        labels=("target","mae","mfe") if name in ("lightgbm_excursions","recovery_survival","regime_recovery") else ("target","downside")
        for label in labels:
            model = (LGBMRegressor(n_estimators=100,num_leaves=7,max_depth=3,
                     learning_rate=.04,min_child_samples=35,random_state=SEED,
                     n_jobs=1,verbosity=-1) if "lightgbm" in name or name in ("regime_recovery","recovery_survival") else
                     make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=20.)))
            valid=tr[label].notna()
            if valid.sum()<100:raise ValueError(f"Amostra insuficiente {name}/{label}")
            model.fit(x.loc[valid],tr.loc[valid,label])
            preds[{"target":"mean","downside":"risk","mae":"risk","mfe":"expected_mfe"}[label]]=model.predict(z)
        if name=="quantile_lightgbm":
            for alpha,col in ((.1,"q10"),(.5,"q50"),(.9,"q90")):
                model=LGBMRegressor(objective="quantile",alpha=alpha,n_estimators=120,
                     num_leaves=7,max_depth=3,learning_rate=.04,min_child_samples=50,
                     random_state=SEED,n_jobs=1,verbosity=-1)
                model.fit(x,tr.target)
                preds[col]=model.predict(z)
            ordered=np.sort(preds[["q10","q50","q90"]].to_numpy(),axis=1)
            preds[["q10","q50","q90"]]=ordered
            preds["risk"]=(-preds.q10).clip(lower=0)
        if name=="regime_recovery":
            valid=tr.regime.notna()
            if tr.loc[valid,"regime"].nunique()!=3:
                raise ValueError("Amostra insuficiente para tres regimes")
            model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,
                 learning_rate=.04,min_child_samples=50,random_state=SEED,n_jobs=1,verbosity=-1)
            model.fit(x.loc[valid],tr.loc[valid,"regime"].astype(int))
            prob=model.predict_proba(z)
            for i,cl in enumerate(model.classes_):
                preds[["p_down","p_sideways","p_up"][int(cl)]]=prob[:,i]
            # Magnitudes condicionais no treino; expectativa ponderada por regime.
            means=tr.loc[valid].groupby("regime").target.mean()
            preds["mean"]=sum(preds[col]*float(means.loc[k]) for k,col in enumerate(["p_down","p_sideways","p_up"]))
            valid=tr.recovery_observed.notna()
            if tr.loc[valid,"recovery_observed"].nunique()!=2:
                raise ValueError("Amostra insuficiente para recuperacao")
            recovery=LGBMClassifier(n_estimators=80,num_leaves=7,max_depth=3,
                 learning_rate=.04,min_child_samples=50,random_state=SEED,n_jobs=1,verbosity=-1)
            recovery.fit(x.loc[valid],tr.loc[valid,"recovery_observed"].astype(int))
            preds["p_recovery"]=recovery.predict_proba(z)[:,1]
        if name=="recovery_survival":
            # Risco discreto por intervalos de 30 min (6 barras). Uma linha
            # por intervalo em risco; censura no fechamento, sem usar o
            # tempo efetivo futuro como feature. Somente a duracao planejada.
            rows=[];labels=[]
            valid=tr.recovery_steps.notna() & tr.recovery_observed.notna()
            for idx,r in tr.loc[valid].iterrows():
                duration=int(r.recovery_steps);hit=int(r.recovery_observed)
                for start in range(0,duration,6):
                    width=min(6,int(r.remaining_minutes/5)-start)
                    if width<=0:break
                    # Intervalo final parcial censurado nao vira nao-evento completo.
                    event=bool(hit and duration<=start+width)
                    if not event and duration<start+width:break
                    vals=x.loc[idx].to_dict();vals["hazard_bin"]=start//6;vals["hazard_width"]=width
                    rows.append(vals);labels.append(int(event))
                    if event:break
            if len(rows)<100 or len(set(labels))!=2:
                raise ValueError("Amostra insuficiente para sobrevivencia")
            model=LGBMClassifier(n_estimators=80,num_leaves=7,max_depth=3,
                 learning_rate=.04,min_child_samples=50,random_state=SEED,n_jobs=1,verbosity=-1)
            hx=pd.DataFrame(rows)
            model.fit(hx,labels)
            # Horizonte conhecido: numero de barras restantes segundo calendario.
            remaining=ev["remaining_minutes"]
            survival=np.ones(len(ev))
            for start in range(0,78,6):
                width=np.minimum(6,np.maximum(0,(remaining.to_numpy()/5).astype(int)-start))
                active=width>0
                if not active.any():break
                zz=z.loc[active].copy()
                zz["hazard_bin"]=start//6;zz["hazard_width"]=width[active]
                hazard=model.predict_proba(zz[hx.columns])[:,1]
                survival[active]*=1-hazard
            preds["p_recovery"]=1-survival
        preds["actual_recovery"]=ev.recovery_observed
        preds["actual_regime"]=ev.regime
        preds["actual_mae"]=ev.mae
        preds["actual_mfe"]=ev.mfe
        preds["risk"] = preds.risk.clip(lower=0)
        # Nao autorizar ordem com estado de tendencia completamente ausente.
        essential = {"kalman":"k_slope","multirrolling":"rolling","bocpd":"bocpd",
                     "lightgbm":"r6","kalman_lightgbm":"k_slope","lightgbm_excursions":"k_slope","cusum_adaptive":"cusum_neg","quantile_lightgbm":"r6","regime_recovery":"r6","recovery_survival":"r6"}[name]
        preds = preds[z[essential].notna()] if essential in z else preds
        out[name] = preds
    return out


def signal_map(preds, risk, frequency=60):
    if risk is None:
        return {}
    preds=preds[preds.minutes_elapsed.mod(frequency).eq(0)] if "minutes_elapsed" in preds else preds
    # Permanecer vs vender: vender custa COST; sem stop por % de preco.
    # Vender e recomprar tem custo dos dois lados.
    score = preds["mean"] - risk*preds["risk"] + np.log((1+COST)/(1-COST))
    if "expected_mfe" in preds:
        # Potencial favoravel estimado nao e lucro realizavel; peso fixo moderado.
        score += .25*preds.expected_mfe.clip(lower=0)*preds.get("p_recovery",1.)
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
            preclose = tg is not None and not tg.empty and t==tg.timestamp.iloc[-1]
            def quote(sym):
                g = by_symbol.get(sym)
                prior = t-pd.Timedelta(minutes=5)
                if g is None or t not in g.index:
                    return None,0.
                p = g.loc[t,"open"]
                volume = g.loc[prior,"volume"] if prior in g.index else np.nan
                return (float(p) if valid(p) else None,
                        PARTICIPATION*float(volume) if valid(volume) else 0.)
            # Concluir venda residual durante todo o pregao; novas compras so na abertura.
            if qty>1e-10 and held!=target:
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
            if pending and not preclose and qty>1e-10 and held==target:
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
            # Recompra programada do alvo conhecido no inicio do dia.
            # A capacidade do candle anterior limita o fill, inclusive na recompra.
            if ((opening and not triggered) or (preclose and triggered)) and held in (None,target) and cash>1e-8:
                px,capacity=quote(target)
                if px is not None:
                    q=min(cash/(px*(1+cost)),capacity)
                    if q>0:
                        fee=q*px*cost
                        cash-=q*px+fee;qty+=q;held=target;mark=px;fees+=fee
                        events.append(dict(session=day,symbol=target,timestamp=t,side="buy_scheduled_preclose" if preclose else "buy_winner",qty=q,price=px,fee=fee))
            # Barra atual encerrou. O score nunca executa no seu proprio open.
            if qty>1e-10 and held is not None:
                g=by_symbol.get(held)
                if g is not None and t in g.index and valid(g.loc[t,"close"]):
                    mark=float(g.loc[t,"close"])
                value=cash+qty*(mark or 0.)
                peak=max(peak,value)
                s=signals.get((day,held,t))
                if held==target and not triggered and s is not None and s<0 and tg is not None and t+pd.Timedelta(minutes=5)<tg.timestamp.iloc[-1]:
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
    # Nunca apresentar patrimonio congelado como resultado financeiro valido.
    stat["valid_valuation"]=not bool(missing_marks)
    if missing_marks:
        stat.update(capital_final=np.nan,return_pct=np.nan,max_drawdown_daily=np.nan,sharpe=np.nan)
        d["equity_unverified_proxy"]=d.equity
        d.loc[d.stale_mark,["equity","ret","drawdown"]]=np.nan
    return stat,d,pd.DataFrame(events)


def anchored_replay(pos,sessions,signals,capital=CAPITAL,cost=COST):
    """Overlay proporcional ancorado nos fatores diarios oficiais.
    A segunda camada pode emitir apenas uma venda. Recompra programada
    do MESMO alvo da Winner no open da ultima barra da sessao regular.
    A Winner congelada continua soberana na abertura seguinte.
    Nao e decisao independente no fechamento, nem fill real garantido.
    Sem venda, fator diario oficial permanece identico.
    """
    eq=float(capital);daily=[];events=[];invalid=False
    by_day={}
    for (day,symbol,ts),score in signals.items():
        if score<0 and np.isfinite(score):
            by_day.setdefault((day,symbol),[]).append((ts,score))
    for r in pos.itertuples(index=False):
        previous_eq=eq;factor=float(r.ref_factor)
        adjust=1.;status="hold";exit_px=None;entry_px=None
        key=(r.session,r.symbol)
        candidates=by_day.get(key,[])
        if candidates:
            ts,score=min(candidates,key=lambda p:p[0])
            g=sessions.get(key)
            if g is None or g.empty:
                invalid=True;status="missing_signal_session"
            else:
                last=g.iloc[-1]
                # A ultima barra e definida pelo calendario, inclusive sessoes curtas.
                cutoff=last.timestamp
                if ts+pd.Timedelta(minutes=5)<cutoff:
                    after=g[(g.timestamp>ts)&(g.timestamp<cutoff)&g.open.notna()&g.open.gt(0)]
                    if after.empty:
                        status="signal_without_fill"
                    else:
                        fill=after.iloc[0];exit_px=float(fill.open)
                        if pd.isna(last.open) or last.open<=0:
                            invalid=True;status="missing_reentry_price"
                        elif not getattr(sessions,"scale_ok",{}).get(key,True):
                            invalid=True;status="unreconciled_session"
                        else:
                            entry_px=float(last.open)
                            # Mesma serie ajustada, sem comparar RAW e SPLIT.
                            # Compra: cash/(preco*(1+custo)); venda: preco*(1-custo).
                            adjust=exit_px/entry_px*(1-cost)/(1+cost)
                            factor*=adjust;status="reentered"
                            events.extend([
                                dict(session=r.session,symbol=r.symbol,side="sell_protection",
                                     signal_timestamp=ts+pd.Timedelta(minutes=5),timestamp=fill.timestamp,
                                     price=exit_px,score=score),
                                dict(session=r.session,symbol=r.symbol,side="buy_scheduled_preclose",
                                     timestamp=cutoff,price=entry_px,exit_factor=adjust)])
        eq=eq*factor if not invalid else np.nan
        daily.append(dict(session=r.session,target=r.symbol,equity=eq,
            ret=factor-1 if not invalid else np.nan,reference_factor=r.ref_factor,
            exit_factor=adjust,exit_price=exit_px,reentry_price=entry_px,status=status,
            cash_at_close=False if status in ("hold","reentered","signal_without_fill") else None,
            valid_valuation=not invalid,exit_advantage=adjust-1 if status=="reentered" else np.nan))
    d=pd.DataFrame(daily);r=d.ret;final=float(d.equity.iloc[-1])
    peak=np.maximum.accumulate(np.r_[capital,d.equity])[1:]
    d["drawdown"]=d.equity/peak-1
    return dict(capital_initial=capital,capital_final=final,return_pct=100*(final/capital-1),
        max_drawdown_daily=float(d.drawdown.min()) if not invalid else np.nan,
        sharpe=float(r.mean()/r.std()*np.sqrt(252)) if not invalid and r.std()>0 else np.nan,
        signals=int(d.status.ne("hold").sum()),sessions_with_exit=int(d.status.eq("reentered").sum()),
        positive_exits=int(d.exit_advantage.gt(0).sum()),premature_exits=int(d.exit_advantage.lt(0).sum()),
        valid_valuation=not invalid,sessions=len(d)),d,pd.DataFrame(events)


def oracle_candidates(frame,sessions):
    """Apenas auditoria retrospectiva. Mesmo conjunto de checkpoints e
    mesmo fill posterior da politica causal; sem usar high como fill."""
    rows=[]
    for (day,symbol),g in frame.groupby(["session","symbol"],sort=False):
        key=(day,symbol)
        raw=sessions.get(key)
        if raw is None or raw.empty or not getattr(sessions,"scale_ok",{}).get(key,True):
            continue
        last=raw.iloc[-1]
        if pd.isna(last.open) or last.open<=0:continue
        valid=raw[(raw.timestamp<last.timestamp)&raw.open.notna()&raw.open.gt(0)]
        times=np.array([t.value for t in valid.timestamp],dtype=np.int64)
        for row in g[g.horario].itertuples(index=False):
            i=np.searchsorted(times,row.timestamp.value,side="right")
            if i>=len(valid):continue
            fill=valid.iloc[i]
            rows.append(dict(session=day,symbol=symbol,timestamp=row.timestamp,
                fill_timestamp=fill.timestamp,minutes_elapsed=row.minutes_elapsed,
                sell_price=float(fill.open),reentry_price=float(last.open),
                reentry_timestamp=last.timestamp,ratio=float(fill.open/last.open)))
    if not rows:raise ValueError("Sem checkpoints conciliados para Oracle")
    return pd.DataFrame(rows)


def oracle_map(candidates,frequency,cost):
    d=candidates[candidates.minutes_elapsed.mod(frequency).eq(0)].copy()
    d["net_factor"]=d.ratio*(1-cost)/(1+cost)
    best=d.sort_values(["session","net_factor","timestamp"],ascending=[True,False,True]).drop_duplicates("session")
    best=best.copy();best["opportunity"]=best.net_factor.gt(1)
    signals={(r.session,r.symbol,r.timestamp):-1. for r in best[best.opportunity].itertuples(index=False)}
    return signals,best


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
    assert d.cash.iloc[-1]<1e-8 and d.shares.iloc[-1]>0
    assert e.loc[e.side.eq("buy_scheduled_preclose"),"timestamp"].tolist()==[ts[-1]]
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
    # Ancora: sem saida, fatores oficiais. Com saida, recompra antes do fechamento.
    nextday="2026-07-29"
    g2=g.copy()
    g2["timestamp"]=g2.timestamp+pd.Timedelta(days=1)
    g2[["open","close"]]=99.
    m[(nextday,"DEMO")]=g2
    ref=pd.DataFrame(dict(session=["2026-07-28",nextday],symbol=["DEMO","DEMO"],
        previous=[None,"DEMO"],ref_factor=[.9,1.1],ref_close=[90.,99.],
        ref_previous_close=[np.nan,90.],ref_sell_open=[np.nan,np.nan]))
    a,_,_=anchored_replay(ref,m,{},cost=0.)
    assert np.isclose(a["capital_final"],9900.)
    b,_,e=anchored_replay(ref,m,{("2026-07-28","DEMO",ts[11]):-1.},cost=0.)
    assert np.isclose(b["capital_final"],9900.)
    assert e.side.tolist()==["sell_protection","buy_scheduled_preclose"]
    # Venda a 100 e recompra a 90: ganho proporcional de 100/90.
    g.loc[12,["open","close"]]=100.
    b,_,e=anchored_replay(ref,m,{("2026-07-28","DEMO",ts[11]):-1.},cost=0.)
    assert np.isclose(b["capital_final"],11000.)
    # Multiplicar toda a serie intradiaria por 40 nao deve produzir ganhos ficticios.
    scaled={key:g.assign(open=g.open*40,close=g.close*40) for key,g in m.items()}
    c,_,_=anchored_replay(ref,scaled,{("2026-07-28","DEMO",ts[11]):-1.},cost=0.)
    assert np.isclose(b["capital_final"],c["capital_final"])
    # Oracle nunca escolhe fill na propria barra do sinal nem no open de recompra.
    demo=g.assign(session="2026-07-28",symbol="DEMO",horario=False,minutes_elapsed=np.arange(1,79)*5)
    demo.loc[[11,76],"horario"]=True
    oc=oracle_candidates(demo,m)
    assert len(oc)==1 and oc.fill_timestamp.iloc[0]==ts[12]
    om,od=oracle_map(oc,60,0.)
    assert len(om)==1 and od.opportunity.iloc[0]
    return "synthetic_passed_not_market_result"


def run():
    files={}
    smoke_result=smoke()
    log(f"Testes sinteticos: {smoke_result}")
    pos,bars,sessions,coverage,source=load_data()
    files["auditoria_cobertura.csv"]=csv(coverage)
    files["control_oficial_mct.csv"]=csv(pos[["session","decision_session","symbol","previous",
        "walk_forward_fold","strategy_equity","ref_close","ref_factor","ref_buy_open","ref_sell_open"]])
    # Verificar identidade matematica da ancora, nao certificar motor independente.
    official,official_curve,_=anchored_replay(pos,sessions,{},capital=source["initial"])
    np.testing.assert_allclose(official_curve.equity,pos.strategy_equity,rtol=1e-10)
    f=frame_features(bars,pos)
    tr,va,te=split(pos)
    log(f"Treino {len(tr)}; validacao {len(va)}; teste fold3 {len(te)}")
    pred=make_scores(f,tr,va,te)
    select=pos[pos.session.isin(va)]
    test=pos[pos.session.isin(te)]
    grid=[];summary=[];curves=[];events=[];forecasts=[]
    restricted=[];restricted_curves=[];restricted_events=[]
    # Capital acumulado antes do fold 3; sem overlay durante treino/validacao.
    prior_capital=float(test.iloc[0].ref_previous_equity)
    baseline,bd,be=anchored_replay(test,sessions,{},capital=prior_capital)
    np.testing.assert_allclose(bd.equity,test.strategy_equity,rtol=1e-10)
    oracle=oracle_candidates(f,sessions)
    oracle_summaries=[];oracle_days=[];oracle_curves=[]
    for frequency in FREQUENCIES:
        for label,scope,capital in (("full_hindsight",pos,source["initial"]),("fold3_hindsight",test,prior_capital),("validation_hindsight",select,CAPITAL)):
            sig,best=oracle_map(oracle[oracle.session.isin(scope.session)],frequency,COST)
            stat,curve,_=anchored_replay(scope,sessions,sig,capital=capital)
            ref_stat,_,_=anchored_replay(scope,sessions,{},capital=capital)
            oracle_summaries.append(dict(period=label,frequency=frequency,covered_sessions=len(best),
                missing_sessions=len(scope)-len(best),opportunities=int(best.opportunity.sum()),
                control_final=ref_stat["capital_final"],delta_control_pct=100*(stat["capital_final"]/ref_stat["capital_final"]-1),**stat))
            oracle_days.append(best.assign(period=label,frequency=frequency))
            oracle_curves.append(curve.assign(period=label,frequency=frequency))
    files["oracle_comparacao.csv"]=csv(pd.DataFrame(oracle_summaries))
    files["oracle_oportunidades_diarias.csv"]=csv(pd.concat(oracle_days,ignore_index=True))
    files["oracle_patrimonio.csv"]=csv(pd.concat(oracle_curves,ignore_index=True))
    log("Oracle retrospectivo concluido; nunca usado para escolher parametros.")
    summary.append(dict(method="Control_MCT_anchored",risk=None,**baseline))
    curves.append(bd.assign(method="Control_MCT_anchored"))
    selected={};maps={"Control_MCT_anchored":{}}
    for name,p in pred.items():
        v=p[p.session.isin(va)];t=p[p.session.isin(te)]
        candidates=[]
        for frequency,risk in [(60,None)]+[(f,r) for f in FREQUENCIES for r in RISKS]:
            stat,_,_=anchored_replay(select,sessions,signal_map(v,risk,frequency))
            objective=np.log(stat["capital_final"]/CAPITAL)+.5*stat["max_drawdown_daily"] if stat["valid_valuation"] else -np.inf
            candidates.append((objective,risk,frequency))
            grid.append(dict(method=name,risk=risk,frequency=frequency,objective=objective,**stat))
        _,best,frequency=max(candidates,key=lambda x:x[0])
        selected[name]={"risk":best,"frequency":frequency}
        maps[name]=signal_map(t,best,frequency)
        stat,d,e=anchored_replay(test,sessions,maps[name],capital=prior_capital)
        actual=t.target.notna()
        stat["prediction_mae"]=float((t.loc[actual,"mean"]-t.loc[actual,"target"]).abs().mean()) if actual.any() else None
        if "expected_mfe" in t:
            stat["mfe_prediction_mae"]=float((t.expected_mfe-t.actual_mfe).abs().mean())
            stat["mae_prediction_mae"]=float((t.risk-t.actual_mae).abs().mean())
        summary.append(dict(method=name,risk=best,frequency=frequency,**stat))
        curves.append(d.assign(method=name))
        if not e.empty:events.append(e.assign(method=name))
        forecasts.append(t.assign(method=name))
        log(f"{name}: risco validado={best}; frequencia={frequency}; fold3 capital={stat['capital_final']:.2f}; saidas={stat['sessions_with_exit']}")
    result=pd.DataFrame(summary)
    # Referencia preditiva zero e comparacao em suporte comum, sem selecionar no teste.
    common=None
    for p in forecasts:
        keys=set(zip(p.session,p.symbol,p.timestamp))
        common=keys if common is None else common & keys
    predictive=[]
    for p in forecasts:
        q=p.loc[[k in common for k in zip(p.session,p.symbol,p.timestamp)]].dropna(subset=["target","mean"])
        if q.empty:continue
        predictive.append(dict(method=q.method.iloc[0],samples=len(q),
            mae=float((q["mean"]-q.target).abs().mean()),
            zero_mae=float(q.target.abs().mean()),
            directional_accuracy=float((np.sign(q["mean"])==np.sign(q.target)).mean()),
            recovery_brier=float(((q.p_recovery-q.actual_recovery)**2).mean()) if "p_recovery" in q else np.nan,
            quantile_80pct_coverage=float(((q.target>=q.q10)&(q.target<=q.q90)).mean()) if "q10" in q else np.nan))
    files["qualidade_preditiva_suporte_comum.csv"]=csv(pd.DataFrame(predictive))
    result["fold3_rebased_10000"]=result.capital_final/prior_capital*CAPITAL
    result["full_initial_capital"]=source["initial"]
    result["overlay_applied_only_fold3"]=True
    result["delta_control_usd"]=result.capital_final-baseline["capital_final"]
    result["delta_control_pct"]=100*(result.capital_final/baseline["capital_final"]-1)
    ceiling={r["frequency"]:r["capital_final"] for r in oracle_summaries if r["period"]=="fold3_hindsight"}
    result["oracle_covered_final"]=result.frequency.map(ceiling)
    oracle_gain=np.log(result.oracle_covered_final/baseline["capital_final"])
    result["fraction_oracle_log_gain"]=np.log(result.capital_final/baseline["capital_final"])/oracle_gain.where(oracle_gain>0)
    # Replay separado com liquidez, a MESMA politica congelada. Pode ficar
    # residual, mas sempre busca cotacoes do ticker realmente mantido.
    for name,sig in maps.items():
        log(f"Replay com liquidez: {name}")
        stat,d,e=replay(test,sessions,sig)
        restricted.append(dict(method=name,**stat))
        restricted_curves.append(d.assign(method=name))
        if not e.empty:restricted_events.append(e.assign(method=name))
    stress=[]
    for bps in (5,10,25,50):
        for name,sig in maps.items():
            st,_,_=anchored_replay(test,sessions,sig,capital=prior_capital,cost=bps/10000)
            stress.append(dict(method=name,extra_cost_bps=bps,**st))
    # Curva continua: antes do fold 3, referencia oficial sem overlay.
    pre=official_curve[~official_curve.session.isin(te)]
    full_curves=pd.concat([pd.concat([pre.assign(method=d.method.iloc[0]),d],ignore_index=True) for d in curves],ignore_index=True)
    files["patrimonio_continuo.csv"]=csv(full_curves)
    full_stats=[]
    for name,d in full_curves.groupby("method",sort=False):
        eq=d.equity.to_numpy(float);rr=d.ret.to_numpy(float)
        valid=np.isfinite(eq).all() and np.isfinite(rr).all()
        peaks=np.maximum.accumulate(np.r_[source["initial"],eq])[1:]
        full_stats.append(dict(method=name,full_max_drawdown=float((eq/peaks-1).min()) if valid else np.nan,
            full_sharpe=float(rr.mean()/rr.std(ddof=1)*np.sqrt(252)) if valid and rr.std(ddof=1)>0 else np.nan,
            full_return_pct=100*(eq[-1]/source["initial"]-1) if valid else np.nan))
    result=result.merge(pd.DataFrame(full_stats),on="method",how="left")
    files.update({"comparacao_estrategias.csv":csv(result),
                  "comparacao_com_liquidez.csv":csv(pd.DataFrame(restricted)),
                  "patrimonio_com_liquidez.csv":csv(pd.concat(restricted_curves,ignore_index=True)),
                  "ordens_com_liquidez.csv":csv(pd.concat(restricted_events,ignore_index=True) if restricted_events else pd.DataFrame()),
                  "selecao_validacao.csv":csv(pd.DataFrame(grid)),
                  "patrimonio_diario.csv":csv(pd.concat(curves,ignore_index=True)),
                  "ordens_simuladas.csv":csv(pd.concat(events,ignore_index=True) if events else pd.DataFrame()),
                  "previsoes_teste.csv":csv(pd.concat(forecasts,ignore_index=True)),
                  "sensibilidade_custos.csv":csv(pd.DataFrame(stress)),
                  "cotacoes_ausentes.csv":csv(pd.DataFrame(sorted(sessions.missing),columns=["session","symbol"]))})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6))
    for name,d in full_curves.groupby("method",sort=False):
        ax.plot(pd.to_datetime(d.session),d.equity,label=name)
    ax.set_title("Patrimonio continuo: overlay apenas no fold 3")
    ax.set_yscale("log")
    ax.set_ylabel("USD");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    buffer=io.BytesIO();fig.savefig(buffer,format="png",dpi=150);plt.close(fig)
    files["comparacao_patrimonio.png"]=buffer.getvalue()
    cov=coverage[coverage.session.isin(te)]
    mismatches=int(cov.close_difference_pct.abs().gt(100*PRICE_TOLERANCE).sum())
    status={"status":"completed_research_proxy" if cov.final_bar.all() and cov.rotation_quote.all()
            and result.valid_valuation.all() and not sessions.missing and mismatches==0
            and all(x["valid_valuation"] for x in restricted) else "partial_data_or_unreconciled_prices",
            "smoke":smoke_result,"source":source,
            "cache_hashes":sessions.hashes,"cache_errors":sessions.errors,
            "split":{"train":tr,"validation":va,"test":te},
            "official_full_control_identity":official["capital_final"],
            "capital_initial_test":prior_capital,"capital_initial_full":source["initial"],
            "restricted_initial_capital":CAPITAL,"extra_cost_per_side":COST,
            "oracle":"hindsight only; best next observed open before scheduled same-symbol reentry; missing sessions left unchanged; no liquidity limit",
            "volume_participation_proxy":PARTICIPATION,"selected_risk":selected,
            "first_layer":"MCT decision_session earlier than execution session, frozen selection",
            "cash_rule":"one protection exit, scheduled reentry of known Winner target at final bar open, next official opening selection",
            "control":"official continuous daily factors; no overlay in training/validation; fold3 uses accumulated capital; not independent replication",
            "limitations":["anchored overlay assumes proportional full fills; base fees retained conservatively even when cash avoids a sale",
                           "restricted replay separate; next-open/previous volume are execution proxies",
                           "SPLIT adjusted SIP; closing auction differences remain; dividends not independently reconciled; no certification",
                           "no independent closing Winner signal in daily export; scheduled preclose buy of same target is a separate proxy",
                           "models and cadence selected on validation; fold3 is exploratory because previous research inspected this period"],
            "validation_sessions_missing_final":int((~coverage.loc[coverage.session.isin(va),"final_bar"]).sum()),
            "validation_scale_discrepancies":int((~coverage.loc[coverage.session.isin(va),"scale_ok"]).sum()),
            "test_sessions_missing_final":int((~cov.final_bar).sum()),
            "test_missing_rotation_quote":int((~cov.rotation_quote).sum()),
            "test_close_discrepancies_over_tolerance":mismatches,
            "test_close_discrepancies_over_0_1pct":int(cov.close_difference_pct.abs().gt(.1).sum())}
    files["manifesto.json"]=json.dumps(status,ensure_ascii=False,indent=2,default=str).encode()
    return files

# Diagnostico independente de acesso e representatividade IEX/SIP.
# Nao altera os modelos, nao envia ordens, nao requer plano pago.
FEED_SYMBOLS=("DNN","CAPR","MKSI","NVDA")
FEED_FREQUENCIES=(1,5,15,30,60)


def feed_grid(raw, start, end):
    idx=pd.date_range(start,end,freq="min",inclusive="left",name="timestamp")
    cols=["open","high","low","close","volume"]
    if raw is None or raw.empty:
        return pd.DataFrame(index=idx,columns=cols,dtype=float)
    data=raw.copy()
    if "timestamp" in data.columns:data=data.set_index("timestamp")
    data.index=pd.to_datetime(data.index,utc=True)
    data=data[~data.index.duplicated(keep="last")].sort_index()
    return data[cols].apply(pd.to_numeric,errors="coerce").reindex(idx)


def feed_aggregate(g, frequency):
    # Janelas alinhadas a abertura regular, rotuladas pelo FIM da barra.
    # Sem forward-fill de preco, sem inventar volume.
    out=g.resample(f"{frequency}min",origin=g.index[0],label="right",closed="left").agg(
        open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume",lambda x:x.sum(min_count=1)),
        observed_minutes=("close","count"))
    last=pd.Series(g.index,index=g.index).where(g.close.notna()).resample(
        f"{frequency}min",origin=g.index[0],label="right",closed="left").max()
    out["last_observed_minute_age"]=(out.index-last-pd.Timedelta(minutes=1)).dt.total_seconds()/60
    return out[out.index<=g.index[-1]+pd.Timedelta(minutes=1)]


def feed_day_compare(sip,iex,day,symbol,frequency):
    a=feed_aggregate(sip,frequency);b=feed_aggregate(iex,frequency)
    j=a.add_prefix("sip_").join(b.add_prefix("iex_"),how="outer")
    common=j.sip_close.notna() & j.iex_close.notna()
    j["close_error_bps"]=10000*(j.iex_close/j.sip_close-1)
    j["high_error_bps"]=10000*(j.iex_high/j.sip_high-1)
    j["low_error_bps"]=10000*(j.iex_low/j.sip_low-1)
    sr=np.log(j.sip_close/j.sip_close.shift())
    ir=np.log(j.iex_close/j.iex_close.shift())
    paired=sr.notna() & ir.notna()
    j["sip_return"]=sr;j["iex_return"]=ir
    j["direction_agrees"]=np.where(paired,np.sign(sr)==np.sign(ir),np.nan)
    # Regra diagnostica SIMPLES e congelada: retorno negativo da janela.
    # Nao e a politica Winner nem um modelo aprovado de venda.
    j["negative_return_agrees"]=np.where(paired,(sr<0)==(ir<0),np.nan)
    # Oracle: todos os fills avaliados em SIP, com mesma recompra 5min antes
    # do fechamento. IEX apenas restringe os checkpoints observaveis.
    # Espera de 1 min apos fechamento da barra como proxy de latencia.
    reentry=sip.index[-1]-pd.Timedelta(minutes=4)
    reprice=sip.loc[reentry,"open"]
    ts=j.index+pd.Timedelta(minutes=1)
    fills=sip.open.reindex(ts).to_numpy()
    j["sip_fill_timestamp"]=ts
    j["sip_fill_open"]=fills
    j["oracle_net_factor"]=fills/reprice*(1-COST)/(1+COST) if pd.notna(reprice) and reprice>0 else np.nan
    allowed=(j.index>=sip.index[0]+pd.Timedelta(minutes=30)) & (ts<reentry)
    valid=allowed & j.sip_close.notna() & j.oracle_net_factor.notna()
    def best(mask):
        q=j.loc[mask,"oracle_net_factor"]
        return max(1.,float(q.max())) if len(q) else np.nan
    v_sip=float(sip.volume.sum(min_count=1));v_iex=float(iex.volume.sum(min_count=1))
    result=dict(session=day,symbol=symbol,frequency=frequency,
        expected_bars=len(j),sip_bars=int(j.sip_close.notna().sum()),
        iex_bars=int(j.iex_close.notna().sum()),common_bars=int(common.sum()),
        sip_minutes=int(sip.close.notna().sum()),iex_minutes=int(iex.close.notna().sum()),
        expected_minutes=len(sip),sip_volume=v_sip,iex_volume=v_iex,
        iex_volume_share=v_iex/v_sip if v_sip>0 else np.nan,
        mean_abs_close_error_bps=float(j.loc[common,"close_error_bps"].abs().mean()),
        p95_abs_close_error_bps=float(j.loc[common,"close_error_bps"].abs().quantile(.95)),
        mean_abs_high_error_bps=float(j.loc[common,"high_error_bps"].abs().mean()),
        mean_abs_low_error_bps=float(j.loc[common,"low_error_bps"].abs().mean()),
        iex_p95_last_observed_minute_age=float(j.iex_last_observed_minute_age.quantile(.95)),
        paired_returns=int(paired.sum()),direction_agreement=float(j.direction_agrees.mean()),
        negative_return_agreement=float(j.negative_return_agrees.mean()),
        oracle_sip_checkpoints=int(valid.sum()),
        oracle_iex_checkpoints=int((valid & j.iex_close.notna()).sum()),
        oracle_sip_factor=best(valid),oracle_iex_observable_factor=best(valid & j.iex_close.notna()),
        reentry_timestamp=reentry,reentry_sip_open=reprice)
    j=j.reset_index().assign(session=day,symbol=symbol,frequency=frequency)
    return result,j


def feed_test_smoke():
    start=pd.Timestamp("2026-07-28T13:30:00Z")
    end=start+pd.Timedelta(minutes=390)
    idx=pd.date_range(start,end,freq="min",inclusive="left")
    raw=pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=1000.),index=idx)
    a=feed_grid(raw,start,end);b=feed_grid(raw.iloc[::2],start,end)
    r,j=feed_day_compare(a,b,"2026-07-28","DEMO",1)
    assert r["sip_bars"]==390 and r["iex_bars"]==195
    assert r["iex_volume_share"]==.5 and r["mean_abs_close_error_bps"]==0.
    assert r["oracle_sip_factor"]==1. and r["oracle_iex_observable_factor"]==1.
    assert feed_aggregate(a,5).index[0]==start+pd.Timedelta(minutes=5)
    # Falta de barra nao vira preco inventado.
    assert b.close.isna().sum()==195
    return "feed_diagnostic_synthetic_passed"


def run_feed_test():
    import exchange_calendars as xc
    import time
    now=pd.Timestamp.now(tz="UTC")
    cutoff=now-pd.Timedelta(minutes=20)
    date=now.tz_convert("America/New_York").date()
    cal=xc.get_calendar("XNYS",start=pd.Timestamp(date)-pd.Timedelta(days=45),
                       end=pd.Timestamp(date)+pd.Timedelta(days=7))
    dates=[d for d in cal.sessions if cal.session_close(d)<=cutoff][-10:]
    if len(dates)<10:raise ValueError("Calendario insuficiente")
    start=cal.session_open(dates[0]);end=cal.session_close(dates[-1])
    smoke_result=feed_test_smoke()
    download,creds=base._market_reader_local()
    errors=[];requests=[];grids={};live=[]
    def query(symbol,feed,start,end,phase,adjustment="split"):
        log(f"Teste {phase}: {symbol} {feed} 1Min {start} / {end}")
        tic=time.monotonic();status="ok";message=""
        try:
            data=download(**creds,symbol=symbol,timeframe="1Min",start=start,end=end,
                          feed=feed,adjustment=adjustment)
            if data is None or data.empty:status="empty"
        except Exception as exc:
            status="error";data=None
            # Apenas tipo de erro; nao exportar credenciais ou respostas brutas.
            message=type(exc).__name__
            msg=str(exc).lower()
            if "subscription" in msg or "recent sip" in msg:message="subscription_restriction"
            elif "429" in msg or "rate limit" in msg:message="rate_limit"
            elif "401" in msg or "403" in msg:message="authentication_or_permission"
            errors.append(dict(phase=phase,symbol=symbol,feed=feed,error=message))
        requests.append(dict(phase=phase,symbol=symbol,feed=feed,status=status,
            elapsed_seconds=time.monotonic()-tic,rows=0 if data is None else len(data),error=message))
        return data,status
    for sym in FEED_SYMBOLS:
        for feed in ("sip","iex"):
            data,status=query(sym,feed,start,end,"historical")
            for d in dates:
                day=str(d.date())
                grids[(sym,feed,day)]=feed_grid(data,cal.session_open(d),cal.session_close(d))
    rows=[];details=[];minute_frames=[]
    for sym in FEED_SYMBOLS:
        for d in dates:
            day=str(d.date());a=grids[(sym,"sip",day)];b=grids[(sym,"iex",day)]
            for feed,g in (("sip",a),("iex",b)):
                minute_frames.append(g.reset_index().assign(session=day,symbol=sym,feed=feed))
            for freq in FEED_FREQUENCIES:
                r,j=feed_day_compare(a,b,day,sym,freq)
                rows.append(r);details.append(j)
    report=pd.DataFrame(rows);detail=pd.concat(details,ignore_index=True)
    summary=[]
    for (sym,freq),g in report.groupby(["symbol","frequency"]):
        q=detail[(detail.symbol==sym)&(detail.frequency==freq)]
        good=g.oracle_sip_factor.notna() & g.oracle_iex_observable_factor.notna()
        # Soma log dos fatores somente nos dias comparaveis. Sem capitalizacao ficticia.
        oracle_sip=float(np.log(g.loc[good,"oracle_sip_factor"]).sum())
        oracle_iex=float(np.log(g.loc[good,"oracle_iex_observable_factor"]).sum())
        summary.append(dict(symbol=sym,frequency=freq,sessions=len(g),
            sip_coverage_pct=100*g.sip_bars.sum()/g.expected_bars.sum(),
            iex_coverage_pct=100*g.iex_bars.sum()/g.expected_bars.sum(),
            iex_volume_share_pct=100*g.iex_volume.sum(min_count=1)/g.sip_volume.sum(min_count=1),
            mean_abs_close_error_bps=float(q.close_error_bps.abs().mean()),
            p95_abs_close_error_bps=float(q.close_error_bps.abs().quantile(.95)),
            iex_p95_last_observed_minute_age=float(q.iex_last_observed_minute_age.quantile(.95)),
            paired_returns=int(g.paired_returns.sum()),direction_agreement_pct=100*float(q.direction_agrees.mean()),
            negative_return_agreement_pct=100*float(q.negative_return_agrees.mean()),
            oracle_comparable_sessions=int(good.sum()),oracle_sip_covered_sessions=int(g.oracle_sip_factor.notna().sum()),
            oracle_iex_covered_sessions=int(g.oracle_iex_observable_factor.notna().sum()),
            oracle_sip_log_gain=oracle_sip if good.any() else np.nan,
            oracle_iex_observable_log_gain=oracle_iex if good.any() else np.nan,
            oracle_fraction_on_common_days=oracle_iex/oracle_sip if oracle_sip>0 else np.nan))
    # Prova de acesso recente via REST historico. So mede durante sessao
    # regular, com pelo menos 10 min decorridos. Nao e teste WebSocket.
    probe=pd.Timestamp.now(tz="UTC")
    label=pd.Timestamp(probe.tz_convert("America/New_York").date())
    regular=cal.is_session(label) and cal.session_open(label)+pd.Timedelta(minutes=10)<=probe<cal.session_close(label)
    if regular:
        for sym in FEED_SYMBOLS:
            for feed in ("iex","sip"):
                t=pd.Timestamp.now(tz="UTC").floor("min")
                if t>=cal.session_close(label):break
                data,status=query(sym,feed,t-pd.Timedelta(minutes=10),t,"recent",adjustment="raw")
                idx=pd.DatetimeIndex([]) if data is None or data.empty else pd.to_datetime(data.index,utc=True)
                last=idx[idx<t].max() if len(idx) else pd.NaT
                age=(pd.Timestamp.now(tz="UTC")-last-pd.Timedelta(minutes=1)).total_seconds() if pd.notna(last) else np.nan
                live.append(dict(symbol=sym,feed=feed,status=status,last_bar_start=last,
                    last_completed_bar_age_seconds=age,query_end=t))
    else:
        live.append(dict(status="outside_regular_session_or_first_10_minutes",checked_at=probe))
    manifest=dict(test="IEX_vs_SIP_read_only",created_at=now.isoformat(),symbols=FEED_SYMBOLS,
        frequencies=FEED_FREQUENCIES,sessions=[str(d.date()) for d in dates],historical_adjustment="split",
        sip_historical_end_at_least_20min_old=True,recent_adjustment="raw",
        regular_session_recent_probe=bool(regular),synthetic=smoke_result,
        errors=errors,limitations=["historical IEX can include corrections unavailable live",
        "recent REST probe is not continuous latency or websocket test",
        "oracle uses future SIP fills, one minute execution delay, same reentry five minutes before close",
        "oracle IEX limits checkpoints only; not a trained trading rule or realized execution",
        "return sign agreement is a diagnostic baseline, not Winner model agreement",
        "no orders, no subscription change"])
    files={"feed_test_manifesto.json":json.dumps(manifest,indent=2,default=str).encode(),
        "feed_test_resumo.csv":csv(pd.DataFrame(summary)),
        "feed_test_diario.csv":csv(report),"feed_test_checkpoints.csv":csv(detail),
        "feed_test_barras_1min.csv":csv(pd.concat(minute_frames,ignore_index=True)),
        "feed_test_requisicoes.csv":csv(pd.DataFrame(requests)),
        "feed_test_recente.csv":csv(pd.DataFrame(live))}
    # Preservar resultados da pesquisa principal no MESMO ZIP se ja existir.
    # Verificar integridade e hashes antes de reutilizar, substituir apenas diagnostico.
    if OUT.exists():
        with ZipFile(OUT) as z:
            if z.testzip() is not None:raise ValueError("ZIP anterior corrompido")
            hashes=json.loads(z.read("sha256.json"))
            for name,h in hashes.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError("Hash anterior invalido")
            prior={name:z.read(name) for name in z.namelist()
                   if name!="sha256.json" and not name.startswith("feed_test_")}
        prior.update(files);files=prior
    return files

# Monitoramento prospectivo REST. Um unico pregao, sem ordens.
# IEX atual e SIP atrasado; a coleta SIP continua 20 min apos fechar
# para permitir comparar o mesmo minuto de negociacao.
MONITOR_SYMBOLS=(*FEED_SYMBOLS,"AMS")
MONITOR_DELAY=20
MONITOR_GRACE=10
MONITOR_LOOKBACK=30


def monitor_jobs(slot,opening,closing,symbols=MONITOR_SYMBOLS):
    jobs=[]
    for feed,delay in (("iex",0),("sip",MONITOR_DELAY)):
        end=slot-pd.Timedelta(minutes=delay)
        if opening<end<=closing:
            start=max(opening,end-pd.Timedelta(minutes=MONITOR_LOOKBACK))
            jobs.extend((symbol,feed,delay,start,end) for symbol in symbols)
    return jobs


def monitor_error(exc):
    # Nao exportar mensagens arbitrarias, URLs ou credenciais.
    msg=str(exc).lower()
    for tokens,category in (
        (("subscription","recent sip"),"subscription_restriction"),
        (("429","rate limit","too many requests"),"rate_limit"),
        (("401","unauthorized"),"authentication"),
        (("403","forbidden"),"permission"),
        (("timeout","timed out"),"timeout"),
        (("connection","network"),"connection"),
        (("500","502","503","504"),"server_error")):
        if any(t in msg for t in tokens):return category
    return type(exc).__name__


def monitor_observation(data,start,end,received):
    target=end-pd.Timedelta(minutes=1)
    grid=feed_grid(data,start,end)
    valid=grid.close.notna() & np.isfinite(grid.close) & grid.close.gt(0)
    latest=grid.loc[valid]
    last=latest.index[-1] if len(latest) else pd.NaT
    target_present=bool(valid.loc[target])
    age=(received-last-pd.Timedelta(minutes=1)).total_seconds() if pd.notna(last) else np.nan
    return dict(target_bar_start=target,target_present=target_present,
        target_close=float(grid.loc[target,"close"]) if target_present else np.nan,
        last_bar_start=last,last_close=float(latest.close.iloc[-1]) if len(latest) else np.nan,
        age_seconds=age,lag_from_requested_end_seconds=(end-last-pd.Timedelta(minutes=1)).total_seconds()
            if pd.notna(last) else np.nan,
        observed_minutes=int(valid.sum()),expected_minutes=len(grid)),grid.loc[valid]


def monitor_smoke():
    opening=pd.Timestamp("2026-11-24T14:30:00Z")
    closing=opening+pd.Timedelta(minutes=390)
    assert len(monitor_jobs(opening+pd.Timedelta(minutes=1),opening,closing,("DEMO",)))==1
    jobs=monitor_jobs(opening+pd.Timedelta(minutes=21),opening,closing,("DEMO",))
    assert len(jobs)==2 and jobs[1][4]==opening+pd.Timedelta(minutes=1)
    assert [j[1] for j in monitor_jobs(closing+pd.Timedelta(minutes=20),opening,closing,("DEMO",))]==["sip"]
    assert not monitor_jobs(closing+pd.Timedelta(minutes=21),opening,closing,("DEMO",))
    end=opening+pd.Timedelta(minutes=3)
    raw=pd.DataFrame(dict(open=100.,high=100.,low=100.,close=100.,volume=1.),
                     index=[opening,opening+pd.Timedelta(minutes=1),end])
    row,g=monitor_observation(raw,opening,end,end+pd.Timedelta(seconds=10))
    assert not row["target_present"] and row["age_seconds"]==70 and len(g)==2
    row,_=monitor_observation(None,opening,end,end)
    assert not row["target_present"] and pd.isna(row["age_seconds"])
    return "monitor_synthetic_passed"


def monitor_report(rows,skipped,expected_slots,opening,closing):
    p=pd.DataFrame(rows)
    if p.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    summary=[]
    for (symbol,feed),g in p.groupby(["symbol","feed"],sort=True):
        g=g.sort_values("slot")
        fresh=g.target_present.fillna(False).astype(bool)
        # Falta de chamada, erro HTTP e candle ausente sao coisas distintas.
        # Uma sequencia interrompida por slot nao consultado nao e imputada.
        longest=streak=0;prev=None
        for r in g.itertuples():
            if prev is not None and r.slot-prev!=pd.Timedelta(minutes=1):streak=0
            streak=0 if r.target_present else streak+1
            longest=max(longest,streak);prev=r.slot
        expected=sum(1 for t in expected_slots if opening<t-pd.Timedelta(minutes=MONITOR_DELAY if feed=="sip" else 0)<=closing)
        summary.append(dict(symbol=symbol,feed=feed,expected_polls=expected,polls=len(g),
            first_poll=g.slot.min(),last_poll=g.slot.max(),
            successful_requests=int(g.status.isin(["ok","empty"]).sum()),
            errors=int(g.status.eq("error").sum()),empty_responses=int(g.status.eq("empty").sum()),
            exact_target_bars=int(fresh.sum()),target_availability_pct=100*fresh.mean(),
            target_availability_of_expected_pct=100*fresh.sum()/expected if expected else np.nan,
            longest_consecutive_missing_target_polls=longest,
            median_age_seconds=float(g.age_seconds.median()),p95_age_seconds=float(g.age_seconds.quantile(.95)),
            p95_excess_age_seconds=float(g.lag_from_requested_end_seconds.quantile(.95)),
            p95_response_seconds=float(g.response_seconds.quantile(.95)),
            p95_dispatch_lateness_seconds=float(g.dispatch_lateness_seconds.quantile(.95)),
            responses_after_next_poll=int((g.received_at>=g.slot+pd.Timedelta(seconds=MONITOR_GRACE+60)).sum())))
    # Mesmo minuto de mercado, observado em tempos DIFERENTES. Diagnostico
    # retrospectivo, nao uma feature disponivel no instante IEX original.
    cols=["symbol","target_bar_start","target_close","received_at","slot"]
    iex=p[p.feed.eq("iex") & p.target_present][cols].rename(columns={
        "target_close":"iex_close","received_at":"iex_received_at","slot":"iex_slot"})
    sip=p[p.feed.eq("sip") & p.target_present][cols].rename(columns={
        "target_close":"sip_close","received_at":"sip_received_at","slot":"sip_slot"})
    paired=iex.merge(sip,on=["symbol","target_bar_start"],validate="one_to_one")
    paired["price_difference_bps"]=10000*(paired.iex_close/paired.sip_close-1)
    paired["receipt_difference_seconds"]=(paired.sip_received_at-paired.iex_received_at).dt.total_seconds()
    comparison=[]
    for symbol,g in paired.groupby("symbol",sort=True):
        g=g.sort_values("target_bar_start")
        adjacent=g.target_bar_start.diff().eq(pd.Timedelta(minutes=1))
        ir=np.sign(g.iex_close.diff());sr=np.sign(g.sip_close.diff())
        comparison.append(dict(symbol=symbol,paired_minutes=len(g),
            mean_abs_price_error_bps=float(g.price_difference_bps.abs().mean()),
            p95_abs_price_error_bps=float(g.price_difference_bps.abs().quantile(.95)),
            adjacent_pairs=int(adjacent.sum()),
            direction_agreement_pct=100*float((ir[adjacent]==sr[adjacent]).mean()),
            median_receipt_difference_seconds=float(g.receipt_difference_seconds.median())))
    return pd.DataFrame(summary),paired,pd.DataFrame(comparison)


def run_feed_monitor():
    import exchange_calendars as xc
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    now=pd.Timestamp.now(tz="UTC")
    day=now.tz_convert("America/New_York").date()
    cal=xc.get_calendar("XNYS",start=pd.Timestamp(day)-pd.Timedelta(days=7),
                       end=pd.Timestamp(day)+pd.Timedelta(days=7))
    label=pd.Timestamp(day)
    if not cal.is_session(label):
        raise ValueError("Hoje nao e pregao XNYS. Execute em um dia de negociacao.")
    opening=cal.session_open(label);closing=cal.session_close(label)
    first=opening+pd.Timedelta(minutes=1)
    last=closing+pd.Timedelta(minutes=MONITOR_DELAY)
    grace=pd.Timedelta(seconds=MONITOR_GRACE)
    if now>closing:
        raise ValueError("Pregao regular ja encerrou. Inicie antes ou durante o proximo pregao.")
    smoke_result=monitor_smoke()
    download,creds=base._market_reader_local()
    prior={}
    if OUT.exists():
        with ZipFile(OUT) as z:
            if z.testzip() is not None:raise ValueError("ZIP anterior corrompido")
            hashes=json.loads(z.read("sha256.json"))
            if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):
                raise ValueError("Hash anterior invalido")
            prior={n:z.read(n) for n in z.namelist() if n!="sha256.json" and not n.startswith("feed_monitor_")}
    expected=list(pd.date_range(first,last,freq="min"))
    rows=[];bars=[];skipped=[];done=set()
    status="running"
    def poll(slot,job):
        symbol,feed,delay,start,end=job
        sent=pd.Timestamp.now(tz="UTC")
        tic=time.monotonic()
        error="";state="ok";data=None
        try:
            data=download(**creds,symbol=symbol,timeframe="1Min",start=start,end=end,
                          feed=feed,adjustment="raw")
            if data is None or data.empty:state="empty"
            received=pd.Timestamp.now(tz="UTC")
            obs,g=monitor_observation(data,start,end,received)
            if g.empty and state=="ok":state="empty"
        except Exception as exc:
            state="error";error=monitor_error(exc)
            received=pd.Timestamp.now(tz="UTC")
            obs,g=monitor_observation(None,start,end,received)
        row=dict(session=str(day),symbol=symbol,feed=feed,slot=slot,
            requested_delay_minutes=delay,query_start=start,query_end=end,
            sent_at=sent,received_at=received,response_seconds=time.monotonic()-tic,
            dispatch_lateness_seconds=max(0.,(sent-slot-grace).total_seconds()),
            status=state,error=error,**obs)
        # Arquivar a janela realmente recebida no instante da consulta.
        # Repeticoes intencionais permitem estudar revisoes e barras tardias.
        snapshot=g.reset_index().assign(symbol=symbol,feed=feed,slot=slot,received_at=received)
        return row,snapshot
    def checkpoint(state):
        summary,paired,comp=monitor_report(rows,skipped,expected,opening,closing)
        manifest=dict(test="prospective_rest_monitor",status=state,session=str(day),
            started_at=now,updated_at=pd.Timestamp.now(tz="UTC"),opening=opening,closing=closing,
            symbols=MONITOR_SYMBOLS,cadence_seconds=60,grace_seconds=MONITOR_GRACE,
            sip_delay_minutes=MONITOR_DELAY,scheduled_slots=len(expected),finished_slots=len(done),
            skipped_slots=len(skipped),request_count=len(rows),
            request_errors=sum(r["status"]=="error" for r in rows),
            expected_request_count=sum(len(monitor_jobs(t,opening,closing)) for t in expected),
            synthetic=smoke_result,
            limitations=["REST polls not websocket or exchange latency; age is based on completed minute bar",
                         "IEX and SIP are separate sources; same-minute price comparison is retrospective",
                         "no backfill of missed live observations; skipped slots counted separately",
                         "reader retries may exceed cadence; no concurrent batches; raw bars, no real orders"])
        files=dict(prior)
        files.update({"feed_monitor_manifesto.json":json.dumps(manifest,indent=2,default=str).encode(),
            "feed_monitor_consultas.csv":csv(pd.DataFrame(rows)),
            "feed_monitor_barras_recebidas.csv":csv(pd.concat(bars,ignore_index=True) if bars else pd.DataFrame()),
            "feed_monitor_slots_ausentes.csv":csv(pd.DataFrame(skipped,columns=["slot","reason"])),
            "feed_monitor_resumo.csv":csv(summary),"feed_monitor_minutos_pareados.csv":csv(paired),
            "feed_monitor_comparacao.csv":csv(comp),
            "feed_monitor_codigo_executado.py":Path(__file__).read_bytes(),
            "feed_monitor_execucao.log":LOG.read_bytes()})
        pack(files)
    log(f"Monitor local {day}: abertura={opening} fechamento={closing}; SIP ate {last}; sem ordens")
    log("Pode interromper com Ctrl+C; checkpoint parcial a cada cinco ciclos.")
    pool=ThreadPoolExecutor(max_workers=5)
    try:
        for slot in expected:
            current=pd.Timestamp.now(tz="UTC")
            due=slot+grace
            if current>=due+pd.Timedelta(minutes=1):
                skipped.append(dict(slot=slot,reason="before_start" if due<now else "previous_cycle_overrun"))
                continue
            while current<due:
                time.sleep(min(5.,max(0.,(due-current).total_seconds())))
                current=pd.Timestamp.now(tz="UTC")
            jobs=monitor_jobs(slot,opening,closing)
            for future in as_completed([pool.submit(poll,slot,j) for j in jobs]):
                row,snapshot=future.result();rows.append(row)
                if not snapshot.empty:bars.append(snapshot)
            done.add(slot)
            recent=[r for r in rows[-len(jobs):] if r["status"]=="error"]
            log(f"Slot {slot}: {len(jobs)} consultas, {len(recent)} erros; {len(done)} ciclos; {len(skipped)} ausentes")
            if len(done)==1 or len(done)%5==0:checkpoint("running")
        status="completed" if not skipped else "completed_partial_slots"
    except KeyboardInterrupt:
        status="interrupted"
        log("Interrompido pelo operador; salvando observacoes ja recebidas.")
    except Exception:
        status="failed_partial"
        log("Falha no monitor; salvando observacoes ja recebidas.")
        raise
    finally:
        pool.shutdown(wait=False,cancel_futures=True)
        checkpoint(status)
    print("ZIP:",OUT)
    return status

# Comparacao Yahoo / Alpaca. Yahoo e uma fonte externa nao oficial.
# A coleta automatizada depende de permissao do provedor. Alternativa:
# CSV autorizado, com timestamp ISO com fuso, symbol, open/high/low/close/volume.
def yahoo_normalize(raw):
    if raw is None or raw.empty:return pd.DataFrame(columns=["open","high","low","close","volume"])
    data=raw.copy()
    data.columns=[str(c).strip().lower() for c in data.columns]
    if "timestamp" in data.columns:data=data.set_index("timestamp")
    idx=pd.DatetimeIndex(pd.to_datetime(data.index,errors="raise"))
    if idx.tz is None:raise ValueError("Yahoo: timestamps sem fuso")
    data.index=idx.tz_convert("UTC")
    data.index.name="timestamp"
    data=data[["open","high","low","close","volume"]].apply(pd.to_numeric,errors="coerce")
    if data.index.duplicated().any():raise ValueError("Yahoo: timestamps duplicados")
    return data.sort_index()


def yahoo_pair(sip,other,day,symbol,frequency,feed):
    r,j=feed_day_compare(sip,other,day,symbol,frequency)
    rename={"iex_"+k:"source_"+k for k in (
        "bars","minutes","volume","volume_share","p95_last_observed_minute_age")}
    rename.update({"oracle_iex_checkpoints":"oracle_source_checkpoints",
                   "oracle_iex_observable_factor":"oracle_source_observable_factor"})
    r={rename.get(k,k):v for k,v in r.items()}
    r["source"]=feed
    paired=sip.close.notna() & other.close.notna() & sip.close.gt(0) & other.close.gt(0)
    ratio=(other.loc[paired,"close"]/sip.loc[paired,"close"]).median()
    r["median_price_ratio"]=float(ratio)
    r["scale_ok"]=bool(pd.notna(ratio) and abs(ratio-1)<=PRICE_TOLERANCE)
    if not r["scale_ok"]:
        r["oracle_source_observable_factor"]=np.nan
    j=j.rename(columns={k:k.replace("iex_","source_",1) for k in j.columns if k.startswith("iex_")})
    j["source"]=feed
    # Extremos diarios observados: comparar preco e horario, nao provar
    # que um algoritmo causal teria identificado a reversao naquele momento.
    for label,operation in (("high","max"),("low","min")):
        a=sip[label].dropna();b=other[label].dropna()
        aa=getattr(a,operation)() if len(a) else np.nan
        bb=getattr(b,operation)() if len(b) else np.nan
        ta=(a.idxmax() if label=="high" else a.idxmin()) if len(a) else pd.NaT
        tb=(b.idxmax() if label=="high" else b.idxmin()) if len(b) else pd.NaT
        r["daily_"+label+"_error_bps"]=10000*(bb/aa-1) if pd.notna(aa) and aa>0 else np.nan
        r["daily_"+label+"_time_error_minutes"]=(tb-ta).total_seconds()/60 if pd.notna(ta) and pd.notna(tb) else np.nan
    return r,j


def yahoo_test_smoke():
    start=pd.Timestamp("2026-11-24T14:30:00Z");end=start+pd.Timedelta(minutes=390)
    idx=pd.date_range(start,end,freq="min",inclusive="left")
    raw=pd.DataFrame(dict(Open=100.,High=101.,Low=99.,Close=100.,Volume=1000.),index=idx)
    a=feed_grid(yahoo_normalize(raw),start,end)
    b=feed_grid(yahoo_normalize(raw.iloc[::2]),start,end)
    r,j=yahoo_pair(a,b,"2026-11-24","DEMO",1,"yahoo")
    assert r["source_bars"]==195 and r["sip_bars"]==390
    assert r["mean_abs_close_error_bps"]==0 and r["daily_high_error_bps"]==0
    assert r["oracle_source_observable_factor"]==1
    assert j.source.eq("yahoo").all()
    assert not b.close.iloc[1:2].notna().any()
    return "yahoo_comparison_synthetic_passed"


def run_yahoo_test():
    import exchange_calendars as xc
    import time
    authorized="--yahoo-authorized" in sys.argv
    imported=None
    if "--yahoo-csv" in sys.argv:
        p=Path(sys.argv[sys.argv.index("--yahoo-csv")+1]).expanduser()
        imported=pd.read_csv(p)
        imported.columns=[str(c).strip().lower() for c in imported.columns]
        imported["symbol"]=imported.symbol.astype(str).str.upper().str.strip()
    elif not authorized:
        raise ValueError("Use --yahoo-authorized somente com permissao do provedor, ou --yahoo-csv CAMINHO")
    else:
        import yfinance as yf
    now=pd.Timestamp.now(tz="UTC")
    day=now.tz_convert("America/New_York").date()
    cal=xc.get_calendar("XNYS",start=pd.Timestamp(day)-pd.Timedelta(days=15),
                       end=pd.Timestamp(day)+pd.Timedelta(days=7))
    # Janela curta: candles Yahoo 1m tem retencao/limite por requisicao.
    dates=[d for d in cal.sessions if cal.session_close(d)<=now-pd.Timedelta(minutes=20)
           and cal.session_open(d)>=now-pd.Timedelta(days=7)][-5:]
    if not dates:raise ValueError("Nenhum pregao completo no recorte recente")
    start=cal.session_open(dates[0]);end=cal.session_close(dates[-1])
    symbols=MONITOR_SYMBOLS
    download,creds=base._market_reader_local()
    requests=[];grids={};recent=[]
    def query(symbol,source,a,b,phase):
        tic=time.monotonic();status="ok";error="";raw=None
        try:
            if source=="yahoo":
                if imported is not None:
                    raw=yahoo_normalize(imported[imported.symbol.eq(symbol)])
                else:
                    raw=yahoo_normalize(yf.Ticker(symbol).history(start=a.to_pydatetime(),
                        end=b.to_pydatetime(),interval="1m",prepost=False,
                        auto_adjust=False,actions=False,repair=False,timeout=10,raise_errors=True))
            else:
                raw=download(**creds,symbol=symbol,timeframe="1Min",start=a,end=b,
                             feed=source,adjustment="raw")
            if raw is None or raw.empty:status="empty"
            else:
                raw=raw.copy()
                raw.index=pd.to_datetime(raw.index,utc=True)
                raw=raw[(raw.index>=a)&(raw.index<b)]
                if raw.empty:status="empty"
        except Exception as exc:
            status="error";error=monitor_error(exc);raw=None
        received=pd.Timestamp.now(tz="UTC")
        requests.append(dict(symbol=symbol,source=source,phase=phase,status=status,error=error,
            query_start=a,query_end=b,received_at=received,rows=0 if raw is None else len(raw),
            response_seconds=time.monotonic()-tic))
        log(f"Yahoo comparison {phase} {symbol} {source}: {status}, {requests[-1]['rows']} barras")
        return raw,status,received
    for symbol in symbols:
        for source in ("sip","iex","yahoo"):
            raw,status,_=query(symbol,source,start,end,"historical")
            for d in dates:
                grids[symbol,source,str(d.date())]=feed_grid(raw,cal.session_open(d),cal.session_close(d))
    rows=[];details=[];minutes=[];support=[]
    for symbol in symbols:
        for d in dates:
            date=str(d.date())
            sip=grids[symbol,"sip",date];iex=grids[symbol,"iex",date];yahoo=grids[symbol,"yahoo",date]
            for source,g in (("sip",sip),("iex",iex),("yahoo",yahoo)):
                minutes.append(g.reset_index().assign(symbol=symbol,session=date,source=source))
            for frequency in FEED_FREQUENCIES:
                a,ja=yahoo_pair(sip,iex,date,symbol,frequency,"iex")
                b,jb=yahoo_pair(sip,yahoo,date,symbol,frequency,"yahoo")
                rows.extend([a,b]);details.extend([ja,jb])
                # Comparacao de erros nos MESMOS checkpoints dos tres feeds.
                m=ja[["timestamp","sip_close","source_close","sip_return","source_return"]].merge(
                    jb[["timestamp","source_close","source_return"]],on="timestamp",
                    suffixes=("_iex","_yahoo"),validate="one_to_one")
                good=m[["sip_close","source_close_iex","source_close_yahoo"]].notna().all(axis=1)
                ret=m[["sip_return","source_return_iex","source_return_yahoo"]].notna().all(axis=1)
                for source in ("iex","yahoo"):
                    support.append(dict(symbol=symbol,session=date,frequency=frequency,source=source,
                        common_bars=int(good.sum()),common_returns=int(ret.sum()),
                        absolute_error_sum_bps=float((10000*(m.loc[good,"source_close_"+source]/m.loc[good,"sip_close"]-1)).abs().sum()),
                        direction_matches=int((np.sign(m.loc[ret,"source_return_"+source])==np.sign(m.loc[ret,"sip_return"])).sum())))
    report=pd.DataFrame(rows);detail=pd.concat(details,ignore_index=True);common=pd.DataFrame(support)
    summary=[]
    for (symbol,frequency,source),g in report.groupby(["symbol","frequency","source"]):
        q=detail[(detail.symbol==symbol)&(detail.frequency==frequency)&(detail.source==source)]
        c=common[(common.symbol==symbol)&(common.frequency==frequency)&(common.source==source)]
        good=g.oracle_sip_factor.notna() & g.oracle_source_observable_factor.notna()
        # Oracle entre fontes no mesmo conjunto de sessoes (tres feeds).
        other=report[(report.symbol==symbol)&(report.frequency==frequency)&(report.source!=source)]
        common_days=set(other.loc[other.oracle_source_observable_factor.notna()&other.oracle_sip_factor.notna(),"session"])
        good &= g.session.isin(common_days)
        denom=float(np.log(g.loc[good,"oracle_sip_factor"]).sum())
        num=float(np.log(g.loc[good,"oracle_source_observable_factor"]).sum())
        summary.append(dict(symbol=symbol,frequency=frequency,source=source,sessions=len(g),
            sip_coverage_pct=100*g.sip_bars.sum()/g.expected_bars.sum(),
            scale_ok_sessions=int(g.scale_ok.sum()),
            source_coverage_pct=100*g.source_bars.sum()/g.expected_bars.sum(),
            volume_ratio_pct=100*g.source_volume.sum(min_count=1)/g.sip_volume.sum(min_count=1) if g.sip_volume.sum()>0 else np.nan,
            mean_abs_close_error_bps=float(q.close_error_bps.abs().mean()),
            p95_abs_close_error_bps=float(q.close_error_bps.abs().quantile(.95)),
            p95_last_observed_minute_age=float(q.source_last_observed_minute_age.quantile(.95)),
            daily_high_mae_bps=float(g.daily_high_error_bps.abs().mean()),
            daily_low_mae_bps=float(g.daily_low_error_bps.abs().mean()),
            daily_high_time_mae_min=float(g.daily_high_time_error_minutes.abs().mean()),
            daily_low_time_mae_min=float(g.daily_low_time_error_minutes.abs().mean()),
            direction_agreement_pct=100*float(q.direction_agrees.mean()),
            common_bars=int(c.common_bars.sum()),
            common_mae_bps=c.absolute_error_sum_bps.sum()/c.common_bars.sum() if c.common_bars.sum() else np.nan,
            common_direction_agreement_pct=100*c.direction_matches.sum()/c.common_returns.sum() if c.common_returns.sum() else np.nan,
            oracle_comparable_sessions=int(good.sum()),oracle_sip_log_gain=denom if good.any() else np.nan,
            oracle_source_log_gain=num if good.any() else np.nan,
            oracle_fraction=num/denom if denom>0 else np.nan))
    # Snapshot recente: nao confundir dados historicos com acesso live.
    t=pd.Timestamp.now(tz="UTC");label=pd.Timestamp(t.tz_convert("America/New_York").date())
    regular=cal.is_session(label) and cal.session_open(label)+pd.Timedelta(minutes=30)<=t<cal.session_close(label)
    if regular:
        for symbol in symbols:
            for source,delay in (("iex",0),("sip",20),("yahoo",0)):
                if source=="yahoo" and imported is not None:continue
                now2=pd.Timestamp.now(tz="UTC")
                if now2>=cal.session_close(label):break
                end2=now2.floor("min")-pd.Timedelta(minutes=delay)
                start2=max(cal.session_open(label),end2-pd.Timedelta(minutes=10))
                raw,status,received=query(symbol,source,start2,end2,"recent")
                obs,_=monitor_observation(raw,start2,end2,received)
                recent.append(dict(symbol=symbol,source=source,status=status,
                    requested_delay_minutes=delay,received_at=received,query_end=end2,**obs))
    if not recent:recent.append(dict(status="outside_regular_session_or_first_30_minutes",checked_at=t))
    manifest=dict(test="yahoo_vs_alpaca",created_at=now,symbols=symbols,
        sessions=[str(d.date()) for d in dates],frequencies=FEED_FREQUENCIES,
        yahoo_mode="authorized_csv" if imported is not None else "authorized_yfinance",
        yfinance_version=None if imported is not None else yf.__version__,
        recent_probe=bool(regular),synthetic=yahoo_test_smoke(),
        limitations=["Yahoo unofficial; permission and data license required",
        "Alpaca RAW and Yahoo auto_adjust=False; split handling may differ, audit price scale",
        "historical data may include later corrections; recent snapshot does not prove continuous latency",
        "volume ratio is not market share for Yahoo; feed definitions may differ",
        "oracle uses future SIP prices; not realized returns or a trained signal",
        "no orders or production changes"])
    files={"yahoo_test_manifesto.json":json.dumps(manifest,indent=2,default=str).encode(),
        "yahoo_test_resumo.csv":csv(pd.DataFrame(summary)),
        "yahoo_test_diario.csv":csv(report),"yahoo_test_checkpoints.csv":csv(detail),
        "yahoo_test_suporte_comum.csv":csv(common),
        "yahoo_test_barras_1min.csv":csv(pd.concat(minutes,ignore_index=True)),
        "yahoo_test_requisicoes.csv":csv(pd.DataFrame(requests)),
        "yahoo_test_recente.csv":csv(pd.DataFrame(recent))}
    if OUT.exists():
        with ZipFile(OUT) as z:
            if z.testzip() is not None:raise ValueError("ZIP anterior corrompido")
            hashes=json.loads(z.read("sha256.json"))
            if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):
                raise ValueError("Hash anterior invalido")
            prior={n:z.read(n) for n in z.namelist() if n!="sha256.json" and not n.startswith("yahoo_test_")}
        prior.update(files);files=prior
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
        if "--yahoo-self-test" in sys.argv:
            print(yahoo_test_smoke())
            sys.exit(0)
        if "--feed-monitor-self-test" in sys.argv:
            print(feed_test_smoke())
            print(monitor_smoke())
            sys.exit(0)
        if "--self-test" in sys.argv:
            print(smoke())
            print(feed_test_smoke())
            print(monitor_smoke())
            sys.exit(0)
        if "--feed-monitor" in sys.argv:
            run_feed_monitor()
            sys.exit(0)
        feed_only="--feed-test" in sys.argv
        yahoo_only="--yahoo-test" in sys.argv
        files=run_yahoo_test() if yahoo_only else (run_feed_test() if feed_only else run())
        log("Pesquisa concluida. Gravando ZIP.")
        prefix="yahoo_test_" if yahoo_only else ("feed_test_" if feed_only else "")
        files[prefix+"execucao.log"]=LOG.read_bytes()
        files[prefix+"codigo_executado.py"]=Path(__file__).read_bytes()
        pack(files)
        print("ZIP:",OUT)
    except Exception:
        LOGGER.exception("Falha na pesquisa; resultado anterior preservado.")
        raise
