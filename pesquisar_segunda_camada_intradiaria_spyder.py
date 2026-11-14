"""TCC: comparador causal de segunda camada de permanencia intradiaria.

Spyder F5. Nao acessa Alpaca, Mongo, MCT Live nem envia ordens.
Entrada: dados_intraday/*.csv, com timestamp UTC, symbol, open, high, low,
close, volume e barras de 1 ou 5 minutos. Em 1 minuto, usa apenas grupos
COMPLETOS de 5 minutos. Sinais as 10h30, 11h30 ... 15h30 de NY, calculados
sobre o candle anterior ja encerrado.

Metodos:
  multirrolling: regressao causal em 3 escalas
  bocpd: Bayesian Online Change Point Detection Normal-Normal
  hmm: filtro sequencial de HMM gaussiano 2 estados
  lightgbm: modelo supervisionado de retorno 30 minutos
  pca_lightgbm: PCA preco/volume + LightGBM (proxy, NAO deep PVB-FPCA)
  fitted_q: aproximacao de optimal stopping, CASH terminal
  ofi_l1: INDISPONIVEL sem quotes de bid/ask com tamanhos
Nenhum metodo escolhe outro ativo. Comparacao exploratoria apenas; para
resultados da Winner faltam replay integrado e dataset intradiario completo.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/"dados_intraday"
OUTPUT=ROOT/"output"/"pesquisa_vantagem_rotacao.zip"
SEED=20261114
H=6  # 30 min em candles de cinco minutos
CUSTO_BPS=10
METHODS=("multirrolling","bocpd","hmm","lightgbm","pca_lightgbm","fitted_q","ofi_l1")
FEATURES=("r1","r3","r6","r12","vol6","vol12","slope6","slope12",
          "slope24","accel","volume_rel","vwap_gap","range_pos","atr_rel","time_frac")


def preparar(d):
    req={"timestamp","symbol","open","high","low","close","volume"}
    if not req.issubset(d):
        raise ValueError(f"Colunas ausentes: {sorted(req-set(d.columns))}")
    d=d.copy()
    d["timestamp"]=pd.to_datetime(d.timestamp,utc=True,errors="raise")
    d["symbol"]=d.symbol.astype(str).str.strip().str.upper()
    if d.duplicated(["symbol","timestamp"]).any():
        raise ValueError("Candles repetidos")
    for k in ("open","high","low","close","volume"):
        d[k]=pd.to_numeric(d[k],errors="raise")
    if (d[["open","high","low","close"]]<=0).any().any() or (d.volume<0).any():
        raise ValueError("OHLCV invalido")
    if (d.high<d[["open","close","low"]].max(axis=1)).any() or (d.low>d[["open","close","high"]].min(axis=1)).any():
        raise ValueError("OHLC invalido")
    parts=[]
    for symbol,src in d.sort_values("timestamp").groupby("symbol",sort=True):
        src=src.set_index("timestamp")
        ny=src.index.tz_convert("America/New_York")
        src=src.assign(session=ny.date,minute=ny.hour*60+ny.minute)
        src=src[(src.minute>=570)&(src.minute<960)]
        for session,s in src.groupby("session",sort=True):
            if len(s)<2:continue
            f=s[["open","high","low","close","volume"]]
            freq=f.index.to_series().diff().median()
            if freq<=pd.Timedelta(minutes=1):
                x=f.resample("5min",label="left",closed="left").agg({
                    "open":"first","high":"max","low":"min","close":"last","volume":"sum"
                }).dropna()
                counts=f.close.resample("5min",label="left",closed="left").count()
                x=x.loc[counts[counts>=5].index]
            elif freq<=pd.Timedelta(minutes=5):
                x=f
            else:raise ValueError(f"Frequencia incompatível: {symbol}/{session}")
            x=x.reset_index()
            x["symbol"]=symbol
            x["session"]=str(session)
            parts.append(x)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def t_slope(values):
    v=np.asarray(values,dtype=float)
    if len(v)<5 or not np.isfinite(v).all():return np.nan
    x=np.arange(len(v),dtype=float);x-=x.mean()
    z=v-v.mean();beta=float(x@z)/float(x@x)
    err=z-beta*x
    se=math.sqrt(float(err@err)/max(1,len(v)-2)/float(x@x))
    return beta/se if se>1e-12 else 0.


def features(bars):
    result=[]
    for (symbol,session),d in bars.groupby(["symbol","session"],sort=True):
        d=d.sort_values("timestamp").copy().reset_index(drop=True)
        c=d.close.astype(float);log=np.log(c);r=log.diff()
        d["r1"]=r
        for w in (3,6,12):d[f"r{w}"]=log.diff(w)
        for w in (6,12):d[f"vol{w}"]=r.rolling(w,min_periods=w).std()
        for w in (6,12,24):
            d[f"slope{w}"]=log.rolling(w,min_periods=w).apply(t_slope,raw=True)
        d["accel"]=d.slope6-d.slope12
        d["volume_rel"]=d.volume/d.volume.shift().rolling(12,min_periods=6).mean().replace(0,np.nan)
        typical=(d.high+d.low+d.close)/3
        vwap=(typical*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)
        d["vwap_gap"]=c/vwap-1
        hi=d.high.rolling(12,min_periods=12).max()
        lo=d.low.rolling(12,min_periods=12).min()
        d["range_pos"]=(c-lo)/(hi-lo).replace(0,np.nan)
        tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
        d["atr_rel"]=tr.rolling(12,min_periods=12).mean()/c
        ny=d.timestamp.dt.tz_convert("America/New_York")
        mins=ny.dt.hour*60+ny.dt.minute
        d["time_frac"]=(mins-570)/390
        # Os labels futuros sao excluidos de todas as features.
        d["ret_futuro_30m"]=c.shift(-H)/c-1
        d["alvo_alta"]=np.where(d.ret_futuro_30m.notna(),
                                  (d.ret_futuro_30m>CUSTO_BPS/10000).astype(int),np.nan)
        d["horario"]=mins.isin([625,685,745,805,865,925])
        result.append(d)
    return pd.concat(result,ignore_index=True) if result else pd.DataFrame()


def split_cronologico(d):
    sessions=sorted(d.session.unique())
    if len(sessions)<9:
        return d.iloc[:0],d.iloc[:0],d.iloc[:0],{"status":"menos_de_9_sessoes"}
    i=max(3,int(len(sessions)*.6));j=max(i+2,int(len(sessions)*.8))
    if j>=len(sessions):
        return d.iloc[:0],d.iloc[:0],d.iloc[:0],{"status":"nao_particionavel"}
    tr=d[d.session.isin(sessions[:i-1])].copy()
    va=d[d.session.isin(sessions[i:j-1])].copy()
    te=d[d.session.isin(sessions[j:])].copy()
    return tr,va,te,{"treino":sorted(tr.session.unique()),
                      "validacao":sorted(va.session.unique()),
                      "teste":sorted(te.session.unique()),
                      "purge_sessoes":[sessions[i-1],sessions[j-1]],
                      "nota":"Particao exploratoria; substituir pelos folds congelados da Winner no replay completo"}


def rolling_scores(d):
    return (.5*np.tanh(d.slope6.clip(-15,15)/4)
           +.3*np.tanh(d.slope12.clip(-15,15)/4)
           +.2*np.tanh(d.slope24.clip(-15,15)/4)).clip(-1,1)


def bocpd(d,variance,hazard=1/40,max_run=100):
    """BOCPD Normal-Normal filtrado online e reiniciado em cada sessao."""
    variance=max(float(variance),1e-8)
    out=pd.Series(np.nan,index=d.index)
    for _,block in d.groupby(["symbol","session"],sort=False):
        run=np.array([1.]);n=np.array([0.]);mu=np.array([0.])
        for idx,ret in zip(block.index,block.r1.to_numpy(dtype=float)):
            if not np.isfinite(ret):continue
            pred=variance*(1+1/(4+n))
            like=np.exp(-.5*(ret-mu)**2/pred)/np.sqrt(2*np.pi*pred)
            p=np.r_[float(np.sum(run*like*hazard)),run[:max_run]*like[:max_run]*(1-hazard)]
            p/=max(p.sum(),1e-300)
            new_n=np.r_[1.,n[:max_run]+1]
            new_mu=np.r_[ret/5,(mu[:max_run]*n[:max_run]+ret)/(n[:max_run]+1)]
            run,n,mu=p,new_n,new_mu
            out.at[idx]=float(np.tanh((run@mu)/(math.sqrt(variance)*.75)))
    return out


def hmm(train,test):
    """Emissoes GaussianMixture + transicoes empiricas; filtro forward causal."""
    try:from sklearn.mixture import GaussianMixture
    except ImportError:return None
    x=(train.r1.dropna()*100).to_numpy().reshape(-1,1)
    if len(x)<80 or np.std(x)<1e-6:return None
    gm=GaussianMixture(n_components=2,random_state=SEED,reg_covar=1e-5).fit(x)
    order=np.argsort(gm.means_.ravel())
    mu=gm.means_.ravel()[order];var=gm.covariances_.reshape(-1)[order]
    old=gm.predict(x);states=np.array([int(np.where(order==s)[0][0]) for s in old])
    trans=np.ones((2,2))*2
    for a,b in zip(states[:-1],states[1:]):trans[a,b]+=1
    trans/=trans.sum(axis=1,keepdims=True)
    out=pd.Series(np.nan,index=test.index)
    for _,block in test.groupby(["symbol","session"],sort=False):
        p=np.array([.5,.5])
        for idx,rr in zip(block.index,block.r1.to_numpy(dtype=float)):
            if not np.isfinite(rr):continue
            like=np.exp(-.5*(rr*100-mu)**2/var)/np.sqrt(2*np.pi*var)
            p=(p@trans)*like;p/=max(p.sum(),1e-300)
            out.at[idx]=float(p[1]-p[0])
    return out


def lgb_scores(train,test,pca=False):
    try:
        from lightgbm import LGBMClassifier
        from sklearn.impute import SimpleImputer
    except ImportError:return None
    t=train[train.alvo_alta.notna()].copy()
    if len(t)<180 or t.alvo_alta.nunique()<2:return None
    imp=SimpleImputer(strategy="median")
    x=imp.fit_transform(t[list(FEATURES)].replace([np.inf,-np.inf],np.nan))
    z=imp.transform(test[list(FEATURES)].replace([np.inf,-np.inf],np.nan))
    if pca:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler=StandardScaler();x=scaler.fit_transform(x);z=scaler.transform(z)
        p=PCA(n_components=min(8,x.shape[1]),random_state=SEED)
        x=p.fit_transform(x);z=p.transform(z)
    model=LGBMClassifier(n_estimators=100,learning_rate=.04,num_leaves=7,
                         min_child_samples=35,random_state=SEED,n_jobs=1,verbose=-1)
    model.fit(x,t.alvo_alta.astype(int))
    return pd.Series(model.predict_proba(z)[:,1]*2-1,index=test.index)


def fqi(train,test):
    """Fitted Q: HOLD ate o proximo checkpoint, CASH estado absorvente."""
    try:
        from lightgbm import LGBMRegressor
        from sklearn.impute import SimpleImputer
    except ImportError:return None
    t=train[train.horario].copy().sort_values(["symbol","session","timestamp"])
    if len(t)<24:return None
    next_price=t.groupby(["symbol","session"],sort=False).close.shift(-1)
    last_price=train.groupby(["symbol","session"],sort=False).close.last()
    terminal=next_price.isna()
    for idx in next_price[terminal].index:
        row=t.loc[idx];next_price.loc[idx]=float(last_price.loc[(row.symbol,row.session)])
    next_features=t.groupby(["symbol","session"],sort=False)[list(FEATURES)].shift(-1)
    imp=SimpleImputer(strategy="median")
    x=imp.fit_transform(t[list(FEATURES)].replace([np.inf,-np.inf],np.nan))
    xn=imp.transform(next_features.replace([np.inf,-np.inf],np.nan))
    xt=imp.transform(test[list(FEATURES)].replace([np.inf,-np.inf],np.nan))
    reward=next_price.to_numpy()/t.close.to_numpy()-1-CUSTO_BPS/10000
    continuation=np.zeros(len(t))
    for _ in range(5):
        model=LGBMRegressor(n_estimators=80,learning_rate=.04,num_leaves=5,
                            min_child_samples=12,random_state=SEED,n_jobs=1,verbose=-1)
        model.fit(x,reward+.97*continuation)
        continuation=np.maximum(0,model.predict(xn))
        continuation[terminal.to_numpy()]=0
    return pd.Series(np.tanh(model.predict(xt)*10),index=test.index)


def metricas(y,s):
    from sklearn.metrics import balanced_accuracy_score,brier_score_loss,roc_auc_score
    valid=y.notna()&s.notna()
    if valid.sum()<3 or y[valid].nunique()<2:
        return {"status":"amostra_insuficiente","observacoes":int(valid.sum())}
    truth=y[valid].astype(int);score=((s[valid]+1)/2).clip(0,1)
    return {"status":"ok","observacoes":int(valid.sum()),
            "auc":float(roc_auc_score(truth,score)),
            "balanced_accuracy":float(balanced_accuracy_score(truth,score>=.5)),
            "brier":float(brier_score_loss(truth,score))}


def benchmark(d):
    tr,va,te,split=split_cronologico(d)
    if te.empty:return pd.DataFrame(),pd.DataFrame(),split
    history=pd.concat([tr,va]).sort_values(["symbol","timestamp"])
    variance=history.r1.dropna().var()
    results=[];predictions=[]
    for method in METHODS:
        if method=="ofi_l1":
            results.append({"metodo":method,"status":"nao_avaliavel_sem_quotes_L1"});continue
        if method=="multirrolling":score=rolling_scores(te)
        elif method=="bocpd":score=bocpd(te,variance)
        elif method=="hmm":score=hmm(history,te)
        elif method=="lightgbm":score=lgb_scores(history,te)
        elif method=="pca_lightgbm":score=lgb_scores(history,te,pca=True)
        else:score=fqi(history,te)
        if score is None:
            results.append({"metodo":method,"status":"treino_insuficiente_ou_dependencia_ausente"});continue
        hourly=te[te.horario]
        s=score.reindex(hourly.index)
        results.append({"metodo":method,**metricas(hourly.alvo_alta,s)})
        for idx,v in s.dropna().items():
            row=hourly.loc[idx]
            predictions.append({"metodo":method,"symbol":row.symbol,
                "timestamp_utc":row.timestamp.isoformat(),"session":row.session,
                "score":float(v),
                "retorno_futuro_30min_apenas_auditoria":
                None if not np.isfinite(row.ret_futuro_30m) else float(row.ret_futuro_30m)})
    return pd.DataFrame(results),pd.DataFrame(predictions),split


def teste_causalidade():
    rng=np.random.default_rng(SEED)
    ts=pd.date_range("2026-07-28T13:30:00Z",periods=78,freq="5min")
    c=100*np.exp(np.cumsum(np.r_[0,rng.normal(0,.003,77)]))
    rows=[dict(timestamp=t,symbol="DEMO",open=c[i],high=c[i]*1.001,
               low=c[i]*.999,close=c[i],volume=1000) for i,t in enumerate(ts)]
    a=features(preparar(pd.DataFrame(rows)))
    more=rows+[dict(timestamp=pd.Timestamp("2026-07-29T13:30:00Z"),
                    symbol="DEMO",open=100,high=100.1,low=99.9,close=100,volume=1000),
               dict(timestamp=pd.Timestamp("2026-07-29T13:35:00Z"),
                    symbol="DEMO",open=100,high=100.1,low=99.9,close=100,volume=1000)]
    b=features(preparar(pd.DataFrame(more)))
    original=b[b.session=="2026-07-28"]
    assert np.allclose(rolling_scores(a).fillna(0),rolling_scores(original).fillna(0))
    assert a.loc[a.horario,"timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M").tolist()==[
        "10:25","11:25","12:25","13:25","14:25","15:25"]
    try:preparar(pd.concat([pd.DataFrame(rows[:1]),pd.DataFrame(rows[:1])]))
    except ValueError:pass
    else:raise AssertionError("Candles duplicados foram aceitos")
    return {"status":"aprovado","barras_sinteticas":78,"decisoes_horarias":6,
            "nunca_apresentar_como_resultado_de_mercado":True}


def zip_atomico(files):
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    files["arquivos_sha256.json"]=json.dumps(hashes,indent=2).encode()
    temp=OUTPUT.with_suffix(".zip.tmp")
    with ZipFile(temp,"w",ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr("segunda_camada_intradiaria/"+name,data)
    with ZipFile(temp) as z:
        if z.testzip() is not None:raise RuntimeError("ZIP invalido")
        for name,sha in hashes.items():
            got=hashlib.sha256(z.read("segunda_camada_intradiaria/"+name)).hexdigest()
            if got!=sha:raise RuntimeError("SHA divergente")
    os.replace(temp,OUTPUT)


def executar():
    smoke=teste_causalidade()
    paths=sorted(INPUT.glob("*.csv"))
    if not paths:
        result={"status":"aguardando_candles_intradiarios_reais","metodos":METHODS,
                "teste_causalidade":smoke,"universo_winner_configurado":77,
                "patrimonio_final":None,"motivo":"Sem OHLCV intradiario completo; nao inventar patrimonio",
                "nenhuma_ordem_real":True}
        zip_atomico({"status.json":json.dumps(result,indent=2,ensure_ascii=False).encode(),
                    "codigo_executado.py":Path(__file__).read_bytes()})
        return result
    raw=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    frame=features(preparar(raw))
    ranking,events,split=benchmark(frame)
    result={"status":"piloto" if not ranking.empty else "amostra_insuficiente",
            "teste_causalidade":smoke,"sessoes":int(frame.session.nunique()),
            "ativos":sorted(frame.symbol.unique().tolist()),"split":split,
            "patrimonio_final":None,
            "motivo_patrimonio":"Sem replay do Control e execucao/liquidez integrada",
            "nenhuma_ordem_real":True}
    zip_atomico({"status.json":json.dumps(result,indent=2,ensure_ascii=False,default=str).encode(),
                 "comparacao_metodos.csv":ranking.to_csv(index=False).encode(),
                 "previsoes_horarias.csv":events.to_csv(index=False).encode(),
                 "codigo_executado.py":Path(__file__).read_bytes()})
    return result


if __name__=="__main__":
    print(json.dumps(executar(),ensure_ascii=False,indent=2,default=str))
    print("ZIP:",OUTPUT)
