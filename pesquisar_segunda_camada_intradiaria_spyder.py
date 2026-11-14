"""TCC: comparador causal de segunda camada de permanencia intradiaria.

Spyder F5. Extrai posicoes do Control congelado e consulta OHLCV 5Min RAW/SIP
usando apenas o leitor de dados historicos da API local INSTALADA LOCALMENTE.
Nao modifica o local, nao chama endpoints de ordem nem envia operacoes reais.
Os candles sao armazenados em dados_intraday/cotacoes_5min/ e reutilizados.
Fora da maquina do usuario, sem as credenciais e o snapshot, a rotina falha
informando a ausencia de dados, sem inventar retornos ou patrimonio.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/"dados_intraday"
OUTPUT=ROOT/"output"/"pesquisa_vantagem_rotacao.zip"
LOG_FILE=OUTPUT.parent/"pesquisa_segunda_camada.log"
_LOG_LINES=[]
_LOG_ACTIVE=False


def registrar(etapa, mensagem, nivel="INFO"):
    """Console imediato e log local, sem expor credenciais nem respostas privadas."""
    line=(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
          f"{nivel} [{etapa}] {mensagem}")
    print(line, flush=True)
    if _LOG_ACTIVE:
        _LOG_LINES.append(line)
        LOG_FILE.parent.mkdir(parents=True,exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line+"\n")


def log_blob():
    return ("\n".join(_LOG_LINES)+"\n").encode("utf-8")
SEED=20261114
H=6  # 30 min em candles de cinco minutos
CUSTO_BPS=10
METHODS=("multirrolling","bocpd","hmm","lightgbm","pca_lightgbm","fitted_q","ofi_l1")
FEATURES=("r1","r3","r6","r12","vol6","vol12","slope6","slope12",
          "slope24","accel","volume_rel","vwap_gap","range_pos","atr_rel","time_frac")


def preparar(d, *, informar_lacunas=False):
    """Normaliza barras observadas em uma grade temporal fixa de cinco minutos.

    Um ativo com poucos negocios pode ter intervalos de 10, 15 ou mais minutos
    entre barras 5Min VALIDAS. Nao se deduz o timeframe pela mediana dessas
    distancias; diferencas grandes sao lacunas reais, nao candles mais longos.
    A grade representa lacunas com NaN, nunca ffill ou preco sintetico. Assim,
    shift(-6) continua correspondendo a 30 minutos REAIS, e nao seis trades.
    """
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
    sessao_com_lacunas=0
    barras_ausentes=0
    sessoes_descartadas=0
    for symbol,src in d.sort_values("timestamp").groupby("symbol",sort=True):
        src=src.set_index("timestamp")
        ny=src.index.tz_convert("America/New_York")
        src=src.assign(session=ny.date,minute=ny.hour*60+ny.minute)
        src=src[(src.minute>=570)&(src.minute<960)]
        for session,s in src.groupby("session",sort=True):
            if len(s)<2:
                sessoes_descartadas+=1
                continue
            f=s[["open","high","low","close","volume"]]
            # O feed de 5 minutos usa timestamps 09:30, 09:35, ...,
            # independentemente de quantos intervalos nao tiveram negocios.
            ny_index=f.index.tz_convert("America/New_York")
            alinhado_5min=(
                (ny_index.minute%5==0).all()
                and (ny_index.second==0).all()
                and (ny_index.microsecond==0).all()
                and (f.index.to_series().diff().iloc[1:]>=pd.Timedelta(minutes=5)).all()
            )
            if alinhado_5min:
                x=f
            else:
                # Alternativa: 1Min com barras realmente consecutivas.
                # Somente agregados COM CINCO barras sao aceitos.
                freq=f.index.to_series().diff().median()
                if freq>pd.Timedelta(minutes=1):
                    sessoes_descartadas+=1
                    if informar_lacunas:
                        registrar("cadencia",f"{symbol}/{session}: horario irregular nao "
                                 "identificado como 5Min nem 1Min completo; sessao ignorada.", "WARN")
                    continue
                counts=f.close.resample("5min",label="left",closed="left").count()
                x=f.resample("5min",label="left",closed="left").agg({
                    "open":"first","high":"max","low":"min","close":"last","volume":"sum"
                })
                x=x.loc[counts[counts==5].index].dropna()
            # Grade de 78 intervalos: sessao regular NY 09:30 a 15:55.
            start=pd.Timestamp(str(session),tz="America/New_York")+pd.Timedelta(hours=9,minutes=30)
            grade=pd.date_range(start.tz_convert("UTC"),periods=78,freq="5min",name="timestamp")
            x=x.reindex(grade)
            missing=int(x.close.isna().sum())
            if missing:
                sessao_com_lacunas+=1
                barras_ausentes+=missing
                if informar_lacunas and (missing>=12 or len(s)<12):
                    registrar("lacunas",f"{symbol}/{session}: {missing}/78 intervalos "
                             "sem barra observada; mantendo NaN, sem preencher preco.", "WARN")
            x=x.reset_index()
            x["symbol"]=symbol
            x["session"]=str(session)
            parts.append(x)
    if informar_lacunas:
        registrar("cadencia",f"Sessoes com lacunas={sessao_com_lacunas}; "
                 f"intervalos ausentes={barras_ausentes}; sessoes ignoradas={sessoes_descartadas}.")
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
        # Nao emitir decisao se nao houve candle no checkpoint nem seis
        # barras reais consecutivas no contexto imediato.
        recentes=c.rolling(6,min_periods=6).count().eq(6)
        d["horario"]=mins.isin([625,685,745,805,865,925]) & c.notna() & recentes
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
    registrar("validacao",f"Treino={tr.session.nunique()} dias, validacao={va.session.nunique()} dias, "
             f"teste={te.session.nunique()} dias.")
    if te.empty:
        registrar("validacao","Amostra insuficiente para separar treino e teste.", "WARN")
        return pd.DataFrame(),pd.DataFrame(),split
    history=pd.concat([tr,va]).sort_values(["symbol","timestamp"])
    variance=history.r1.dropna().var()
    results=[];predictions=[]
    for method in METHODS:
        registrar("modelo",f"Iniciando metodo={method}")
        tic=time.monotonic()
        if method=="ofi_l1":
            results.append({"metodo":method,"status":"nao_avaliavel_sem_quotes_L1"})
            registrar("modelo","ofi_l1: sem historico de quotes L1; metodo nao avaliavel.", "WARN")
            continue
        if method=="multirrolling":score=rolling_scores(te)
        elif method=="bocpd":score=bocpd(te,variance)
        elif method=="hmm":score=hmm(history,te)
        elif method=="lightgbm":score=lgb_scores(history,te)
        elif method=="pca_lightgbm":score=lgb_scores(history,te,pca=True)
        else:score=fqi(history,te)
        if score is None:
            results.append({"metodo":method,"status":"treino_insuficiente_ou_dependencia_ausente"})
            registrar("modelo",f"{method}: sem treino suficiente ou biblioteca ausente.", "WARN")
            continue
        hourly=te[te.horario]
        s=score.reindex(hourly.index)
        measures=metricas(hourly.alvo_alta,s)
        results.append({"metodo":method,**measures})
        registrar("modelo",f"{method}: status={measures['status']} "
                 f"amostras={measures['observacoes']} tempo={time.monotonic()-tic:.1f}s")
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



# Fonte historica no PC: somente leitura, usando o leitor Alpaca do local.
# NENHUMA alteracao no local, endpoints, conta de corretora ou Winner.
FORCAR_DOWNLOAD=False
MAX_GRUPOS_DOWNLOAD=None  # None: processar todos os meses ausentes.
SNAPSHOT_CONTROL=ROOT/"pesquisa_permanencia_minima.zip"

# Posicoes originais OOS extraidas do arquivo congelado de permanencia.
# Somente pregao e ativo mantido; nenhum preco ou score foi recalculado.
_POSICOES_CONTROL_SHA256="91f3570ab3d0fd479f4cab36b00a5b14baf9638459181bf3363f7fbc81f2469a"
_POSICOES_CONTROL_LZMA_B64="""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"""


def _posicoes_congeladas_embutidas():
    import base64
    import lzma
    original=lzma.decompress(base64.b64decode(_POSICOES_CONTROL_LZMA_B64))
    if hashlib.sha256(original).hexdigest()!=_POSICOES_CONTROL_SHA256:
        raise RuntimeError("Snapshot embutido de posicoes Control nao passou no SHA-256")
    from io import BytesIO
    data=pd.read_csv(BytesIO(original),dtype=str)
    if (len(data)!=1560 or data.session.iloc[0]!="2020-07-22"
        or data.session.iloc[-1]!="2026-10-06"
        or data.duplicated("session").any()):
        raise RuntimeError("Snapshot embutido de posicoes Control tem calendario invalido")
    return original




def preparar_posicoes_control():
    file=INPUT/"winner_positions.csv"
    if not file.is_file():
        snapshot=next((p for p in [SNAPSHOT_CONTROL,ROOT.parent/"pesquisa_permanencia_minima.zip"]
                       if p.is_file()),None)
        if snapshot is None:
            # Posicoes Control congeladas acompanham o proprio script.
            original=_posicoes_congeladas_embutidas()
            INPUT.mkdir(parents=True,exist_ok=True)
            temp=file.with_suffix(".tmp")
            temp.write_bytes(original)
            os.replace(temp,file)
            origin={"status":"posicoes_control_congeladas_embutidas",
                    "origem":"Control original 2 pregoes / comparacao_diaria_ativos_e_patrimonio",
                    "snapshot_posicoes_sha256":_POSICOES_CONTROL_SHA256}
            positions=pd.read_csv(file,dtype=str)
            origin.update({"sessoes":int(len(positions)),"ativos":int(positions.symbol.nunique()),
                           "sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
            return positions,origin
        import io
        with ZipFile(snapshot) as z:
            names=[n for n in z.namelist() if n.endswith("/comparacao_diaria_ativos_e_patrimonio.csv")]
            if len(names)!=1:raise ValueError("Snapshot sem historico diario das posicoes")
            original=pd.read_csv(io.BytesIO(z.read(names[0])))
        if not {"data","ativo_2_pregoes"}.issubset(original):
            raise ValueError("Colunas do Control congelado ausentes")
        positions=pd.DataFrame({
            "session":pd.to_datetime(original["data"],utc=True).dt.date.astype(str),
            "symbol":original["ativo_2_pregoes"].astype(str).str.upper().str.strip()})
        positions=positions[positions.symbol.str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,15}") &
                            (positions.symbol!="CASH")].copy()
        if positions.duplicated("session").any():raise ValueError("Control duplicou posicoes por sessao")
        INPUT.mkdir(parents=True,exist_ok=True)
        temp=file.with_suffix(".tmp")
        positions.to_csv(temp,index=False)
        os.replace(temp,file)
        origin={"status":"derivado_do_control_congelado",
                "snapshot_sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    else:
        origin={"status":"arquivo_existente_reutilizado"}
    positions=pd.read_csv(file,dtype=str)[["session","symbol"]]
    positions["session"]=pd.to_datetime(positions.session,errors="raise").dt.date.astype(str)
    positions["symbol"]=positions.symbol.str.upper().str.strip()
    positions=positions[positions.symbol.str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,15}") &
                            (positions.symbol!="CASH")].copy()
    if positions.duplicated("session").any():raise ValueError("winner_positions.csv contem sessoes duplicadas")
    origin.update({"sessoes":len(positions),"ativos":int(positions.symbol.nunique()),
                   "sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
    return positions,origin


def _market_reader_local():
    """Nao usa endpoint de grafico recente, que nao retorna OHLCV historico."""
    import sys
    from importlib.util import find_spec
    candidates=[
        ROOT.parent/"market_cycle_trader_api"/"src",
        ROOT.parent/"extrema_backtest_dashboard_mongo"/"market_cycle_trader"/"market_cycle_trader_api"/"src",
        ROOT.parent/"market_cycle_trader"/"market_cycle_trader_api"/"src",
        Path.home()/"des"/"gpt"/"extrema_backtest_dashboard_mongo"/"market_cycle_trader"/"market_cycle_trader_api"/"src",
    ]
    candidate=next((p for p in candidates if
        (p/"market_cycle_trader_api"/"infrastructure"/"market_data"/"alpaca.py").exists()),None)
    if candidate is not None and str(candidate) not in sys.path:sys.path.insert(0,str(candidate))
    if find_spec("market_cycle_trader_api") is None:
        raise RuntimeError("Codigo da API local nao encontrado na pasta de projetos")
    from market_cycle_trader_api.core.environment import load_project_environment
    from market_cycle_trader_api.infrastructure.market_data.alpaca import download_stock_bars
    load_project_environment(candidate.parent.parent/".env" if candidate is not None else None)
    # Usa a configuracao de mercado existente, sem carregar pymongo/Mongo.
    key=next((os.getenv(k) for k in ("ALPACA_API_KEY_ID","APCA_API_KEY_ID") if os.getenv(k)),None)
    secret=next((os.getenv(k) for k in ("ALPACA_SECRET_KEY","ALPACA_API_SECRET_KEY","APCA_API_SECRET_KEY")
                 if os.getenv(k)),None)
    if not key or not secret:
        raise RuntimeError("Credenciais historicas nao estao configuradas na API local")
    return download_stock_bars,{"api_key_id":key,"secret_key":secret}


def preencher_cache_control(positions):
    """Baixa somente meses/ativos que o Control manteve; reutiliza CSVs."""
    if positions.empty:
        registrar("posicoes","Nenhuma sessao do Control disponivel para consulta.", "WARN")
        return {"status":"sem_posicoes"}
    folder=INPUT/"cotacoes_5min"
    folder.mkdir(parents=True,exist_ok=True)
    work=positions.assign(month=positions.session.str[:7])
    groups=[(a,m,sorted(g.session.unique().tolist()))
            for (a,m),g in work.groupby(["symbol","month"],sort=True)]
    pending=[(a,m,sessions) for a,m,sessions in groups
             if FORCAR_DOWNLOAD or not (folder/f"{a}_{m}.csv").is_file()]
    report={"status":"cache_completo" if not pending else "pendente",
            "total_grupos":len(groups),"em_cache":len(groups)-len(pending),
            "novos_baixados":0,"falhas":[],"fonte":"servico_historico_local_local_Alpaca_RAW_SIP_5Min",
            "alteracoes_local":False,"operacoes_reais":False}
    registrar("cache",f"Grupos ativo/mes={len(groups)}; cache={len(groups)-len(pending)}; "
             f"faltam={len(pending)}; download_forcado={FORCAR_DOWNLOAD}")
    if not pending:
        registrar("cache","Todos os grupos ja estao em cache.")
        return report
    try:
        download,creds=_market_reader_local()
    except Exception as error:
        report["status"]="local_local_indisponivel"
        report["falhas"].append({"etapa":"configuracao_local","tipo":type(error).__name__,
                                 "motivo":"local nao encontrado ou credenciais Alpaca nao disponiveis"})
        registrar("fonte",f"Leitor historico indisponivel: {type(error).__name__}. "
                 "Conferir instalacao da API local e permissao de dados.", "ERROR")
        return report
    requests=pending if MAX_GRUPOS_DOWNLOAD is None else pending[:MAX_GRUPOS_DOWNLOAD]
    registrar("coleta",f"Iniciando {len(requests)} consultas historicas 5Min RAW/SIP; "
             "a primeira execucao pode ser demorada.")
    for n,(symbol,month,sessions) in enumerate(requests,1):
        tic=time.monotonic()
        registrar("coleta",f"[{n}/{len(requests)}] {symbol} {month}, sessoes={len(sessions)}: consultando")
        file=folder/f"{symbol}_{month}.csv"
        try:
            start=pd.Timestamp(f"{month}-01",tz="America/New_York")
            end=start+pd.offsets.MonthBegin(1)
            frame=download(api_key_id=creds["api_key_id"],secret_key=creds["secret_key"],
                           symbol=symbol,timeframe="5Min",start=start.tz_convert("UTC"),
                           end=end.tz_convert("UTC"),feed="sip",adjustment="raw")
            if frame is None or frame.empty:
                report["falhas"].append({"ativo":symbol,"mes":month,"motivo":"sem_candles"})
                registrar("coleta",f"[{n}/{len(requests)}] {symbol} {month}: sem candles.", "WARN")
                continue
            frame=frame.reset_index()
            frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
            local=frame.timestamp.dt.tz_convert("America/New_York")
            mins=local.dt.hour*60+local.dt.minute
            frame=frame[(local.dt.date.astype(str).isin(set(sessions)))&(mins>=570)&(mins<960)].copy()
            if frame.empty:
                report["falhas"].append({"ativo":symbol,"mes":month,"motivo":"sem_candles_na_sessao"})
                registrar("coleta",f"[{n}/{len(requests)}] {symbol} {month}: sem candles nas sessoes Control.", "WARN")
                continue
            frame["symbol"]=symbol
            frame=frame[["timestamp","symbol","open","high","low","close","volume"]]
            tmp=file.with_suffix(".csv.tmp")
            frame.to_csv(tmp,index=False)
            os.replace(tmp,file)
            report["novos_baixados"]+=1
            registrar("coleta",f"[{n}/{len(requests)}] {symbol} {month}: "
                     f"{len(frame)} candles salvos em {time.monotonic()-tic:.1f}s.")
        except Exception as error:
            report["falhas"].append({"ativo":symbol,"mes":month,"motivo":"erro_consulta",
                                    "tipo":type(error).__name__})
            registrar("coleta",f"[{n}/{len(requests)}] {symbol} {month}: "
                     f"falhou ({type(error).__name__}), em {time.monotonic()-tic:.1f}s.", "ERROR")
    report["grupos_faltantes"]=sum(not(folder/f"{a}_{m}.csv").is_file() for a,m,_ in groups)
    report["status"]="cache_completo" if not report["grupos_faltantes"] else "cache_parcial"
    registrar("cache",f"Download concluido: novos={report['novos_baixados']}, "
             f"faltantes={report['grupos_faltantes']}, falhas={len(report['falhas'])}.")
    return report


def medir_cobertura(positions,raw):
    if positions.empty:return {"sessoes_control":0,"sessoes_cobertas":0}
    if raw.empty:return {"sessoes_control":len(positions),"sessoes_cobertas":0}
    local=pd.to_datetime(raw.timestamp,utc=True).dt.tz_convert("America/New_York")
    got=pd.DataFrame({"session":local.dt.date.astype(str),"symbol":raw.symbol})
    got=got.drop_duplicates()
    covered=positions.merge(got,on=["session","symbol"],how="inner").session.nunique()
    return {"sessoes_control":len(positions),"sessoes_cobertas":int(covered),
            "sessoes_faltantes":int(len(positions)-covered),
            "cobertura_pct":round(100*covered/len(positions),2)}


def executar():
    global _LOG_ACTIVE
    _LOG_ACTIVE=False
    _LOG_LINES.clear()
    LOG_FILE.parent.mkdir(parents=True,exist_ok=True)
    LOG_FILE.write_text("",encoding="utf-8")
    _LOG_ACTIVE=True
    registrar("inicio","Pesquisa intradiaria iniciada, somente leitura; nenhuma ordem sera enviada.")
    tic=time.monotonic()
    smoke=teste_causalidade()
    registrar("teste",f"Causalidade sintetica: {smoke['status']}.")
    positions,origem=preparar_posicoes_control()
    registrar("posicoes",f"Origem={origem['status']}; sessoes={len(positions)}; "
             f"ativos={positions.symbol.nunique() if not positions.empty else 0}.")
    aquisicao=preencher_cache_control(positions)
    registrar("fonte",f"Aquisicao={aquisicao['status']}.")
    paths=(sorted(p for p in INPUT.glob("*.csv") if p.name!="winner_positions.csv")
           +sorted((INPUT/"cotacoes_5min").glob("*.csv")))
    if not paths:
        registrar("dados","Nenhum CSV intradiario real encontrado; sem treinamento.", "WARN")
        result={"status":"aguardando_candles_intradiarios_reais","metodos":METHODS,
                "origem_control":origem,"aquisicao":aquisicao,
                "cobertura":medir_cobertura(positions,pd.DataFrame()),
                "teste_causalidade":smoke,"universo_winner_configurado":77,
                "patrimonio_final":None,"motivo":"Sem OHLCV intradiario completo; nao inventar patrimonio",
                "nenhuma_ordem_real":True}
        registrar("fim",f"Status={result['status']}, duracao={time.monotonic()-tic:.1f}s.")
        zip_atomico({"status.json":json.dumps(result,indent=2,ensure_ascii=False).encode(),
                    "codigo_executado.py":Path(__file__).read_bytes(),
                    "execucao.log":log_blob()})
        registrar("zip",f"Arquivo gerado: {OUTPUT}")
        _LOG_ACTIVE=False
        return result
    registrar("dados",f"CSV intradiarios encontrados={len(paths)}; iniciando leitura.")
    raw=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    cobertura=medir_cobertura(positions,raw)
    registrar("cobertura",f"Sessoes cobertas={cobertura.get('sessoes_cobertas',0)}/"
             f"{cobertura.get('sessoes_control',0)}; faltantes={cobertura.get('sessoes_faltantes','-')}.")
    bars=preparar(raw,informar_lacunas=True)
    registrar("dados",f"Intervalos de grade={len(bars)}; "
             f"candles efetivamente observados={int(bars.close.notna().sum())}; "
             "iniciando features.")
    positions_file=INPUT/"winner_positions.csv"
    conditioned=False
    if positions_file.exists():
        positions=pd.read_csv(positions_file)
        if not {"session","symbol"}.issubset(positions.columns):
            raise ValueError("winner_positions.csv precisa de session,symbol")
        positions["session"]=positions.session.astype(str).str[:10]
        positions["symbol"]=positions.symbol.astype(str).str.upper().str.strip()
        if positions.duplicated(["session"]).any():
            raise ValueError("Uma unica posicao selecionada por sessao")
        bars=bars.merge(positions[["session","symbol"]],on=["session","symbol"],how="inner")
        conditioned=True
    if bars.empty:
        raise ValueError("Nenhuma barra coincide com as selecoes da Winner")
    frame=features(bars)
    registrar("features",f"Linhas={len(frame)}; decisoes horarias={int(frame.horario.sum())}.")
    ranking,events,split=benchmark(frame)
    registrar("avaliacao",f"Metodos avaliados={len(ranking)}; previsoes={len(events)}.")
    # Camada economica exploratoria. Codigo independente, sem acesso a ordens.
    import importlib.util
    source_politicas=ROOT/"pesquisar_politicas_intradiarias.py"
    spec=importlib.util.spec_from_file_location("politicas_intradiarias_tcc",source_politicas)
    if spec is None or spec.loader is None:
        raise RuntimeError("Modulo de pesquisa das politicas nao encontrado")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    funcoes={"rolling":rolling_scores,"bocpd":bocpd,"hmm":hmm,
             "lgb":lgb_scores,"fqi":fqi,"split":split_cronologico}
    extras,manifesto_politicas=mod.executar(frame,positions,funcoes,registrar=registrar)
    registrar("politicas",f"Replay concluido: periodo={manifesto_politicas['periodo_teste']}; "
             "capital intradiario comparado ao Control intradiario, "
             "nao ao Control diario original.")
    result={"status":("piloto" if cobertura.get("sessoes_faltantes",1)==0 else "piloto_parcial")
            if not ranking.empty else "amostra_insuficiente",
            "origem_control":origem,"aquisicao":aquisicao,"cobertura":cobertura,
            "teste_causalidade":smoke,"sessoes":int(frame.session.nunique()),
            "ativos":sorted(frame.symbol.unique().tolist()),"split":split,
            "condicionado_a_posicoes_winner":conditioned,
            "patrimonio_final":None,
            "aviso": "Sem winner_positions.csv: comparacao metodologica, nao politica de permanencia da Winner" if not conditioned else "Amostra condicionada ao ativo selecionado; sem replay de patrimonio",
            "motivo_patrimonio":"Patrimonio do Control original nao reproduzido pelo replay intradiario: "
                                "simulacao continua com compra integral hipotetica no fechamento; "
                                "nao e fill comprovado; consultar manifesto das politicas",
            "politicas_intradiarias":{"status":"proxy_gerado",
                "capital_final_original_comparavel":False,
                "cenarios":[10000.0,68160546.38689804]},
            "nenhuma_ordem_real":True}
    registrar("fim",f"Status={result['status']}, duracao={time.monotonic()-tic:.1f}s.")
    zip_atomico({"status.json":json.dumps(result,indent=2,ensure_ascii=False,default=str).encode(),
                 "comparacao_metodos.csv":ranking.to_csv(index=False).encode(),
                 "previsoes_horarias.csv":events.to_csv(index=False).encode(),
                 "codigo_executado.py":Path(__file__).read_bytes(),
                 "codigo_politicas_executado.py":source_politicas.read_bytes(),
                 "execucao.log":log_blob(), **extras})
    registrar("zip",f"Arquivo gerado: {OUTPUT}")
    _LOG_ACTIVE=False
    return result


if __name__=="__main__":
    try:
        print(json.dumps(executar(),ensure_ascii=False,indent=2,default=str))
        print("ZIP:",OUTPUT)
    except Exception as exc:
        registrar("erro",f"Execucao interrompida: {type(exc).__name__}; "
                 f"consulte o traceback no console. Log parcial: {LOG_FILE}", "ERROR")
        raise
