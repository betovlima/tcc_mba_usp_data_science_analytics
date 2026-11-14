"""Testes de pesquisa: sem Alpaca, chaves, dados reais ou ordens."""
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/"pesquisar_segunda_camada_intradiaria_spyder.py"
SPEC=importlib.util.spec_from_file_location("segunda_camada_intradiaria",SCRIPT)
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def make_bars(days=18,symbol="SIM"):
    rng=np.random.default_rng(20261114)
    days_index=pd.bdate_range("2026-03-02",periods=days)
    rows=[]
    for day in days_index:
        # 09h30 NY (em marco/2026 ja havia horario de verao)
        start=pd.Timestamp(f"{day.date()}T09:30:00",tz="America/New_York").tz_convert("UTC")
        price=100.
        for i in range(78):
            nxt=price*np.exp(rng.normal(0,.005))
            rows.append({"timestamp":start+pd.Timedelta(minutes=i*5),
                         "symbol":symbol,"open":price,"high":max(price,nxt)*1.001,
                         "low":min(price,nxt)*.999,"close":nxt,
                         "volume":int(rng.integers(1000,20000))})
            price=nxt
    return pd.DataFrame(rows)


def test_causalidade_e_horarios():
    result=mod.teste_causalidade()
    assert result["status"]=="aprovado"
    assert result["decisoes_horarias"]==6


def test_rejeita_candles_duplicados():
    x=make_bars(days=1).iloc[:1]
    with pytest.raises(ValueError,match="repetidos"):
        mod.preparar(pd.concat([x,x],ignore_index=True))


def test_todas_as_tecnicas_mesmo_teste_cronologico():
    pytest.importorskip("sklearn")
    pytest.importorskip("lightgbm")
    frame=mod.features(mod.preparar(make_bars()))
    scores,preds,split=mod.benchmark(frame)
    assert set(mod.METHODS)==set(scores.metodo)
    assert scores.loc[scores.metodo=="ofi_l1","status"].iloc[0]=="nao_avaliavel_sem_quotes_L1"
    assert (scores[scores.metodo!="ofi_l1"].status=="ok").all()
    assert split["treino"][-1]<split["teste"][0]
    assert not preds.empty
    assert all(h in {"10:25","11:25","12:25","13:25","14:25","15:25"}
               for h in pd.to_datetime(preds.timestamp_utc,utc=True)
                          .dt.tz_convert("America/New_York").dt.strftime("%H:%M").unique())


def test_sem_dados_nao_inventa_patrimonio(monkeypatch,tmp_path):
    monkeypatch.setattr(mod,"INPUT",tmp_path/"sem_dados")
    monkeypatch.setattr(mod,"OUTPUT",tmp_path/"output"/"pesquisa_vantagem_rotacao.zip")
    x=mod.executar()
    assert x["patrimonio_final"] is None
    assert x["status"]=="aguardando_candles_intradiarios_reais"
    with ZipFile(mod.OUTPUT) as z:
        assert z.testzip() is None
        report=json.loads(z.read("segunda_camada_intradiaria/status.json"))
        assert report["patrimonio_final"] is None


def test_restringe_ao_ativo_selecionado(monkeypatch,tmp_path):
    frames=[make_bars(days=9,symbol="AAA"),make_bars(days=9,symbol="BBB")]
    path=tmp_path/"dados_intraday";path.mkdir()
    pd.concat(frames).to_csv(path/"barras.csv",index=False)
    sessions=sorted(pd.concat(frames).timestamp.dt.tz_convert("America/New_York")
                    .dt.date.astype(str).unique())
    pd.DataFrame({"session":sessions,"symbol":["AAA"]*len(sessions)}).to_csv(
        path/"winner_positions.csv",index=False)
    monkeypatch.setattr(mod,"INPUT",path)
    monkeypatch.setattr(mod,"OUTPUT",tmp_path/"output"/"pesquisa_vantagem_rotacao.zip")
    result=mod.executar()
    assert result["condicionado_a_posicoes_winner"] is True
    assert result["ativos"]==["AAA"]
    assert result["patrimonio_final"] is None
