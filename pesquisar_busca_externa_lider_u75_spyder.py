"""TCC: descobrir alternativas externas ao líder do U75.

Spyder: abra e execute F5. Usa apenas snapshots CSV e ZIPs CONGELADOS.
NÃO executa ordens, NÃO usa MongoDB, NÃO baixa preços, NÃO promove MCT.

Antes do F5:
  1. Ter o snapshot Strategy #30 validado em dados/strategy30_snapshot.
  2. Ter a shortlist congelada (24 ativos) e respectivo snapshot auditado em
     dados/cluster_candidate_search (produzido pelas pesquisas anteriores).
  3. Copiar o ZIP MCT com o ranking U75 para evidencias/, ou definir
     MCT_BACKTEST_ZIP com seu caminho absoluto.
  4. Disponibilizar o ZIP Control Strategy #30 conforme o executor original
     pesquisar_diagnostico_u75_spyder.py (CONTROL_STRATEGY30_ZIP).

Há duas hipóteses avaliadas sobre o retorno abertura->fechamento em t+1:
  A) o candidato externo MAIS SIMILAR ao líder U75 ganha do líder;
  B) um modelo Ridge treinado só no passado identifica melhor substituição.
A segunda hipótese só escolhe candidato quando prevê vantagem positiva.
Esse é um diagnóstico por pregão, NÃO um replay de carteira com retenção,
slippage real, impostos ou rotações. A lista de 24 candidatos foi selecionada
no passado e não equivale a descoberta nested de todo o mercado.
"""
from __future__ import annotations

from datetime import datetime,timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pesquisa.busca_externa_lider_u75 import (
    FEATURES, SCRIPT_VERSION, ResearchSettings,
    make_panel,evaluate_expanding_years,summarize,
)
from pesquisa.control_operacional_strategy30 import carregar_contrato, validar_control_strategy30
from pesquisa.paridade_strategy30 import load_snapshot_strategy30,config_from_frozen_manifest
from reproducao.dados import SnapshotPaths,validate_snapshot
from reproducao.preparacao import load_raw_bar_file,load_actions_file,split_normalize
from pesquisa.referencia_strategy30.rotacao import construir_quadro_rotacao
from pesquisar_controle_strategy30_spyder import localizar_zip_original

# %% 0. Contrato de pesquisa
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'/'busca_externa_lider_u75'
CANDIDATE_ROOT=ROOT/'dados'/'cluster_candidate_search'
CANDIDATE_SNAPSHOT=SnapshotPaths.from_root(CANDIDATE_ROOT/'snapshot')
FREEZE=CANDIDATE_ROOT/'cluster_shortlist_freeze.json'
RESEARCH=ResearchSettings()
# Se múltiplos ZIPs MCT existirem, informe explicitamente a variável de ambiente.
BACKTEST_ZIP=os.environ.get('MCT_BACKTEST_ZIP','').strip()


def locate_mct_zip()->Path:
    if BACKTEST_ZIP:
        path=Path(BACKTEST_ZIP).expanduser()
        if not path.is_file():raise FileNotFoundError(path)
        return path
    roots=[ROOT/'evidencias',ROOT,Path.home()/'Downloads']
    found=sorted({p.resolve() for root in roots if root.exists() for p in root.glob('market_cycle_trader_*.zip')})
    if len(found)!=1:
        raise RuntimeError(f'Esperado um ZIP MCT ou MCT_BACKTEST_ZIP explícito; encontrados={len(found)}')
    return found[0]


def read_leaders(path:Path,universe:set[str])->tuple[pd.DataFrame,dict]:
    data=path.read_bytes()
    with ZipFile(BytesIO(data)) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP corrompido')
        preds=[n for n in z.namelist() if n.endswith('_predictions.csv')]
        if len(preds)!=1:raise RuntimeError('ZIP deve ter uma previsão U75 única')
        manifest=json.loads(z.read('strategy_manifest.json'))
        params=manifest.get('effective_configuration') or {}
        if params.get('research_market_data_protocol')!='raw_total_causal_v1':
            raise RuntimeError('Protocolo do backtest não coincide com RAW causal')
        assets=set(params.get('assets') or [])
        if universe-assets or assets-universe not in (set(), {'DOC','CLMT'}):
            raise RuntimeError(f'Universo do backtest não corresponde ao U75: extras={assets-universe}, ausentes={universe-assets}')
        df=pd.read_csv(BytesIO(z.read(preds[0])))
        required={'timestamp','decision_date','raw_best_asset'}
        if not required<=set(df.columns):raise RuntimeError(f'Diagnóstico de ranking ausente {required-set(df.columns)}')
        # O líder bruto é o MAIOR SCORE, NÃO necessariamente a posição mantida.
        leaders=df[['decision_date','timestamp','raw_best_asset']].rename(columns={
            'timestamp':'execution_date','raw_best_asset':'leader'})
        if leaders['leader'].isna().any():raise RuntimeError('Há pregão sem líder bruto')
        return leaders,{'zip_name':path.name,'zip_sha256':sha256(data).hexdigest(),
                        'strategy_hash':manifest.get('strategy_configuration_hash'),
                        'source':'raw_best_asset', 'predictions_rows':len(df)}


def load_snapshots():
    # Mesma validação oficial da pesquisa anterior. Não criar snapshot novo.
    contract=carregar_contrato()
    reference_zip=localizar_zip_original()
    validar_control_strategy30(reference_zip,contrato=contract)
    with ZipFile(reference_zip) as z:
        strategy=json.loads(z.read('strategy_manifest.json'))
        experiment=json.loads(z.read('experiment_manifest.json'))
    reference=load_snapshot_strategy30(ROOT/'dados'/'strategy30_snapshot',
                                     contract=contract,experiment_manifest=experiment)
    config=config_from_frozen_manifest(strategy,contract)
    if not FREEZE.is_file():raise FileNotFoundError(f'Shortlist congelada ausente: {FREEZE}')
    frozen=json.loads(FREEZE.read_text(encoding='utf-8'))
    if (frozen.get('script_version')!='cluster-shortlist-freeze-v1.0.0' or
        frozen.get('selection_locked') is not True or
        frozen.get('financial_outcomes_used_for_selection') is not False or
        frozen.get('backtest_executed_before_freeze') is not False):
        raise RuntimeError('Shortlist não comprova seleção prévia sem resultados financeiros')
    external_manifest=validate_snapshot(CANDIDATE_SNAPSHOT)
    candidate_audit=CANDIDATE_ROOT/'cluster_candidate_snapshot.json'
    if not candidate_audit.is_file():
        raise RuntimeError('Manifesto de auditoria do snapshot de candidatos não encontrado')
    audit_payload=json.loads(candidate_audit.read_text(encoding='utf-8'))
    if audit_payload.get('script_version')!='cluster-candidate-snapshot-v1.0.0' or audit_payload.get('status')!='candidate_snapshot_ready' or audit_payload.get('structural_exclusions'):
        raise RuntimeError('Snapshot de candidatos sem auditoria estrutural válida')
    candidates=[str(x).strip().upper() for x in frozen.get('assets') or []]
    if len(candidates)!=24 or len(set(candidates))!=24:raise RuntimeError('A shortlist auditada exige 24 ativos únicos')
    if set(candidates)!=set(external_manifest.get('assets') or []):
        raise RuntimeError('Ativos de shortlist e snapshot diferem')
    external={}
    for sym in candidates:
        raw=load_raw_bar_file(CANDIDATE_SNAPSHOT.raw_bars/f'{sym}.csv',float_precision='round_trip')
        actions=load_actions_file(CANDIDATE_SNAPSHOT.corporate_actions/f'{sym}.csv')
        bars,_=split_normalize(raw,actions)
        external[sym]=bars
    universe=set(contract['assets'])
    overlap=universe.intersection(external)
    external={k:v for k,v in external.items() if k not in universe}
    if not external:raise RuntimeError('Todos os candidatos da shortlist já estão no U75')
    return reference.frames,external,universe,config,{
        'reference_zip':reference_zip.name,
        'reference_snapshot':experiment.get('market_data_signature_sha256'),
        'candidate_snapshot':external_manifest.get('snapshot_sha256'),
        'candidate_shortlist':candidates,
        'candidates_overlapping_u75':sorted(overlap),
        'external_candidates':sorted(external),
    }


def build_feature_rows(leaders:pd.DataFrame,frames:dict,config)->pd.DataFrame:
    """Apenas os 12 atributos da rotação; preço de t+1 fica separado."""
    dates=leaders[['decision_date','execution_date']].copy()
    dates['decision_date']=pd.to_datetime(dates['decision_date'],utc=True).dt.strftime('%Y-%m-%d')
    dates['execution_date']=pd.to_datetime(dates['execution_date'],utc=True).dt.strftime('%Y-%m-%d')
    if dates.duplicated('decision_date').any():raise RuntimeError('Decisão duplicada')
    parts=[]
    for n,(symbol,bars) in enumerate(sorted(frames.items()),1):
        feat=construir_quadro_rotacao(bars,config)
        feat=feat.copy()
        feat.index=pd.to_datetime(feat.index,utc=True).strftime('%Y-%m-%d')
        if feat.index.has_duplicates:raise RuntimeError(f'Duplicatas em {symbol}')
        missing=set(FEATURES)-set(feat.columns)
        if missing:raise RuntimeError(f'{symbol}: faltam features originais {missing}')
        # Lookup do outcome por data de EXECUÇÃO, não pelo último fechamento.
        rows=dates.copy()
        for name in FEATURES:
            rows[name]=feat[name].reindex(rows['decision_date']).to_numpy()
        rows['execution_open']=feat['open'].reindex(rows['execution_date']).to_numpy()
        rows['execution_close']=feat['close'].reindex(rows['execution_date']).to_numpy()
        rows['symbol']=symbol
        parts.append(rows)
        if n%25==0 or n==len(frames):print(f'[features] {n}/{len(frames)}',flush=True)
    return pd.concat(parts,ignore_index=True)


def plot_years(picks:pd.DataFrame,out:Path)->None:
    metric=picks.groupby('year',sort=True).agg(
        nearest=('nearest_delta','mean'),model=('model_delta','mean'))
    fig,ax=plt.subplots(figsize=(10,4))
    metric.plot(ax=ax,marker='o')
    ax.axhline(0,linewidth=1)
    ax.set_title('Excesso médio de retorno diário abertura-fechamento sobre líder U75 (OOS)')
    ax.set_ylabel('Diferença em log-retorno por pregão')
    ax.set_xlabel('Ano OOS')
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


def main()->dict:
    OUT.mkdir(parents=True,exist_ok=True)
    # %% 1. Dados/identidade
    base,external,universe,config,source=load_snapshots()
    mct=locate_mct_zip()
    leaders,archive_meta=read_leaders(mct,universe)
    # %% 2. União por data, sem reamostrar e sem reconstruir séries
    raw=build_feature_rows(leaders,{**base,**external},config)
    # %% 3. Ranking de similaridade e meta-regressor com expanding-window
    panel,coverage=make_panel(leaders,raw,universe,settings=RESEARCH)
    picks,folds=evaluate_expanding_years(panel,settings=RESEARCH)
    if picks.empty:raise RuntimeError('Não há treino cronológico suficiente para gerar OOS; consulte cobertura')
    summary=summarize(picks)
    # %% 4. Auditoria e exportação
    for name,df in [('candidate_pairs.csv',panel),('decision_coverage.csv',coverage),
                    ('oos_decisions.csv',picks),('expanding_folds.csv',folds)]:
        df.to_csv(OUT/name,index=False)
    plot_years(picks,OUT/'excesso_medio_anual.png')
    result={'version':SCRIPT_VERSION,'status':'completed','source_backtest':archive_meta,
            'data':source,'settings':RESEARCH.__dict__,'summary':summary,
            'limitations':['Candidatos externos vêm de shortlist histórica de 24, não de busca nested do mercado inteiro.',
                          'O líder do U75 usa raw_best_asset da execução existente; elegibilidade no motor original precisa de auditoria separada de futuro.',
                          'Outcome é retorno abertura-fechamento da sessão seguinte; não é replay de capital, política de holding, slippage ou taxas de operação.',
                          'Somente preços/targets já maduros entram no treino de cada ano; ausência de preço futuro é marcada, nunca usada para selecionar o candidato.']}
    (OUT/'resultado.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    (OUT/'resumo.md').write_text('# Pesquisa de candidato externo versus líder U75\n\n'+
         'Versão: '+SCRIPT_VERSION+'\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+
         '\n'.join('- '+v for v in result['limitations'])+'\n',encoding='utf-8')
    pack=OUT/f'pacote_busca_externa_lider_u75_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.zip'
    with ZipFile(pack,'w',ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            if f.is_file() and f != pack and f.suffix in ('.csv','.json','.md','.png'):
                z.write(f,arcname=f.name)
    print('[concluído] ',pack, '\n',json.dumps(summary,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    main()
