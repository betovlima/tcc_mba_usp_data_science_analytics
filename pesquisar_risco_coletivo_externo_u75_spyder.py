"""TCC / Spyder F5: risco coletivo estatistico + busca externa condicionada.

Fase P1 shadow: NAO opera, NAO baixa dados, NAO abre banco, NAO altera MCT.
Reutiliza os dois snapshots congelados, top-1 do MCT e 24 candidatos.
O detector nao e parametrizado pelo capital historico; nao cria replay.
"""
# %% 0. Contrato e parametros cientificos
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pesquisa.localizacao_zip_u75 import locate_mct_zip
from pesquisar_busca_externa_lider_u75_spyder import (
    load_snapshots, read_leaders, build_feature_rows,
)
from pesquisa.risco_coletivo_externo_u75 import (
    VERSION, StatisticalSettings, build_collective_panel,
    evaluate_regime_walkforward, shadow_external_choices,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'risco_coletivo_externo_u75'
SETTINGS = StatisticalSettings(
    covariance_window=126, horizon=5, min_assets=68,
    min_train_dates=300, target_tail_quantile=0.05,
)


def _store_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False, float_format='%.17g')


def _report(oos: pd.DataFrame, choices: pd.DataFrame) -> dict:
    measured = oos.loc[oos.future_observable & oos.realized_tail_event.notna()].copy()
    rows = []
    for year, group in measured.groupby(measured.decision_date.str[:4], sort=True):
        pos=group.loc[group.risk_alert]
        negative=group.realized_tail_event.astype(bool)
        hits=int((group.risk_alert & negative).sum())
        false=int((group.risk_alert & ~negative).sum())
        misses=int((~group.risk_alert & negative).sum())
        rows.append({'year':year, 'measured_sessions':len(group),
                     'alerts':int(group.risk_alert.sum()), 'tail_events':int(negative.sum()),
                     'hits':hits,'false_alerts':false,'missed_events':misses,
                     'event_recall':hits/(hits+misses) if hits+misses else None,
                     'alert_precision':hits/(hits+false) if hits+false else None,
                     'brier_score':float(np.mean((group.predicted_tail_probability - negative.astype(float))**2)),
                     'baseline_brier':float(np.mean((group.prior_event_probability - negative.astype(float))**2))})
    table=pd.DataFrame(rows)
    _store_csv(OUT/'metricas_risco_por_ano.csv',table)
    sc=choices.loc[choices.status.eq('ok')] if len(choices) else choices
    return {'total_oos_signals':len(oos),
            'measured_oos_signals':len(measured),
            'risk_alerts':int(oos.risk_alert.sum()),
            'measured_alerts':int(measured.risk_alert.sum()),
            'tail_events_measured':int(measured.realized_tail_event.fillna(False).sum()),
            'shadow_choices_with_outcome':int(sc.outcome_available.sum()) if len(sc) else 0,
            'shadow_decisions':sc.decision_shadow.value_counts().to_dict() if len(sc) else {},
            'mean_intraday_delta_observed_log':float(sc.loc[sc.outcome_available,'delta_intraday_log'].mean())
                if len(sc) and bool(sc.outcome_available.any()) else None,
            'per_year':rows}


def _plot(oos: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = pd.to_datetime(oos.decision_date, utc=True)
    ax.plot(x, oos.predicted_tail_probability, label='P(queda extrema em 5 pregoes)', lw=1)
    ax.plot(x, oos.prior_event_probability, label='Risco historico incondicional', alpha=.75)
    selected=oos.loc[oos.risk_alert]
    ax.scatter(pd.to_datetime(selected.decision_date,utc=True),
               selected.predicted_tail_probability, s=9, label='Alertas predefinidos')
    ax.set_ylabel('Probabilidade estimada'); ax.set_xlabel('Data da decisao')
    ax.set_title('Risco estatistico do universo U75: sinal calculado no fechamento')
    ax.legend(loc='best'); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output, dpi=140); plt.close(fig)


def _run():
    OUT.mkdir(parents=True,exist_ok=True)
    # %% 1. Identidade e snapshots originais: mesmo protocolo auditado da busca anterior
    base, external, universe, config, source = load_snapshots()
    mct = locate_mct_zip(ROOT, explicit=os.environ.get('MCT_BACKTEST_ZIP',''))
    leaders, archive = read_leaders(mct, universe)
    if len(universe)!=75 or len(external)!=24:
        raise RuntimeError('U75 ou shortlist externa nao corresponde ao congelamento')
    dates=pd.DatetimeIndex(pd.to_datetime(leaders.decision_date, utc=True).dt.normalize())
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise RuntimeError('Datas do ranking nao sao cronologicas e unicas')
    print(f'[fonte] U75={len(universe)} externos={len(external)} decisoes={len(dates)}',flush=True)

    # %% 2. Covariancia causal Ledoit-Wolf, Mahalanobis, PCA e breadth
    collective=build_collective_panel(base, universe, dates, settings=SETTINGS)
    _store_csv(OUT/'indice_coletivo_u75.csv',collective)
    print('[matematica] covariancia, turbulencia e PCA calculados', flush=True)

    # %% 3. Previsao por analogos estatisticos, folds anuais e rótulos maturados
    risk,folds=evaluate_regime_walkforward(collective,settings=SETTINGS)
    if risk.empty:
        raise RuntimeError('Sem teste temporal OOS: verifique historico e cobertura')
    _store_csv(OUT/'previsoes_risco_oos.csv',risk)
    _store_csv(OUT/'folds_cronologicos.csv',folds)
    _plot(risk,OUT/'probabilidade_queda_u75.png')
    print(f'[risco] OOS={len(risk)} alertas={int(risk.risk_alert.sum())}',flush=True)

    # %% 4. Somente para alertas: assinatura do Top-1 versus candidatos externos
    # Esta parte nao altera o detector e nao usa outcome futuro para a escolha.
    if bool(risk.risk_alert.any()):
        alerts=leaders.loc[pd.to_datetime(leaders.decision_date,utc=True).dt.strftime('%Y-%m-%d').isin(
            set(risk.loc[risk.risk_alert,'decision_date']))]
        all_bars={**base,**external}
        feature_rows=build_feature_rows(alerts,all_bars,config)
        choices=shadow_external_choices(risk,leaders,feature_rows,universe)
    else:
        choices=pd.DataFrame(columns=['decision_date','status','decision_shadow','outcome_available'])
        print('[externo] Nenhum alerta: nao relaxar o limiar para inventar sinais',flush=True)
    _store_csv(OUT/'decisoes_externas_shadow.csv',choices)

    # %% 5. Auditoria e relatorio sem transformar ganho intradiario em patrimonio
    summary=_report(risk,choices)
    report={'script_version':VERSION,'status':'completed','settings':SETTINGS.__dict__,
            'reference_backtest':archive,'source':source,
            'method':'LedoitWolf previous 126 sessions; contemporaneous Mahalanobis + PC1; historical 5-session tail and kNN sqrt(N) analogues with yearly OOS; one-sided 95% approximate signal; risk-adjusted 20-day momentum external benchmark',
            'summary':summary,
            'limitations':[
                'Primeira fase P1 SHADOW: intraday da sessao seguinte nao e replay financeiro de posicoes, CASH ou custos.',
                'Top-1 vem de raw_best_asset do ZIP MCT, e nao da posicao efetiva; a elegibilidade historica de t+1 pode contaminar o ranking e exige auditoria separada.',
                'Os 24 ativos externos foram escolhidos em pesquisa anterior; isto nao demonstra busca prospectiva independente de todo o mercado.',
                'Covariancia e PCA sao observacoes do fechamento t. Sua capacidade de prever risco depende exclusivamente da validacao temporal.',
                'O criterio de alerta e aproximacao unilateral com analogos dependentes no tempo; validar intervalo por blocos antes de transformar em decisao.',
                'Retorno futuro de cinco pregoes serve somente para rotular. A escolha externa observa exclusivamente atributos presentes em t.',
                'A avaliacao externa usa apenas retorno abertura-fechamento seguinte e ignora overnight, permanencia, custos e slippage.',
                'O universo U75 de referencia e reequilibrado equal-weight apenas no indice diagnostico; nao e o portfolio Control.',
            ]}
    (OUT/'resultado.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    (OUT/'resumo.md').write_text('# Pesquisa estatistica de regime e oportunidades externas ao U75\n\n'+
        json.dumps(summary,indent=2,ensure_ascii=False,default=str)+'\n\n'+
        '\n'.join('- '+s for s in report['limitations']),encoding='utf-8')
    filename=OUT/f'pacote_risco_coletivo_externo_u75_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.zip'
    with ZipFile(filename,'w',ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            if f.is_file() and f != filename and f.suffix in ('.csv','.json','.png','.md'):
                z.write(f,arcname=f.name)
    print('[pacote]', filename, flush=True)
    return report


def main():
    try: return _run()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'ultimo_erro.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
