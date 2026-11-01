"""Pesquisa estatistica U75: risco coletivo e alternativas externas em modo shadow.

Nenhuma decisao consulta retornos futuros. Os retornos posteriores sao gerados
em coluna isolada para avaliar sinais que ja foram produzidos.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

VERSION = 'u75-risk-gated-external-v1.0.0'
RISK_COLUMNS = ('negative_breadth_20', 'negative_index_return_20',
                'mahalanobis', 'pc1_variance_share', 'negative_index_return_1')


@dataclass(frozen=True)
class StatisticalSettings:
    covariance_window: int = 126
    horizon: int = 5
    min_assets: int = 68
    min_train_dates: int = 300
    target_tail_quantile: float = 0.05
    one_sided_z: float = 1.645


def _date_index(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).normalize()
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError('Calendario deve ser crescente, sem dias duplicados')
    return idx


def build_collective_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    universe: set[str],
    decision_dates,
    *,
    settings: StatisticalSettings = StatisticalSettings(),
) -> pd.DataFrame:
    """Calcula atributos em t e um rotulo ex post completamente separado.

    A matriz de covariancia usa exclusivamente t-126 .. t-1; o vetor da
    distancia usa o retorno disponivel no fechamento t. Nao preenche series.
    O indice equal-weight e rebalanceado a cada fechamento, apenas diagnostico.
    """
    if len(universe) != 75 or set(bars_by_symbol) != set(universe):
        raise ValueError('A pesquisa requer exatamente os 75 ativos congelados')
    dates = _date_index(decision_dates)
    # Uniao de calendarios historicos sem interpolacao de fechamento ou volumes.
    closes = {}
    for sym in sorted(universe):
        f = bars_by_symbol[sym]
        idx = _date_index(f.index)
        series = pd.Series(pd.to_numeric(f['close'], errors='coerce').to_numpy(float), index=idx)
        series = series.where(series > 0)
        closes[sym] = series
    close = pd.DataFrame(closes).sort_index()
    # Historico completo fica acessivel apenas para rotulos ex post; nunca para features em t.
    if not len(dates):
        raise ValueError('Nenhuma data de decisao')
    # pct_change default de pandas depende da versao: preenchimento proibido.
    pct = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    logs = np.log1p(pct)
    coverage = pct.notna().sum(axis=1)
    daily_index = np.log1p(pct.mean(axis=1))
    daily_index = daily_index.where(coverage >= settings.min_assets)
    # Return de dia t+1...t+h: rotulo exclusivamente de avaliacao.
    future = sum(daily_index.shift(-j) for j in range(1, settings.horizon + 1))
    historical_20 = np.log(close / close.shift(20))
    records = []
    for i, date in enumerate(dates):
        if i and i % 100 == 0:
            print(f'[matematica] U75 dia={i}/{len(dates)}', flush=True)
        row = {'decision_date':date.strftime('%Y-%m-%d'), 'status':'insufficient_history',
               'valid_assets_t':int(coverage.get(date, 0)),
               'maturity_date':None, 'realized_future_log_return':np.nan}
        if date not in close.index:
            row['status'] = 'missing_calendar_date'; records.append(row); continue
        pos = close.index.get_loc(date)
        if pos + settings.horizon < len(close.index):
            maturity = close.index[pos + settings.horizon]
            row['maturity_date'] = maturity.strftime('%Y-%m-%d')
            row['realized_future_log_return'] = float(future.loc[date])
        if pos < settings.covariance_window + 20:
            records.append(row); continue
        old = logs.iloc[pos - settings.covariance_window:pos]
        current = logs.iloc[pos]
        # Mesmo conjunto completo de ativos e dias para covariancia e PCA.
        valid = old.notna().all(axis=0) & current.notna()
        eligible = valid[valid].index
        if len(eligible) < settings.min_assets:
            row['status'] = 'insufficient_coverage'; records.append(row); continue
        samples = old[eligible].to_numpy(float)
        x = current[eligible].to_numpy(float)
        if not np.isfinite(samples).all() or not np.isfinite(x).all():
            row['status'] = 'nonfinite_market_data'; records.append(row); continue
        model = LedoitWolf().fit(samples)
        residual = x - model.location_
        covariance = model.covariance_
        turbulence = max(0., float(residual @ np.linalg.solve(covariance, residual)))
        diagonal = np.sqrt(np.diag(covariance))
        corr = covariance / np.outer(diagonal, diagonal)
        eigenvalues = np.linalg.eigvalsh(corr)
        pc1 = float(max(eigenvalues[-1], 0.) / np.trace(corr))
        h20 = historical_20.loc[date, eligible].dropna()
        if len(h20) < settings.min_assets:
            row['status'] = 'insufficient_breadth'; records.append(row);continue
        ret20 = daily_index.iloc[pos-19:pos+1]
        if not ret20.notna().all() or not np.isfinite(daily_index.iloc[pos]):
            row['status'] = 'insufficient_index';records.append(row);continue
        row.update(status='ok', negative_breadth_20=float((h20 < 0).mean()),
                   negative_index_return_20=float(-ret20.sum()),
                   negative_index_return_1=float(-daily_index.iloc[pos]),
                   mahalanobis=float(np.log1p(turbulence)),
                   pc1_variance_share=pc1,
                   daily_index_log_return=float(daily_index.iloc[pos]),
                   covariance_assets=int(len(eligible)))
        records.append(row)
    return pd.DataFrame(records)


def _es95_of_losses(future_returns: np.ndarray, alpha: float) -> float:
    losses = -np.asarray(future_returns, dtype=float)
    if not len(losses):
        return float('nan')
    q = float(np.quantile(losses, 1 - alpha))
    return float(losses[losses >= q].mean())


def evaluate_regime_walkforward(panel: pd.DataFrame, *, settings=StatisticalSettings()):
    """Fold anual; refit do scaler e analogo somente em datas ANTERIORES.

    Risco do quantil 5% definido usando TREINO por fold. Evento em teste
    sempre mede o limiar do treino, nao um quantil conhecido a posteriori.
    """
    p = panel.copy().sort_values('decision_date')
    p['year'] = p['decision_date'].str.slice(0, 4).astype(int)
    oos, audit = [], []
    for year in sorted(p.year.unique()):
        test = p.loc[(p.year == year) & p.status.eq('ok')]
        if test.empty: continue
        first = str(test.decision_date.min())
        train = p.loc[p.status.eq('ok') & (p.maturity_date < first) &
                      p.realized_future_log_return.notna()].copy()
        if train.decision_date.nunique() < settings.min_train_dates:
            continue
        assert train['maturity_date'].max() < first
        train = train.dropna(subset=list(RISK_COLUMNS))
        x_train = train[list(RISK_COLUMNS)].to_numpy(float)
        scaler = StandardScaler().fit(x_train)
        z_train = scaler.transform(x_train)
        observed = train.realized_future_log_return.to_numpy(float)
        cutoff = float(np.quantile(observed, settings.target_tail_quantile))
        events = observed <= cutoff
        p0 = float((events.sum() + 1) / (len(events) + 2))
        k = min(len(train), max(10, ceil(sqrt(len(train)))))
        for _, record in test.iterrows():
            features = record[list(RISK_COLUMNS)].to_numpy(dtype=float).reshape(1, -1)
            z = scaler.transform(features)[0]
            distance2 = ((z_train-z)**2).sum(axis=1)
            nearest = np.argsort(distance2, kind='stable')[:k]
            matched = observed[nearest]
            hits = int((matched <= cutoff).sum())
            p1 = float((hits + 1) / (k + 2))
            # Teste de sinal unilateral 95% aproximado contra risco incondicional.
            # A aproximacao e declarada; dependencias temporais exigem auditoria.
            se = float(sqrt(p0*(1-p0)/k))
            alert = bool(p1 > p0 + settings.one_sided_z*se and matched.mean() < 0)
            available = bool(pd.notna(record.realized_future_log_return))
            oos.append({**record.to_dict(), 'train_end_maturity':str(train.maturity_date.max()),
                        'tail_cutoff_log':cutoff, 'prior_event_probability':p0,
                        'predicted_tail_probability':p1, 'conditional_es95':
                        _es95_of_losses(matched, settings.target_tail_quantile),
                        'unconditional_es95':_es95_of_losses(observed, settings.target_tail_quantile),
                        'neighbors':k, 'risk_alert':alert,
                        'future_observable':available,
                        'realized_tail_event':(bool(float(record.realized_future_log_return) <= cutoff)
                                               if available else None)})
        audit.append({'year':year,'train_dates':len(train), 'train_maturity_max':train.maturity_date.max(),
                      'test_start':first,'test_dates':len(test),'neighbors':k,
                      'tail_cutoff_log':cutoff,'base_rate':p0})
    return pd.DataFrame(oos), pd.DataFrame(audit)


def shadow_external_choices(risk: pd.DataFrame, leaders: pd.DataFrame,
                            feature_rows: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """Decisao em t: momentum padronizado pelo risco, sem ler precos de t+1.

    E somente um benchmark de elegibilidade do candidato. Resultado futuro
    da proxima sessao e acrescentado DEPOIS da decisao para diagnostico.
    """
    from pesquisa.busca_externa_lider_u75 import _date
    l = leaders.copy()
    l['decision_date'] = l['decision_date'].map(_date)
    leader_by_day = l.set_index('decision_date')['leader'].to_dict()
    rows = feature_rows.copy()
    rows['decision_date'] = rows['decision_date'].map(_date)
    rows['symbol'] = rows['symbol'].astype(str).str.upper()
    result = []
    for record in risk.itertuples(index=False):
        if not bool(record.risk_alert):continue
        day = record.decision_date
        top = leader_by_day.get(day)
        d = rows.loc[rows.decision_date.eq(day)]
        d = d.drop_duplicates(subset=['symbol']).set_index('symbol')
        if not top or top not in universe or top not in d.index:
            result.append({'decision_date':day,'status':'missing_top1'});continue
        def score(rec):
            m = float(rec['return_20']); vol = float(rec['vol_20'])
            if not (np.isfinite(m) and np.isfinite(vol) and vol > 0):
                return float('nan')
            return m/(sqrt(20)*vol)
        top_score=score(d.loc[top])
        if not np.isfinite(top_score):
            result.append({'decision_date':day,'status':'missing_top1_features'});continue
        candidates=[]
        for symbol in sorted(set(d.index)-universe):
            value=score(d.loc[symbol])
            if np.isfinite(value):candidates.append((value,symbol))
        # Abertura e fechamento seguintes NAO entram na pontuacao acima.
        candidates.sort(key=lambda x:(-x[0],x[1]))
        best_score,best = candidates[0] if candidates else (float('nan'),None)
        if not candidates:
            result.append({'decision_date':day, 'status':'no_external_snapshot',
                           'top1':top, 'top1_score':top_score});continue
        if best is not None and best_score > max(0., top_score):
            choice='EXTERNAL'; selected=best
        elif top_score <= 0.:
            choice='CASH';selected='CASH'
        else:
            choice='TOP1';selected=top
        def outcome(sym):
            if sym == 'CASH':return 0.
            v = d.loc[sym]
            op,cl = float(v.execution_open),float(v.execution_close)
            return float(np.log(cl/op)) if (np.isfinite(op) and np.isfinite(cl) and min(op,cl)>0) else float('nan')
        top_ret = outcome(top)
        selected_ret = outcome(selected)
        result.append({'decision_date':day,'status':'ok','top1':top,'top1_score':top_score,
                       'best_external':best,'best_external_score':best_score,
                       'external_candidates_valid_t':len(candidates),
                       'decision_shadow':choice,'shadow_symbol':selected,
                       'top1_next_intraday_log':top_ret,
                       'shadow_next_intraday_log':selected_ret,
                       'delta_intraday_log':selected_ret-top_ret if np.isfinite(top_ret) and np.isfinite(selected_ret) else np.nan,
                       'outcome_available':bool(np.isfinite(top_ret) and np.isfinite(selected_ret))})
    return pd.DataFrame(result)
