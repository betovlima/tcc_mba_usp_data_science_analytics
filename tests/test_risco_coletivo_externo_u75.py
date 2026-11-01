"""Testes de causalidade, integridade e seleção shadow sem preços futuros."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pesquisa.risco_coletivo_externo_u75 import (
    StatisticalSettings, RISK_COLUMNS, build_collective_panel,
    evaluate_regime_walkforward, shadow_external_choices,
)


class RiskTests(unittest.TestCase):
    def _bars(self):
        d=pd.date_range('2021-01-04',periods=175,freq='B',tz='UTC')
        gen=np.random.default_rng(42)
        bars={}
        for i in range(75):
            ret=gen.normal(.0001,.01,len(d))
            price=100*np.exp(np.cumsum(ret))
            bars[f'A{i:02d}']=pd.DataFrame({'close':price},index=d)
        return bars,d

    def test_future_change_cannot_affect_current_features(self):
        bars,d=self._bars()
        settings=StatisticalSettings(min_assets=68,covariance_window=126)
        today=d[155]
        before=build_collective_panel(bars,set(bars),[today],settings=settings).iloc[0]
        self.assertEqual(before.status,'ok')
        for asset,df in bars.items():
            df.loc[d[156]:,'close'] *= (1.17 if asset.endswith('0') else .73)
        after=build_collective_panel(bars,set(bars),[today],settings=settings).iloc[0]
        for col in RISK_COLUMNS:
            self.assertAlmostEqual(float(before[col]),float(after[col]),places=11)
        self.assertNotEqual(float(before.realized_future_log_return),float(after.realized_future_log_return))

    def test_partial_coverage_is_not_interpreted_as_safe(self):
        bars,d=self._bars()
        for i in range(10):
            bars[f'A{i:02d}'].loc[d[140:],'close']=np.nan
        frame=build_collective_panel(bars,set(bars),[d[155]])
        self.assertEqual(frame.iloc[0].status,'insufficient_coverage')

    def test_yearly_fold_trains_only_matured_targets(self):
        d=pd.date_range('2019-01-01',periods=1550,freq='B',tz='UTC')
        g=np.random.default_rng(7)
        values=g.normal(0,0.01,len(d))
        p=pd.DataFrame({'decision_date':d.strftime('%Y-%m-%d'),
                        'maturity_date':(d+pd.offsets.BDay(5)).strftime('%Y-%m-%d'),
                        'status':'ok','realized_future_log_return':values})
        for name in RISK_COLUMNS:p[name]=g.normal(size=len(d))
        choices,folds=evaluate_regime_walkforward(p)
        self.assertGreater(len(choices),500)
        self.assertTrue(all(folds.train_maturity_max < folds.test_start))
        self.assertFalse(choices['risk_alert'].isna().any())
        self.assertTrue(((choices.predicted_tail_probability >= 0) & (choices.predicted_tail_probability <= 1)).all())

    def test_future_candidate_prices_do_not_change_shadow_choice(self):
        u={f'A{i:02d}' for i in range(75)}
        top='A00'; outside='Z99'
        leaders=pd.DataFrame([{'decision_date':'2025-01-02','execution_date':'2025-01-03','leader':top}])
        risk=pd.DataFrame([{'decision_date':'2025-01-02','risk_alert':True}])
        vals=[{'decision_date':'2025-01-02','symbol':top,'return_20':0.01,'vol_20':0.015,
               'execution_open':100.,'execution_close':105.},
              {'decision_date':'2025-01-02','symbol':outside,'return_20':0.1,'vol_20':0.01,
               'execution_open':100.,'execution_close':95.}]
        raw=pd.DataFrame(vals)
        x=shadow_external_choices(risk,leaders,raw,u).iloc[0]
        self.assertEqual(x.decision_shadow,'EXTERNAL')
        self.assertEqual(x.shadow_symbol,outside)
        raw.loc[raw.symbol.eq(outside),'execution_close']=10000.
        y=shadow_external_choices(risk,leaders,raw,u).iloc[0]
        self.assertEqual((x.decision_shadow,x.shadow_symbol),(y.decision_shadow,y.shadow_symbol))
        self.assertNotEqual(float(x.delta_intraday_log),float(y.delta_intraday_log))

    def test_no_external_data_is_not_treated_as_cash_signal(self):
        u={f'A{i:02d}' for i in range(75)}
        raw=pd.DataFrame([{'decision_date':'2025-01-02','symbol':'A00','return_20':-.1,
                           'vol_20':.02,'execution_open':100.,'execution_close':105.}])
        l=pd.DataFrame([{'decision_date':'2025-01-02','leader':'A00'}])
        risk=pd.DataFrame([{'decision_date':'2025-01-02','risk_alert':True}])
        choice=shadow_external_choices(risk,l,raw,u).iloc[0]
        self.assertEqual(choice.status,'no_external_snapshot')


if __name__=='__main__':unittest.main()
