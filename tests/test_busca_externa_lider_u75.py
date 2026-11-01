"""Testes independentes de Alpaca, Mongo e modelos financeiros originais."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pesquisa.busca_externa_lider_u75 import (
    FEATURES,ResearchSettings,make_panel,evaluate_expanding_years,summarize,
)


class ExternalLeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe={f'U{i:02d}' for i in range(75)}
        # Treino 2020-2021 e ano OOS 2022.
        dates=pd.bdate_range('2020-01-02',periods=650,tz='UTC')
        dates2=dates+pd.Timedelta(days=1)
        cls.leaders=pd.DataFrame({'decision_date':dates,'execution_date':dates2,
                                  'leader':['U00']*len(dates)})
        raw=[]
        for d,e in zip(dates,dates2):
            for symbol in sorted(cls.universe|{'OUTA','OUTB'}):
                level=0.001 if symbol=='OUTA' else 0.10 if symbol=='OUTB' else 0.0
                v={'decision_date':d,'execution_date':e,'symbol':symbol,
                   'execution_open':100.0,'execution_close':101.5 if symbol=='OUTA' else 99.5 if symbol=='OUTB' else 100.5}
                v.update({col:level for col in FEATURES})
                raw.append(v)
        cls.rows=pd.DataFrame(raw)
        cls.s=ResearchSettings(min_training_dates=100,min_training_years=2)

    def test_nearest_and_outcomes_are_distinct(self):
        p,coverage=make_panel(self.leaders.iloc[:20],self.rows,self.universe,settings=self.s)
        self.assertEqual(len(p),40)
        self.assertTrue((coverage.status=='ok').all())
        nearest=p.sort_values('distance_u75_leader').groupby('decision_date').first()
        self.assertTrue((nearest.candidate=='OUTA').all())
        self.assertTrue((nearest.delta_log_open_close>0).all())

    def test_future_price_does_not_change_discovery(self):
        small=self.leaders.iloc[:10]
        p1,_=make_panel(small,self.rows,self.universe,settings=self.s)
        modified=self.rows.copy()
        modified.loc[modified.symbol=='OUTA','execution_close']=1e6
        p2,_=make_panel(small,modified,self.universe,settings=self.s)
        keys=['decision_date','candidate','distance_u75_leader',*('relative_'+f for f in FEATURES)]
        pd.testing.assert_frame_equal(p1[keys],p2[keys])
        self.assertFalse(p1['delta_log_open_close'].equals(p2['delta_log_open_close']))

    def test_chronological_oos_and_no_future_gate(self):
        p,_=make_panel(self.leaders,self.rows,self.universe,settings=self.s)
        picks,folds=evaluate_expanding_years(p,settings=self.s)
        self.assertFalse(picks.empty)
        self.assertTrue((folds.train_last_execution<folds.first_test_decision).all())
        self.assertTrue((picks.nearest=='OUTA').all())
        self.assertTrue((picks.nearest_delta>0).all())
        self.assertGreater(summarize(picks)['nearest']['mean_daily_excess_log'],0)
        # O candidato deve continuar elegível se seu preço futuro faltar:
        changed=self.rows.copy()
        changed.loc[(changed.symbol=='OUTA')&(changed.decision_date==self.leaders.iloc[5].decision_date), 'execution_close']=np.nan
        broken,_=make_panel(self.leaders.iloc[:10],changed,self.universe,settings=self.s)
        day=broken[(broken.candidate=='OUTA')&(broken.decision_date==pd.Timestamp(self.leaders.iloc[5].decision_date).strftime('%Y-%m-%d'))].iloc[0]
        self.assertFalse(bool(day.target_available))
        self.assertTrue(np.isnan(day.delta_log_open_close))

    def test_structural_rejections(self):
        with self.assertRaisesRegex(ValueError,'75 ativos'):
            make_panel(self.leaders.iloc[:2],self.rows,{'U00'},settings=self.s)
        bad=self.leaders.iloc[:2].copy();bad.loc[bad.index[0],'leader']='OUTA'
        with self.assertRaisesRegex(ValueError,'fora do U75'):
            make_panel(bad,self.rows,self.universe,settings=self.s)


if __name__=='__main__':unittest.main()
