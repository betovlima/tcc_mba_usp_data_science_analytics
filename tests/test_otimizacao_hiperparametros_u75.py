"""Testes sem APIs, sem modelos de mercado e sem snapshots externos."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pesquisa.otimizacao_hiperparametros_u75 import (
    BOUNDS, PARAMETERS, SearchSettings, decode_point, choose_best,
    propose, parameter_signature, validate_fold_windows,
)

FROZEN = {
    "n_estimators":329, "learning_rate":0.020731, "max_depth":3,
    "num_leaves":6, "min_child_samples":18, "colsample_bytree":0.88067,
    "reg_alpha":0.050837, "reg_lambda":3.596305,
    "min_child_weight":5.0, "random_state":42,
}


class U75SearchTest(unittest.TestCase):
    def test_dimensions_and_parameter_types(self):
        self.assertEqual(len(BOUNDS),8)
        p=decode_point(np.full(8,0.5),FROZEN)
        self.assertEqual(set(p),set(FROZEN))
        self.assertEqual(p['random_state'],42)
        self.assertLessEqual(p['num_leaves'],2**p['max_depth'])
        self.assertIsInstance(p['n_estimators'],int)

    def test_cannot_escape_research_space(self):
        with self.assertRaises(ValueError):
            decode_point(np.full(8,1.01),FROZEN)
        with self.assertRaises(ValueError):
            decode_point(np.full(4,0.5),FROZEN)

    def test_proposals_are_deterministic_and_distinct(self):
        settings=SearchSettings(budget=6,lhs_proposals=3,acquisition_candidates=64)
        def campaign():
            hist=[]
            for step in range(settings.budget):
                x,p,method=propose(hist,FROZEN,settings=settings,fold_id=2)
                self.assertEqual(method, 'frozen_anchor' if step==0 else method)
                hist.append({'trial':step,'normalized':x.tolist(), 'parameters':p,
                             'objective':float(step%3)+0.0001*step})
            return hist
        first,second=campaign(),campaign()
        self.assertEqual(first,second)
        self.assertEqual(first[0]['parameters'],FROZEN)
        self.assertEqual(len({parameter_signature(r['parameters']) for r in first}),6)
        self.assertEqual(first[1]['parameters']['min_child_weight'],FROZEN['min_child_weight'])

    def test_chosen_only_on_training_scores_with_anchored_ties(self):
        trials=[{'trial':0,'objective':1.1},{'trial':1,'objective':1.1},{'trial':2,'objective':0.8}]
        self.assertEqual(choose_best(trials)['trial'],0)
        trials[2]['objective']=1.2
        self.assertEqual(choose_best(trials)['trial'],2)

    def test_expanding_windows_reject_future_and_overlapping(self):
        dates=pd.bdate_range('2018-01-01',periods=350,tz='UTC')
        validate_fold_windows(dates[:140],dates[200:260],dates[262:300],dates[320:])
        with self.assertRaises(ValueError):
            validate_fold_windows(dates[:140],dates[200:270],dates[269:300],dates[320:])
        with self.assertRaises(ValueError):
            validate_fold_windows(dates[:140],dates[300:320],dates[262:300],dates[320:])

    def test_budget_guard(self):
        with self.assertRaises(ValueError):
            SearchSettings(budget=2,lhs_proposals=3)


if __name__=='__main__':
    unittest.main()
