"""Causal and stateful tests for historical-motif CASH research.

Synthetic data only. These tests do not claim real financial superiority.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from pesquisa.motivos_historicos_cash_u75 import (
    PatternSettings, _dtw, build_series, causal_analogues,
    HistoricalMotifRisk, wrap_cash_policy,
)


class HistoricalMotifTests(unittest.TestCase):
    def setUp(self):
        self.settings = PatternSettings(
            window=10, horizon=3, minimum_history=50,
            neighbours=5, candidates=20, dtw_radius=2,
        )
        self.days = pd.date_range("2024-01-02", periods=180, freq="B", tz="UTC")
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.015, 180)
        self.market = np.cumsum(r)
        self.asset = np.cumsum(.7*r + rng.normal(0, .007, 180))

    def test_dtw_zero_distance_for_identical_path(self):
        path = np.stack([np.sin(np.arange(11)/3), np.arange(11)], axis=1)
        self.assertAlmostEqual(_dtw(path,path,2),0,places=12)

    def test_changes_to_future_do_not_change_signal_or_analogues(self):
        t = 110
        before = causal_analogues(self.market,self.asset,t,self.settings)
        self.assertEqual(before["status"],"ok")
        market2=self.market.copy()
        asset2=self.asset.copy()
        market2[t+1:] += 99
        asset2[t+1:] -= 99
        after=causal_analogues(market2,asset2,t,self.settings)
        self.assertEqual(before,after)

    def test_matured_outcomes_cannot_end_in_future(self):
        t=100
        p=causal_analogues(self.market,self.asset,t,self.settings)
        self.assertEqual(p["status"],"ok")
        self.assertLessEqual(max(p["analogue_ends"])+self.settings.horizon,t)

    def test_excludes_missing_future_history(self):
        t=110
        broken=self.asset.copy()
        broken[50:112] = np.nan
        got=causal_analogues(self.market,broken,t,self.settings)
        self.assertEqual(got["status"],"missing_current_price")
        self.assertFalse(got["signal"])

    def test_future_prices_cannot_change_index_features(self):
        days=self.days
        names=[f"U{i:02d}" for i in range(75)]
        raw={}
        for i,s in enumerate(names):
            values=100*np.exp(self.market + .00001*i*np.arange(len(days)))
            raw[s]=pd.DataFrame({"close":values},index=days)
        market,assets,audit=build_series(raw,names,days)
        self.assertEqual(len(assets),75)
        self.assertTrue(np.isfinite(market[100]))
        before=causal_analogues(market,None,100,self.settings)
        self.assertEqual(before["status"],"ok")
        for df in raw.values():
            df.loc[days[101]:,"close"] *= 0.1
        market2,assets2,audit2=build_series(raw,names,days)
        after=causal_analogues(market2,None,100,self.settings)
        self.assertEqual(before,after)

    def test_missing_universe_asset_is_not_silently_removed(self):
        raw={"U00":pd.DataFrame({"close":np.exp(self.market)},index=self.days)}
        with self.assertRaises(ValueError):
            build_series(raw,["U00"],self.days)

    def test_policy_receives_real_position_and_can_enter_cash(self):
        class Dummy:
            def decision(self, date, symbol, *, include_asset):
                return {"decision_date":date,"base_asset":symbol,
                        "override_cash":True, "market_signal":True,
                        "asset_signal":False}
        actual=[]
        def base(dt,position,holding):
            actual.append((position,holding))
            return 1,0.45
        log=[]
        policy=wrap_cash_policy(base,Dummy(),["ABC"],include_asset=False,log=log)
        decision,score=policy(self.days[50],1,7)
        self.assertEqual(decision,0)
        self.assertEqual(actual,[(1,7)])
        self.assertTrue(log[0]["override_cash"])

    def test_policy_reenters_without_cash_dwell_lock(self):
        class Dummy:
            def decision(self, date, symbol, *, include_asset):
                return {"decision_date":date,"base_asset":symbol,
                        "override_cash":False,"market_signal":False,
                        "asset_signal":False}
        log=[]
        policy=wrap_cash_policy(lambda dt,pos,holding:(2,.12),
                                Dummy(),["AAA","BBB"],include_asset=True,log=log)
        decision,score=policy(self.days[50],0,0)
        self.assertEqual(decision,2)
        self.assertEqual(score,.12)

    def test_market_detector_has_strict_calendar_alignment(self):
        assets={"A":self.asset}
        detector=HistoricalMotifRisk(self.market,assets,self.days,self.settings)
        with self.assertRaises(KeyError):
            detector.decision(pd.Timestamp("2030-01-01",tz="UTC"),"A",include_asset=True)


if __name__=="__main__":
    unittest.main()
