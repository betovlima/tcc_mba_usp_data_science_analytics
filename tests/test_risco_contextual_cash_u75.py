"""Synthetic causal tests for U75 context, next-open outcomes, CASH utility."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pesquisa.risco_contextual_cash_u75 import (
    AdvantageSettings, ContextualCashRisk, construct_market_context,
    contextual_cash_policy,
)


def synthetic(n_symbols: int=65, n_sessions: int=420):
    dates=pd.date_range("2021-01-04",periods=n_sessions,freq="B",tz="UTC")
    rng=np.random.default_rng(131)
    common=np.cumsum(rng.normal(.0002,.011,n_sessions))
    raw={}
    for i in range(n_symbols):
        symbol=f"S{i:03d}"
        drift=rng.normal(0,.003,n_sessions)
        logclose=np.log(100)+common+np.cumsum(drift)
        logopen=logclose-rng.normal(0,.003,n_sessions)
        raw[symbol]=pd.DataFrame({
            "open":np.exp(logopen),"close":np.exp(logclose),
        },index=dates)
    return dates,raw,sorted(raw)


@pytest.mark.parametrize("count", [65,75,82])
def test_universe_size_not_fixed(count):
    dates,raw,symbols=synthetic(count,300)
    data=construct_market_context(raw,symbols,dates)
    assert len(data.asset_features)==count
    assert data.coverage["universe_assets"].iloc[-1]==count
    assert data.coverage["required_assets"].iloc[-1]==int(np.ceil(.9*count))


def test_rejects_missing_symbol_without_rebuilding_history():
    dates,raw,symbols=synthetic()
    with pytest.raises(ValueError,match="exactly"):
        construct_market_context(raw,symbols[:-1],dates)


def test_context_features_are_available_as_of_close_not_future():
    dates,raw,symbols=synthetic()
    old=construct_market_context(raw,symbols,dates)
    t=360
    original_market=old.market_features[t].copy()
    original_asset=old.asset_features[symbols[0]][t].copy()
    for frame in raw.values():
        frame.loc[dates[t+1]:,"close"] *= 2.5
        frame.loc[dates[t+1]:,"open"] *= .4
    new=construct_market_context(raw,symbols,dates)
    np.testing.assert_array_equal(old.market_features[:t+1],new.market_features[:t+1])
    np.testing.assert_array_equal(old.asset_features[symbols[0]][:t+1],
                                  new.asset_features[symbols[0]][:t+1])
    np.testing.assert_array_equal(original_market,new.market_features[t])
    np.testing.assert_array_equal(original_asset,new.asset_features[symbols[0]][t])


@pytest.mark.parametrize("mode", ["market","joint","joint_shape"])
def test_matured_historical_outcomes_and_future_invariance(mode):
    dates,raw,symbols=synthetic()
    setting=AdvantageSettings()
    t=370
    before=construct_market_context(raw,symbols,dates)
    risk=ContextualCashRisk(before,setting)
    prior=risk.decision(dates[t],symbols[0],mode)
    assert prior["status"]=="ok",prior
    assert prior["analogues"]>=setting.minimum_episodes
    assert prior["latest_outcome_observed_at"]<=t
    for frame in raw.values():
        frame.loc[dates[t+1]:,"open"] *= 1.9
        frame.loc[dates[t+1]:,"close"] *= .2
    after=construct_market_context(raw,symbols,dates)
    later=ContextualCashRisk(after,setting).decision(dates[t],symbols[0],mode)
    assert prior==later


def test_true_gap_in_historical_next_open_window_excluded():
    dates,raw,symbols=synthetic()
    symbol=symbols[0]
    raw[symbol].loc[dates[250:300],"open"]=np.nan
    data=construct_market_context(raw,symbols,dates)
    result=ContextualCashRisk(data,AdvantageSettings()).decision(
        dates[370],symbol,"joint",
    )
    assert result["status"] in ("ok","few_historical_rows","few_independent_analogues")
    if result["status"]=="ok":
        assert result["latest_outcome_observed_at"]<=370


def test_query_horizon_does_not_peek_at_next_session_open():
    dates,raw,symbols=synthetic()
    data=construct_market_context(raw,symbols,dates)
    t=340
    risk=ContextualCashRisk(data,AdvantageSettings())
    query=risk.decision(dates[t],symbols[0],"joint")
    assert query["status"]=="ok"
    assert query["latest_outcome_observed_at"]<=t
    assert query["latest_outcome_observed_at"]!=t+1


def test_runtime_position_and_reentry_are_preserved():
    dates,raw,symbols=synthetic()
    class FakeRisk:
        def __init__(self):self.alert=True;self.calls=[]
        def decision(self,date,symbol,mode):
            self.calls.append((date,symbol,mode))
            return {"decision_date":date,"candidate_asset":symbol,
                    "cash_signal":self.alert,"status":"ok"}
    fake=FakeRisk()
    state=[]
    def base(dt,position,holding):
        state.append((position,holding))
        return 1,.50
    log=[]
    policy=contextual_cash_policy(
        base,symbols,fake,mode="joint",log=log,
    )
    target,_=policy(dates[300],1,9)
    assert target==0
    fake.alert=False
    target,_=policy(dates[301],0,0)
    assert target==1
    assert state==[(1,9),(0,0)]
    assert [x["cash_signal"] for x in log]==[True,False]


def test_static_settings_cannot_permit_negative_cost_or_invalid_horizon():
    with pytest.raises(ValueError):
        AdvantageSettings(round_trip_buffer_bps=-1)
    with pytest.raises(ValueError):
        AdvantageSettings(horizon=0)
    with pytest.raises(ValueError):
        AdvantageSettings(min_coverage_fraction=1.5)
