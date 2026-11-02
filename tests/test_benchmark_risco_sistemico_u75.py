"""Testes sinteticos, independentes de dados privados e da execucao operacional."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pesquisa.benchmark_risco_sistemico_u75 import (
    METHODS, Settings, _bocpd_series, _cusum_series,
    evaluate, make_panel,
)


def _raw(n=150, assets=8, seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    shared = rng.normal(.0002, .01, n)
    result = {}
    for i in range(assets):
        p = 30 * np.exp(np.cumsum(shared + rng.normal(0, .003 + i*.0001, n)))
        result[f"A{i:02d}"] = pd.DataFrame({
            "open":p*1.0005, "close":p, "volume":1000,
        }, index=dates)
    return result


def test_universe_not_fixed_to_75_and_no_future_leakage():
    cfg=Settings(horizons=(1, 5), min_coverage_fraction=.75,
                 covariance_window=25, covariance_step=5,
                 evt_window=100, evt_step=10)
    raw = _raw()
    names = list(raw)
    initial, labels = make_panel(raw, names, cfg)
    assert len(initial) == 150
    assert {"pc1_share", "bocpd_change_probability", "evt_es95"}.issubset(initial.columns)
    assert labels["future_market_5"].iloc[-5:].isna().all()
    changed={k:v.copy() for k,v in raw.items()}
    for frame in changed.values():
        frame.iloc[-10:, frame.columns.get_loc("close")] *= 3
    newer,_=make_panel(changed,names,cfg)
    compare=initial.iloc[:125]
    pd.testing.assert_frame_equal(compare,newer.iloc[:125],check_exact=False,
                                  rtol=1e-10,atol=1e-12)


def test_bocpd_and_cusum_causal_prefix():
    cfg=Settings()
    rng=np.random.default_rng(11)
    observations=rng.normal(size=180)
    cp1, neg1 = _bocpd_series(observations, cfg)
    changed=observations.copy()
    changed[150:]+=30
    cp2, neg2 = _bocpd_series(changed, cfg)
    np.testing.assert_allclose(cp1[:150],cp2[:150])
    np.testing.assert_allclose(neg1[:150],neg2[:150])
    scores=_cusum_series(observations,np.ones(180))
    assert np.isfinite(scores).all()
    assert np.min(scores)>=0


def _feature_label_frame(n=560):
    rng=np.random.default_rng(19)
    index=pd.date_range("2021-01-01",periods=n,freq="B",tz="UTC")
    names=set.union(*(set(c) for c in __import__(
        "pesquisa.benchmark_risco_sistemico_u75",fromlist=["METHOD_FEATURES"]
    ).METHOD_FEATURES.values()))
    names |= {"market_r1","breadth_down5"}
    features=pd.DataFrame({
        s:rng.normal(0,.5,n) for s in names
    },index=index)
    # Inject simple contemporaneous relationships and future events.
    features["market_r1"]=rng.normal(0,.02,n)
    features["breadth_down5"]=np.clip(rng.normal(.5,.15,n),0,1)
    labels=pd.DataFrame(index=index)
    labels["future_market_1"] = features["market_r1"].shift(-1)
    labels["future_breadth_down_1"] = np.where(
        labels["future_market_1"]<0,.8,.2)
    labels["future_breadth_down_1"] = labels["future_breadth_down_1"].where(
        labels["future_market_1"].notna())
    labels["maturity_1"]=pd.Series(index,index=index).shift(-1)
    return features,labels


def test_chronological_comparison_same_dates_and_matured_targets():
    cfg=Settings(horizons=(1,),min_training_dates=130,
                 min_positive_events=3)
    features, labels=_feature_label_frame()
    predictions, summary, audit=evaluate(features,labels,cfg)
    assert not summary.empty
    assert "Base_historica" in set(predictions.method)
    assert "Markov_filtrado" in set(predictions.method)
    assert "LightGBM_multivariado" in set(predictions.method)
    counts=predictions.groupby(["year","horizon","method"]).size()
    assert counts.groupby(level=["year","horizon"]).nunique().eq(1).all()
    valid=audit.loc[audit.status.eq("ok")]
    assert (pd.to_datetime(valid.last_training_maturity)
            <pd.to_datetime(valid.first_test_date)).all()
    assert set(predictions.method)==set(METHODS)


def test_invalid_settings():
    with pytest.raises(ValueError):
        Settings(min_coverage_fraction=0.)
    with pytest.raises(ValueError):
        Settings(horizons=(0,))
