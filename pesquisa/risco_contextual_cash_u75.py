"""Causal context analogues: movement scale, volatility, breadth, and CASH.

Scientific pilot only. Query features end at the closing decision t. Each
historical *next-open* investment outcome j+1 ... j+h+1 must be entirely
observable by t, i.e. j+h+1 <= t. No knowledge of query's next open.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, sqrt
from typing import Any

import numpy as np
import pandas as pd

from pesquisa.motivos_historicos_cash_u75 import _dtw, _normalized_paths

VERSION = "u75-contextual-cash-advantage-v1.0.0"

MARKET_FEATURES = (
    "market_log_return_5", "market_log_return_20",
    "market_log_volatility_20", "market_drawdown_20",
    "breadth_positive_20", "breadth_below_ma20",
    "dispersion_return_20",
)
ASSET_FEATURES = (
    "asset_log_return_5", "asset_log_return_20",
    "asset_log_volatility_20", "asset_drawdown_20",
    "asset_volatility_ratio_5_20",
)


@dataclass(frozen=True)
class AdvantageSettings:
    window: int = 20
    horizon: int = 5
    minimum_history: int = 252
    min_coverage_fraction: float = .90
    pool: int = 96
    neighbours: int = 16
    minimum_episodes: int = 8
    dtw_radius: int = 4
    shape_weight: float = .35
    risk_penalty: float = 1.0
    lower_bound_z: float = 1.645
    minimum_loss_fraction: float = .65
    round_trip_buffer_bps: float = 15.0

    def __post_init__(self):
        if (self.window < 5 or self.horizon < 1
                or self.minimum_history < self.window + self.horizon + 1
                or not (0 < self.min_coverage_fraction <= 1)
                or self.neighbours < self.minimum_episodes
                or self.minimum_episodes < 3 or self.pool < self.neighbours
                or self.dtw_radius < 0 or self.shape_weight < 0
                or self.risk_penalty < 0 or self.lower_bound_z <= 0
                or not (0.5 <= self.minimum_loss_fraction <= 1)
                or self.round_trip_buffer_bps < 0):
            raise ValueError("Invalid prespecified context risk protocol")


@dataclass(frozen=True)
class MarketContext:
    calendar: pd.DatetimeIndex
    market_log_level: np.ndarray
    market_features: np.ndarray
    asset_log_levels: dict[str, np.ndarray]
    asset_log_opens: dict[str, np.ndarray]
    asset_features: dict[str, np.ndarray]
    coverage: pd.DataFrame


def _log_safe(frame: pd.DataFrame, kind: str, calendar: pd.DatetimeIndex) -> pd.Series:
    original = pd.to_numeric(frame[kind], errors="coerce")
    original.index = pd.DatetimeIndex(pd.to_datetime(original.index, utc=True))
    if original.index.has_duplicates or not original.index.is_monotonic_increasing:
        raise ValueError(f"Invalid or duplicated {kind} timestamps")
    return np.log(original.where(original > 0).reindex(calendar))


def construct_market_context(
    raw: dict[str, pd.DataFrame],
    symbols: list[str],
    calendar: pd.DatetimeIndex,
    *,
    coverage_fraction: float = .90,
) -> MarketContext:
    """Daily close-t market and asset features, with no backward/forward fill."""
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar, utc=True))
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Invalid calendar")
    if (not symbols or len(set(symbols)) != len(symbols)
            or set(raw) != set(symbols)):
        raise ValueError("Raw series must match the declared universe exactly")
    if not 0 < coverage_fraction <= 1:
        raise ValueError("Invalid coverage fraction")
    required = ceil(len(symbols)*coverage_fraction)
    log_close = pd.DataFrame({
        s: _log_safe(raw[s], "close", calendar) for s in symbols
    }, index=calendar)
    log_open = {
        s: _log_safe(raw[s], "open", calendar).to_numpy(dtype=float)
        for s in symbols
    }
    log_daily = log_close.diff()
    daily_simple = np.expm1(log_daily)
    count_day = daily_simple.notna().sum(axis=1)
    equal_daily_simple = daily_simple.mean(axis=1).where(count_day >= required)
    equal_daily_log = np.log1p(equal_daily_simple)

    market_daily = equal_daily_log.to_numpy(dtype=float)
    cumulative = np.cumsum(np.where(np.isfinite(market_daily), market_daily, 0.0))
    market_level = pd.Series(cumulative, index=calendar)
    market_level = market_level.where(equal_daily_log.notna())
    # Window functions require every observation (so gaps are not bridged).
    market_return_5 = market_level.diff(5)
    market_return_20 = market_level.diff(20)
    market_vol20 = equal_daily_log.rolling(20, min_periods=20).std()
    market_dd20 = market_level - market_level.rolling(20, min_periods=20).max()

    asset_return20 = log_close.diff(20)
    positive = asset_return20.notna()
    enough20 = positive.sum(axis=1) >= required
    breadth_positive20 = (asset_return20 > 0).sum(axis=1).div(
        positive.sum(axis=1).replace(0, np.nan)
    ).where(enough20)
    ma20_log = np.log(
        np.exp(log_close).rolling(20, min_periods=20).mean()
    )
    valid_below = log_close.notna() & ma20_log.notna()
    breadth_below_ma20 = (log_close < ma20_log).where(
        valid_below, other=False
    ).sum(axis=1).div(valid_below.sum(axis=1).replace(0, np.nan)).where(
        valid_below.sum(axis=1) >= required
    )
    dispersion20 = asset_return20.std(axis=1).where(enough20)

    market_feature_frame = pd.concat([
        market_return_5, market_return_20, market_vol20, market_dd20,
        breadth_positive20, breadth_below_ma20, dispersion20,
    ], axis=1)
    market_feature_frame.columns = list(MARKET_FEATURES)

    asset_features: dict[str,np.ndarray] = {}
    asset_levels: dict[str,np.ndarray] = {}
    for s in symbols:
        logp = log_close[s]
        daily = logp.diff()
        vol20 = daily.rolling(20,min_periods=20).std()
        vol5 = daily.rolling(5,min_periods=5).std()
        feature_frame = pd.concat([
            logp.diff(5), logp.diff(20), vol20,
            logp - logp.rolling(20,min_periods=20).max(),
            (vol5/vol20.where(vol20 > 1e-12)),
        ],axis=1)
        feature_frame.columns = list(ASSET_FEATURES)
        asset_features[s] = feature_frame.to_numpy(dtype=float)
        asset_levels[s] = logp.to_numpy(dtype=float)

    coverage = pd.DataFrame({
        "date": calendar.strftime("%Y-%m-%d"),
        "universe_assets":len(symbols),
        "required_assets":required,
        "valid_daily_returns":count_day.to_numpy(int),
        "valid_return_20":positive.sum(axis=1).to_numpy(int),
        "market_features_valid":np.isfinite(
            market_feature_frame.to_numpy(dtype=float)
        ).all(axis=1),
        "breadth_positive_20":breadth_positive20.to_numpy(float),
        "market_log_return_20":market_return_20.to_numpy(float),
        "market_log_volatility_20":market_vol20.to_numpy(float),
    })
    return MarketContext(
        calendar=calendar,
        market_log_level=market_level.to_numpy(float),
        market_features=market_feature_frame.to_numpy(float),
        asset_log_levels=asset_levels,
        asset_log_opens=log_open,
        asset_features=asset_features,
        coverage=coverage,
    )


def _status(reason: str, *, observed: int = 0) -> dict[str, Any]:
    return {"status":reason, "cash_signal":False, "analogues":observed,
            "loss_fraction":None, "gross_mean_log":None,
            "cash_advantage_mean":None, "cash_advantage_lower_bound":None,
            "es95_gross_log":None, "latest_outcome_observed_at":None,
            "context_distance":None}


def _match(
    context: MarketContext, symbol: str, t: int, settings: AdvantageSettings,
    *, mode: str,
) -> dict[str, Any]:
    """Return recommendation using only decisions/realized outcomes at or before t."""
    if mode not in ("market", "joint", "joint_shape"):
        raise ValueError("Invalid declared matching mode")
    n = len(context.calendar)
    if symbol not in context.asset_log_opens or t < 0 or t >= n:
        raise ValueError("Symbol/index not present in research context")
    w, h = settings.window, settings.horizon
    if t < settings.minimum_history or t < w+20+h+1:
        return _status("warmup")
    market = context.market_features
    asset = context.asset_features[symbol]
    asset_level = context.asset_log_levels[symbol]
    market_level = context.market_log_level

    market_only = mode == "market"
    joint = np.concatenate([market, asset], axis=1) if not market_only else market
    query = joint[t]
    if not np.isfinite(query).all():
        return _status("missing_current_state")
    # A trade entered next-open j+1, exited at j+h+1. Both must be
    # historical and fully known by close t; never use opening t+1.
    candidates = np.arange(max(w+20, 1), t-h, dtype=int)
    if candidates.size == 0:
        return _status("no_matured_history")
    past = joint[candidates]
    valid = np.isfinite(past).all(axis=1)
    log_opens = context.asset_log_opens[symbol]
    nan_open = (~np.isfinite(log_opens)).astype(int)
    prefix = np.r_[0, np.cumsum(nan_open)]
    valid &= ((prefix[candidates+h+2]-prefix[candidates+1]) == 0)
    if not valid.any():
        return _status("no_complete_history")
    candidates = candidates[valid]
    past = past[valid]
    if len(candidates) < settings.minimum_episodes:
        return _status("few_historical_rows", observed=len(candidates))
    # Robust scaling only from historical rows that are already observable.
    # Amplitude and volatility stay in the context: they are NOT z-normalized
    # per pattern, unlike the supplemental shape score.
    historical_median = np.median(past, axis=0)
    mad = 1.4826*np.median(abs(past-historical_median),axis=0)
    fallback = np.std(past,axis=0)
    scale = np.where(mad > 1e-6, mad, np.maximum(fallback,1e-4))
    distances = np.mean(((past-query)/scale)**2,axis=1)

    if mode == "joint_shape":
        ends = candidates
        mh, mgood = _normalized_paths(market_level, ends, w)
        ah, agood = _normalized_paths(asset_level, ends, w)
        mq, mqvalid = _normalized_paths(market_level, np.array([t]), w)
        aq, aqvalid = _normalized_paths(asset_level, np.array([t]), w)
        if not mqvalid[0] or not aqvalid[0]:
            return _status("missing_shape_query")
        valid_shape = mgood & agood
        ends=ends[valid_shape]
        distances=distances[valid_shape]
        mh=mh[valid_shape]; ah=ah[valid_shape]
        if len(ends) < settings.minimum_episodes:
            return _status("few_valid_shapes", observed=len(ends))
        candidate_n=min(len(ends),settings.pool)
        selected=np.argpartition(distances,candidate_n-1)[:candidate_n]
        # Shape refinement ONLY among closest *state* matches. DTW band fixed.
        shape_q=np.stack([mq[0],aq[0]],axis=-1)
        reranked=[]
        for i in selected:
            shape_old=np.stack([mh[i],ah[i]],axis=-1)
            ds=_dtw(shape_old,shape_q,settings.dtw_radius)
            if np.isfinite(ds):
                reranked.append((float(distances[i])+settings.shape_weight*ds,
                                 int(ends[i])))
        ranked=sorted(reranked,key=lambda x:(x[0],x[1]))
    else:
        take=min(len(candidates),settings.pool)
        closest=np.argpartition(distances,take-1)[:take]
        ranked=sorted(
            [(float(distances[i]),int(candidates[i])) for i in closest],
            key=lambda x:(x[0],x[1]),
        )

    # Avoid counting the same future return window repeatedly.
    chosen: list[int]=[]
    scores: list[float]=[]
    for dist,j in ranked:
        if all(abs(j-old)>=h+1 for old in chosen):
            chosen.append(j)
            scores.append(dist)
            if len(chosen)>=settings.neighbours:
                break
    if len(chosen)<settings.minimum_episodes:
        return _status("few_independent_analogues",observed=len(chosen))
    ends=np.asarray(chosen,dtype=int)
    if (ends+h+1>t).any():
        raise RuntimeError("Causal leakage: analogue response not matured")
    gross_log=log_opens[ends+h+1]-log_opens[ends+1]
    if not np.isfinite(gross_log).all():
        raise RuntimeError("Unexpected missing analogue price")
    # Utility proxy, NOT the exact portfolio simulation:
    # Invest: forward log return - downside risk penalty.
    # CASH: zero return - conservative synthetic round-trip friction buffer.
    cash_buffer=settings.round_trip_buffer_bps/10000.0
    advantage = (
        -gross_log
        +settings.risk_penalty*np.maximum(-gross_log,0.0)
        -cash_buffer
    )
    avg=float(np.mean(advantage))
    se=float(np.std(advantage,ddof=1)/sqrt(len(advantage)))
    lower=avg-settings.lower_bound_z*se
    fraction=float(np.mean(gross_log<0))
    loss_threshold=float(np.quantile(gross_log,.05))
    es95=float(np.mean(gross_log[gross_log<=loss_threshold]))
    signal=bool(
        lower>0 and fraction>=settings.minimum_loss_fraction
        and float(np.mean(gross_log))<0
    )
    return {
        "status":"ok", "cash_signal":signal,
        "analogues":len(chosen),
        "loss_fraction":fraction,
        "gross_mean_log":float(np.mean(gross_log)),
        "cash_advantage_mean":avg,
        "cash_advantage_lower_bound":float(lower),
        "es95_gross_log":es95,
        "latest_outcome_observed_at":int(np.max(ends+h+1)),
        "closest_analogue_end_index":int(np.max(ends)),
        "context_distance":float(np.mean(scores)),
        "mode":mode,
    }


class ContextualCashRisk:
    """Causal, memoized queries by daily decision and selected symbol."""
    def __init__(self, context: MarketContext, settings: AdvantageSettings):
        self.context=context
        self.settings=settings
        self.index={
            pd.Timestamp(dt):i for i,dt in enumerate(context.calendar)
        }

    @lru_cache(maxsize=20000)
    def at(self, t: int, symbol: str, mode: str):
        return _match(self.context,symbol,t,self.settings,mode=mode)

    def decision(self, date: pd.Timestamp, symbol: str, mode: str) -> dict:
        key=pd.Timestamp(date)
        if key not in self.index:
            raise KeyError(f"No date in reference calendar: {key}")
        result=self.at(self.index[key],symbol,mode)
        return {"decision_date":key,"candidate_asset":symbol,**result}


def contextual_cash_policy(base_policy, symbols: list[str],
                           risk: ContextualCashRisk, *, mode: str,
                           log: list[dict]):
    """Control sees actual portfolio state, including CASH on prior session."""
    def policy(date,position,holding):
        target,score=base_policy(date,position,holding)
        if target==0:
            log.append({
                "decision_date":pd.Timestamp(date),
                "candidate_asset":"CASH",
                **_status("control_cash"),
                "mode":mode,
            })
            return target,score
        if not 1<=target<=len(symbols):
            raise ValueError("Selected position outside the declared universe")
        row=risk.decision(date,symbols[target-1],mode)
        log.append(row)
        if row["cash_signal"]:
            return 0,0.0
        return target,score
    return policy
