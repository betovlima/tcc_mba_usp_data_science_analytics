"""Causal historical-motif research for U75 risk and hypothetical CASH.

This is an expanding, past-only distance profile + constrained multivariate DTW,
NOT an exact STUMPY/STOMP full matrix-profile implementation. Data at t+h may
only label an analogue when t_analogue+h <= current decision t.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt, ceil
from typing import Any

import numpy as np
import pandas as pd

VERSION = "u75-historical-motifs-cash-v1.0.1"


@dataclass(frozen=True)
class PatternSettings:
    window: int = 20
    horizon: int = 5
    minimum_history: int = 252
    candidates: int = 48
    neighbours: int = 16
    dtw_radius: int = 4
    upper_bound_z: float = 1.645
    minimum_coverage_fraction: float = 0.90

    def __post_init__(self):
        if (self.window < 5 or self.horizon < 1 or self.minimum_history < self.window
                or self.candidates < self.neighbours or self.neighbours < 3
                or self.dtw_radius < 0 or self.upper_bound_z <= 0
                or not 0 < self.minimum_coverage_fraction <= 1):
            raise ValueError("Invalid prespecified pattern protocol")


def _as_utc_index(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("Daily index must be sorted and unique")
    return idx


def build_series(
    raw: dict[str, pd.DataFrame],
    symbols: list[str],
    calendar: pd.DatetimeIndex,
    *,
    minimum_coverage_fraction: float = 0.90,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """Equal-weight *diagnostic* index and per-symbol log close, no fill."""
    if (not symbols or len(symbols) != len(set(symbols))
            or set(symbols) != set(raw)):
        raise ValueError("Active study universe must exactly match the available series")
    if not 0 < minimum_coverage_fraction <= 1:
        raise ValueError("Coverage fraction must be between zero and one")
    minimum_assets = max(1, ceil(len(symbols) * minimum_coverage_fraction))
    calendar = _as_utc_index(calendar)
    close = {}
    for symbol in symbols:
        s = pd.to_numeric(raw[symbol]["close"], errors="coerce")
        s.index = _as_utc_index(s.index)
        s = s.where(s > 0).reindex(calendar)
        close[symbol] = s
    prices = pd.DataFrame(close, index=calendar)
    returns = prices.pct_change(fill_method=None)
    coverage = returns.notna().sum(axis=1)
    mean_return = returns.mean(axis=1).where(coverage >= minimum_assets)
    market_log_daily = np.log1p(mean_return).to_numpy(dtype=float)
    # Missing dates remain NaN; never silently infer a no-loss session.
    market_level = np.cumsum(np.where(np.isfinite(market_log_daily),
                                      market_log_daily, 0.0))
    market_level[~np.isfinite(market_log_daily)] = np.nan
    asset_levels = {s: np.log(prices[s].to_numpy(dtype=float)) for s in symbols}
    audit = pd.DataFrame({
        "date": calendar.strftime("%Y-%m-%d"),
        "valid_assets": coverage.to_numpy(dtype=int),
        "required_assets": minimum_assets,
        "universe_assets": len(symbols),
        "equal_weight_log_return": market_log_daily,
        "status": np.where(coverage >= minimum_assets, "ok", "insufficient_coverage"),
    })
    return market_level, asset_levels, audit


def _normalized_paths(level: np.ndarray, ends: np.ndarray,
                      window: int) -> tuple[np.ndarray, np.ndarray]:
    """All windows end exactly at given indices, normalized independently."""
    segments = np.lib.stride_tricks.sliding_window_view(level, window + 1)[ends-window]
    good = np.isfinite(segments).all(axis=1)
    anchor = segments[:, :1]
    centered = segments - anchor
    means = np.mean(centered, axis=1, keepdims=True)
    stdev = np.std(centered, axis=1, keepdims=True)
    good &= np.isfinite(stdev[:, 0]) & (stdev[:, 0] > 1e-12)
    z = (centered-means)/np.where(stdev > 1e-12, stdev, 1.0)
    return z, good


def _dtw(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    """Sakoe-Chiba constrained DTW with normalized vector inputs."""
    n, m = len(a), len(b)
    if abs(n-m) > radius:
        return float("inf")
    last = np.full(m+1, np.inf)
    last[0] = 0.0
    for i in range(1, n+1):
        curr = np.full(m+1, np.inf)
        for j in range(max(1, i-radius), min(m, i+radius)+1):
            delta = float(np.mean((a[i-1]-b[j-1])**2))
            curr[j] = delta + min(last[j], curr[j-1], last[j-1])
        last = curr
    return float(last[m]/(n+m))


def causal_analogues(market: np.ndarray, asset: np.ndarray | None, t: int,
                    settings: PatternSettings) -> dict[str, Any]:
    """Find nearest earlier motifs with outcomes matured BEFORE decision t.

    Uses a distance profile (Euclidean prefilter) and banded DTW reranking.
    Each returned analogue's future outcome is known at decision t.
    """
    n = len(market)
    if not (0 <= t < n) or (asset is not None and len(asset) != n):
        raise ValueError("Misaligned decision index/asset")
    w, h = settings.window, settings.horizon
    if t < settings.minimum_history or t < w+h+2:
        return {"status": "warmup", "signal": False, "count": 0}
    if not np.isfinite(market[t]) or (asset is not None and not np.isfinite(asset[t])):
        return {"status": "missing_current_price", "signal": False, "count": 0}
    eligible = np.arange(w, t-h+1, dtype=int)
    # No analogue whose response window reaches beyond decision t.
    if len(eligible) == 0:
        return {"status": "no_matured_analogue", "signal": False, "count": 0}
    sequences = [market] if asset is None else [market, asset]
    query_end = np.array([t], dtype=int)
    history_parts = []
    query_parts = []
    valid = np.ones(len(eligible), dtype=bool)
    for series in sequences:
        hx, ok = _normalized_paths(series, eligible, w)
        qx, qok = _normalized_paths(series, query_end, w)
        valid &= ok
        if not bool(qok[0]):
            return {"status": "flat_or_missing_query", "signal": False, "count": 0}
        history_parts.append(hx)
        query_parts.append(qx[0])
    observed = market if asset is None else asset
    good_outcomes = (np.isfinite(observed[eligible])
                     & np.isfinite(observed[eligible+h]))
    # Ensure every intervening observation is present: no bridge over gaps.
    bad = (~np.isfinite(observed)).astype(int)
    prefix = np.r_[0, bad.cumsum()]
    good_outcomes &= (prefix[eligible+h+1]-prefix[eligible]) == 0
    valid &= good_outcomes
    eligible = eligible[valid]
    if len(eligible) < settings.neighbours:
        return {"status": "insufficient_analogues", "signal": False,
                "count": int(len(eligible))}
    history_parts = [x[valid] for x in history_parts]
    query = np.stack(query_parts, axis=-1)
    paths = np.stack(history_parts, axis=-1)
    distance_profile = np.mean((paths-query[None, ...])**2, axis=(1, 2))
    take = min(settings.candidates, len(eligible))
    nearest_euclidean = np.argpartition(distance_profile, take-1)[:take]
    refined = sorted(
        [(_dtw(paths[i], query, settings.dtw_radius), int(eligible[i]))
         for i in nearest_euclidean],
        key=lambda x: (x[0], x[1]),
    )
    # Distinct forward outcome windows: reduce repeated evidence from same drop.
    selected: list[int] = []
    for distance, endpoint in refined:
        if not np.isfinite(distance): continue
        if all(abs(endpoint-prior) >= h for prior in selected):
            selected.append(endpoint)
        if len(selected) >= settings.neighbours:break
    if len(selected) < 3:
        return {"status": "insufficient_distinct_episodes", "signal": False,
                "count": len(selected)}
    ends = np.asarray(selected, dtype=int)
    future = observed[ends+h]-observed[ends]
    mean = float(np.mean(future))
    std = float(np.std(future, ddof=1)) if len(future)>1 else float("nan")
    upper = mean + settings.upper_bound_z*std/sqrt(len(future))
    losses = -future
    es95 = float(np.mean(losses[losses >= np.quantile(losses, .95)]))
    return {"status":"ok", "signal": bool(upper < 0),
            "count":len(selected), "mean_future_log":mean,
            "upper_mean_log":float(upper), "negative_fraction":float((future<0).mean()),
            "es95_loss":es95, "closest_matured_end":int(max(selected)),
            "oldest_matured_end":int(min(selected)),
            "analogue_ends":selected,
            "mean_dtw":float(np.mean([d for d,j in refined if j in selected]))}


class HistoricalMotifRisk:
    """Memoized, causal risk lookup for one risk protocol."""

    def __init__(self, market: np.ndarray, assets: dict[str, np.ndarray],
                 calendar: pd.DatetimeIndex, settings: PatternSettings):
        self.market = np.asarray(market,dtype=float)
        self.assets = assets
        self.calendar = _as_utc_index(calendar)
        self.settings = settings
        self.positions = {pd.Timestamp(dt): i for i,dt in enumerate(self.calendar)}
        if len(self.market) != len(self.calendar):
            raise ValueError("Market series and calendar differ")
        if any(len(v)!=len(self.market) for v in assets.values()):
            raise ValueError("Asset series and calendar differ")

    @lru_cache(maxsize=6000)
    def market_at(self, index: int):
        return causal_analogues(self.market, None, index, self.settings)

    @lru_cache(maxsize=12000)
    def asset_at(self, index: int, symbol: str):
        if symbol not in self.assets: raise KeyError(symbol)
        return causal_analogues(self.market, self.assets[symbol], index,
                                self.settings)

    def decision(self, dt: pd.Timestamp, symbol: str, *, include_asset: bool):
        key = pd.Timestamp(dt)
        if key not in self.positions:
            raise KeyError(f"Decision outside market calendar: {key}")
        idx = self.positions[key]
        market = self.market_at(idx)
        asset = self.asset_at(idx, symbol) if include_asset else None
        market_signal = bool(market["signal"]) if market["status"] == "ok" else False
        asset_signal = bool(asset["signal"]) if asset is not None and asset["status"] == "ok" else False
        return {"decision_date":key, "base_asset":symbol, "market_status":market["status"],
                "market_signal":market_signal, "market_neighbours":market["count"],
                "market_mean_future_log":market.get("mean_future_log"),
                "market_upper_mean_log":market.get("upper_mean_log"),
                "asset_status":asset["status"] if asset is not None else "not_requested",
                "asset_signal":asset_signal, "asset_neighbours":asset["count"] if asset is not None else 0,
                "asset_mean_future_log":asset.get("mean_future_log") if asset is not None else None,
                "asset_upper_mean_log":asset.get("upper_mean_log") if asset is not None else None,
                "override_cash": bool(market_signal or (include_asset and asset_signal))}


def wrap_cash_policy(base_policy, detector: HistoricalMotifRisk,
                     symbols: list[str], *, include_asset: bool, log: list[dict]):
    """One decision per closing; simulator passes *actual* current position."""
    def policy(dt, position, holding):
        target, score = base_policy(dt, position, holding)
        if target == 0:
            log.append({"decision_date":pd.Timestamp(dt), "base_asset":"CASH",
                        "override_cash":False, "market_signal":False,
                        "asset_signal":False,"source":"Control_cash"})
            return target, score
        symbol = symbols[target-1]
        finding = detector.decision(dt, symbol, include_asset=include_asset)
        finding["source"] = "pattern_risk"
        log.append(finding)
        if finding["override_cash"]:
            return 0, 0.0
        return target, score
    return policy
