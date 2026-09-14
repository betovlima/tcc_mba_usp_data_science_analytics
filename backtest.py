"""Backtest acadêmico reproduzível.

A única fonte externa usada durante a execução é a coleção MongoDB
``alpaca_market_bars`` com candles OHLCV. A estratégia, as features, o target,
os folds walk-forward, os modelos LightGBM, a calibração, a política de rotação,
as operações e as métricas são construídos novamente por este arquivo.

Abra no Spyder e pressione F5, ou execute:

    python backtest.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from pymongo import MongoClient


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DATABASE = "extrema_backtest"
MARKET_COLLECTION = "alpaca_market_bars"

START_DATE = "2016-01-01"
END_DATE = "2026-09-04"

# Universo do experimento. Ele é parte da definição do estudo e não é lido de
# Strategy, backtest_runs ou qualquer outra coleção do Market Cycle Trader.
ASSETS = (
    "NVDA", "MSFT", "META", "TSLA", "AMD", "JPM", "SPY", "AVGO", "NFLX",
    "ORCL", "COST", "LLY", "XOM", "CAT", "WMT", "V", "HD", "ADC", "ADEA",
    "ADI", "ADM", "GKOS", "VNCE", "CORT", "UNFI", "DNN", "MKSI", "APD",
    "DDS", "RACE", "UNF", "TX", "CEF", "YANG", "KKR", "BXMT", "SCSC",
)

ROTATION_FEATURES = [
    "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
    "return_40", "return_60", "return_120",
    "vol_5", "vol_10", "vol_20", "vol_40", "vol_60",
    "vol_ratio_5_20", "vol_ratio_10_40", "vol_ratio_20_60",
    "ema_distance_5", "ema_distance_10", "ema_distance_20", "ema_distance_50",
    "ema_distance_100", "ema_5_vs_20", "ema_20_vs_50", "ema_50_vs_100",
    "ema_slope_20_5", "ema_slope_50_10", "ema_slope_100_20",
    "rsi_14", "atr_pct_14",
    "distance_from_high_20", "distance_from_low_20",
    "distance_from_high_50", "distance_from_low_50",
    "distance_from_high_100", "distance_from_low_100",
    "distance_from_high_200", "distance_from_low_200",
    "channel_position_20", "channel_position_50", "channel_position_100",
    "channel_position_200",
    "trend_efficiency_10", "trend_efficiency_20", "trend_efficiency_40",
    "trend_efficiency_60",
    "momentum_acceleration_5_20", "momentum_acceleration_20_60",
    "range_expansion_5_20", "volume_zscore_20", "volume_zscore_60",
    "volume_ratio_5_20",
]


@dataclass(frozen=True)
class ExperimentConfig:
    initial_capital: float = 10_000.0

    target_horizons: tuple[int, ...] = (5, 10, 20, 40, 60)
    target_weights: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.25)
    downside_penalty: float = 0.20
    drawdown_penalty: float = 0.35
    movement_capture_weight: float = 0.35
    trend_persistence_weight: float = 0.20

    minimum_training_rows: int = 700
    calibration_days: int = 126
    test_days: int = 504
    minimum_test_days: int = 126
    purge_days: int = 60

    minimum_holding_days: int = 2
    minimum_expected_edge: float = 0.001
    cash_threshold: float = 0.0
    switch_margin: float = 0.005
    switch_margin_candidates: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01)

    lightgbm_n_estimators: int = 300
    lightgbm_learning_rate: float = 0.035
    lightgbm_max_depth: int = 3
    lightgbm_num_leaves: int = 8
    lightgbm_min_child_samples: int = 20
    lightgbm_min_child_weight: float = 5.0
    lightgbm_subsample: float = 0.85
    lightgbm_subsample_freq: int = 0
    lightgbm_colsample_bytree: float = 0.85
    lightgbm_reg_alpha: float = 0.10
    lightgbm_reg_lambda: float = 2.0
    lightgbm_max_bin: int = 255
    lightgbm_n_jobs: int = -1
    random_state: int = 42

    slippage_bps: float = 0.0
    commission_rate: float = 0.0
    sec_fee_rate: float = 2.06e-5
    taf_fee_per_share: float = 0.000195
    taf_fee_cap: float = 9.79
    cat_fee_per_share: float = 3e-6
    fractional_shares: bool = True


CONFIG = ExperimentConfig()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = safe_divide(avg_gain, avg_loss)
    return 100 - 100 / (1 + rs)


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def load_market_data() -> dict[str, pd.DataFrame]:
    """Load only OHLCV market data from local MongoDB."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    mongo_uri = str(os.getenv("TCC_MONGO_URI") or MONGO_URI).strip()
    database_name = str(os.getenv("TCC_MONGO_DATABASE") or MONGO_DATABASE).strip()

    if not any(host in mongo_uri.lower() for host in ("localhost", "127.0.0.1", "::1")):
        raise RuntimeError("O projeto do TCC aceita somente MongoDB local.")

    start = pd.Timestamp(START_DATE, tz="UTC").to_pydatetime()
    end = (pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        collection = client[database_name][MARKET_COLLECTION]
        query = {
            "symbol": {"$in": list(ASSETS)},
            "interval": "1Day",
            "feed": "sip",
            "adjustment": "all",
            "timestamp": {"$gte": start, "$lt": end},
        }
        projection = {
            "_id": 0,
            "symbol": 1,
            "timestamp": 1,
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "volume": 1,
        }
        rows = list(
            collection.find(query, projection).sort(
                [("symbol", 1), ("timestamp", 1)]
            )
        )
    finally:
        client.close()

    if not rows:
        raise RuntimeError(
            "Nenhum candle foi encontrado em alpaca_market_bars para o período configurado."
        )

    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = (
            group.drop(columns=["symbol"])
            .set_index("timestamp")
            .sort_index()
        )
        frame = frame[~frame.index.duplicated(keep="last")]
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
        positive_prices = (
            (frame["open"] > 0)
            & (frame["high"] > 0)
            & (frame["low"] > 0)
            & (frame["close"] > 0)
            & (frame["volume"] >= 0)
        )
        frames[str(symbol)] = frame.loc[positive_prices].copy()

    missing = [symbol for symbol in ASSETS if symbol not in frames or frames[symbol].empty]
    if missing:
        raise RuntimeError("Sem histórico OHLCV para: " + ", ".join(missing))

    for index, symbol in enumerate(ASSETS, start=1):
        frame = frames[symbol]
        if index == 1 or index % 5 == 0 or index == len(ASSETS):
            log(
                f"[1/8] Dados {index:02d}/{len(ASSETS)} | {symbol} | "
                f"{len(frame)} candles"
            )
    return frames


def build_rotation_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Create all features and the future utility target from raw OHLCV."""
    cfg = CONFIG
    horizons = list(cfg.target_horizons)
    weights = np.asarray(cfg.target_weights, dtype=float)
    weights = weights / weights.sum()
    max_horizon = max(horizons)

    data = bars.copy().sort_index()
    data.index = pd.to_datetime(data.index, utc=True)
    close = data["close"].astype(float)
    open_price = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    daily_return = close.pct_change(fill_method=None)
    daily_range = safe_divide(high - low, close)

    for period in (1, 2, 3, 5, 10, 20, 40, 60, 120):
        data[f"return_{period}"] = close.pct_change(period, fill_method=None)

    for period in (5, 10, 20, 40, 60):
        data[f"vol_{period}"] = daily_return.rolling(period).std()

    data["vol_ratio_5_20"] = safe_divide(data["vol_5"], data["vol_20"])
    data["vol_ratio_10_40"] = safe_divide(data["vol_10"], data["vol_40"])
    data["vol_ratio_20_60"] = safe_divide(data["vol_20"], data["vol_60"])

    ema: dict[int, pd.Series] = {}
    for period in (5, 10, 20, 50, 100):
        ema[period] = close.ewm(span=period, adjust=False).mean()
        data[f"ema_distance_{period}"] = safe_divide(close, ema[period]) - 1

    data["ema_5_vs_20"] = safe_divide(ema[5], ema[20]) - 1
    data["ema_20_vs_50"] = safe_divide(ema[20], ema[50]) - 1
    data["ema_50_vs_100"] = safe_divide(ema[50], ema[100]) - 1
    data["ema_slope_20_5"] = ema[20].pct_change(5, fill_method=None)
    data["ema_slope_50_10"] = ema[50].pct_change(10, fill_method=None)
    data["ema_slope_100_20"] = ema[100].pct_change(20, fill_method=None)

    data["rsi_14"] = rsi(close) / 100.0
    atr = true_range(data).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_pct_14"] = safe_divide(atr, close)

    for period in (20, 50, 100, 200):
        rolling_high = high.rolling(period).max()
        rolling_low = low.rolling(period).min()
        data[f"distance_from_high_{period}"] = safe_divide(close, rolling_high) - 1
        data[f"distance_from_low_{period}"] = safe_divide(close, rolling_low) - 1
        data[f"channel_position_{period}"] = safe_divide(
            close - rolling_low,
            rolling_high - rolling_low,
        )

    absolute_change = close.pct_change(fill_method=None).abs()
    for period in (10, 20, 40, 60):
        net_move = close.pct_change(period, fill_method=None).abs()
        traveled = absolute_change.rolling(period).sum()
        data[f"trend_efficiency_{period}"] = safe_divide(net_move, traveled)

    data["momentum_acceleration_5_20"] = (
        data["return_5"] - data["return_20"] * (5.0 / 20.0)
    )
    data["momentum_acceleration_20_60"] = (
        data["return_20"] - data["return_60"] * (20.0 / 60.0)
    )
    data["range_expansion_5_20"] = safe_divide(
        daily_range.rolling(5).mean(),
        daily_range.rolling(20).mean(),
    )

    for period in (20, 60):
        volume_mean = volume.rolling(period).mean()
        volume_std = volume.rolling(period).std()
        data[f"volume_zscore_{period}"] = safe_divide(
            volume - volume_mean,
            volume_std,
        )
    data["volume_ratio_5_20"] = safe_divide(
        volume.rolling(5).mean(),
        volume.rolling(20).mean(),
    )

    round_trip_cost = min(
        0.25,
        2.0
        * (
            max(0.0, cfg.slippage_bps) / 10_000.0
            + max(0.0, cfg.commission_rate)
        ),
    )
    net_cost_log = math.log(max(1e-12, 1.0 - round_trip_cost))

    lows = low.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = open_price.to_numpy(dtype=float)

    utility_components = np.full((len(data), len(horizons)), np.nan, dtype=float)
    net_return_components = np.full((len(data), len(horizons)), np.nan, dtype=float)
    movement_capture = np.full(len(data), np.nan, dtype=float)
    trend_persistence = np.full(len(data), np.nan, dtype=float)

    for idx in range(max(0, len(data) - max_horizon)):
        entry = opens[idx + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue

        full_closes = closes[idx + 1 : idx + max_horizon + 1]
        full_highs = highs[idx + 1 : idx + max_horizon + 1]
        if len(full_closes) != max_horizon:
            continue

        max_upside = max(0.0, float(np.nanmax(full_highs) / entry - 1.0))
        positive_share = float(np.mean(full_closes > entry))
        directional_changes = np.diff(full_closes)
        positive_steps = (
            float(np.mean(directional_changes > 0))
            if len(directional_changes)
            else 0.0
        )
        movement_capture[idx] = math.log1p(max_upside)
        trend_persistence[idx] = 0.5 * positive_share + 0.5 * positive_steps

        for component_idx, horizon in enumerate(horizons):
            future_lows = lows[idx + 1 : idx + horizon + 1]
            future_closes = closes[idx + 1 : idx + horizon + 1]
            if len(future_closes) != horizon:
                continue

            gross_log_return = math.log(
                max(float(future_closes[-1]) / entry, 1e-12)
            )
            minimum_low = float(np.nanmin(future_lows))
            downside = max(0.0, 1.0 - minimum_low / entry)

            path = np.concatenate(([entry], future_closes))
            running_peak = np.maximum.accumulate(path)
            drawdowns = 1.0 - np.divide(
                path,
                running_peak,
                out=np.ones_like(path),
                where=running_peak > 0,
            )
            path_drawdown = max(0.0, float(np.nanmax(drawdowns)))

            net_log_return = gross_log_return + net_cost_log
            net_return_components[idx, component_idx] = net_log_return
            utility_components[idx, component_idx] = (
                net_log_return
                - cfg.downside_penalty * downside
                - cfg.drawdown_penalty * path_drawdown
            )

    weighted_utility = np.nansum(
        utility_components * weights.reshape(1, -1),
        axis=1,
    )
    weighted_net_log_return = np.nansum(
        net_return_components * weights.reshape(1, -1),
        axis=1,
    )
    invalid_rows = np.isnan(utility_components).any(axis=1)
    weighted_utility[invalid_rows] = np.nan
    weighted_net_log_return[invalid_rows] = np.nan

    data["forward_net_log_return"] = weighted_net_log_return
    data["forward_movement_capture"] = movement_capture
    data["forward_trend_persistence"] = trend_persistence
    data["forward_risk_adjusted_utility"] = (
        weighted_utility
        + cfg.movement_capture_weight * data["forward_movement_capture"]
        + cfg.trend_persistence_weight * data["forward_trend_persistence"]
    )

    required = [*ROTATION_FEATURES, "open", "high", "low", "close", "volume"]
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=required)


def prepare_panel(
    raw_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    log("[2/8] Construindo features e targets diretamente dos candles OHLCV...")
    frames: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(ASSETS, start=1):
        frames[symbol] = build_rotation_frame(raw_frames[symbol])
        if index == 1 or index % 5 == 0 or index == len(ASSETS):
            log(
                f"[2/8] Features {index:02d}/{len(ASSETS)} | {symbol} | "
                f"{len(frames[symbol])} linhas utilizáveis"
            )

    common: pd.DatetimeIndex | None = None
    for symbol in sorted(frames):
        index = pd.DatetimeIndex(frames[symbol].index)
        common = index if common is None else common.intersection(index)

    if common is None or len(common) < 1_000:
        raise RuntimeError(
            "Histórico comum insuficiente depois da engenharia de features."
        )

    common = common.sort_values()
    aligned = {symbol: frame.reindex(common).copy() for symbol, frame in frames.items()}
    log(
        f"[2/8] Painel comum: {len(common)} sessões | "
        f"{common[0].date()} -> {common[-1].date()}"
    )
    return aligned, common


def build_walk_forward_folds(
    common_dates: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    cfg = CONFIG
    purge = max(cfg.purge_days, max(cfg.target_horizons))
    first_test_start = (
        cfg.minimum_training_rows
        + purge
        + cfg.calibration_days
        + purge
    )
    if first_test_start >= len(common_dates) - cfg.minimum_test_days:
        raise RuntimeError("Histórico insuficiente para o protocolo walk-forward.")

    ranges: list[tuple[int, int]] = []
    test_start = first_test_start
    while test_start < len(common_dates):
        test_end = min(len(common_dates), test_start + cfg.test_days)
        if test_end - test_start < cfg.minimum_test_days:
            if ranges:
                previous_start, _ = ranges[-1]
                ranges[-1] = (previous_start, len(common_dates))
            break
        ranges.append((test_start, test_end))
        test_start = test_end

    folds: list[dict[str, Any]] = []
    for fold_id, (test_start, test_end) in enumerate(ranges, start=1):
        calibration_end = test_start - purge
        calibration_start = calibration_end - cfg.calibration_days
        train_end = calibration_start - purge
        final_fit_end = test_start - purge
        if train_end < cfg.minimum_training_rows:
            raise RuntimeError(
                f"Fold {fold_id}: apenas {train_end} sessões de treino."
            )
        folds.append(
            {
                "fold_id": fold_id,
                "train_end_index": train_end,
                "calibration_start_index": calibration_start,
                "calibration_end_index": calibration_end,
                "final_fit_end_index": final_fit_end,
                "test_start_index": test_start,
                "test_end_index": test_end,
                "train_start": common_dates[0],
                "train_end": common_dates[train_end - 1],
                "calibration_start": common_dates[calibration_start],
                "calibration_end": common_dates[calibration_end - 1],
                "test_start": common_dates[test_start],
                "test_end": common_dates[test_end - 1],
                "decision_dates": common_dates[test_start - 1 : test_end],
            }
        )

    if not folds:
        raise RuntimeError("Nenhum fold walk-forward válido foi criado.")

    log(f"[3/8] Walk-forward criado com {len(folds)} folds.")
    for fold in folds:
        log(
            f"[3/8] Fold {fold['fold_id']} | treino até {fold['train_end'].date()} | "
            f"calibração {fold['calibration_start'].date()} -> "
            f"{fold['calibration_end'].date()} | "
            f"teste {fold['test_start'].date()} -> {fold['test_end'].date()}"
        )
    return folds


def fit_models(
    frames: dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    *,
    fold_id: int,
    phase: str,
) -> dict[str, LGBMRegressor]:
    cfg = CONFIG
    models: dict[str, LGBMRegressor] = {}

    for position, symbol in enumerate(sorted(frames), start=1):
        frame = frames[symbol].loc[train_dates].dropna(
            subset=["forward_risk_adjusted_utility", *ROTATION_FEATURES]
        )
        if len(frame) < cfg.minimum_training_rows:
            continue

        model = LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            n_estimators=cfg.lightgbm_n_estimators,
            learning_rate=cfg.lightgbm_learning_rate,
            max_depth=cfg.lightgbm_max_depth,
            num_leaves=cfg.lightgbm_num_leaves,
            min_child_samples=cfg.lightgbm_min_child_samples,
            min_child_weight=cfg.lightgbm_min_child_weight,
            subsample=cfg.lightgbm_subsample,
            subsample_freq=cfg.lightgbm_subsample_freq,
            colsample_bytree=cfg.lightgbm_colsample_bytree,
            reg_alpha=cfg.lightgbm_reg_alpha,
            reg_lambda=cfg.lightgbm_reg_lambda,
            max_bin=cfg.lightgbm_max_bin,
            random_state=cfg.random_state,
            n_jobs=cfg.lightgbm_n_jobs,
            verbosity=-1,
        )
        model.fit(
            frame[ROTATION_FEATURES],
            frame["forward_risk_adjusted_utility"],
        )
        models[symbol] = model

        if position == 1 or position % 5 == 0 or position == len(frames):
            log(
                f"[4/8] Fold {fold_id} {phase}: "
                f"{position:02d}/{len(frames)} modelos processados"
            )

    if len(models) < 2:
        raise RuntimeError(
            f"Fold {fold_id}: modelos insuficientes na fase {phase}."
        )
    return models


def model_utilities(
    models: dict[str, LGBMRegressor],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
) -> np.ndarray:
    values = [0.0]  # CASH
    for symbol in symbols:
        model = models.get(symbol)
        frame = frames[symbol]
        if model is None or timestamp not in frame.index:
            values.append(float("-inf"))
            continue
        row = frame.loc[[timestamp], ROTATION_FEATURES]
        if row.empty or row.isna().any(axis=None):
            values.append(float("-inf"))
            continue
        values.append(float(model.predict(row)[0]))
    return np.asarray(values, dtype=np.float64)


def make_policy(
    models: dict[str, LGBMRegressor],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibrated_margin: float,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    cfg = CONFIG
    required_margin = max(cfg.switch_margin, calibrated_margin)

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        utilities = model_utilities(models, frames, symbols, timestamp)
        if not np.isfinite(utilities[1:]).any():
            return 0, 0.0

        best = int(np.nanargmax(utilities))
        best_value = float(utilities[best])
        current_value = float(utilities[current_position])
        entry_threshold = cfg.cash_threshold + cfg.minimum_expected_edge

        if (
            current_position > 0
            and np.isfinite(current_value)
            and holding_days < cfg.minimum_holding_days
        ):
            return current_position, current_value

        if best == 0 or best_value <= cfg.cash_threshold:
            return 0, 0.0

        if current_position == 0:
            if best_value >= entry_threshold:
                return best, best_value
            return 0, 0.0

        if best == current_position:
            return current_position, current_value

        if best_value >= current_value + required_margin:
            return best, best_value

        return current_position, current_value

    return policy


def training_transition_log_return(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    date_now: pd.Timestamp,
    date_next: pd.Timestamp,
    from_position: int,
    to_position: int,
) -> float:
    cfg = CONFIG
    gross = 1.0

    if from_position > 0:
        symbol = symbols[from_position - 1]
        close_now = float(frames[symbol].loc[date_now, "close"])
        open_next = float(frames[symbol].loc[date_next, "open"])
        if close_now > 0:
            gross *= open_next / close_now

    if from_position != to_position:
        one_side = max(0.0, cfg.slippage_bps) / 10_000.0
        one_side += max(0.0, cfg.commission_rate)
        sides = int(from_position != 0) + int(to_position != 0)
        switch_cost = min(0.25, one_side * sides)
        gross *= max(1e-8, 1.0 - switch_cost)

    if to_position > 0:
        symbol = symbols[to_position - 1]
        open_next = float(frames[symbol].loc[date_next, "open"])
        close_next = float(frames[symbol].loc[date_next, "close"])
        if open_next > 0:
            gross *= close_next / open_next

    return float(np.log(max(gross, 1e-12)))


def risk_adjusted_reward(
    log_return: float,
    wealth_before: float,
    peak_before: float,
) -> tuple[float, float, float]:
    cfg = CONFIG
    wealth_after = wealth_before * math.exp(log_return)
    peak_after = max(peak_before, wealth_after)
    previous_drawdown = max(
        0.0,
        1.0 - wealth_before / max(peak_before, 1e-12),
    )
    current_drawdown = max(
        0.0,
        1.0 - wealth_after / max(peak_after, 1e-12),
    )
    drawdown_increase = max(0.0, current_drawdown - previous_drawdown)
    downside = max(0.0, -log_return)
    reward = (
        log_return
        - cfg.downside_penalty * downside
        - cfg.drawdown_penalty * drawdown_increase
    )
    return float(reward), float(wealth_after), float(peak_after)


def calibration_score(
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> float:
    wealth = 1.0
    peak = 1.0
    position = 0
    holding = 0
    utility = 0.0

    for idx in range(len(dates) - 1):
        now = dates[idx]
        nxt = dates[idx + 1]
        action, _ = policy(now, position, holding)
        log_return = training_transition_log_return(
            frames,
            symbols,
            now,
            nxt,
            position,
            action,
        )
        reward, wealth, peak = risk_adjusted_reward(
            log_return,
            wealth,
            peak,
        )
        utility += reward

        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0

    return float(utility)


def calibrate_margin(
    models: dict[str, LGBMRegressor],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    fold_id: int,
) -> tuple[float, float]:
    best_margin = CONFIG.switch_margin_candidates[0]
    best_score = float("-inf")

    for candidate in CONFIG.switch_margin_candidates:
        policy = make_policy(
            models,
            frames,
            symbols,
            calibrated_margin=candidate,
        )
        score = calibration_score(
            policy,
            frames,
            symbols,
            calibration_dates,
        )
        log(
            f"[5/8] Fold {fold_id} | margem={candidate:.4f} | "
            f"score de calibração={score:.6f}"
        )
        if score > best_score:
            best_score = score
            best_margin = candidate

    return float(best_margin), float(best_score)


def round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


def calculate_fees(side: str, quantity: float, price: float) -> dict[str, float]:
    cfg = CONFIG
    if quantity <= 0 or price <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }

    side = side.upper()
    trade_value = quantity * price
    commission = round_fee_to_cent(trade_value * cfg.commission_rate)
    cat = round_fee_to_cent(quantity * cfg.cat_fee_per_share)
    sec = 0.0
    taf = 0.0

    if side == "SELL":
        sec = round_fee_to_cent(trade_value * cfg.sec_fee_rate)
        taf = round_fee_to_cent(
            min(quantity * cfg.taf_fee_per_share, cfg.taf_fee_cap)
        )
    elif side != "BUY":
        raise ValueError(f"Lado de operação inválido: {side}")

    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def apply_slippage(price: float, side: str) -> float:
    adjustment = CONFIG.slippage_bps / 10_000.0
    return price * (1 + adjustment if side == "BUY" else 1 - adjustment)


def execute_buy(cash: float, raw_price: float) -> tuple[float, float, dict[str, float]]:
    price = apply_slippage(raw_price, "BUY")
    quantity = cash / price

    for _ in range(25):
        fees = calculate_fees("BUY", quantity, price)
        next_quantity = max(0.0, (cash - fees["total_fee"]) / price)
        if not CONFIG.fractional_shares:
            next_quantity = float(math.floor(next_quantity))
        if abs(next_quantity - quantity) < 1e-10:
            quantity = next_quantity
            break
        quantity = next_quantity

    fees = calculate_fees("BUY", quantity, price)
    return float(quantity), float(price), fees


def equal_weight_benchmark(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    execution_dates: pd.DatetimeIndex,
) -> pd.Series:
    first = execution_dates[0]
    last = execution_dates[-1]
    capital_per_asset = CONFIG.initial_capital / len(symbols)

    quantities: dict[str, float] = {}
    residual = 0.0
    for symbol in symbols:
        price = apply_slippage(
            float(frames[symbol].loc[first, "open"]),
            "BUY",
        )
        quantity = capital_per_asset / price
        for _ in range(20):
            fees = calculate_fees("BUY", quantity, price)
            next_quantity = max(
                0.0,
                (capital_per_asset - fees["total_fee"]) / price,
            )
            if abs(next_quantity - quantity) < 1e-10:
                quantity = next_quantity
                break
            quantity = next_quantity
        fees = calculate_fees("BUY", quantity, price)
        quantities[symbol] = quantity
        residual += capital_per_asset - (
            quantity * price + fees["total_fee"]
        )

    values: list[float] = []
    for timestamp in execution_dates:
        equity = residual
        for symbol, quantity in quantities.items():
            equity += quantity * float(frames[symbol].loc[timestamp, "close"])
        values.append(float(equity))

    final_cash = residual
    for symbol, quantity in quantities.items():
        price = apply_slippage(
            float(frames[symbol].loc[last, "close"]),
            "SELL",
        )
        fees = calculate_fees("SELL", quantity, price)
        final_cash += quantity * price - fees["total_fee"]
    values[-1] = float(final_cash)

    return pd.Series(values, index=execution_dates, dtype=float)


def simulate_portfolio(
    policies: dict[int, Callable[[pd.Timestamp, int, int], tuple[int, float]]],
    decision_to_fold: dict[pd.Timestamp, int],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    execution_dates = decision_dates[1:]
    benchmark = equal_weight_benchmark(frames, symbols, execution_dates)

    cash = CONFIG.initial_capital
    position = 0
    quantity = 0.0
    entry_price = float("nan")
    entry_time: pd.Timestamp | None = None
    holding_days = 0
    total_fees = 0.0
    rotation_count = 0

    trades: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for idx in range(len(decision_dates) - 1):
        decision_date = pd.Timestamp(decision_dates[idx])
        execution_date = pd.Timestamp(decision_dates[idx + 1])
        fold_id = int(decision_to_fold[decision_date])
        policy = policies[fold_id]
        previous_position = position

        target_position, score = policy(
            decision_date,
            position,
            holding_days,
        )

        day_actions: list[str] = []
        if target_position != position:
            old_symbol = symbols[position - 1] if position > 0 else None
            new_symbol = symbols[target_position - 1] if target_position > 0 else None

            if position > 0:
                symbol = symbols[position - 1]
                price = apply_slippage(
                    float(frames[symbol].loc[execution_date, "open"]),
                    "SELL",
                )
                fees = calculate_fees("SELL", quantity, price)
                gross = quantity * price
                realized = (
                    quantity * (price - entry_price)
                    - fees["total_fee"]
                )
                cash += gross - fees["total_fee"]
                total_fees += fees["total_fee"]
                position_return = (
                    price / entry_price - 1
                    if np.isfinite(entry_price) and entry_price > 0
                    else 0.0
                )
                trades.append(
                    {
                        "timestamp": execution_date,
                        "decision_timestamp": decision_date,
                        "fold_id": fold_id,
                        "action": "SELL",
                        "asset": symbol,
                        "execution_price": price,
                        "quantity": quantity,
                        "gross_trade_value": gross,
                        "total_fee": fees["total_fee"],
                        "realized_pnl": realized,
                        "position_return": position_return,
                        "holding_bars": holding_days,
                        "entry_timestamp": entry_time,
                        "entry_price": entry_price,
                        "cash_after_trade": cash,
                    }
                )
                day_actions.append("SELL")
                quantity = 0.0
                entry_price = float("nan")
                entry_time = None
                holding_days = 0

            position = target_position
            if position > 0:
                symbol = symbols[position - 1]
                quantity, price, fees = execute_buy(
                    cash,
                    float(frames[symbol].loc[execution_date, "open"]),
                )
                gross = quantity * price
                cash -= gross + fees["total_fee"]
                total_fees += fees["total_fee"]
                entry_price = price
                entry_time = execution_date
                holding_days = 1
                trades.append(
                    {
                        "timestamp": execution_date,
                        "decision_timestamp": decision_date,
                        "fold_id": fold_id,
                        "action": "BUY",
                        "asset": symbol,
                        "execution_price": price,
                        "quantity": quantity,
                        "gross_trade_value": gross,
                        "total_fee": fees["total_fee"],
                        "realized_pnl": 0.0,
                        "position_return": 0.0,
                        "holding_bars": 0,
                        "entry_timestamp": execution_date,
                        "entry_price": price,
                        "cash_after_trade": cash,
                    }
                )
                day_actions.append("BUY")

            if previous_position > 0 and target_position > 0:
                rotation_count += 1
        elif position > 0:
            holding_days += 1

        if position > 0:
            symbol = symbols[position - 1]
            close_price = float(frames[symbol].loc[execution_date, "close"])
            equity = cash + quantity * close_price
            selected_asset = symbol
        else:
            equity = cash
            selected_asset = "CASH"

        curve_rows.append(
            {
                "timestamp": execution_date,
                "decision_timestamp": decision_date,
                "fold_id": fold_id,
                "selected_asset": selected_asset,
                "selected_score": float(score),
                "trade_action": "+".join(day_actions) if day_actions else "",
                "strategy_equity": float(equity),
                "benchmark_equity": float(benchmark.loc[execution_date]),
                "rotation_count": rotation_count,
                "total_fees": total_fees,
            }
        )

    if position > 0:
        final_date = pd.Timestamp(execution_dates[-1])
        symbol = symbols[position - 1]
        price = apply_slippage(
            float(frames[symbol].loc[final_date, "close"]),
            "SELL",
        )
        fees = calculate_fees("SELL", quantity, price)
        gross = quantity * price
        realized = quantity * (price - entry_price) - fees["total_fee"]
        cash += gross - fees["total_fee"]
        total_fees += fees["total_fee"]

        trades.append(
            {
                "timestamp": final_date,
                "decision_timestamp": final_date,
                "fold_id": curve_rows[-1]["fold_id"],
                "action": "FINAL_SELL",
                "asset": symbol,
                "execution_price": price,
                "quantity": quantity,
                "gross_trade_value": gross,
                "total_fee": fees["total_fee"],
                "realized_pnl": realized,
                "position_return": price / entry_price - 1,
                "holding_bars": holding_days,
                "entry_timestamp": entry_time,
                "entry_price": entry_price,
                "cash_after_trade": cash,
            }
        )
        curve_rows[-1]["strategy_equity"] = float(cash)
        curve_rows[-1]["total_fees"] = total_fees
        curve_rows[-1]["trade_action"] = (
            curve_rows[-1]["trade_action"] + "+FINAL_SELL"
        ).strip("+")

    curve = pd.DataFrame(curve_rows)
    trade_frame = pd.DataFrame(trades)
    return curve, trade_frame, benchmark


def maximum_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def annualized_sharpe(curve: pd.Series) -> float:
    returns = curve.pct_change(fill_method=None).dropna()
    if returns.empty or float(returns.std()) <= 0:
        return float("nan")
    return float(np.sqrt(252.0) * returns.mean() / returns.std())


def cagr(curve: pd.Series, initial_capital: float) -> float:
    if len(curve) < 2:
        return float("nan")
    years = max(
        (pd.Timestamp(curve.index[-1]) - pd.Timestamp(curve.index[0])).days
        / 365.25,
        1 / 365.25,
    )
    return float(
        (float(curve.iloc[-1]) / initial_capital) ** (1 / years) - 1
    )


def calculate_metrics(
    curve_frame: pd.DataFrame,
    trades: pd.DataFrame,
    folds: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    strategy_curve = pd.Series(
        curve_frame["strategy_equity"].to_numpy(dtype=float),
        index=pd.to_datetime(curve_frame["timestamp"], utc=True),
        dtype=float,
    )
    benchmark_curve = pd.Series(
        curve_frame["benchmark_equity"].to_numpy(dtype=float),
        index=strategy_curve.index,
        dtype=float,
    )

    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        subset = curve_frame.loc[curve_frame["fold_id"] == fold_id]
        if subset.empty:
            continue

        first_index = int(subset.index[0])
        strategy_start = (
            CONFIG.initial_capital
            if first_index == 0
            else float(curve_frame.loc[first_index - 1, "strategy_equity"])
        )
        benchmark_start = (
            CONFIG.initial_capital
            if first_index == 0
            else float(curve_frame.loc[first_index - 1, "benchmark_equity"])
        )
        strategy_end = float(subset.iloc[-1]["strategy_equity"])
        benchmark_end = float(subset.iloc[-1]["benchmark_equity"])
        fold_rows.append(
            {
                "fold_id": fold_id,
                "test_start": pd.Timestamp(subset.iloc[0]["timestamp"]),
                "test_end": pd.Timestamp(subset.iloc[-1]["timestamp"]),
                "strategy_starting_capital": strategy_start,
                "strategy_ending_capital": strategy_end,
                "strategy_return": strategy_end / strategy_start - 1.0,
                "benchmark_return": benchmark_end / benchmark_start - 1.0,
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    ending = float(strategy_curve.iloc[-1])
    benchmark_ending = float(benchmark_curve.iloc[-1])
    cash_days = int((curve_frame["selected_asset"] == "CASH").sum())
    buys = int((trades["action"] == "BUY").sum()) if not trades.empty else 0
    sells = (
        int(trades["action"].isin(["SELL", "FINAL_SELL"]).sum())
        if not trades.empty
        else 0
    )
    rotations = int(curve_frame["rotation_count"].iloc[-1]) if not curve_frame.empty else 0

    metrics = {
        "initial_capital": CONFIG.initial_capital,
        "ending_capital": ending,
        "total_return": ending / CONFIG.initial_capital - 1.0,
        "cagr": cagr(strategy_curve, CONFIG.initial_capital),
        "sharpe": annualized_sharpe(strategy_curve),
        "maximum_drawdown": maximum_drawdown(strategy_curve),
        "benchmark_ending_capital": benchmark_ending,
        "benchmark_return": benchmark_ending / CONFIG.initial_capital - 1.0,
        "benchmark_cagr": cagr(benchmark_curve, CONFIG.initial_capital),
        "benchmark_sharpe": annualized_sharpe(benchmark_curve),
        "benchmark_maximum_drawdown": maximum_drawdown(benchmark_curve),
        "cash_days": cash_days,
        "market_exposure": 1.0 - cash_days / max(1, len(curve_frame)),
        "buys": buys,
        "sells": sells,
        "rotations": rotations,
        "total_transaction_fees": (
            float(trades["total_fee"].sum()) if not trades.empty else 0.0
        ),
        "oos_sessions": int(len(curve_frame)),
        "oos_start": pd.Timestamp(curve_frame.iloc[0]["timestamp"]),
        "oos_end": pd.Timestamp(curve_frame.iloc[-1]["timestamp"]),
        "worst_fold_return": (
            float(fold_frame["strategy_return"].min())
            if not fold_frame.empty
            else None
        ),
    }
    return metrics, fold_frame


def write_outputs(
    metrics: dict[str, Any],
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    folds: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "folds.csv", index=False)

    payload = {
        "project": "tcc_mba_usp_data_science_analytics",
        "method": "standalone_walk_forward_lightgbm_rotation",
        "data_source": {
            "database": os.getenv("TCC_MONGO_DATABASE") or MONGO_DATABASE,
            "collection": MARKET_COLLECTION,
            "allowed_input": "OHLCV market bars only",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "assets": list(ASSETS),
        },
        "experiment": {
            "target_horizons": list(CONFIG.target_horizons),
            "target_weights": list(CONFIG.target_weights),
            "minimum_training_rows": CONFIG.minimum_training_rows,
            "calibration_days": CONFIG.calibration_days,
            "test_days": CONFIG.test_days,
            "purge_days": CONFIG.purge_days,
            "random_state": CONFIG.random_state,
        },
        "result": metrics,
        "elapsed_seconds": elapsed_seconds,
    }
    (OUTPUT_DIR / "backtest_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def success_sound() -> None:
    if platform.system().lower() != "windows":
        return
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        pass


def run() -> dict[str, Any]:
    started = time.perf_counter()
    log("TCC MBA USP — backtest independente")
    log(
        "Fonte permitida: somente candles OHLCV de alpaca_market_bars. "
        "Nenhuma Strategy/modelo/resultado salvo será lido."
    )

    raw_frames = load_market_data()
    frames, common_dates = prepare_panel(raw_frames)
    folds = build_walk_forward_folds(common_dates)
    symbols = sorted(frames)

    policies: dict[int, Callable[[pd.Timestamp, int, int], tuple[int, float]]] = {}
    decision_to_fold: dict[pd.Timestamp, int] = {}

    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        log(
            f"[4/8] Fold {fold_position}/{len(folds)} — "
            "treinando modelos para calibração..."
        )
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"])
            : int(fold["calibration_end_index"])
        ]

        calibration_models = fit_models(
            frames,
            train_dates,
            fold_id=fold_id,
            phase="calibração",
        )
        best_margin, best_score = calibrate_margin(
            calibration_models,
            frames,
            symbols,
            calibration_dates,
            fold_id,
        )

        log(
            f"[5/8] Fold {fold_id}: margem escolhida={best_margin:.4f} "
            f"(score={best_score:.6f})"
        )
        log(
            f"[6/8] Fold {fold_position}/{len(folds)} — "
            "treinando modelos finais do fold..."
        )
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        final_models = fit_models(
            frames,
            final_fit_dates,
            fold_id=fold_id,
            phase="final",
        )
        policies[fold_id] = make_policy(
            final_models,
            frames,
            symbols,
            calibrated_margin=best_margin,
        )

        for timestamp in fold["decision_dates"][:-1]:
            decision_to_fold[pd.Timestamp(timestamp)] = fold_id

    execution_start = int(folds[0]["test_start_index"])
    execution_end = int(folds[-1]["test_end_index"])
    decision_dates = common_dates[execution_start - 1 : execution_end]

    log(
        f"[7/8] Simulando {len(decision_dates) - 1} sessões OOS "
        f"({decision_dates[1].date()} -> {decision_dates[-1].date()})..."
    )
    curve, trades, _ = simulate_portfolio(
        policies,
        decision_to_fold,
        frames,
        symbols,
        decision_dates,
    )

    log("[8/8] Calculando métricas e gravando resultados...")
    metrics, fold_frame = calculate_metrics(curve, trades, folds)
    elapsed = time.perf_counter() - started
    write_outputs(metrics, curve, trades, fold_frame, elapsed)

    log("Backtest concluído.")
    log(f"Capital inicial : US$ {CONFIG.initial_capital:,.2f}")
    log(f"Capital final   : US$ {metrics['ending_capital']:,.2f}")
    log(f"Retorno total   : {metrics['total_return']:.2%}")
    log(f"CAGR            : {metrics['cagr']:.2%}")
    log(f"Sharpe          : {metrics['sharpe']:.4f}")
    log(f"Max Drawdown    : {metrics['maximum_drawdown']:.2%}")
    log(f"Rotações        : {metrics['rotations']}")
    log(f"Dias em CASH    : {metrics['cash_days']}")
    log(f"Pior fold       : {metrics['worst_fold_return']:.2%}")
    log(f"Tempo total     : {elapsed:.1f}s")
    log(f"Artefatos       : {OUTPUT_DIR}")

    success_sound()
    return metrics


if __name__ == "__main__":
    run()
