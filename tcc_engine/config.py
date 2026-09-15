"""Frozen standalone configuration for the certified historical TCC replay.

The values in this module reconstruct the *execution request* that produced the
certified historical Strategy #10 replay.  They are declared locally on
purpose: the TCC runtime must not read Strategy documents, model profiles or
processed artifacts from Market Cycle Trader.

The final 37-asset universe is the frozen output of the already-finished
historical universe study.  No asset search is performed at runtime.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

START_DATE = "2016-01-01"
END_DATE = "2026-09-04"

ASSETS = (
    "NVDA", "MSFT", "META", "TSLA", "AMD", "JPM", "SPY", "AVGO", "NFLX",
    "ORCL", "COST", "LLY", "XOM", "CAT", "WMT", "V", "HD", "ADC", "ADEA",
    "ADI", "ADM", "GKOS", "VNCE", "CORT", "UNFI", "DNN", "MKSI", "APD",
    "DDS", "RACE", "UNF", "TX", "CEF", "YANG", "KKR", "BXMT", "SCSC",
)

# Provenance only.  These fingerprints are never used as trading inputs.
HISTORICAL_SOURCE_COMMIT = "17019d95bfce6f0fbcd153e097b1968d9cfce1ca"
HISTORICAL_STRATEGY_ID = "strategy-87713a05860748719ec18d0a086dcce7"
HISTORICAL_STRATEGY_SEQUENCE = 10
HISTORICAL_STRATEGY_REVISION_AT_CERTIFICATION = 15
HISTORICAL_STRATEGY_CONFIGURATION_SHA256 = (
    "509b940659a89a7348be3690882213c839ce1a43b7e44057656074f5b2517a6e"
)
HISTORICAL_MODEL_SETTINGS_SHA256 = (
    "b4d112d678f79ca931c24630831e6464ebfe492f46364f54632c2980618803ab"
)
HISTORICAL_EXECUTION_REQUEST_SHA256 = (
    "8aa99e2c5a9e4cdf666cbfa406896b1aee82f2fbe9ea65d68ad077e8b8be73a6"
)
HISTORICAL_MARKET_OHLCV_SHA256 = (
    "2db920471bc6ff8925081735c4d8218adf879a1363fae7fd239da940d6ebe30c"
)
HISTORICAL_ENDING_CAPITAL = 43_759_854.819224246


def _lightgbm_settings() -> dict[str, Any]:
    """Exact model snapshot bound to Strategy #10 during the certified replay."""
    return {
        "schema_version": 3,
        "settings_revision": 2,
        "profile_id": "strategy",
        "lightgbm": {
            "n_estimators": 329,
            "learning_rate": 0.020731,
            "max_depth": 3,
            "num_leaves": 6,
            "min_child_samples": 18,
            "min_child_weight": 5.0,
            "subsample": 0.85,
            "subsample_freq": 0,
            "colsample_bytree": 0.88067,
            "reg_alpha": 0.050837,
            "reg_lambda": 3.596305,
            "max_bin": 255,
            "n_jobs": -1,
            "repetitions": 1,
            "seed_step": 1000,
            "random_state": 42,
        },
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StandaloneBacktestConfig:
    """Local equivalent of the certified BacktestExecutionRequest."""

    assets: tuple[str, ...] = ASSETS
    strategy_mode: str = "COMPOUND_ROTATION_SWING_XGBOOST"
    start_date: str = START_DATE
    end_date: str | None = END_DATE
    timeframe: str = "1Day"

    market_data_provider: str = "alpaca"
    alpaca_historical_feed: str = "sip"
    alpaca_live_feed: str = "iex"
    alpaca_adjustment: str = "all"
    market_data_history_backfill_enabled: bool = True
    market_data_history_backfill_provider: str = "alpaca"
    market_data_history_start_tolerance_days: int = 10
    market_data_require_complete_history: bool = True

    # Historical schema name.  The bound research_model_family below is the
    # actual LightGBM implementation used by the replay.
    rotation_models: tuple[str, ...] = ("xgboost_utility",)
    rotation_horizon_days: int = 40
    rotation_target_horizons: tuple[int, ...] = (5, 10, 20, 40, 60)
    rotation_target_horizon_weights: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.25)
    rotation_movement_capture_weight: float = 0.35
    rotation_trend_persistence_weight: float = 0.20

    rotation_minimum_training_rows: int = 700
    rotation_walk_forward_enabled: bool = True
    rotation_walk_forward_calibration_days: int = 126
    rotation_walk_forward_test_days: int = 504
    rotation_walk_forward_min_test_days: int = 126
    rotation_purge_days: int = 60

    rotation_downside_penalty: float = 0.20
    rotation_drawdown_penalty: float = 0.35
    rotation_min_holding_days: int = 2
    rotation_min_expected_edge: float = 0.001
    rotation_cash_threshold: float = 0.0
    rotation_switch_margin: float = 0.0005
    rotation_switch_margin_candidates: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01)

    opportunity_utility_entry_threshold: float = 0.28
    opportunity_utility_exit_threshold: float = 0.27
    allocation_lookback_days: int = 126
    allocation_max_asset_weight: float = 1.0
    allocation_cvar_confidence: float = 0.95
    allocation_cvar_penalty: float = 1.0
    allocation_turnover_penalty: float = 0.0025
    allocation_minimum_utility: float = 0.0
    allocation_signal_scale: float = 1.0

    # Strategy-owned legacy fields are retained because they are part of the
    # certified request fingerprint.  LightGBM hyperparameters come from the
    # research_model_settings snapshot, not these compatibility values.
    rotation_xgb_n_estimators: int = 300
    rotation_xgb_learning_rate: float = 0.035
    rotation_xgb_max_depth: int = 3
    rotation_accelerator: str = "cpu"
    rotation_allow_cpu_fallback: bool = True
    rotation_xgb_repetitions: int = 1
    rotation_seed_step: int = 1000

    initial_capital: float = 10_000.0
    whole_shares: bool = False
    slippage_bps: float = 0.0
    commission_rate: float = 0.0
    sec_fee_rate: float = 2.06e-5
    taf_fee_per_share: float = 0.000195
    taf_fee_cap: float = 9.79
    cat_fee_per_share: float = 3e-6

    xgb_min_child_weight: float = 5.0
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_alpha: float = 0.10
    xgb_reg_lambda: float = 2.0
    xgb_n_jobs: int = -1
    deterministic_execution: bool = False
    numeric_thread_limit: int = 1

    mongo_cache_enabled: bool = True
    mongo_refresh_overlap_days: int = 7
    mongo_write_batch_size: int = 1000
    random_state: int = 42

    analysis_start_date: str = START_DATE
    analysis_end_date: str | None = END_DATE
    calendar_anchor_assets: tuple[str, ...] = ASSETS
    research_reference_assets: tuple[str, ...] = ASSETS
    research_candidate_assets: tuple[str, ...] = ()
    research_model_family: str = "lightgbm_utility"
    research_model_settings: dict[str, Any] = field(default_factory=_lightgbm_settings)
    research_market_data_mode: str = "database_only"
    expected_market_data_signature_sha256: str | None = None
    research_market_data_snapshot_id: str | None = None
    walk_forward_fold_count_override: int | None = None

    @property
    def fractional_shares(self) -> bool:
        """Historical BacktestRequest exposes this as the inverse property."""
        return not self.whole_shares

    def model_copy(self, *, update: dict[str, Any] | None = None) -> "StandaloneBacktestConfig":
        """Small Pydantic-compatible adapter required by the vendored engine."""
        return replace(self, **dict(update or {}))

    def execution_request_payload(self) -> dict[str, Any]:
        """Mirror BacktestExecutionRequest.model_dump(mode='json') exactly."""
        return {
            "assets": list(self.assets),
            "strategy_mode": self.strategy_mode,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timeframe": self.timeframe,
            "market_data_provider": self.market_data_provider,
            "alpaca_historical_feed": self.alpaca_historical_feed,
            "alpaca_live_feed": self.alpaca_live_feed,
            "alpaca_adjustment": self.alpaca_adjustment,
            "market_data_history_backfill_enabled": self.market_data_history_backfill_enabled,
            "market_data_history_backfill_provider": self.market_data_history_backfill_provider,
            "market_data_history_start_tolerance_days": self.market_data_history_start_tolerance_days,
            "market_data_require_complete_history": self.market_data_require_complete_history,
            "rotation_models": list(self.rotation_models),
            "rotation_horizon_days": self.rotation_horizon_days,
            "rotation_target_horizons": list(self.rotation_target_horizons),
            "rotation_target_horizon_weights": list(self.rotation_target_horizon_weights),
            "rotation_movement_capture_weight": self.rotation_movement_capture_weight,
            "rotation_trend_persistence_weight": self.rotation_trend_persistence_weight,
            "rotation_minimum_training_rows": self.rotation_minimum_training_rows,
            "rotation_walk_forward_enabled": self.rotation_walk_forward_enabled,
            "rotation_walk_forward_calibration_days": self.rotation_walk_forward_calibration_days,
            "rotation_walk_forward_test_days": self.rotation_walk_forward_test_days,
            "rotation_walk_forward_min_test_days": self.rotation_walk_forward_min_test_days,
            "rotation_purge_days": self.rotation_purge_days,
            "rotation_downside_penalty": self.rotation_downside_penalty,
            "rotation_drawdown_penalty": self.rotation_drawdown_penalty,
            "rotation_min_holding_days": self.rotation_min_holding_days,
            "rotation_min_expected_edge": self.rotation_min_expected_edge,
            "rotation_cash_threshold": self.rotation_cash_threshold,
            "rotation_switch_margin": self.rotation_switch_margin,
            "rotation_switch_margin_candidates": list(self.rotation_switch_margin_candidates),
            "opportunity_utility_entry_threshold": self.opportunity_utility_entry_threshold,
            "opportunity_utility_exit_threshold": self.opportunity_utility_exit_threshold,
            "allocation_lookback_days": self.allocation_lookback_days,
            "allocation_max_asset_weight": self.allocation_max_asset_weight,
            "allocation_cvar_confidence": self.allocation_cvar_confidence,
            "allocation_cvar_penalty": self.allocation_cvar_penalty,
            "allocation_turnover_penalty": self.allocation_turnover_penalty,
            "allocation_minimum_utility": self.allocation_minimum_utility,
            "allocation_signal_scale": self.allocation_signal_scale,
            "rotation_xgb_n_estimators": self.rotation_xgb_n_estimators,
            "rotation_xgb_learning_rate": self.rotation_xgb_learning_rate,
            "rotation_xgb_max_depth": self.rotation_xgb_max_depth,
            "rotation_accelerator": self.rotation_accelerator,
            "rotation_allow_cpu_fallback": self.rotation_allow_cpu_fallback,
            "rotation_xgb_repetitions": self.rotation_xgb_repetitions,
            "rotation_seed_step": self.rotation_seed_step,
            "initial_capital": self.initial_capital,
            "whole_shares": self.whole_shares,
            "slippage_bps": self.slippage_bps,
            "commission_rate": self.commission_rate,
            "sec_fee_rate": self.sec_fee_rate,
            "taf_fee_per_share": self.taf_fee_per_share,
            "taf_fee_cap": self.taf_fee_cap,
            "cat_fee_per_share": self.cat_fee_per_share,
            "xgb_min_child_weight": self.xgb_min_child_weight,
            "xgb_subsample": self.xgb_subsample,
            "xgb_colsample_bytree": self.xgb_colsample_bytree,
            "xgb_reg_alpha": self.xgb_reg_alpha,
            "xgb_reg_lambda": self.xgb_reg_lambda,
            "xgb_n_jobs": self.xgb_n_jobs,
            "deterministic_execution": self.deterministic_execution,
            "numeric_thread_limit": self.numeric_thread_limit,
            "mongo_cache_enabled": self.mongo_cache_enabled,
            "mongo_refresh_overlap_days": self.mongo_refresh_overlap_days,
            "mongo_write_batch_size": self.mongo_write_batch_size,
            "random_state": self.random_state,
            "analysis_start_date": self.analysis_start_date,
            "analysis_end_date": self.analysis_end_date,
            "calendar_anchor_assets": list(self.calendar_anchor_assets),
            "research_reference_assets": list(self.research_reference_assets),
            "research_candidate_assets": list(self.research_candidate_assets),
            "research_model_family": self.research_model_family,
            "research_model_settings": self.research_model_settings,
            "research_market_data_mode": self.research_market_data_mode,
            "expected_market_data_signature_sha256": self.expected_market_data_signature_sha256,
            "research_market_data_snapshot_id": self.research_market_data_snapshot_id,
            "walk_forward_fold_count_override": self.walk_forward_fold_count_override,
        }

    def execution_request_sha256(self) -> str:
        return _canonical_sha256(self.execution_request_payload())

    def model_settings_sha256(self) -> str:
        return _canonical_sha256(self.research_model_settings)

    def validate_reference_fingerprints(self) -> None:
        model_hash = self.model_settings_sha256()
        if model_hash != HISTORICAL_MODEL_SETTINGS_SHA256:
            raise RuntimeError(
                "Standalone LightGBM snapshot drifted from the certified Strategy #10 snapshot: "
                f"{model_hash} != {HISTORICAL_MODEL_SETTINGS_SHA256}."
            )
        request_hash = self.execution_request_sha256()
        if request_hash != HISTORICAL_EXECUTION_REQUEST_SHA256:
            raise RuntimeError(
                "Standalone execution request drifted from the certified Strategy #10 replay: "
                f"{request_hash} != {HISTORICAL_EXECUTION_REQUEST_SHA256}."
            )


CONFIG = StandaloneBacktestConfig()
CONFIG.validate_reference_fingerprints()
