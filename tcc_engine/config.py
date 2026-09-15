"""Configuração explícita do experimento acadêmico.

Os parâmetros do modelo, da validação temporal e da simulação financeira ficam
neste repositório. A fonte externa de mercado é o OHLCV diário baixado do Yahoo
Finance pelo ``backtest.py``.
"""
from __future__ import annotations

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


def _configuracoes_lightgbm() -> dict[str, Any]:
    """Retorna os hiperparâmetros declarados para o LightGBM."""
    return {
        "schema_version": 3,
        "settings_revision": 1,
        "profile_id": "tcc",
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


@dataclass(frozen=True)
class StandaloneBacktestConfig:
    assets: tuple[str, ...] = ASSETS
    strategy_mode: str = "COMPOUND_ROTATION_SWING_XGBOOST"
    start_date: str = START_DATE
    end_date: str | None = END_DATE
    timeframe: str = "1Day"

    # Fonte de mercado declarada pelo experimento.
    market_data_provider: str = "yahoo"
    yahoo_interval: str = "1d"
    yahoo_auto_adjust: bool = True
    market_data_history_backfill_enabled: bool = False
    market_data_history_backfill_provider: str = "yahoo"
    market_data_history_start_tolerance_days: int = 10
    market_data_require_complete_history: bool = True

    # Alvo e validação temporal.
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

    # Política de rotação.
    rotation_downside_penalty: float = 0.20
    rotation_drawdown_penalty: float = 0.35
    rotation_min_holding_days: int = 2
    rotation_min_expected_edge: float = 0.001
    rotation_cash_threshold: float = 0.0
    rotation_switch_margin: float = 0.0005
    rotation_switch_margin_candidates: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01)

    # Nomes mantidos por compatibilidade com a interface interna do motor.
    opportunity_utility_entry_threshold: float = 0.28
    opportunity_utility_exit_threshold: float = 0.27
    allocation_lookback_days: int = 126
    allocation_max_asset_weight: float = 1.0
    allocation_cvar_confidence: float = 0.95
    allocation_cvar_penalty: float = 1.0
    allocation_turnover_penalty: float = 0.0025
    allocation_minimum_utility: float = 0.0
    allocation_signal_scale: float = 1.0

    rotation_xgb_n_estimators: int = 300
    rotation_xgb_learning_rate: float = 0.035
    rotation_xgb_max_depth: int = 3
    rotation_accelerator: str = "cpu"
    rotation_allow_cpu_fallback: bool = True
    rotation_xgb_repetitions: int = 1
    rotation_seed_step: int = 1000

    # Simulação financeira.
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
    random_state: int = 42

    analysis_start_date: str = START_DATE
    analysis_end_date: str | None = END_DATE
    calendar_anchor_assets: tuple[str, ...] = ASSETS
    research_reference_assets: tuple[str, ...] = ASSETS
    research_candidate_assets: tuple[str, ...] = ()
    research_model_family: str = "lightgbm_utility"
    research_model_settings: dict[str, Any] = field(default_factory=_configuracoes_lightgbm)
    research_market_data_mode: str = "yahoo_snapshot"
    expected_market_data_signature_sha256: str | None = None
    research_market_data_snapshot_id: str | None = None
    walk_forward_fold_count_override: int | None = None

    @property
    def fractional_shares(self) -> bool:
        return not self.whole_shares

    def model_copy(self, *, update: dict[str, Any] | None = None) -> "StandaloneBacktestConfig":
        return replace(self, **dict(update or {}))


CONFIG = StandaloneBacktestConfig()
