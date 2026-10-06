from __future__ import annotations

import math

import pandas as pd

from pesquisas.eventos_substituicao_u67 import (
    _asset_bucket,
    _future_target,
)


def test_asset_holdout_bucket_is_deterministic() -> None:
    assert _asset_bucket("AAPL") == _asset_bucket("AAPL")
    assert 0 <= _asset_bucket("AAPL") < 5
    assert 0 <= _asset_bucket("MSFT") < 5


def test_future_target_uses_relative_utility_and_switch_penalty() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    candidate = pd.DataFrame(
        {
            "forward_risk_adjusted_utility": [0.20],
            "forward_net_log_return": [0.18],
            "forward_horizon_utility_5": [0.10],
            "forward_horizon_utility_10": [0.12],
            "forward_horizon_utility_20": [0.14],
            "forward_horizon_utility_40": [0.16],
            "forward_horizon_utility_60": [0.18],
        },
        index=[timestamp],
    )
    incumbent = pd.DataFrame(
        {
            "forward_risk_adjusted_utility": [0.08],
            "forward_net_log_return": [0.07],
            "forward_horizon_utility_5": [0.02],
            "forward_horizon_utility_10": [0.03],
            "forward_horizon_utility_20": [0.04],
            "forward_horizon_utility_40": [0.05],
            "forward_horizon_utility_60": [0.06],
        },
        index=[timestamp],
    )

    penalty = math.log(0.999)
    target = _future_target(
        candidate,
        incumbent,
        timestamp,
        penalty,
    )

    expected = 0.20 - 0.08 + penalty
    assert math.isclose(
        target["target_delta_utility_multi"],
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert target["target_positive_multi"] == 1.0


def test_substitution_runner_declares_non_random_validation() -> None:
    source = (
        pd.io.common.get_handle(
            "pesquisas/eventos_substituicao_u67.py",
            "r",
            encoding="utf-8",
        ).handle.read()
    )
    assert '"random_row_split_allowed": False' in source
    assert "financial_replay=NO" in source
    assert "alpaca_download=NO" in source
