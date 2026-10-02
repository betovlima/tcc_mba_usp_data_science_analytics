from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from pesquisas.directional_change_lightgbm import (
    DC_FEATURES,
    _directional_change_state,
    _envolver_politica_reversao,
    adicionar_directional_change,
    calcular_peak_exit,
)


def _base_frame(close_values: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-02",
        periods=len(close_values),
        freq="B",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "close": close_values,
            "atr_pct_14": [0.02] * len(close_values),
        },
        index=index,
    )


def test_directional_change_state_detects_up_down_up_transitions() -> None:
    frame = _base_frame([100.0, 103.0, 106.0, 100.0, 97.0, 102.0, 104.0])
    state = _directional_change_state(frame["close"], 0.05)

    assert state.iloc[2]["dc_regime_05pct"] == 1.0
    assert state.iloc[3]["dc_regime_05pct"] == -1.0
    assert state.iloc[5]["dc_regime_05pct"] == 1.0


def test_directional_change_features_are_causal() -> None:
    shared = [100.0, 102.0, 105.0, 104.0, 106.0, 103.0]
    left = adicionar_directional_change(
        _base_frame(shared + [90.0, 88.0, 86.0, 85.0, 84.0])
    )
    right = adicionar_directional_change(
        _base_frame(shared + [112.0, 115.0, 117.0, 120.0, 122.0])
    )

    pd.testing.assert_frame_equal(
        left.loc[left.index[: len(shared)], list(DC_FEATURES)],
        right.loc[right.index[: len(shared)], list(DC_FEATURES)],
    )


def test_overlay_exits_when_reversal_probability_is_high() -> None:
    timestamp = pd.Timestamp("2026-01-05T00:00:00Z")
    frames = {
        "AAA": pd.DataFrame(
            {"dc_regime_04pct": [1.0]},
            index=pd.DatetimeIndex([timestamp]),
        )
    }
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(_timestamp, _current_position, _holding_days):
        diagnostics[timestamp] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_reversao(
        base_policy,
        probabilities={timestamp: {"AAA": 0.82}},
        frames=frames,
        symbols=["AAA"],
        probability_threshold=0.70,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
    )

    target, score = policy(timestamp, 1, 4)

    assert target == 0
    assert score == 0.0
    assert diagnostics[timestamp]["directional_change_exit_triggered"] is True
    assert diagnostics[timestamp]["decision_reason"] == (
        "DIRECTIONAL_CHANGE_REVERSAL_EXIT"
    )
    assert diagnostics[timestamp]["final_action_asset"] == "CASH"


def test_overlay_preserves_control_rotation() -> None:
    timestamp = pd.Timestamp("2026-01-05T00:00:00Z")
    frames = {
        "AAA": pd.DataFrame(
            {"dc_regime_04pct": [1.0]},
            index=pd.DatetimeIndex([timestamp]),
        ),
        "BBB": pd.DataFrame(
            {"dc_regime_04pct": [1.0]},
            index=pd.DatetimeIndex([timestamp]),
        ),
    }
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(_timestamp, _current_position, _holding_days):
        diagnostics[timestamp] = {
            "decision_reason": "ROTATE_TO_BEST_ASSET",
            "final_action_asset": "BBB",
        }
        return 2, 0.30

    policy = _envolver_politica_reversao(
        base_policy,
        probabilities={timestamp: {"AAA": 0.99}},
        frames=frames,
        symbols=["AAA", "BBB"],
        probability_threshold=0.70,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
    )

    target, score = policy(timestamp, 1, 5)

    assert target == 2
    assert score == 0.30
    assert diagnostics[timestamp]["directional_change_exit_triggered"] is False


def test_peak_exit_normal_sell_excludes_exit_session_high() -> None:
    index = pd.to_datetime(
        [
            "2026-01-05T05:00:00Z",
            "2026-01-06T05:00:00Z",
            "2026-01-07T05:00:00Z",
            "2026-01-08T05:00:00Z",
            "2026-01-09T05:00:00Z",
            "2026-01-12T05:00:00Z",
            "2026-01-13T05:00:00Z",
            "2026-01-14T05:00:00Z",
            "2026-01-15T05:00:00Z",
            "2026-01-16T05:00:00Z",
            "2026-01-19T05:00:00Z",
            "2026-01-20T05:00:00Z",
        ],
        utc=True,
    )
    frames = {
        "AAA": pd.DataFrame(
            {
                "high": [
                    12.0,
                    15.0,
                    18.0,
                    14.0,
                    15.0,
                    16.0,
                    15.0,
                    15.5,
                    16.0,
                    15.0,
                    14.5,
                    14.0,
                ],
            },
            index=index,
        )
    }
    trades = pd.DataFrame(
        [
            {
                "timestamp": index[2],
                "action": "SELL",
                "asset": "AAA",
                "entry_timestamp": index[0],
                "entry_price": 10.0,
                "execution_price": 12.0,
            }
        ]
    )

    summary, detail = calcular_peak_exit(trades, frames)

    assert summary["closed_positions"] == 1
    assert abs(detail.iloc[0]["peak_price_while_held"] - 15.0) < 1e-12
    assert abs(detail.iloc[0]["exit_distance_from_peak_pct"] - 20.0) < 1e-12
    assert abs(detail.iloc[0]["peak_capture_pct"] - 40.0) < 1e-12
    assert abs(detail.iloc[0]["max_runup_pct"] - 50.0) < 1e-12
