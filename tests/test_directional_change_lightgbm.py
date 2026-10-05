from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import numpy as np
import pandas as pd

from pesquisas.directional_change_lightgbm import (
    DC_FEATURES,
    HSMMAssetModel,
    _agrupar_gatilhos_ablation,
    _bocpd_downward_scores,
    _bottom_turn_targets,
    _directional_change_state,
    _envolver_politica_bocpd,
    _envolver_politica_bottom_turn,
    _envolver_politica_cooldown_pos_top_turn,
    _envolver_politica_hazard,
    _envolver_politica_hsmm,
    _filter_hsmm_asset,
    _envolver_politica_top_turn,
    _top_turn_targets,
    adicionar_top_turn_features,
    calcular_bottom_entry,
    calcular_metricas_peak_gatilhos,
    calcular_peak_exit,
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)


def _base_frame(close_values: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-02",
        periods=len(close_values),
        freq="B",
        tz="UTC",
    )
    close = pd.Series(close_values, index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
            "atr_pct_14": 0.02,
            "distance_from_high_20": -0.01,
            "distance_from_low_20": 0.01,
            "return_20": 0.10,
        },
        index=index,
    )
    return frame


def test_completion_sound_does_not_require_snd_sync(monkeypatch) -> None:
    calls: list[str] = []

    fake_winsound = SimpleNamespace(
        SND_ALIAS=65536,
        MB_ICONASTERISK=64,
        PlaySound=lambda _name, _flags: calls.append("PlaySound"),
        MessageBeep=lambda _kind: calls.append("MessageBeep"),
        Beep=lambda _freq, _duration: calls.append("Beep"),
    )
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)

    sinal_sonoro_conclusao()

    assert "PlaySound" in calls
    assert "MessageBeep" in calls
    assert calls.count("Beep") == 2


def test_directional_change_state_detects_up_down_up_transitions() -> None:
    frame = _base_frame([100.0, 103.0, 106.0, 100.0, 97.0, 102.0, 104.0])
    state = _directional_change_state(frame["close"], 0.05)

    assert state.iloc[2]["dc_regime_05pct"] == 1.0
    assert state.iloc[3]["dc_regime_05pct"] == -1.0
    assert state.iloc[5]["dc_regime_05pct"] == 1.0


def test_directional_change_features_are_causal() -> None:
    shared = [100.0 + index for index in range(30)]
    left = adicionar_top_turn_features(
        _base_frame(shared + [120.0, 115.0, 110.0, 108.0, 106.0])
    )
    right = adicionar_top_turn_features(
        _base_frame(shared + [132.0, 135.0, 138.0, 140.0, 142.0])
    )

    pd.testing.assert_frame_equal(
        left.loc[left.index[: len(shared)], list(DC_FEATURES)],
        right.loc[right.index[: len(shared)], list(DC_FEATURES)],
    )


def test_top_turn_target_marks_drop_before_continuation() -> None:
    frame = _base_frame([100.0, 99.0, 97.0, 95.0, 96.0, 97.0])
    target = _top_turn_targets(frame, horizon=5)

    assert target.iloc[0]["forward_down_reversal"] == 1.0


def test_top_turn_target_rejects_when_continuation_happens_first() -> None:
    frame = _base_frame([100.0, 102.0, 104.0, 99.0, 96.0, 95.0])
    target = _top_turn_targets(frame, horizon=5)

    assert target.iloc[0]["forward_down_reversal"] == 0.0


def test_bottom_turn_target_marks_recovery_before_new_drop() -> None:
    frame = _base_frame([100.0, 101.0, 103.0, 105.0, 104.0, 103.0])
    target = _bottom_turn_targets(frame, horizon=5)

    assert target.iloc[0]["forward_up_reversal"] == 1.0


def test_bottom_turn_target_rejects_when_downside_continues_first() -> None:
    frame = _base_frame([100.0, 99.0, 98.0, 104.0, 105.0, 106.0])
    target = _bottom_turn_targets(frame, horizon=5)

    assert target.iloc[0]["forward_up_reversal"] == 0.0


def test_bottom_turn_gates_cash_entry_until_two_confirmations() -> None:
    dates = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {"bottom_turn_eligible": [True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "ENTER_BEST_ASSET",
            "final_action_asset": "AAA",
        }
        return 1, 0.40

    policy = _envolver_politica_bottom_turn(
        base_policy,
        probabilities={
            dates[0]: {"AAA": 0.80},
            dates[1]: {"AAA": 0.82},
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        probability_threshold=0.75,
        decision_diagnostics=diagnostics,
    )

    first_target, _ = policy(dates[0], 0, 0)
    second_target, _ = policy(dates[1], 0, 0)

    assert first_target == 0
    assert second_target == 1
    assert diagnostics[dates[0]]["bottom_turn_entry_blocked"] is True
    assert diagnostics[dates[1]]["bottom_turn_entry_triggered"] is True


def test_fixed_cooldown_blocks_exactly_five_sessions_after_top_exit() -> None:
    dates = pd.date_range(
        "2026-01-05",
        periods=7,
        freq="B",
        tz="UTC",
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, current_position, _holding_days):
        key = pd.Timestamp(timestamp)
        if current_position > 0:
            diagnostics[key] = {
                "decision_reason": "DIRECTIONAL_CHANGE_TOP_TURN_EXIT",
                "directional_change_exit_triggered": True,
                "final_action_asset": "CASH",
            }
            return 0, 0.0
        diagnostics[key] = {
            "decision_reason": "ENTER_BEST_ASSET",
            "directional_change_exit_triggered": False,
            "final_action_asset": "AAA",
        }
        return 1, 0.40

    policy = _envolver_politica_cooldown_pos_top_turn(
        base_policy,
        decision_diagnostics=diagnostics,
        cooldown_sessions=5,
    )

    assert policy(dates[0], 1, 10)[0] == 0
    blocked = [policy(date, 0, 0)[0] for date in dates[1:6]]
    released = policy(dates[6], 0, 0)[0]

    assert blocked == [0, 0, 0, 0, 0]
    assert diagnostics[dates[5]]["cooldown_gate_expired"] is True
    assert diagnostics[dates[5]]["cooldown_entry_blocked"] is True
    assert released == 1
    assert diagnostics[dates[6]]["cooldown_gate_active"] is False


def test_fixed_cooldown_does_not_block_initial_cash_entry() -> None:
    date = pd.Timestamp("2026-01-05T00:00:00Z")
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "ENTER_BEST_ASSET",
            "directional_change_exit_triggered": False,
            "final_action_asset": "AAA",
        }
        return 1, 0.40

    policy = _envolver_politica_cooldown_pos_top_turn(
        base_policy,
        decision_diagnostics=diagnostics,
        cooldown_sessions=5,
    )

    target, score = policy(date, 0, 0)

    assert target == 1
    assert score == 0.40
    assert diagnostics[date]["cooldown_gate_active"] is False
    assert diagnostics[date]["cooldown_entry_blocked"] is False


def test_bottom_turn_v2_does_not_block_initial_cash_entry() -> None:
    date = pd.Timestamp("2026-01-05T00:00:00Z")
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "ENTER_BEST_ASSET",
            "final_action_asset": "AAA",
        }
        return 1, 0.40

    policy = _envolver_politica_bottom_turn(
        base_policy,
        probabilities={date: {"AAA": 0.10}},
        frames={
            "AAA": pd.DataFrame(
                {"bottom_turn_eligible": [False]},
                index=[date],
            )
        },
        symbols=["AAA"],
        probability_threshold=0.75,
        decision_diagnostics=diagnostics,
        activate_only_after_top_turn=True,
        max_wait_sessions=5,
    )

    target, score = policy(date, 0, 0)

    assert target == 1
    assert score == 0.40
    assert diagnostics[date]["bottom_turn_gate_active"] is False
    assert diagnostics[date]["bottom_turn_entry_blocked"] is False


def test_bottom_turn_v2_expires_after_five_post_top_sessions() -> None:
    dates = pd.date_range(
        "2026-01-05",
        periods=7,
        freq="B",
        tz="UTC",
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, current_position, _holding_days):
        key = pd.Timestamp(timestamp)
        if current_position > 0:
            diagnostics[key] = {
                "decision_reason": "DIRECTIONAL_CHANGE_TOP_TURN_EXIT",
                "directional_change_exit_triggered": True,
                "final_action_asset": "CASH",
            }
            return 0, 0.0
        diagnostics[key] = {
            "decision_reason": "ENTER_BEST_ASSET",
            "directional_change_exit_triggered": False,
            "final_action_asset": "AAA",
        }
        return 1, 0.40

    frame = pd.DataFrame(
        {"bottom_turn_eligible": [False] * len(dates)},
        index=dates,
    )
    policy = _envolver_politica_bottom_turn(
        base_policy,
        probabilities={
            date: {"AAA": 0.10}
            for date in dates
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        probability_threshold=0.75,
        decision_diagnostics=diagnostics,
        activate_only_after_top_turn=True,
        max_wait_sessions=5,
    )

    assert policy(dates[0], 1, 10)[0] == 0
    blocked = [
        policy(date, 0, 0)[0]
        for date in dates[1:6]
    ]
    released = policy(dates[6], 0, 0)[0]

    assert blocked == [0, 0, 0, 0, 0]
    assert diagnostics[dates[5]]["bottom_turn_gate_expired"] is True
    assert released == 1
    assert diagnostics[dates[6]]["bottom_turn_entry_blocked"] is False


def test_bottom_turn_does_not_change_asset_to_asset_rotation() -> None:
    date = pd.Timestamp("2026-01-05T00:00:00Z")
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "ROTATE_TO_BEST_ASSET",
            "final_action_asset": "BBB",
        }
        return 2, 0.55

    policy = _envolver_politica_bottom_turn(
        base_policy,
        probabilities={},
        frames={
            "AAA": pd.DataFrame(index=[date]),
            "BBB": pd.DataFrame(index=[date]),
        },
        symbols=["AAA", "BBB"],
        probability_threshold=0.75,
        decision_diagnostics=diagnostics,
    )

    target, score = policy(date, 1, 5)

    assert target == 2
    assert score == 0.55
    assert diagnostics[date]["bottom_turn_entry_candidate"] is False


def test_overlay_requires_two_confirmations() -> None:
    dates = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {"top_turn_eligible": [True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_top_turn(
        base_policy,
        probabilities={
            dates[0]: {"AAA": 0.80},
            dates[1]: {"AAA": 0.82},
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        probability_threshold=0.75,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
    )

    first_target, _ = policy(dates[0], 1, 4)
    second_target, _ = policy(dates[1], 1, 5)

    assert first_target == 1
    assert second_target == 0
    assert diagnostics[dates[0]][
        "directional_change_confirmation_streak"
    ] == 1
    assert diagnostics[dates[1]][
        "directional_change_exit_triggered"
    ] is True



def test_overlay_can_suppress_one_trigger_and_require_fresh_confirmation() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-05T00:00:00Z",
            "2026-01-06T00:00:00Z",
            "2026-01-07T00:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {"top_turn_eligible": [True, True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_top_turn(
        base_policy,
        probabilities={
            dates[0]: {"AAA": 0.80},
            dates[1]: {"AAA": 0.82},
            dates[2]: {"AAA": 0.84},
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        probability_threshold=0.75,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
        suppressed_triggers={(dates[1], "AAA")},
    )

    first_target, _ = policy(dates[0], 1, 4)
    second_target, _ = policy(dates[1], 1, 5)
    third_target, _ = policy(dates[2], 1, 6)

    assert first_target == 1
    assert second_target == 1
    assert third_target == 1
    assert diagnostics[dates[1]][
        "directional_change_ablation_suppressed"
    ] is True
    assert diagnostics[dates[2]][
        "directional_change_confirmation_streak"
    ] == 1


def test_overlay_suppresses_all_triggers_for_asset() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-05T00:00:00Z",
            "2026-01-06T00:00:00Z",
            "2026-01-07T00:00:00Z",
            "2026-01-08T00:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {"top_turn_eligible": [True, True, True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_top_turn(
        base_policy,
        probabilities={
            date: {"AAA": 0.90}
            for date in dates
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        probability_threshold=0.75,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
        suppressed_assets={"AAA"},
    )

    targets = [
        policy(date, 1, 4 + index)[0]
        for index, date in enumerate(dates)
    ]

    assert targets == [1, 1, 1, 1]
    assert all(
        diagnostics[date][
            "directional_change_exit_triggered"
        ] is False
        for date in dates
    )
    assert any(
        diagnostics[date][
            "directional_change_ablation_suppressed_by_asset"
        ] is True
        for date in dates
    )


def test_overlay_can_disable_all_triggers() -> None:
    dates = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {"top_turn_eligible": [True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_top_turn(
        base_policy,
        probabilities={
            dates[0]: {"AAA": 0.90},
            dates[1]: {"AAA": 0.92},
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        probability_threshold=0.75,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
        disable_all_triggers=True,
    )

    first_target, _ = policy(dates[0], 1, 4)
    second_target, _ = policy(dates[1], 1, 5)

    assert first_target == 1
    assert second_target == 1
    assert diagnostics[dates[1]][
        "directional_change_ablation_disable_all"
    ] is True
    assert diagnostics[dates[1]][
        "directional_change_exit_triggered"
    ] is False

def test_ablation_groups_original_triggers_by_asset_and_fold() -> None:
    trigger_rows = pd.DataFrame(
        [
            {
                "decision_date": "2023-06-13T00:00:00Z",
                "current_asset": "TSLA",
                "previous_asset": "TSLA",
                "walk_forward_fold": 2,
            },
            {
                "decision_date": "2023-06-20T00:00:00Z",
                "current_asset": "TSLA",
                "previous_asset": "TSLA",
                "walk_forward_fold": 2,
            },
            {
                "decision_date": "2025-07-17T00:00:00Z",
                "current_asset": "NVDA",
                "previous_asset": "NVDA",
                "walk_forward_fold": 3,
            },
        ]
    )

    groups = _agrupar_gatilhos_ablation(trigger_rows)

    assert len(groups[("asset", "TSLA")]) == 2
    assert len(groups[("asset", "NVDA")]) == 1
    assert len(groups[("fold", "2")]) == 2
    assert len(groups[("fold", "3")]) == 1


def test_bottom_entry_measures_distance_from_low_before_entry() -> None:
    index = pd.date_range(
        "2026-01-05",
        periods=25,
        freq="B",
        tz="UTC",
    )
    close = pd.Series(
        [110.0, 106.0, 102.0, 100.0, 101.0, 102.0]
        + [103.0 + index * 0.2 for index in range(19)],
        index=index,
    )
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=index,
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": index[5],
                "action": "BUY",
                "asset": "AAA",
                "execution_price": 102.0,
                "decision_timestamp": index[4],
                "rotation_from_asset": "CASH",
                "rotation_to_asset": "AAA",
                "walk_forward_fold": 1,
                "decision_reason": "BOTTOM_TURN_CONFIRMED_ENTRY",
            }
        ]
    )

    summary, detail = calcular_bottom_entry(
        trades,
        {"AAA": frame},
        oos_start=index[0],
    )

    assert summary["cash_entries"] == 1
    assert detail.iloc[0]["bottom_price_before_entry"] == 99.0
    assert abs(
        detail.iloc[0]["entry_distance_from_bottom_pct"]
        - (102.0 / 99.0 - 1.0) * 100.0
    ) < 1e-12


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
                ]
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


def test_bocpd_scores_are_causal_on_shared_prefix() -> None:
    shared = [100.0 + 0.25 * index for index in range(45)]
    left_close = shared + [108.0, 103.0, 98.0, 96.0, 95.0]
    right_close = shared + [113.0, 116.0, 119.0, 121.0, 123.0]

    def frame(values: list[float]) -> pd.DataFrame:
        index = pd.date_range(
            "2020-01-02",
            periods=len(values),
            freq="B",
            tz="UTC",
        )
        close = pd.Series(values, index=index, dtype=float)
        returns = close.pct_change()
        return pd.DataFrame(
            {
                "return_1": returns,
                "vol_20": returns.rolling(20).std(),
            },
            index=index,
        )

    left = _bocpd_downward_scores(
        frame(left_close),
        hazard_lambda=60,
    )
    right = _bocpd_downward_scores(
        frame(right_close),
        hazard_lambda=60,
    )

    pd.testing.assert_series_equal(
        left.iloc[: len(shared)],
        right.iloc[: len(shared)],
    )


def test_bocpd_overlay_requires_two_confirmations() -> None:
    dates = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {"bocpd_eligible": [True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_bocpd(
        base_policy,
        score_cache={
            "AAA": pd.Series([0.20, 0.22], index=dates),
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        score_threshold=0.10,
        hazard_lambda=60,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
    )

    first_target, _ = policy(dates[0], 1, 4)
    second_target, _ = policy(dates[1], 1, 5)

    assert first_target == 1
    assert second_target == 0
    assert diagnostics[dates[0]]["bocpd_confirmation_streak"] == 1
    assert diagnostics[dates[1]]["bocpd_exit_triggered"] is True


def test_hsmm_filter_is_causal_on_shared_prefix() -> None:
    shared = [100.0 + 0.35 * index for index in range(50)]
    left_values = shared + [116.0, 110.0, 103.0, 99.0]
    right_values = shared + [118.0, 121.0, 124.0, 127.0]

    def frame(values: list[float]) -> pd.DataFrame:
        index = pd.date_range(
            "2020-01-02",
            periods=len(values),
            freq="B",
            tz="UTC",
        )
        close = pd.Series(values, index=index, dtype=float)
        daily = close.pct_change()
        vol = daily.rolling(20).std()
        ema = close.ewm(span=20, adjust=False).mean()
        return pd.DataFrame(
            {
                "return_1": daily,
                "vol_20": vol,
                "ema_slope_20_5": ema.pct_change(5),
            },
            index=index,
        )

    duration = np.ones((3, 60), dtype=float)
    duration = duration / duration.sum(axis=1, keepdims=True)
    model = HSMMAssetModel(
        means=np.asarray(
            [
                [-1.5, -1.0],
                [0.0, 0.0],
                [1.5, 1.0],
            ],
            dtype=float,
        ),
        variances=np.ones((3, 2), dtype=float),
        transition=np.asarray(
            [
                [0.0, 0.7, 0.3],
                [0.5, 0.0, 0.5],
                [0.3, 0.7, 0.0],
            ],
            dtype=float,
        ),
        duration_pmf=duration,
        initial_probabilities=np.asarray(
            [0.2, 0.3, 0.5],
            dtype=float,
        ),
    )

    left = _filter_hsmm_asset(frame(left_values), model)
    right = _filter_hsmm_asset(frame(right_values), model)

    pd.testing.assert_frame_equal(
        left.iloc[: len(shared)],
        right.iloc[: len(shared)],
    )


def test_hsmm_overlay_requires_two_confirmations() -> None:
    dates = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {"hsmm_eligible": [True, True]},
        index=dates,
    )
    filtered = pd.DataFrame(
        {
            "hsmm_reversal_score": [0.55, 0.62],
            "hsmm_down_probability": [0.35, 0.42],
            "hsmm_neutral_probability": [0.45, 0.40],
            "hsmm_up_probability": [0.20, 0.18],
        },
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_hsmm(
        base_policy,
        filtered={"AAA": filtered},
        frames={"AAA": frame},
        symbols=["AAA"],
        score_threshold=0.50,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
    )

    first_target, _ = policy(dates[0], 1, 4)
    second_target, _ = policy(dates[1], 1, 5)

    assert first_target == 1
    assert second_target == 0
    assert diagnostics[dates[0]]["hsmm_confirmation_streak"] == 1
    assert diagnostics[dates[1]]["hsmm_exit_triggered"] is True


def test_hazard_target_marks_competing_event_and_right_censoring() -> None:
    competing = _top_turn_targets(
        _base_frame([100.0, 102.0, 102.2, 102.1, 102.0, 101.9]),
        horizon=5,
    )
    censored = _top_turn_targets(
        _base_frame([100.0, 100.4, 100.2, 100.3, 100.1, 100.5]),
        horizon=5,
    )

    assert competing.iloc[0]["hazard_observed_down_event"] == 0.0
    assert competing.iloc[0]["hazard_competing_event_step"] == 1.0
    assert competing.iloc[0]["hazard_right_censored"] == 0.0

    assert pd.isna(censored.iloc[0]["hazard_observed_down_event"])
    assert pd.isna(censored.iloc[0]["hazard_competing_event_step"])
    assert censored.iloc[0]["hazard_right_censored"] == 1.0


def test_hazard_overlay_requires_two_confirmations() -> None:
    dates = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {"hazard_eligible": [True, True]},
        index=dates,
    )
    diagnostics: dict[pd.Timestamp, dict] = {}

    def base_policy(timestamp, _current_position, _holding_days):
        diagnostics[pd.Timestamp(timestamp)] = {
            "decision_reason": "HOLD_CURRENT_BEST",
            "final_action_asset": "AAA",
        }
        return 1, 0.25

    policy = _envolver_politica_hazard(
        base_policy,
        score_cache={
            "AAA": pd.Series([0.60, 0.62], index=dates),
        },
        frames={"AAA": frame},
        symbols=["AAA"],
        score_threshold=0.50,
        config=SimpleNamespace(rotation_min_holding_days=2),
        decision_diagnostics=diagnostics,
    )

    first_target, _ = policy(dates[0], 1, 4)
    second_target, _ = policy(dates[1], 1, 5)

    assert first_target == 1
    assert second_target == 0
    assert diagnostics[dates[0]]["hazard_confirmation_streak"] == 1
    assert diagnostics[dates[1]]["hazard_exit_triggered"] is True


def test_trigger_peak_metrics_match_same_execution_timestamp() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "current_asset": ["AAA", "AAA"],
            "directional_change_exit_triggered": [True, False],
        },
        index=timestamps,
    )
    peak_trades = pd.DataFrame(
        [
            {
                "asset": "AAA",
                "exit_timestamp": timestamps[0],
                "exit_distance_from_peak_pct": 0.5,
                "peak_capture_pct": 90.0,
            },
            {
                "asset": "AAA",
                "exit_timestamp": timestamps[1],
                "exit_distance_from_peak_pct": 9.0,
                "peak_capture_pct": 10.0,
            },
        ]
    )

    metrics = calcular_metricas_peak_gatilhos(
        peak_trades,
        predictions,
        "directional_change_exit_triggered",
    )

    assert metrics["median_exit_distance_from_peak_pct"] == 0.5
    assert metrics["median_peak_capture_pct"] == 90.0


def test_analysis_package_uses_one_stable_zip(tmp_path: Path) -> None:
    output = tmp_path / "directional_change"
    output.mkdir()
    (output / "intelligent_candidate_screen.json").write_text(
        '{"research_version":"test","execution_schema":"intelligent-candidate-screen-u62-v1"}',
        encoding="utf-8",
    )
    (output / "intelligent_selected_candidates.csv").write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    archive = criar_pacote_analise(output)

    assert archive == output / "pacote_analise.zip"
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(zipped.namelist())
    assert names == sorted(
        [
            "intelligent_candidate_screen.json",
            "intelligent_selected_candidates.csv",
        ]
    )


def test_analysis_package_refuses_stale_v1_schema(tmp_path: Path) -> None:
    output = tmp_path / "directional_change"
    output.mkdir()
    (output / "comparison_cycle.json").write_text(
        '{"research_version":"1.8.0-dev.1","execution_schema":"top-bottom-cycle-v1"}',
        encoding="utf-8",
    )

    try:
        criar_pacote_analise(output)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("stale v1 package should have been refused")

    assert "intelligent_candidate_screen.json" in message
    assert "copia antiga" in message
