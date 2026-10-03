from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd

from pesquisas.directional_change_lightgbm import (
    DC_FEATURES,
    _agrupar_gatilhos_ablation,
    _directional_change_state,
    _envolver_politica_top_turn,
    _top_turn_targets,
    adicionar_top_turn_features,
    calcular_peak_exit,
    criar_pacote_analise,
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
            "return_20": 0.10,
        },
        index=index,
    )
    return frame


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


def test_analysis_package_uses_one_stable_zip(tmp_path: Path) -> None:
    output = tmp_path / "directional_change"
    output.mkdir()
    (output / "comparison_directional_change.json").write_text(
        '{"ok": true}',
        encoding="utf-8",
    )
    (output / "directional_change_trades.csv").write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    archive = criar_pacote_analise(output)

    assert archive == output / "pacote_analise.zip"
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(zipped.namelist())
    assert names == [
        "comparison_directional_change.json",
        "directional_change_trades.csv",
    ]
