from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd

from pesquisas.directional_change_top_turn import (
    _envolver_politica_top_turn,
    _top_turn_targets,
    adicionar_top_turn_features,
)
from pesquisas.utilitarios import criar_pacote_analise


def _frame(close_values: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-02",
        periods=len(close_values),
        freq="B",
        tz="UTC",
    )
    close = pd.Series(close_values, index=index, dtype=float)
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000.0,
            "atr_pct_14": 0.02,
            "distance_from_high_20": -0.01,
            "return_20": 0.10,
        },
        index=index,
    )


def test_top_turn_target_marks_drop_before_continuation() -> None:
    frame = _frame([100.0, 99.0, 97.0, 95.0, 96.0, 97.0])
    target = _top_turn_targets(frame, horizon=5)
    assert target.iloc[0]["forward_down_reversal"] == 1.0


def test_top_turn_target_rejects_when_continuation_happens_first() -> None:
    frame = _frame([100.0, 102.0, 104.0, 99.0, 96.0, 95.0])
    target = _top_turn_targets(frame, horizon=5)
    assert target.iloc[0]["forward_down_reversal"] == 0.0


def test_top_turn_features_are_causal_before_future_diverges() -> None:
    shared = [100.0 + index for index in range(30)]
    left = _frame(shared + [120.0, 115.0, 110.0, 108.0, 106.0])
    right = _frame(shared + [132.0, 135.0, 138.0, 140.0, 142.0])

    # Supply the rotation features needed by the research helper.
    for frame in (left, right):
        for column in (
            "return_1", "return_2", "return_3", "return_5", "return_10",
            "return_40", "return_60", "return_120", "vol_5", "vol_10",
            "vol_20", "vol_40", "vol_60", "vol_ratio_5_20",
            "vol_ratio_10_40", "vol_ratio_20_60", "ema_distance_5",
            "ema_distance_10", "ema_distance_20", "ema_distance_50",
            "ema_distance_100", "ema_5_vs_20", "ema_20_vs_50",
            "ema_50_vs_100", "ema_slope_20_5", "ema_slope_50_10",
            "ema_slope_100_20", "rsi_14", "distance_from_low_20",
            "distance_from_high_50", "distance_from_low_50",
            "distance_from_high_100", "distance_from_low_100",
            "distance_from_high_200", "distance_from_low_200",
            "channel_position_20", "channel_position_50",
            "channel_position_100", "channel_position_200",
            "trend_efficiency_10", "trend_efficiency_20",
            "trend_efficiency_40", "trend_efficiency_60",
            "momentum_acceleration_5_20", "momentum_acceleration_20_60",
            "range_expansion_5_20", "volume_zscore_20",
            "volume_zscore_60", "volume_ratio_5_20",
        ):
            if column not in frame:
                frame[column] = 0.1

    left_features = adicionar_top_turn_features(left)
    right_features = adicionar_top_turn_features(right)
    feature_columns = [
        column
        for column in left_features.columns
        if column.startswith("dc_") or column == "top_turn_eligible"
    ]

    pd.testing.assert_frame_equal(
        left_features.loc[
            left_features.index[: len(shared)],
            feature_columns,
        ],
        right_features.loc[
            right_features.index[: len(shared)],
            feature_columns,
        ],
    )


def test_top_turn_overlay_requires_two_confirmations() -> None:
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


def test_analysis_package_contains_only_result_directory(tmp_path: Path) -> None:
    output = tmp_path / "v1_3_0_dev_2"
    output.mkdir()
    (output / "comparison_top_turn.json").write_text(
        '{"ok": true}',
        encoding="utf-8",
    )
    (output / "top_turn_trades.csv").write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    archive = criar_pacote_analise(
        output,
        versao="1.3.0-dev.2",
    )

    assert archive.exists()
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(zipped.namelist())
    assert names == [
        "v1_3_0_dev_2/comparison_top_turn.json",
        "v1_3_0_dev_2/top_turn_trades.csv",
    ]
