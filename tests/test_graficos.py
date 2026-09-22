from __future__ import annotations

import pandas as pd
import pytest

from visualizacao.heatmap_mensal import (
    generate_monthly_realized_pnl_artifacts,
    generate_monthly_return_artifacts,
    monthly_realized_pnl,
    monthly_return_sums,
    monthly_returns_from_predictions,
)
from visualizacao import VISUALIZATION_VERSION


def test_monthly_returns_follow_mct_month_end_formula() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": [
                "2020-07-30T04:00:00+00:00",
                "2020-07-31T04:00:00+00:00",
                "2020-08-31T04:00:00+00:00",
                "2020-09-30T04:00:00+00:00",
            ],
            "strategy_equity": [
                9800.0,
                10000.0,
                11000.0,
                9900.0,
            ],
            "buy_hold_equity": [
                9900.0,
                10000.0,
                10500.0,
                10710.0,
            ],
        }
    )

    monthly = monthly_returns_from_predictions(predictions)

    assert list(monthly["month"]) == ["2020-08", "2020-09"]
    assert monthly.loc[0, "simulation_return"] == pytest.approx(0.10)
    assert monthly.loc[0, "reference_return"] == pytest.approx(0.05)
    assert monthly.loc[0, "excess_return"] == pytest.approx(0.05)
    assert monthly.loc[1, "simulation_return"] == pytest.approx(-0.10)
    assert monthly.loc[1, "reference_return"] == pytest.approx(0.02)
    assert monthly.loc[1, "excess_return"] == pytest.approx(-0.12)


def test_monthly_return_heatmap_exports_png_svg_and_csv(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-31T04:00:00+00:00",
                "2024-02-29T04:00:00+00:00",
                "2024-03-28T04:00:00+00:00",
            ],
            "strategy_equity": [10000.0, 11000.0, 10450.0],
            "buy_hold_equity": [10000.0, 10200.0, 10302.0],
        }
    )
    source = tmp_path / "control_predictions.csv"
    predictions.to_csv(source, index=False)

    artifacts = generate_monthly_return_artifacts(
        source,
        tmp_path / "graficos",
        variant="control",
        modes=("simulation",),
    )

    assert VISUALIZATION_VERSION == "1.2.0-dev.3"
    assert artifacts["data"].exists()
    assert artifacts["simulation_png"].exists()
    assert artifacts["simulation_svg"].exists()
    assert artifacts["simulation_png"].stat().st_size > 0
    assert artifacts["simulation_svg"].stat().st_size > 0


def test_monthly_realized_pnl_keeps_zero_months_from_oos_calendar() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-31T04:00:00+00:00",
                "2024-02-29T04:00:00+00:00",
                "2024-03-28T04:00:00+00:00",
            ],
            "strategy_equity": [10000.0, 11000.0, 10450.0],
            "buy_hold_equity": [10000.0, 10200.0, 10302.0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-15T04:00:00+00:00",
                "2024-03-20T04:00:00+00:00",
            ],
            "realized_pnl": [250.0, -50.0],
        }
    )

    monthly = monthly_realized_pnl(predictions, trades)

    assert list(monthly["month"]) == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]
    assert list(monthly["realized_pnl"]) == [
        250.0,
        0.0,
        -50.0,
    ]


def test_realized_pnl_heatmap_exports_png_svg_and_csv(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-31T04:00:00+00:00",
                "2024-02-29T04:00:00+00:00",
            ],
            "strategy_equity": [10000.0, 11000.0],
            "buy_hold_equity": [10000.0, 10200.0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-15T04:00:00+00:00",
                "2024-02-20T04:00:00+00:00",
            ],
            "realized_pnl": [250.0, -75.0],
        }
    )
    predictions_path = tmp_path / "control_predictions.csv"
    trades_path = tmp_path / "control_trades.csv"
    predictions.to_csv(predictions_path, index=False)
    trades.to_csv(trades_path, index=False)

    artifacts = generate_monthly_realized_pnl_artifacts(
        predictions_path,
        trades_path,
        tmp_path / "graficos",
        variant="control",
    )

    assert artifacts["data"].exists()
    assert artifacts["pnl_png"].exists()
    assert artifacts["pnl_svg"].exists()
    assert artifacts["pnl_png"].stat().st_size > 0
    assert artifacts["pnl_svg"].stat().st_size > 0


def test_monthly_return_sums_are_arithmetic_not_compounded() -> None:
    monthly = pd.DataFrame(
        {
            "month": [
                "2024-01",
                "2024-02",
                "2025-01",
                "2025-02",
            ],
            "simulation_return": [
                0.10,
                0.05,
                -0.02,
                0.03,
            ],
            "reference_return": [
                0.01,
                0.02,
                0.03,
                0.04,
            ],
            "excess_return": [
                0.09,
                0.03,
                -0.05,
                -0.01,
            ],
        }
    )

    years, matrix, year_sums, month_sums, grand_sum = (
        monthly_return_sums(monthly, mode="simulation")
    )

    assert years == [2024, 2025]
    assert matrix[0, 0] == pytest.approx(0.10)
    assert matrix[0, 1] == pytest.approx(0.05)
    assert year_sums.tolist() == pytest.approx([0.15, 0.01])
    assert month_sums[0] == pytest.approx(0.08)
    assert month_sums[1] == pytest.approx(0.08)
    assert grand_sum == pytest.approx(0.16)
    assert year_sums[0] != pytest.approx((1.10 * 1.05) - 1.0)
