from __future__ import annotations

import pandas as pd
import pytest

from visualizacao.heatmap_mensal import (
    generate_monthly_return_artifacts,
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

    assert VISUALIZATION_VERSION == "1.2.0-dev.1"
    assert artifacts["data"].exists()
    assert artifacts["simulation_png"].exists()
    assert artifacts["simulation_svg"].exists()
    assert artifacts["simulation_png"].stat().st_size > 0
    assert artifacts["simulation_svg"].stat().st_size > 0
