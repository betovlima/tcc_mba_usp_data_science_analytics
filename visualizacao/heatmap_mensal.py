from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402


MONTH_LABELS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

MODE_COLUMNS = {
    "simulation": "simulation_return",
    "reference": "reference_return",
    "excess": "excess_return",
}

VARIANT_LABELS = {
    "control": "Control",
    "soft": "Soft Horizon Consensus",
}


def monthly_returns_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reproduz a regra mensal usada pelo MCT.

    Para cada mes, usa o ultimo valor de capital observado e calcula:
    capital_fim_mes / capital_fim_mes_anterior - 1.

    O primeiro mes e descartado porque nao existe um mes anterior para
    comparacao.
    """
    required = {
        "timestamp",
        "strategy_equity",
        "buy_hold_equity",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            "Predictions sem colunas obrigatorias para retorno mensal: "
            + ", ".join(missing)
        )

    frame = predictions.loc[
        :,
        ["timestamp", "strategy_equity", "buy_hold_equity"],
    ].copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["strategy_equity"] = pd.to_numeric(
        frame["strategy_equity"],
        errors="coerce",
    )
    frame["buy_hold_equity"] = pd.to_numeric(
        frame["buy_hold_equity"],
        errors="coerce",
    )
    frame = frame.dropna(
        subset=["timestamp", "strategy_equity", "buy_hold_equity"]
    ).sort_values("timestamp")

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "simulation_return",
                "reference_return",
                "excess_return",
            ]
        )

    frame["month"] = (
        frame["timestamp"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
    )

    month_ends = (
        frame.groupby("month", sort=True)
        .tail(1)
        .sort_values("month")
        .reset_index(drop=True)
    )

    result = month_ends.loc[:, ["month"]].copy()
    result["simulation_return"] = month_ends["strategy_equity"].pct_change(
        fill_method=None
    )
    result["reference_return"] = month_ends["buy_hold_equity"].pct_change(
        fill_method=None
    )
    result["excess_return"] = (
        result["simulation_return"] - result["reference_return"]
    )

    if len(result) > 1:
        return result.iloc[1:].reset_index(drop=True)
    return result.iloc[0:0].reset_index(drop=True)


def load_monthly_returns(predictions_csv: str | Path) -> pd.DataFrame:
    path = Path(predictions_csv)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de predictions nao encontrado: {path}")
    return monthly_returns_from_predictions(pd.read_csv(path))


def _heatmap_values(
    monthly_returns: pd.DataFrame,
    *,
    mode: str,
) -> tuple[list[int], np.ndarray]:
    if mode not in MODE_COLUMNS:
        raise ValueError(
            f"Modo invalido: {mode}. Use {', '.join(MODE_COLUMNS)}."
        )

    if monthly_returns.empty:
        return [], np.empty((0, 12), dtype=float)

    rows = monthly_returns.copy()
    month_parts = rows["month"].astype(str).str.extract(
        r"^(?P<year>\d{4})-(?P<month>\d{2})$"
    )
    rows["year"] = pd.to_numeric(month_parts["year"], errors="coerce")
    rows["month_number"] = pd.to_numeric(
        month_parts["month"],
        errors="coerce",
    )
    rows[MODE_COLUMNS[mode]] = pd.to_numeric(
        rows[MODE_COLUMNS[mode]],
        errors="coerce",
    )
    rows = rows.dropna(
        subset=["year", "month_number", MODE_COLUMNS[mode]]
    )

    years = sorted(rows["year"].astype(int).unique().tolist())
    matrix = np.full((len(years), 12), np.nan, dtype=float)
    year_index = {year: index for index, year in enumerate(years)}

    for row in rows.itertuples(index=False):
        year = int(row.year)
        month_number = int(row.month_number)
        if 1 <= month_number <= 12:
            matrix[
                year_index[year],
                month_number - 1,
            ] = float(getattr(row, MODE_COLUMNS[mode]))

    return years, matrix


def _title_for(variant: str, mode: str) -> tuple[str, str]:
    variant_label = VARIANT_LABELS.get(variant, variant)
    if mode == "reference":
        return (
            "Retorno mensal — Comprar e manter",
            "Benchmark equal-weight buy-and-hold.",
        )
    if mode == "excess":
        return (
            f"Excesso mensal — {variant_label} vs comprar e manter",
            "Retorno mensal da estrategia menos o retorno mensal do benchmark.",
        )
    return (
        f"Retorno mensal — {variant_label}",
        "Retorno do capital da estrategia entre fechamentos mensais consecutivos.",
    )


def _cell_style(
    value: float,
    *,
    max_abs: float,
) -> tuple[tuple[float, float, float, float], str]:
    if not np.isfinite(value):
        return (0.96, 0.96, 0.96, 1.0), "#6f6f6f"

    ratio = min(1.0, abs(float(value)) / max_abs) if max_abs > 0 else 0.0
    intensity = 0.20 + ratio * 0.65

    if value > 0:
        face = plt.get_cmap("Greens")(intensity)
    elif value < 0:
        face = plt.get_cmap("Reds")(intensity)
    else:
        face = plt.get_cmap("Greys")(0.20)

    text_color = "white" if ratio > 0.48 else "black"
    return face, text_color


def render_monthly_return_heatmap(
    monthly_returns: pd.DataFrame,
    output_path: str | Path,
    *,
    variant: str = "control",
    mode: str = "simulation",
) -> Path:
    years, matrix = _heatmap_values(monthly_returns, mode=mode)
    if not years:
        raise ValueError("Nao existem retornos mensais para desenhar o heatmap.")

    finite_values = matrix[np.isfinite(matrix)]
    max_abs = (
        float(np.max(np.abs(finite_values)))
        if finite_values.size
        else 0.0
    )

    width = 16.0
    height = max(4.8, 1.08 + len(years) * 0.82)
    figure, axis = plt.subplots(figsize=(width, height))

    for year_index, _year in enumerate(years):
        for month_index in range(12):
            value = matrix[year_index, month_index]
            face, text_color = _cell_style(
                value,
                max_abs=max_abs,
            )
            axis.add_patch(
                Rectangle(
                    (month_index, year_index),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=2.0,
                )
            )
            label = "—" if not np.isfinite(value) else f"{value * 100:.2f}%"
            axis.text(
                month_index + 0.5,
                year_index + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    axis.set_xlim(0, 12)
    axis.set_ylim(len(years), 0)
    axis.set_xticks(
        np.arange(12) + 0.5,
        MONTH_LABELS,
    )
    axis.set_yticks(
        np.arange(len(years)) + 0.5,
        [str(year) for year in years],
    )
    axis.xaxis.tick_top()
    axis.tick_params(length=0, labelsize=10)
    for spine in axis.spines.values():
        spine.set_visible(False)

    title, subtitle = _title_for(variant, mode)
    axis.set_title(
        title,
        loc="left",
        pad=28,
        fontsize=18,
        fontweight="bold",
    )
    axis.text(
        0,
        -0.35,
        subtitle,
        fontsize=10,
        transform=axis.transData,
    )

    legend_handles = [
        Patch(
            facecolor=plt.get_cmap("Reds")(0.72),
            label="Perda",
        ),
        Patch(
            facecolor=plt.get_cmap("Greys")(0.20),
            label="Próximo de zero",
        ),
        Patch(
            facecolor=plt.get_cmap("Greens")(0.72),
            label="Ganho",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=3,
        frameon=False,
    )

    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        target,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)
    return target


def generate_monthly_return_artifacts(
    predictions_csv: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
    modes: Iterable[str] = ("simulation",),
) -> dict[str, Path]:
    monthly = load_monthly_returns(predictions_csv)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    monthly_csv = destination / f"monthly_returns_{variant}.csv"
    monthly.to_csv(
        monthly_csv,
        index=False,
        float_format="%.17g",
    )
    artifacts["data"] = monthly_csv

    for mode in modes:
        if mode not in MODE_COLUMNS:
            raise ValueError(
                f"Modo invalido: {mode}. Use {', '.join(MODE_COLUMNS)}."
            )
        base = destination / f"monthly_return_heatmap_{variant}_{mode}"
        png_path = render_monthly_return_heatmap(
            monthly,
            base.with_suffix(".png"),
            variant=variant,
            mode=mode,
        )
        svg_path = render_monthly_return_heatmap(
            monthly,
            base.with_suffix(".svg"),
            variant=variant,
            mode=mode,
        )
        artifacts[f"{mode}_png"] = png_path
        artifacts[f"{mode}_svg"] = svg_path

    return artifacts
