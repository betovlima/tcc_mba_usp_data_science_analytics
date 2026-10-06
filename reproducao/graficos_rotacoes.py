"""Visualizacoes limpas e auditaveis das rotacoes do baseline de pesquisa."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib

if (
    os.name != "nt"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .graficos import construir_rotacoes


BASE_COLOR = "#2F6B9A"
HIGHLIGHT_COLOR = "#D97706"
POSITIVE_COLOR = "#2E7D32"
NEGATIVE_COLOR = "#B23A48"
NEUTRAL_COLOR = "#A7B0B7"
GRID_COLOR = "#D8DEE4"


def _visual_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8C0C8",
            "axes.labelcolor": "#20262E",
            "xtick.color": "#343B43",
            "ytick.color": "#343B43",
            "text.color": "#20262E",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_pair(
    fig: Any,
    output_dir: Path,
    stem: str,
    *,
    show: bool,
) -> dict[str, Path]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )
    if show:
        fig.canvas.draw_idle()
        plt.show(block=False)
    else:
        plt.close(fig)
    return {
        f"{stem}_png": png,
        f"{stem}_svg": svg,
    }


def _prediction_frame(result: Any) -> pd.DataFrame:
    frame = result.predictions.reset_index().copy()
    if "timestamp" not in frame.columns:
        raise ValueError("Predictions sem coluna timestamp.")
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )
    frame = frame.sort_values("timestamp", ignore_index=True)
    frame["selected_asset"] = (
        frame["selected_asset"]
        .fillna("CASH")
        .astype(str)
        .str.upper()
    )
    return frame


def _asset_summary(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    session_counts = (
        predictions["selected_asset"]
        .value_counts()
        .rename("selected_sessions")
    )

    rows = trades.copy()
    if rows.empty:
        rows = pd.DataFrame(
            columns=[
                "asset",
                "action",
                "realized_pnl",
                "holding_bars",
            ]
        )

    for column, default in (
        ("asset", "CASH"),
        ("action", ""),
    ):
        if column not in rows.columns:
            rows[column] = default

    rows["asset"] = (
        rows["asset"].fillna("CASH").astype(str).str.upper()
    )
    rows["action"] = (
        rows["action"].fillna("").astype(str).str.upper()
    )
    rows["realized_pnl"] = pd.to_numeric(
        rows.get("realized_pnl"),
        errors="coerce",
    )
    rows["holding_bars"] = pd.to_numeric(
        rows.get("holding_bars"),
        errors="coerce",
    )

    entries = (
        rows.loc[rows["action"] == "BUY"]
        .groupby("asset")
        .size()
        .rename("entries")
    )
    exits_frame = rows.loc[
        rows["action"].isin(["SELL", "FINAL_SELL"])
    ].copy()
    exits = (
        exits_frame.groupby("asset")
        .size()
        .rename("exits")
    )
    pnl = (
        exits_frame.groupby("asset")["realized_pnl"]
        .sum(min_count=1)
        .rename("realized_pnl")
    )
    avg_holding = (
        exits_frame.groupby("asset")["holding_bars"]
        .mean()
        .rename("average_holding_sessions")
    )
    win_rate = (
        exits_frame.assign(
            win=exits_frame["realized_pnl"] > 0.0
        )
        .groupby("asset")["win"]
        .mean()
        .rename("exit_win_rate")
    )

    assets = sorted(
        set(session_counts.index)
        | set(entries.index)
        | set(exits.index)
    )
    summary = pd.DataFrame({"asset": assets}).set_index("asset")
    summary = summary.join(
        [
            session_counts,
            entries,
            exits,
            pnl,
            avg_holding,
            win_rate,
        ],
        how="left",
    )
    summary["selected_sessions"] = (
        summary["selected_sessions"].fillna(0).astype(int)
    )
    summary["entries"] = summary["entries"].fillna(0).astype(int)
    summary["exits"] = summary["exits"].fillna(0).astype(int)
    summary["realized_pnl"] = (
        summary["realized_pnl"].fillna(0.0)
    )
    summary["session_share"] = (
        summary["selected_sessions"] / max(1, len(predictions))
    )
    return (
        summary.reset_index()
        .sort_values(
            ["selected_sessions", "asset"],
            ascending=[False, True],
            ignore_index=True,
        )
    )


def _ordered_assets_by_first_use(
    predictions: pd.DataFrame,
) -> list[str]:
    used = predictions.loc[
        predictions["selected_asset"] != "CASH",
        ["timestamp", "selected_asset"],
    ]
    if used.empty:
        return []
    first = (
        used.groupby("selected_asset")["timestamp"]
        .min()
        .sort_values()
    )
    return first.index.astype(str).tolist()


def _monthly_occupancy(
    predictions: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    count = pd.crosstab(
        frame["selected_asset"],
        frame["month"],
    )
    monthly_sessions = (
        frame.groupby("month").size().reindex(count.columns)
    )
    share = count.div(monthly_sessions, axis=1)
    return share.reindex(index=assets, fill_value=0.0)


def _selected_with_highlights(
    summary: pd.DataFrame,
    highlights: set[str],
    *,
    top_n: int,
) -> pd.DataFrame:
    non_cash = summary.loc[
        summary["asset"].astype(str) != "CASH"
    ].copy()
    top = non_cash.nlargest(top_n, "selected_sessions")
    highlighted = non_cash.loc[
        non_cash["asset"].astype(str).isin(highlights)
    ]
    return (
        pd.concat([top, highlighted], ignore_index=True)
        .drop_duplicates("asset")
        .sort_values(
            ["selected_sessions", "asset"],
            ascending=[True, True],
            ignore_index=True,
        )
    )


def _currency_label(value: float) -> str:
    absolute = abs(float(value))
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000:
        return f"{sign}US$ {absolute / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{sign}US$ {absolute / 1_000:.0f}k"
    return f"{sign}US$ {absolute:.0f}"


def gerar_graficos_rotacoes(
    output_dir: Path,
    *,
    result: Any,
    universe_label: str,
    highlight_assets: Iterable[str] = (),
    show: bool = False,
) -> dict[str, Path]:
    """Gera cinco visoes de rotacao com foco em leitura, nao em densidade.

    O CSV completo e sempre preservado. Os graficos resumem os dados para
    evitar paineis poluidos com dezenas de rotulos ilegíveis.
    """
    _visual_defaults()

    output_dir = Path(output_dir)
    visual_dir = output_dir / "graficos_rotacoes"
    if visual_dir.exists():
        shutil.rmtree(visual_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)

    highlights = {
        str(asset).upper().strip()
        for asset in highlight_assets
    }
    predictions = _prediction_frame(result)
    trades = result.trades.copy()
    rotacoes = construir_rotacoes(trades)
    summary = _asset_summary(predictions, trades)
    missing_highlights = sorted(
        highlights.difference(set(summary["asset"].astype(str)))
    )
    if missing_highlights:
        zero_rows = pd.DataFrame(
            {
                "asset": missing_highlights,
                "selected_sessions": 0,
                "entries": 0,
                "exits": 0,
                "realized_pnl": 0.0,
                "average_holding_sessions": np.nan,
                "exit_win_rate": np.nan,
                "session_share": 0.0,
            }
        )
        summary = pd.concat(
            [summary, zero_rows],
            ignore_index=True,
        )
    paths: dict[str, Path] = {}

    # Dados-base completos.
    rotations_csv = visual_dir / "rotacoes.csv"
    profile_csv = visual_dir / "perfil_ativos.csv"
    rotacoes.to_csv(rotations_csv, index=False)
    summary.to_csv(profile_csv, index=False)
    paths["rotations_csv"] = rotations_csv
    paths["asset_summary_csv"] = profile_csv

    # ------------------------------------------------------------------
    # 1. Mapa temporal de ocupacao por ativo.
    # ------------------------------------------------------------------
    ordered_assets = _ordered_assets_by_first_use(predictions)
    selected_counts = (
        predictions["selected_asset"]
        .value_counts()
        .drop(labels=["CASH"], errors="ignore")
    )
    visible_assets = selected_counts.head(30).index.astype(str).tolist()
    for asset in sorted(highlights):
        if asset not in visible_assets:
            visible_assets.append(asset)
    visible_assets = [
        asset
        for asset in ordered_assets
        if asset in set(visible_assets)
    ] + [
        asset
        for asset in sorted(highlights)
        if asset not in set(ordered_assets)
    ]
    if "CASH" in set(predictions["selected_asset"]):
        visible_assets.append("CASH")

    occupancy = _monthly_occupancy(
        predictions,
        visible_assets,
    )
    occupancy_csv = visual_dir / "ocupacao_mensal.csv"
    occupancy.to_csv(
        occupancy_csv,
        index=True,
        index_label="asset",
    )
    paths["monthly_occupancy_csv"] = occupancy_csv

    if not occupancy.empty:
        fig_height = max(6.2, 0.27 * len(occupancy.index) + 2.2)
        fig, ax = plt.subplots(
            figsize=(15.5, fig_height),
            constrained_layout=True,
        )
        image = ax.imshow(
            100.0 * occupancy.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
            vmin=0.0,
            vmax=max(
                25.0,
                float(
                    np.nanpercentile(
                        100.0 * occupancy.to_numpy(dtype=float),
                        98,
                    )
                ),
            ),
        )

        months = occupancy.columns.astype(str).tolist()
        ticks = list(range(0, len(months), 6))
        ax.set_xticks(
            ticks,
            [months[idx] for idx in ticks],
            rotation=45,
            ha="right",
        )
        ylabels = [
            f"{asset}  ★" if asset in highlights else asset
            for asset in occupancy.index.astype(str)
        ]
        ax.set_yticks(
            range(len(ylabels)),
            ylabels,
            fontsize=9,
        )
        ax.set_xlabel("Mês")
        ax.set_ylabel("Ativo selecionado")
        ax.set_title(
            f"{universe_label}\nMapa temporal de ocupação da carteira",
            loc="left",
            pad=18,
        )
        fig.text(
            0.08,
            0.012,
            "Intensidade = parcela das sessões do mês em que o ativo ficou em carteira. ★ = impulsionador exploratório.",
            fontsize=9,
            color="#5A626B",
        )
        cbar = fig.colorbar(image, ax=ax, pad=0.015)
        cbar.set_label("Ocupação no mês (%)")
        paths.update(
            _save_pair(
                fig,
                visual_dir,
                "01_mapa_temporal_ocupacao",
                show=show,
            )
        )

    # ------------------------------------------------------------------
    # 2. Principais transicoes em vez de uma matriz gigante e ilegivel.
    # ------------------------------------------------------------------
    if not rotacoes.empty:
        transition_pairs = (
            rotacoes.assign(
                par=(
                    rotacoes["from_asset"].astype(str)
                    + " → "
                    + rotacoes["to_asset"].astype(str)
                )
            )
            .groupby("par", as_index=False)
            .agg(
                rotations=("sequence", "count"),
                realized_pnl=("realized_pnl", "sum"),
            )
            .sort_values(
                ["rotations", "par"],
                ascending=[False, True],
                ignore_index=True,
            )
        )
    else:
        transition_pairs = pd.DataFrame(
            columns=["par", "rotations", "realized_pnl"]
        )

    transition_csv = visual_dir / "transicoes_completas.csv"
    transition_pairs.to_csv(transition_csv, index=False)
    paths["transition_pairs_csv"] = transition_csv

    top_transitions = transition_pairs.head(20).sort_values(
        ["rotations", "par"],
        ascending=[True, True],
        ignore_index=True,
    )
    if not top_transitions.empty:
        fig, ax = plt.subplots(
            figsize=(11.5, 7.6),
            constrained_layout=True,
        )
        y = np.arange(len(top_transitions))
        ax.barh(
            y,
            top_transitions["rotations"].to_numpy(dtype=float),
            color=BASE_COLOR,
            alpha=0.9,
        )
        ax.set_yticks(
            y,
            top_transitions["par"].astype(str).tolist(),
        )
        ax.set_xlabel("Número de rotações")
        ax.set_title(
            f"{universe_label}\n20 transições mais frequentes",
            loc="left",
        )
        ax.grid(
            axis="x",
            color=GRID_COLOR,
            linewidth=0.8,
            alpha=0.8,
        )
        ax.set_axisbelow(True)
        for index, value in enumerate(
            top_transitions["rotations"].to_numpy(dtype=int)
        ):
            ax.text(
                value + 0.15,
                index,
                str(value),
                va="center",
                fontsize=9,
            )
        paths.update(
            _save_pair(
                fig,
                visual_dir,
                "02_principais_transicoes",
                show=show,
            )
        )

    # ------------------------------------------------------------------
    # 3. Presenca: top ativos + todos os oito destacados.
    # ------------------------------------------------------------------
    presence = _selected_with_highlights(
        summary,
        highlights,
        top_n=18,
    )
    if not presence.empty:
        fig_height = max(6.4, 0.34 * len(presence) + 2.2)
        fig, ax = plt.subplots(
            figsize=(11.5, fig_height),
            constrained_layout=True,
        )
        y = np.arange(len(presence))
        colors = [
            HIGHLIGHT_COLOR
            if asset in highlights
            else BASE_COLOR
            for asset in presence["asset"].astype(str)
        ]
        values = presence["selected_sessions"].to_numpy(dtype=float)
        ax.barh(y, values, color=colors, alpha=0.92)
        labels = [
            f"{asset}  ★" if asset in highlights else asset
            for asset in presence["asset"].astype(str)
        ]
        ax.set_yticks(y, labels)
        ax.set_xlabel("Sessões em carteira")
        ax.set_title(
            f"{universe_label}\nAtivos com maior presença na carteira",
            loc="left",
        )
        ax.grid(
            axis="x",
            color=GRID_COLOR,
            linewidth=0.8,
            alpha=0.8,
        )
        ax.set_axisbelow(True)
        shares = 100.0 * presence["session_share"].to_numpy(dtype=float)
        for index, (value, share) in enumerate(zip(values, shares)):
            ax.text(
                value + max(values.max() * 0.008, 0.25),
                index,
                f"{int(value)}  ({share:.1f}%)",
                va="center",
                fontsize=9,
            )
        paths.update(
            _save_pair(
                fig,
                visual_dir,
                "03_presenca_por_ativo",
                show=show,
            )
        )

    # ------------------------------------------------------------------
    # 4. Resultado realizado: extremos + todos os oito destacados.
    # ------------------------------------------------------------------
    pnl_source = summary.loc[
        (summary["asset"] != "CASH")
        & (summary["exits"] > 0)
    ].copy()
    if not pnl_source.empty:
        bottom = pnl_source.nsmallest(8, "realized_pnl")
        top = pnl_source.nlargest(10, "realized_pnl")
        highlighted = summary.loc[
            summary["asset"].astype(str).isin(highlights)
        ]
        pnl_plot = (
            pd.concat([bottom, top, highlighted], ignore_index=True)
            .drop_duplicates("asset")
            .sort_values(
                ["realized_pnl", "asset"],
                ascending=[True, True],
                ignore_index=True,
            )
        )
        fig_height = max(6.4, 0.34 * len(pnl_plot) + 2.2)
        fig, ax = plt.subplots(
            figsize=(11.8, fig_height),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.11,
            right=0.97,
            top=0.90,
            bottom=0.14,
        )
        y = np.arange(len(pnl_plot))
        values = pnl_plot["realized_pnl"].to_numpy(dtype=float)
        colors = [
            POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
            for value in values
        ]
        bars = ax.barh(
            y,
            values,
            color=colors,
            alpha=0.88,
        )
        for bar, asset in zip(
            bars,
            pnl_plot["asset"].astype(str),
        ):
            if asset in highlights:
                bar.set_edgecolor(HIGHLIGHT_COLOR)
                bar.set_linewidth(2.0)

        labels = [
            f"{asset}  ★" if asset in highlights else asset
            for asset in pnl_plot["asset"].astype(str)
        ]
        ax.set_yticks(y, labels)
        ax.axvline(0.0, color="#5A626B", linewidth=1.0)
        ax.set_xlabel("PnL realizado (US$ milhões)")
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"{value / 1_000_000:.1f}"
            )
        )
        ax.set_title(
            f"{universe_label}\nPnL realizado por ativo",
            loc="left",
            pad=18,
        )
        fig.text(
            0.08,
            0.025,
            "Visão descritiva das posições fechadas; não é contribuição causal contrafactual. ★ = impulsionador exploratório.",
            fontsize=9,
            color="#5A626B",
        )
        ax.grid(
            axis="x",
            color=GRID_COLOR,
            linewidth=0.8,
            alpha=0.8,
        )
        ax.set_axisbelow(True)
        span = max(
            abs(float(np.nanmin(values))),
            abs(float(np.nanmax(values))),
            1.0,
        )
        current_left, current_right = ax.get_xlim()
        ax.set_xlim(
            min(current_left, float(np.nanmin(values)) - 0.06 * span),
            max(current_right, float(np.nanmax(values)) + 0.14 * span),
        )
        for index, value in enumerate(values):
            if value >= 0:
                x = value + 0.018 * span
                ha = "left"
            else:
                x = value + 0.018 * span
                ha = "left"
            ax.text(
                x,
                index,
                _currency_label(value),
                va="center",
                ha=ha,
                fontsize=8.5,
            )
        paths.update(
            _save_pair(
                fig,
                visual_dir,
                "04_pnl_realizado_por_ativo",
                show=show,
            )
        )

    # ------------------------------------------------------------------
    # 5. Forca das rotacoes: vantagem sobre incumbente / margem exigida.
    # ------------------------------------------------------------------
    needed = {
        "timestamp",
        "selected_asset",
        "previous_asset",
        "trade_action",
        "best_vs_current_gap",
        "best_vs_second_gap",
        "effective_switch_margin",
        "best_asset",
        "current_asset",
    }
    if needed.issubset(predictions.columns):
        strength = predictions.loc[
            predictions["trade_action"].astype(str).str.upper()
            == "ROTATE"
        ][
            [
                "timestamp",
                "selected_asset",
                "previous_asset",
                "best_asset",
                "current_asset",
                "best_vs_current_gap",
                "best_vs_second_gap",
                "effective_switch_margin",
                "walk_forward_fold",
            ]
        ].copy()

        for column in (
            "best_vs_current_gap",
            "best_vs_second_gap",
            "effective_switch_margin",
        ):
            strength[column] = pd.to_numeric(
                strength[column],
                errors="coerce",
            )

        strength["rotation_strength"] = np.where(
            strength["effective_switch_margin"] > 0,
            (
                strength["best_vs_current_gap"]
                / strength["effective_switch_margin"]
            ),
            np.nan,
        )
        strength["is_highlight"] = (
            strength["selected_asset"]
            .astype(str)
            .str.upper()
            .isin(highlights)
        )

        strength_csv = visual_dir / "forca_rotacoes.csv"
        strength.to_csv(strength_csv, index=False)
        paths["rotation_strength_csv"] = strength_csv

        plot = strength.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna(subset=["rotation_strength"])
        if not plot.empty:
            fig, ax = plt.subplots(
                figsize=(14.0, 6.8),
                constrained_layout=False,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.82,
                top=0.88,
                bottom=0.15,
            )
            normal = plot.loc[~plot["is_highlight"]]
            special = plot.loc[plot["is_highlight"]]

            ax.scatter(
                normal["timestamp"],
                normal["rotation_strength"],
                s=28,
                alpha=0.62,
                color=BASE_COLOR,
                label="Demais ativos",
            )
            if not special.empty:
                ax.scatter(
                    special["timestamp"],
                    special["rotation_strength"],
                    s=52,
                    alpha=0.9,
                    color=HIGHLIGHT_COLOR,
                    label="Impulsionadores exploratórios",
                )

            ax.axhline(
                1.0,
                color="#5A626B",
                linewidth=1.1,
                linestyle="--",
            )
            ax.set_xlabel("Data da rotação")
            ax.set_ylabel("Força = (melhor − incumbente) / margem")
            ax.set_yscale("log")
            ax.set_ylim(
                0.9,
                max(
                    10.0,
                    float(plot["rotation_strength"].max()) * 1.25,
                ),
            )
            ax.set_title(
                f"{universe_label}\nForça relativa das rotações executadas",
                loc="left",
                pad=18,
            )
            fig.text(
                0.08,
                0.025,
                "Escala logarítmica. 1,0 = limiar mínimo de troca; valores maiores indicam maior folga. Laranja = impulsionador exploratório.",
                fontsize=9,
                color="#5A626B",
            )
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y")
            )
            ax.grid(
                axis="y",
                color=GRID_COLOR,
                linewidth=0.8,
                alpha=0.8,
            )
            ax.set_axisbelow(True)
            if not special.empty:
                ax.legend(
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                )

            label_rows = (
                plot.nlargest(7, "rotation_strength")
                .sort_values("timestamp")
            )
            for row in label_rows.itertuples(index=False):
                ax.annotate(
                    str(row.selected_asset),
                    (
                        row.timestamp,
                        row.rotation_strength,
                    ),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="#343B43",
                )

            paths.update(
                _save_pair(
                    fig,
                    visual_dir,
                    "05_forca_das_rotacoes",
                    show=show,
                )
            )

    print(
        "[rotation-graphs] "
        f"dir={visual_dir} "
        f"assets_used={int((summary['selected_sessions'] > 0).sum())} "
        f"rotations={len(rotacoes)}",
        flush=True,
    )

    paths["graficos_rotacoes_dir"] = visual_dir
    return paths


def gerar_graficos_rotacoes_u59(
    output_dir: Path,
    *,
    result: Any,
) -> dict[str, Path]:
    """Compatibilidade retroativa com a primeira revisao da branch."""
    return gerar_graficos_rotacoes(
        output_dir,
        result=result,
        universe_label="U59",
        show=False,
    )
