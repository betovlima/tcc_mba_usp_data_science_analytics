"""Visualizacoes exploratorias das rotacoes do universo de referencia.

Este modulo nao altera a politica, o treino ou a simulacao. Ele consome apenas
os artefatos ja produzidos pelo replay e gera visoes por ativo para entender
quando, entre quais ativos e com qual diferenca de score as rotacoes ocorreram.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

import matplotlib

if (
    os.name != "nt"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graficos import construir_rotacoes


def _save_pair(
    fig: Any,
    output_dir: Path,
    stem: str,
    *,
    show: bool,
) -> dict[str, Path]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
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


def _timeline_blocks(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=["asset", "start", "end", "sessions"]
        )

    rows: list[dict[str, Any]] = []
    start_index = 0
    assets = predictions["selected_asset"].tolist()

    for idx in range(1, len(predictions) + 1):
        boundary = (
            idx == len(predictions)
            or assets[idx] != assets[start_index]
        )
        if not boundary:
            continue

        start = pd.Timestamp(
            predictions.iloc[start_index]["timestamp"]
        )
        if idx < len(predictions):
            end = pd.Timestamp(
                predictions.iloc[idx]["timestamp"]
            )
        else:
            end = pd.Timestamp(
                predictions.iloc[idx - 1]["timestamp"]
            ) + pd.Timedelta(days=1)

        rows.append(
            {
                "asset": assets[start_index],
                "start": start,
                "end": end,
                "sessions": int(idx - start_index),
            }
        )
        start_index = idx

    return pd.DataFrame(rows)


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
    total_sessions = max(1, len(predictions))
    summary["session_share"] = (
        summary["selected_sessions"] / total_sessions
    )
    return (
        summary.reset_index()
        .sort_values(
            ["selected_sessions", "asset"],
            ascending=[False, True],
            ignore_index=True,
        )
    )


def gerar_graficos_rotacoes(
    output_dir: Path,
    *,
    result: Any,
    universe_label: str,
    show: bool = False,
) -> dict[str, Path]:
    """Gera as cinco visoes principais das rotacoes do universo informado.

    Saidas:
    1. timeline de permanencia por ativo;
    2. matriz origem -> destino;
    3. presenca/sessoes por ativo;
    4. PnL realizado por ativo;
    5. distancia de score nas rotacoes.

    Os CSVs subjacentes sao gravados junto aos graficos para auditoria.
    """
    output_dir = Path(output_dir)
    visual_dir = output_dir / "graficos_rotacoes"
    if visual_dir.exists():
        shutil.rmtree(visual_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)

    predictions = _prediction_frame(result)
    trades = result.trades.copy()
    rotacoes = construir_rotacoes(trades)
    timeline = _timeline_blocks(predictions)
    summary = _asset_summary(predictions, trades)

    paths: dict[str, Path] = {}

    # Dados-base
    timeline_csv = visual_dir / "timeline_blocos.csv"
    timeline.to_csv(timeline_csv, index=False)
    paths["timeline_csv"] = timeline_csv

    summary_csv = visual_dir / "perfil_ativos.csv"
    summary.to_csv(summary_csv, index=False)
    paths["asset_summary_csv"] = summary_csv

    rotations_csv = visual_dir / "rotacoes.csv"
    rotacoes.to_csv(rotations_csv, index=False)
    paths["rotations_csv"] = rotations_csv

    # 1. Timeline / Gantt
    timeline_assets = [
        asset
        for asset in summary.loc[
            summary["selected_sessions"] > 0,
            "asset",
        ].astype(str)
        if asset != "CASH"
    ]
    if "CASH" in set(timeline["asset"].astype(str)):
        timeline_assets.append("CASH")

    order = {
        asset: idx
        for idx, asset in enumerate(reversed(timeline_assets))
    }
    fig_height = max(6.0, 0.30 * max(1, len(timeline_assets)) + 2.0)
    fig, ax = plt.subplots(figsize=(15.0, fig_height))
    cmap = plt.get_cmap("tab20")
    color_map = {
        asset: cmap(index % 20)
        for index, asset in enumerate(timeline_assets)
    }
    for row in timeline.to_dict(orient="records"):
        asset = str(row["asset"])
        if asset not in order:
            continue
        start = pd.Timestamp(row["start"]).tz_convert(None)
        end = pd.Timestamp(row["end"]).tz_convert(None)
        start_num = mdates.date2num(start.to_pydatetime())
        end_num = mdates.date2num(end.to_pydatetime())
        ax.broken_barh(
            [(start_num, max(end_num - start_num, 0.25))],
            (order[asset] - 0.38, 0.76),
            facecolors=color_map[asset],
        )
    ax.set_yticks(
        [order[a] for a in timeline_assets],
        timeline_assets,
    )
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Data")
    ax.set_ylabel("Ativo em carteira")
    ax.set_title(f"{universe_label} · Linha do tempo das rotacoes")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    paths.update(
        _save_pair(fig, visual_dir, "timeline_rotacoes", show=show)
    )

    # 2. Matriz de transicoes
    transitions = (
        rotacoes.groupby(
            ["from_asset", "to_asset"],
            dropna=False,
        )
        .size()
        .reset_index(name="rotations")
    )
    transition_csv = visual_dir / "matriz_transicoes.csv"
    transitions.to_csv(transition_csv, index=False)
    paths["transition_matrix_csv"] = transition_csv

    if not transitions.empty:
        involved = sorted(
            set(transitions["from_asset"].astype(str))
            | set(transitions["to_asset"].astype(str))
        )
        pivot = (
            transitions.pivot_table(
                index="from_asset",
                columns="to_asset",
                values="rotations",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=involved, columns=involved, fill_value=0)
        )
        n = len(involved)
        size = min(20.0, max(9.0, 0.34 * n + 4.0))
        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            cmap="Blues",
        )
        ax.set_xticks(
            range(n),
            involved,
            rotation=90,
            fontsize=7,
        )
        ax.set_yticks(
            range(n),
            involved,
            fontsize=7,
        )
        ax.set_xlabel("Ativo de destino")
        ax.set_ylabel("Ativo de origem")
        ax.set_title(f"{universe_label} · Matriz de transicao entre ativos")
        fig.colorbar(image, ax=ax, label="Numero de rotacoes")
        fig.tight_layout()
        paths.update(
            _save_pair(
                fig,
                visual_dir,
                "matriz_transicoes",
                show=show,
            )
        )

    # 3. Presenca por ativo
    presence = summary.loc[
        summary["selected_sessions"] > 0
    ].sort_values(
        ["selected_sessions", "asset"],
        ascending=[True, True],
    )
    fig_height = max(6.0, 0.30 * max(1, len(presence)) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    y = np.arange(len(presence))
    ax.barh(
        y,
        presence["selected_sessions"].to_numpy(dtype=float),
    )
    ax.set_yticks(y, presence["asset"].astype(str).tolist())
    ax.set_xlabel("Sessoes selecionado")
    ax.set_ylabel("Ativo")
    ax.set_title(f"{universe_label} · Presenca de cada ativo na carteira")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths.update(
        _save_pair(fig, visual_dir, "presenca_por_ativo", show=show)
    )

    # 4. PnL realizado por ativo
    pnl = summary.loc[
        summary["exits"] > 0
    ].sort_values(
        ["realized_pnl", "asset"],
        ascending=[True, True],
    )
    fig_height = max(6.0, 0.30 * max(1, len(pnl)) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    y = np.arange(len(pnl))
    ax.barh(
        y,
        pnl["realized_pnl"].to_numpy(dtype=float),
    )
    ax.set_yticks(y, pnl["asset"].astype(str).tolist())
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("PnL realizado (US$)")
    ax.set_ylabel("Ativo")
    ax.set_title(
        f"{universe_label} · PnL realizado por ativo\n"
        "(descritivo; nao equivale a contribuicao causal contrafactual)"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths.update(
        _save_pair(
            fig,
            visual_dir,
            "pnl_realizado_por_ativo",
            show=show,
        )
    )

    # 5. Distancia ao topo nas decisoes que efetivamente rotacionaram
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
        distance = predictions.loc[
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
            distance[column] = pd.to_numeric(
                distance[column],
                errors="coerce",
            )

        distance_csv = (
            visual_dir / "distancia_topo_rotacoes.csv"
        )
        distance.to_csv(distance_csv, index=False)
        paths["score_gap_csv"] = distance_csv

        if not distance.empty:
            fig, ax = plt.subplots(figsize=(14.0, 6.5))
            ax.scatter(
                distance["timestamp"],
                distance["best_vs_current_gap"],
                label="Melhor - incumbente",
                s=22,
                alpha=0.75,
            )
            ax.scatter(
                distance["timestamp"],
                distance["best_vs_second_gap"],
                label="Melhor - segundo",
                s=18,
                alpha=0.55,
            )
            ax.plot(
                distance["timestamp"],
                distance["effective_switch_margin"],
                label="Margem exigida",
                linewidth=1.2,
            )
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xlabel("Data de execucao")
            ax.set_ylabel("Diferenca de utilidade prevista")
            ax.set_title(
                f"{universe_label} · Distancia ao topo nas rotacoes efetivas"
            )
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            paths.update(
                _save_pair(
                    fig,
                    visual_dir,
                    "distancia_topo_rotacoes",
                    show=show,
                )
            )

    print(
        "[rotation-graphs] "
        f"dir={visual_dir} "
        f"timeline_blocks={len(timeline)} "
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
    """Compatibilidade com o nome usado na primeira revisao da branch.

    Mantem runners locais momentaneamente defasados funcionando, delegando
    integralmente para a API generalizada. Nao altera calculos ou graficos.
    """
    return gerar_graficos_rotacoes(
        output_dir,
        result=result,
        universe_label="U59",
        show=False,
    )
