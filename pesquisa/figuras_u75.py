"""Visualizacoes auditaveis do U75; PNGs locais sem acesso a rede."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grafico_forca_universo(daily: pd.DataFrame, output: Path) -> None:
    d = daily.sort_values("decision_date")
    x = pd.to_datetime(d["decision_date"], utc=True)
    with plt.rc_context({"figure.figsize": (15, 5), "font.size": 9}):
        fig, ax = plt.subplots()
        ax.plot(x, d["positive_share_5"] * 100, label="Ativos em alta: 5 pregões", lw=1)
        ax.plot(x, d["positive_share_20"] * 100, label="Ativos em alta: 20 pregões", lw=1)
        stress = d["universe_stress"].fillna(False).to_numpy(bool)
        ax.fill_between(x, 0, 100, where=stress, alpha=.13, label="Estresse coletivo")
        ax.set_ylim(0, 100)
        ax.set_ylabel("% de ativos com retorno positivo")
        ax.set_xlabel("Data da decisão (fechamento)")
        ax.set_title("Força interna do universo de 75 ativos")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncols=3)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def grafico_capital_e_estresse(joined: pd.DataFrame, output: Path) -> None:
    d = joined.sort_values("timestamp")
    x = pd.to_datetime(d["timestamp"], utc=True)
    equity = pd.to_numeric(d["strategy_equity"]).to_numpy(float)
    with plt.rc_context({"figure.figsize": (15, 5), "font.size": 9}):
        fig, ax = plt.subplots()
        ax.plot(x, equity, lw=1.1, label="Patrimônio do Control")
        ax.set_yscale("log")
        ax.set_ylabel("Patrimônio em US$ (escala logarítmica)")
        ax.set_xlabel("Pregão da posição efetiva")
        stress = d["universe_stress"].fillna(False).to_numpy(bool)
        ax.fill_between(x, np.min(equity), np.max(equity),
                        where=stress, alpha=.13, label="Estresse no fechamento anterior")
        ax.set_title("Patrimônio e momentos de fraqueza coletiva do U75")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def grafico_mapa_mensal(months: pd.DataFrame, output: Path) -> None:
    years = sorted(months["year"].unique().tolist())
    matrix = np.full((len(years), 12), np.nan)
    text = {}
    for r in months.to_dict("records"):
        iy = years.index(r["year"])
        im = int(r["month_number"]) - 1
        matrix[iy, im] = float(r["dominant_share"])
        text[(iy, im)] = f"{r['dominant_asset']}\n{r['dominant_sessions']} d"
    with plt.rc_context({"figure.figsize": (15, max(3, len(years) * .65 + 2)),
                         "font.size": 8}):
        fig, ax = plt.subplots()
        image = ax.imshow(np.ma.masked_invalid(matrix),
                          cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(12), ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"])
        ax.set_yticks(range(len(years)), [str(y) for y in years])
        for (iy, im), value in text.items():
            ax.text(im, iy, value, ha="center", va="center",
                    color="white" if matrix[iy, im] > .68 else "black",
                    fontsize=7, fontweight="bold")
        ax.set_title("Mapa mensal: ativo efetivamente mantido por mais pregões")
        fig.colorbar(image, ax=ax, label="Fração das sessões no ativo dominante",
                     shrink=.85)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
