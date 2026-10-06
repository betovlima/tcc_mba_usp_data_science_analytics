"""Clusterizacao exploratoria dos candidatos sem novo replay financeiro.

Objetivo
--------
Separar candidatos ativos por padrao de comportamento de score sem usar
resultado financeiro no ajuste dos clusters. Os efeitos financeiros conhecidos
dos Smart20 sao usados somente depois do ajuste, como camada de interpretacao.

Entrada principal
-----------------
output/busca_ativos/intelligent_candidates_ranked.csv

Esse arquivo ja e produzido por buscar_ativos.py. Portanto este runner NAO:
- baixa dados;
- treina LightGBM;
- executa backtest;
- recalcula U67.

Saidas
------
output/clusterizacao_ativos/
- cluster_quality.csv
- cluster_members.csv
- cluster_summary.csv
- inactive_candidates.csv
- known_outcomes_overlay.csv
- cluster_analysis.json
- 01_mapa_clusters_pca.{png,svg}
- 02_qualidade_clusters.{png,svg}
- 03_resultados_conhecidos_por_cluster.{png,svg}
- pacote_clusterizacao_ativos.zip

Execucao no Spyder
------------------
Abra este arquivo e execute com F5.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math
import os
import shutil
import zipfile

import matplotlib

if (
    os.name != "nt"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_RANKED = (
    ROOT
    / "output"
    / "busca_ativos"
    / "intelligent_candidates_ranked.csv"
)
OUTCOMES_FILE = (
    ROOT
    / "dados"
    / "assinatura_matematica"
    / "smart20_outcomes_u59.csv"
)
OUT = ROOT / "output" / "clusterizacao_ativos"

ANALYSIS_VERSION = "1.0.1-dev.1"
ANALYSIS_SCHEMA = "candidate-behavior-clustering-v1"
SHOW_PLOTS = True
RANDOM_STATE = 42
K_VALUES = tuple(range(2, 7))
STABILITY_SEEDS = tuple(range(12))

U67_POSITIVE8 = frozenset(
    {"THO", "WDAY", "EXR", "XEL", "SBFG", "PAYX", "MUX", "SXC"}
)

FEATURES = (
    "candidate_beats_u59_best_share",
    "candidate_score_mean",
    "candidate_score_std",
    "candidate_positive_score_share",
    "abs_score_corr_u59_best",
    "model_score_corr_u59_mean",
)


def _save_pair(fig, stem: str) -> dict[str, str]:
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    if SHOW_PLOTS:
        fig.canvas.draw_idle()
        plt.show(block=False)
    else:
        plt.close(fig)
    return {"png": str(png.relative_to(ROOT)), "svg": str(svg.relative_to(ROOT))}


def _require_input() -> pd.DataFrame:
    if not INPUT_RANKED.exists():
        raise RuntimeError(
            "Arquivo de entrada ausente: "
            f"{INPUT_RANKED}. "
            "Esta fase nao precisa de novo reproduzir_experimento.py, mas "
            "precisa do intelligent_candidates_ranked.csv ja produzido por "
            "buscar_ativos.py."
        )
    frame = pd.read_csv(INPUT_RANKED)
    required = {"asset", "model_score_sessions", *FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "intelligent_candidates_ranked.csv sem colunas necessarias: "
            + ",".join(missing)
        )
    return frame


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["asset"] = (
        data["asset"].astype(str).str.strip().str.upper()
    )
    numeric = {"model_score_sessions", *FEATURES}
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    expected_sessions = int(
        pd.to_numeric(
            data["model_score_sessions"],
            errors="coerce",
        ).max()
    )
    min_sessions = int(math.ceil(expected_sessions * 0.85))

    complete = data.dropna(subset=list(FEATURES)).copy()
    complete = complete.loc[
        complete["model_score_sessions"] >= min_sessions
    ].copy()

    dormant = complete.loc[
        complete["candidate_beats_u59_best_share"] <= 0.0
    ].copy()
    active = complete.loc[
        complete["candidate_beats_u59_best_share"] > 0.0
    ].copy()

    if len(active) < max(K_VALUES) + 2:
        raise RuntimeError(
            f"Apenas {len(active)} candidatos ativos elegiveis; "
            "amostra insuficiente para testar K=2..6."
        )

    return active.reset_index(drop=True), dormant.reset_index(drop=True)


def _seed_stability(x_scaled: np.ndarray, k: int) -> float:
    label_sets = []
    for seed in STABILITY_SEEDS:
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed,
        )
        label_sets.append(model.fit_predict(x_scaled))

    if len(label_sets) < 2:
        return 1.0

    scores = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i, j in combinations(range(len(label_sets)), 2)
    ]
    return float(np.mean(scores))


def _quality_table(x_scaled: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": int(k),
                "silhouette": float(
                    silhouette_score(x_scaled, labels)
                ),
                "davies_bouldin": float(
                    davies_bouldin_score(x_scaled, labels)
                ),
                "calinski_harabasz": float(
                    calinski_harabasz_score(x_scaled, labels)
                ),
                "seed_stability_ari": _seed_stability(
                    x_scaled,
                    k,
                ),
                "inertia": float(model.inertia_),
            }
        )

    quality = pd.DataFrame(rows)
    quality["rank_silhouette"] = quality["silhouette"].rank(
        ascending=False,
        method="min",
    )
    quality["rank_davies_bouldin"] = quality[
        "davies_bouldin"
    ].rank(
        ascending=True,
        method="min",
    )
    quality["rank_calinski_harabasz"] = quality[
        "calinski_harabasz"
    ].rank(
        ascending=False,
        method="min",
    )
    quality["rank_stability"] = quality[
        "seed_stability_ari"
    ].rank(
        ascending=False,
        method="min",
    )
    quality["rank_sum"] = quality[
        [
            "rank_silhouette",
            "rank_davies_bouldin",
            "rank_calinski_harabasz",
            "rank_stability",
        ]
    ].sum(axis=1)
    quality["selected"] = False

    best_index = (
        quality.sort_values(
            [
                "rank_sum",
                "silhouette",
                "seed_stability_ari",
                "k",
            ],
            ascending=[True, False, False, True],
        )
        .index[0]
    )
    quality.loc[best_index, "selected"] = True
    return quality


def _behavior_names(members: pd.DataFrame) -> dict[int, str]:
    """Da nomes humanos aos clusters usando somente variaveis de score."""
    profiles = (
        members.groupby("cluster_id", as_index=False)
        .agg(
            mean_beats=(
                "candidate_beats_u59_best_share",
                "mean",
            ),
            mean_positive_share=(
                "candidate_positive_score_share",
                "mean",
            ),
            mean_score_std=(
                "candidate_score_std",
                "mean",
            ),
            mean_score=(
                "candidate_score_mean",
                "mean",
            ),
        )
    )

    cluster_ids = profiles["cluster_id"].astype(int).tolist()
    if len(cluster_ids) != 3:
        return {
            cluster_id: f"Grupo comportamental {cluster_id + 1}"
            for cluster_id in cluster_ids
        }

    dominant = int(
        profiles.sort_values(
            ["mean_beats", "mean_score_std"],
            ascending=[False, False],
        ).iloc[0]["cluster_id"]
    )
    remaining = profiles.loc[
        profiles["cluster_id"] != dominant
    ].copy()
    persistent = int(
        remaining.sort_values(
            ["mean_positive_share", "mean_score_std"],
            ascending=[False, True],
        ).iloc[0]["cluster_id"]
    )
    opportunistic = int(
        remaining.loc[
            remaining["cluster_id"] != persistent,
            "cluster_id",
        ].iloc[0]
    )

    return {
        dominant: "Dominantes",
        persistent: "Persistentes",
        opportunistic: "Oportunistas",
    }


def _cluster_feature_profiles(
    members: pd.DataFrame,
    behavior_names: dict[int, str],
) -> pd.DataFrame:
    feature_frame = members[list(FEATURES)].copy()
    standardized = StandardScaler().fit_transform(feature_frame)
    z = pd.DataFrame(
        standardized,
        columns=FEATURES,
        index=members.index,
    )
    z["cluster_id"] = members["cluster_id"].to_numpy()

    rows = []
    for cluster_id, group in members.groupby("cluster_id"):
        z_group = z.loc[group.index]
        row = {
            "cluster_id": int(cluster_id),
            "behavior_name": behavior_names[int(cluster_id)],
            "candidate_count": int(len(group)),
        }
        for feature in FEATURES:
            row[f"mean_{feature}"] = float(group[feature].mean())
            row[f"z_{feature}"] = float(z_group[feature].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "cluster_id",
        ignore_index=True,
    )


def _build_cluster_summary(
    members: pd.DataFrame,
    behavior_names: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for cluster_id, group in members.groupby("cluster_id"):
        known = group.loc[
            group["capital_pct_vs_u59"].notna()
        ].copy()

        if known.empty:
            median_effect = np.nan
            mean_effect = np.nan
            positive_rate = np.nan
            harm_rate = np.nan
        else:
            effects = known["capital_pct_vs_u59"].astype(float)
            median_effect = float(effects.median())
            mean_effect = float(effects.mean())
            positive_rate = float((effects > 0.0).mean())
            harm_rate = float((effects <= -0.10).mean())

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "candidate_count": int(len(group)),
                "known_outcomes": int(len(known)),
                "u67_positive8_count": int(
                    group["u67_positive8"].sum()
                ),
                "median_capital_pct_vs_u59": median_effect,
                "mean_capital_pct_vs_u59": mean_effect,
                "positive_rate": positive_rate,
                "harm10_rate": harm_rate,
                "behavior_name": behavior_names[int(cluster_id)],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "cluster_id",
        ignore_index=True,
    )


def _plot_pca(
    members: pd.DataFrame,
    summary: pd.DataFrame,
    explained: np.ndarray,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(11.8, 8.0))
    colors = plt.get_cmap("tab10")

    behavior = summary.set_index("cluster_id")["behavior_name"].to_dict()

    for cluster_id, group in members.groupby("cluster_id"):
        label = (
            f"Grupo {cluster_id + 1} · "
            f"{behavior.get(cluster_id, 'Grupo comportamental')}"
        )
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=36,
            alpha=0.68,
            color=colors(cluster_id % 10),
            label=label,
        )

    special = members.loc[members["u67_positive8"]].copy()
    if not special.empty:
        ax.scatter(
            special["pc1"],
            special["pc2"],
            s=120,
            marker="*",
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="Oito impulsionadores do U67",
        )
        for row in special.itertuples(index=False):
            ax.annotate(
                row.asset,
                (row.pc1, row.pc2),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel(
        f"Componente principal 1 ({explained[0] * 100:.1f}% da variância)"
    )
    ax.set_ylabel(
        f"Componente principal 2 ({explained[1] * 100:.1f}% da variância)"
    )
    ax.set_title(
        "Mapa dos candidatos por comportamento de score\n"
        "Clusters ajustados sem usar resultado financeiro",
        loc="left",
        pad=16,
        fontweight="bold",
    )
    ax.grid(alpha=0.18)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.74)
    return _save_pair(fig, "01_mapa_clusters_pca")


def _plot_quality(
    quality: pd.DataFrame,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    ax.plot(
        quality["k"],
        quality["silhouette"],
        marker="o",
        label="Silhouette",
    )
    ax.plot(
        quality["k"],
        quality["seed_stability_ari"],
        marker="o",
        label="Estabilidade entre sementes (ARI)",
    )
    selected_k = int(
        quality.loc[quality["selected"], "k"].iloc[0]
    )
    ax.axvline(
        selected_k,
        linestyle="--",
        linewidth=1.1,
        label=f"K selecionado = {selected_k}",
    )
    ax.set_xticks(list(K_VALUES))
    ax.set_ylim(
        min(
            0.0,
            float(
                quality[
                    ["silhouette", "seed_stability_ari"]
                ].min().min()
            )
            - 0.05,
        ),
        1.02,
    )
    ax.set_xlabel("Número de grupos (K)")
    ax.set_ylabel("Índice")
    ax.set_title(
        "Qualidade e estabilidade da clusterização",
        loc="left",
        pad=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    return _save_pair(fig, "02_qualidade_clusters")


def _plot_known_outcomes(
    members: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str] | None:
    known = members.loc[
        members["capital_pct_vs_u59"].notna()
    ].copy()
    if known.empty:
        return None

    behavior = summary.set_index("cluster_id")["behavior_name"].to_dict()
    cluster_ids = sorted(known["cluster_id"].unique())
    x_positions = {
        cluster_id: index
        for index, cluster_id in enumerate(cluster_ids)
    }

    category_colors = {
        "Impulsionador observado": "#2E7D32",
        "Sobrevivente observado": "#D08B00",
        "Prejudicial observado": "#B23A48",
    }

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for cluster_id in cluster_ids:
        group = known.loc[
            known["cluster_id"] == cluster_id
        ].sort_values(
            ["capital_pct_vs_u59", "asset"],
            ignore_index=True,
        )
        offsets = np.linspace(
            -0.18,
            0.18,
            max(1, len(group)),
        )
        for offset, row in zip(
            offsets,
            group.itertuples(index=False),
        ):
            x = x_positions[cluster_id] + float(offset)
            y = float(row.capital_pct_vs_u59) * 100.0
            color = category_colors.get(
                row.observed_group,
                "#66717A",
            )
            marker = "*" if row.u67_positive8 else "o"
            size = 110 if row.u67_positive8 else 56
            ax.scatter(
                [x],
                [y],
                s=size,
                marker=marker,
                color=color,
                alpha=0.9,
            )
            if row.u67_positive8:
                ax.annotate(
                    row.asset,
                    (x, y),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.axhline(0.0, linewidth=1.0, color="#5A626B")
    ax.axhline(
        -10.0,
        linewidth=1.0,
        linestyle="--",
        color="#B23A48",
        alpha=0.7,
    )
    ax.set_xticks(
        list(range(len(cluster_ids))),
        [
            f"Grupo {cid + 1}\n{behavior.get(cid, '')}"
            for cid in cluster_ids
        ],
    )
    ax.set_ylabel("Efeito individual conhecido vs U59 (%)")
    ax.set_title(
        "Resultado financeiro conhecido sobreposto aos clusters\n"
        "Os resultados não participaram do ajuste dos grupos",
        loc="left",
        pad=16,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.18)
    return _save_pair(
        fig,
        "03_resultados_conhecidos_por_cluster",
    )


def _plot_cluster_profiles(
    profiles: pd.DataFrame,
) -> dict[str, str]:
    z_columns = [f"z_{feature}" for feature in FEATURES]
    matrix = profiles[z_columns].to_numpy(dtype=float)

    labels = [
        "Vence o melhor U59",
        "Score medio",
        "Volatilidade do score",
        "Score positivo",
        "Correlacao abs. com melhor",
        "Correlacao com media U59",
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    limit = max(
        1.0,
        float(np.nanmax(np.abs(matrix))) * 1.05,
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        range(len(profiles)),
        [
            f"Grupo {int(row.cluster_id) + 1} · {row.behavior_name}"
            for row in profiles.itertuples(index=False)
        ],
    )
    ax.set_title(
        "Perfil comportamental dos clusters\n"
        "Valores padronizados; sem uso de capital",
        loc="left",
        pad=14,
        fontweight="bold",
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Desvio-padrao em relacao ao conjunto")
    fig.tight_layout()
    return _save_pair(fig, "04_perfil_comportamental_clusters")


def main() -> None:
    print("=" * 78, flush=True)
    print("TCC - CLUSTERIZACAO EXPLORATORIA DE CANDIDATOS", flush=True)
    print(
        f"version={ANALYSIS_VERSION} schema={ANALYSIS_SCHEMA}",
        flush=True,
    )
    print("new_backtest=NO", flush=True)
    print("lightgbm_training=NO", flush=True)
    print("financial_outcomes_used_in_fit=NO", flush=True)
    print("=" * 78, flush=True)

    source = _require_input()
    active, dormant = _prepare(source)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(active[list(FEATURES)])

    quality = _quality_table(x_scaled)
    selected_k = int(
        quality.loc[quality["selected"], "k"].iloc[0]
    )

    kmeans = KMeans(
        n_clusters=selected_k,
        n_init=100,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(x_scaled)

    hierarchical = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage="ward",
    ).fit_predict(x_scaled)
    cross_method_ari = float(
        adjusted_rand_score(labels, hierarchical)
    )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )
    coords = pca.fit_transform(x_scaled)

    members = active.copy()
    members["cluster_id"] = labels.astype(int)
    members["pc1"] = coords[:, 0]
    members["pc2"] = coords[:, 1]
    members["u67_positive8"] = members["asset"].isin(
        U67_POSITIVE8
    )

    outcomes = pd.read_csv(OUTCOMES_FILE)
    outcomes["asset"] = (
        outcomes["asset"].astype(str).str.strip().str.upper()
    )
    outcomes["capital_pct_vs_u59"] = pd.to_numeric(
        outcomes["capital_pct_vs_u59"],
        errors="coerce",
    )

    members = members.merge(
        outcomes[
            [
                "asset",
                "capital_pct_vs_u59",
                "observed_group",
                "scientific_role",
            ]
        ],
        on="asset",
        how="left",
        validate="one_to_one",
    )

    behavior_names = _behavior_names(members)
    profiles = _cluster_feature_profiles(
        members,
        behavior_names,
    )
    summary = _build_cluster_summary(
        members,
        behavior_names,
    )
    members["behavior_name"] = members["cluster_id"].map(
        behavior_names
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    quality.to_csv(OUT / "cluster_quality.csv", index=False)
    members.to_csv(OUT / "cluster_members.csv", index=False)
    summary.to_csv(OUT / "cluster_summary.csv", index=False)
    profiles.to_csv(
        OUT / "cluster_behavior_profiles.csv",
        index=False,
    )
    dormant.to_csv(OUT / "inactive_candidates.csv", index=False)
    members.loc[
        members["capital_pct_vs_u59"].notna()
    ].to_csv(
        OUT / "known_outcomes_overlay.csv",
        index=False,
    )

    graphs = {
        "pca": _plot_pca(
            members,
            summary,
            pca.explained_variance_ratio_,
        ),
        "quality": _plot_quality(quality),
        "known_outcomes": _plot_known_outcomes(
            members,
            summary,
        ),
        "behavior_profiles": _plot_cluster_profiles(
            profiles,
        ),
    }

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_schema": ANALYSIS_SCHEMA,
        "input": str(INPUT_RANKED.relative_to(ROOT)),
        "features": list(FEATURES),
        "financial_outcomes_used_in_fit": False,
        "financial_overlay_file": str(
            OUTCOMES_FILE.relative_to(ROOT)
        ),
        "active_candidates": int(len(active)),
        "inactive_candidates": int(len(dormant)),
        "selected_k": selected_k,
        "selection_rule": (
            "minimum rank-sum across silhouette (higher), Davies-Bouldin "
            "(lower), Calinski-Harabasz (higher) and seed-stability ARI "
            "(higher)"
        ),
        "kmeans_vs_hierarchical_ari": cross_method_ari,
        "pca_explained_variance_ratio": [
            float(value)
            for value in pca.explained_variance_ratio_
        ],
        "u67_positive8": sorted(U67_POSITIVE8),
        "cluster_summary": summary.to_dict(orient="records"),
        "graphs": graphs,
        "interpretation": (
            "Clusters are fitted and named behaviorally without capital outcomes. "
            "Smart20 financial effects are merged only afterwards as an "
            "external overlay. The observed outcome labels Impulsionador, "
            "Sobrevivente and Prejudicial are not cluster names because the "
            "current unsupervised groups do not cleanly reproduce those "
            "financial classes."
        ),
    }
    with (
        OUT / "cluster_analysis.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    package = OUT / "pacote_clusterizacao_ativos.zip"
    with zipfile.ZipFile(
        package,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in sorted(OUT.iterdir()):
            if file_path == package or not file_path.is_file():
                continue
            archive.write(
                file_path,
                arcname=file_path.name,
            )

    print(
        f"[cluster] active={len(active)} inactive={len(dormant)} "
        f"selected_k={selected_k} "
        f"kmeans_vs_hierarchical_ari={cross_method_ari:.4f}",
        flush=True,
    )
    print(summary.to_string(index=False), flush=True)
    print(f"[done] output={OUT}", flush=True)
    print(f"[done] package={package}", flush=True)


if __name__ == "__main__":
    main()
