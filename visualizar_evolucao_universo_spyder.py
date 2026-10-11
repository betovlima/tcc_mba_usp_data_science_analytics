"""Visualização exploratória da evolução do universo de ativos no Spyder.

Objetivo
--------
Descobrir, sem assumir previamente o tamanho do universo inicial, as listas de
ativos preservadas no histórico Git e visualizar como os ativos se agrupam por
semelhança à medida que a composição do universo muda.

Este script NÃO:
- treina LightGBM;
- executa backtest;
- altera dados congelados;
- altera a política final;
- consulta internet, Alpaca ou banco de dados.

Execução no Spyder
------------------
1. Abra este arquivo no Spyder.
2. Reinicie o kernel.
3. Execute com F5.

As figuras são exibidas no painel Plots do Spyder por meio de plt.show().
Nenhum backend não interativo é forçado.

Leitura dos gráficos
--------------------
- Cada ponto representa um ativo.
- Pontos próximos têm perfis históricos semelhantes nos 52 atributos.
- Os atributos são resumidos por ativo e expressos relativamente aos demais
  ativos de cada etapa.
- A PCA é ajustada uma única vez sobre todas as combinações ativo-etapa para que
  os eixos sejam comparáveis entre os gráficos.
- Os grupos são exploratórios. Não representam setores econômicos, relação
  causal nem regra automática de seleção de ativos.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS
from engine.rotacao import ROTATION_FEATURES, construir_quadro_rotacao
from reproducao.dados import SnapshotPaths
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuração da visualização
ROOT = Path(__file__).resolve().parent
DATA = SnapshotPaths.u67(ROOT)
OUT = ROOT / "output" / "universe_clustering"

SCRIPT_VERSION = "universe-clustering-visualization-v1.1.0"

# A primeira versão encontrou somente U56 e U67 porque procurava alterações
# literais de ASSETS em engine/configuracao.py. O histórico preservado da
# pesquisa documenta explicitamente a sequência aceita U56 -> U59 -> U67.
# Essa sequência é usada por padrão. A descoberta automática pelo Git continua
# sendo impressa como auditoria e não é tratada como prova de que U56 tenha sido
# o primeiro universo de toda a pesquisa.
USE_DOCUMENTED_RESEARCH_STAGES = True
BUSCAR_EM_TODAS_AS_BRANCHES = True
CONFIG_PATHS = ("engine/configuracao.py",)

# Janela comum para comparar as composições. O corte de 17 set. 2026 coincide
# com o checkpoint histórico em que U56, U59 e U67 foram documentados.
ANALYSIS_START = "2020-07-22"
ANALYSIS_END = "2026-09-17"

# O número de grupos deixa de ser arbitrário. O script testa K=2..6 e escolhe
# o K com maior silhouette no conjunto combinado de ativo-etapa.
K_VALUES = tuple(range(2, 7))

MAX_STAGES_TO_PLOT = 8
SELECTED_STAGE_IDS: list[int] | None = None

SAVE_TABLES = True
SAVE_FIGURES = True

# Ativos que recebem rótulo no gráfico quando estiverem presentes.
HIGHLIGHT_ASSETS = {
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "JPM", "SPY", "ADM", "ADI", "ADEA", "ADC", "DOC", "CLMT",
}


@dataclass(frozen=True)
class UniverseStage:
    stage_id: int
    commit: str
    timestamp: int
    source_path: str
    assets: tuple[str, ...]

    @property
    def date(self) -> str:
        if self.commit.startswith("documented:"):
            return ""
        return datetime.fromtimestamp(
            self.timestamp,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d")


def documented_research_stages() -> list[UniverseStage]:
    """Reconstrói a sequência U56 -> U59 -> U67 documentada na pesquisa."""
    u67 = tuple(U67_REQUESTED_ASSETS)
    additions_u59 = ("COLB", "AMS", "FOXF")
    additions_u67 = (
        "THO", "WDAY", "EXR", "XEL", "SBFG", "PAYX", "MUX", "SXC",
    )
    added = frozenset((*additions_u59, *additions_u67))
    u56 = tuple(symbol for symbol in u67 if symbol not in added)
    u59 = (*u56, *additions_u59)

    if len(u56) != 56:
        raise RuntimeError(f"U56 documentado deveria conter 56 ativos; observado={len(u56)}")
    if len(u59) != 59:
        raise RuntimeError(f"U59 documentado deveria conter 59 ativos; observado={len(u59)}")
    if u67 != (*u59, *additions_u67):
        raise RuntimeError("A ordem atual do U67 não reproduz a sequência documentada U56 -> U59 -> U67.")

    source = "research/rotation-visualizations-v1:README.md"
    return [
        UniverseStage(1, "documented:U56", 0, source, u56),
        UniverseStage(2, "documented:U59", 1, source, u59),
        UniverseStage(3, "documented:U67", 2, source, u67),
    ]


# %% 1 - Descoberta e reconstrução do histórico do universo
def _git(args: list[str]) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return process.stdout


def _extract_assets_from_python(source: str) -> tuple[str, ...] | None:
    """Extrai uma atribuição literal ASSETS = (...) ou ASSETS = [...]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "ASSETS"
                for target in node.targets
            ):
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "ASSETS":
                value_node = node.value

        if value_node is None:
            continue

        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            return None

        if not isinstance(value, (tuple, list)):
            return None

        assets = tuple(
            str(item).strip().upper()
            for item in value
            if str(item).strip()
        )
        return assets or None

    return None


def discover_universe_stages() -> list[UniverseStage]:
    """Descobre listas distintas de ativos preservadas no histórico Git."""
    refs = ["--all"] if BUSCAR_EM_TODAS_AS_BRANCHES else []
    commits: dict[str, int] = {}

    for config_path in CONFIG_PATHS:
        try:
            log = _git([
                "log",
                *refs,
                "--format=%H|%ct",
                "--",
                config_path,
            ])
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

        for line in log.splitlines():
            if "|" not in line:
                continue
            sha, raw_timestamp = line.split("|", 1)
            try:
                commits[sha] = int(raw_timestamp)
            except ValueError:
                continue

    candidates: list[tuple[int, str, str, tuple[str, ...]]] = []
    for sha, timestamp in commits.items():
        for config_path in CONFIG_PATHS:
            try:
                source = _git(["show", f"{sha}:{config_path}"])
            except subprocess.CalledProcessError:
                continue
            assets = _extract_assets_from_python(source)
            if assets:
                candidates.append((timestamp, sha, config_path, assets))
                break

    # Guarda a primeira ocorrência temporal de cada composição literal.
    first_by_assets: dict[tuple[str, ...], tuple[int, str, str]] = {}
    for timestamp, sha, config_path, assets in sorted(candidates):
        first_by_assets.setdefault(assets, (timestamp, sha, config_path))

    stages: list[UniverseStage] = []
    for stage_id, (assets, info) in enumerate(
        sorted(first_by_assets.items(), key=lambda item: item[1][0]),
        start=1,
    ):
        timestamp, sha, config_path = info
        stages.append(
            UniverseStage(
                stage_id=stage_id,
                commit=sha,
                timestamp=timestamp,
                source_path=config_path,
                assets=assets,
            )
        )

    if stages:
        return stages

    # Fallback: permite executar a visualização mesmo em uma cópia sem .git.
    return [
        UniverseStage(
            stage_id=1,
            commit="working-tree",
            timestamp=int(datetime.now(tz=timezone.utc).timestamp()),
            source_path="engine/configuracao.py",
            assets=tuple(U67_REQUESTED_ASSETS),
        )
    ]


def stage_table(stages: list[UniverseStage]) -> pd.DataFrame:
    rows = []
    previous: set[str] = set()
    for stage in stages:
        current = set(stage.assets)
        added = sorted(current - previous)
        removed = sorted(previous - current)
        rows.append(
            {
                "stage_id": stage.stage_id,
                "universe": f"U{len(stage.assets)}",
                "date": stage.date,
                "asset_count": len(stage.assets),
                "added_count": len(added),
                "removed_count": len(removed),
                "added": ",".join(added),
                "removed": ",".join(removed),
                "commit": stage.commit,
                "source_path": stage.source_path,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def plot_stage_timeline(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(table))

    ax.plot(
        x,
        table["asset_count"],
        marker="o",
        linewidth=1.5,
        label="Universo documentado",
    )
    if "represented_count" in table.columns:
        ax.plot(
            x,
            table["represented_count"],
            marker="s",
            linewidth=1.2,
            linestyle="--",
            label="Ativos representados no mapa",
        )

    for i, row in table.iterrows():
        represented = int(row.get("represented_count", row["asset_count"]))
        label = (
            f"{row['universe']}\n"
            f"{int(row['asset_count'])} documentados / {represented} representados\n"
            f"+{int(row['added_count'])} / -{int(row['removed_count'])}"
        )
        ax.annotate(
            label,
            (i, row["asset_count"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(x, table["universe"])
    ax.set_xlabel("Etapas documentadas da expansão do universo")
    ax.set_ylabel("Quantidade de ativos")
    ax.set_title("Evolução documentada da composição do universo de ativos")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if SAVE_FIGURES:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUT / "01_evolucao_tamanho_universo.png",
            dpi=220,
            bbox_inches="tight",
        )

    plt.show()


# %% 2 - Construção da representação de cada ativo
def build_asset_feature_summary() -> tuple[pd.DataFrame, set[str]]:
    """Resume os 52 atributos do modelo para cada ativo efetivamente utilizável."""
    frames_raw, exclusions, _, _ = prepare_model_frames(
        DATA,
        assets=U67_REQUESTED_ASSETS,
        csv_float_precision="round_trip",
    )

    excluded = {
        str(item.get("symbol") or "").strip().upper()
        for item in exclusions
    }

    rows = []
    for symbol in U67_REQUESTED_ASSETS:
        if symbol not in frames_raw:
            continue

        frame = construir_quadro_rotacao(frames_raw[symbol], CONFIG)
        frame = frame.loc[
            (frame.index >= pd.Timestamp(ANALYSIS_START, tz="UTC"))
            & (frame.index <= pd.Timestamp(ANALYSIS_END, tz="UTC"))
        ]
        if frame.empty:
            continue

        values = frame[ROTATION_FEATURES].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        medians = values.median(axis=0, skipna=True)

        row = {"asset": symbol}
        for name in ROTATION_FEATURES:
            row[name] = float(medians[name])
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("asset").sort_index()
    summary = summary.replace([np.inf, -np.inf], np.nan)
    summary = summary.dropna(axis=0, how="any")
    return summary, excluded


def contextual_rows(
    stages: list[UniverseStage],
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, tuple[str, ...]]]:
    """Expressa cada ativo relativamente aos demais de sua etapa."""
    blocks = []
    available_by_stage: dict[int, tuple[str, ...]] = {}

    for stage in stages:
        assets = tuple(
            symbol
            for symbol in stage.assets
            if symbol in summary.index
        )
        if len(assets) < 4:
            continue

        matrix = summary.loc[list(assets), ROTATION_FEATURES].copy()
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0).replace(0, np.nan)
        relative = (matrix - mean) / std
        relative = relative.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        relative.insert(0, "asset", relative.index)
        relative.insert(0, "stage_id", stage.stage_id)
        blocks.append(relative.reset_index(drop=True))
        available_by_stage[stage.stage_id] = assets

    if not blocks:
        raise RuntimeError(
            "Nenhuma etapa possui ativos suficientes com dados válidos."
        )

    return pd.concat(blocks, ignore_index=True), available_by_stage


def cluster_quality(context: pd.DataFrame) -> pd.DataFrame:
    """Testa números de grupos e escolhe o de maior silhouette."""
    matrix = StandardScaler().fit_transform(
        context[ROTATION_FEATURES].to_numpy(dtype=float)
    )
    rows = []
    for k in K_VALUES:
        if k >= len(context):
            continue
        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50,
        ).fit_predict(matrix)
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            }
        )
    quality = pd.DataFrame(rows)
    if quality.empty:
        raise RuntimeError("Não foi possível avaliar a quantidade de grupos.")
    quality["selected"] = False
    best = quality.sort_values(
        ["silhouette", "davies_bouldin", "k"],
        ascending=[False, True, True],
    ).index[0]
    quality.loc[best, "selected"] = True
    return quality


def plot_cluster_quality(quality: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(quality["k"], quality["silhouette"], marker="o", linewidth=1.5)
    best = quality.loc[quality["selected"]].iloc[0]
    ax.scatter([best["k"]], [best["silhouette"]], s=90, zorder=3)
    ax.annotate(
        f"K selecionado = {int(best['k'])}",
        (best["k"], best["silhouette"]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.set_xticks(quality["k"].astype(int))
    ax.set_xlabel("Quantidade de grupos testada")
    ax.set_ylabel("Índice de separação dos grupos (silhouette)")
    ax.set_title("Escolha da quantidade de grupos")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if SAVE_FIGURES:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUT / "02_escolha_quantidade_grupos.png",
            dpi=220,
            bbox_inches="tight",
        )
    plt.show()


def embed_and_cluster(
    context: pd.DataFrame,
    n_groups: int,
) -> pd.DataFrame:
    matrix = context[ROTATION_FEATURES].to_numpy(dtype=float)
    matrix = StandardScaler().fit_transform(matrix)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    groups = KMeans(
        n_clusters=n_groups,
        random_state=42,
        n_init=50,
    ).fit_predict(matrix)

    result = context[["stage_id", "asset"]].copy()
    result["pc1"] = coords[:, 0]
    result["pc2"] = coords[:, 1]
    result["group"] = groups + 1
    result.attrs["pca_explained"] = tuple(
        float(v) for v in pca.explained_variance_ratio_
    )
    return result


# %% 3 - Seleção e gráficos das etapas
def choose_stages(
    stages: list[UniverseStage],
    available_by_stage: dict[int, tuple[str, ...]],
) -> list[UniverseStage]:
    valid = [
        stage
        for stage in stages
        if stage.stage_id in available_by_stage
    ]

    if SELECTED_STAGE_IDS is not None:
        wanted = set(SELECTED_STAGE_IDS)
        selected = [stage for stage in valid if stage.stage_id in wanted]
        if not selected:
            raise ValueError(
                "SELECTED_STAGE_IDS não corresponde a etapas válidas."
            )
        return selected

    if len(valid) <= MAX_STAGES_TO_PLOT:
        return valid

    positions = np.linspace(
        0,
        len(valid) - 1,
        MAX_STAGES_TO_PLOT,
    ).round().astype(int)

    unique_positions = list(dict.fromkeys(int(i) for i in positions))
    return [valid[i] for i in unique_positions]


def plot_stage_map(
    stage: UniverseStage,
    embedding: pd.DataFrame,
    previous_assets: set[str] | None,
    pca_explained: tuple[float, float],
) -> set[str]:
    part = embedding.loc[embedding["stage_id"] == stage.stage_id].copy()
    current = set(part["asset"])

    if previous_assets is None:
        added: set[str] = set()
        removed: set[str] = set()
    else:
        added = current - previous_assets
        removed = previous_assets - current

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    for group_id in sorted(part["group"].unique()):
        group = part.loc[part["group"] == group_id]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=52,
            alpha=0.78,
            label=f"Grupo {group_id}",
        )

    for row in part.itertuples(index=False):
        should_label = (
            row.asset in HIGHLIGHT_ASSETS
            or row.asset in added
        )
        if not should_label:
            continue

        prefix = "+ " if row.asset in added and previous_assets is not None else ""
        ax.annotate(
            prefix + row.asset,
            (row.pc1, row.pc2),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0, linewidth=0.7, alpha=0.25)
    ax.axvline(0, linewidth=0.7, alpha=0.25)

    explained_1 = 100 * pca_explained[0]
    explained_2 = 100 * pca_explained[1]
    ax.set_xlabel(f"Eixo de semelhança 1 ({explained_1:.1f}% da variação)")
    ax.set_ylabel(f"Eixo de semelhança 2 ({explained_2:.1f}% da variação)")

    change_text = ""
    if previous_assets is not None:
        added_text = ", ".join(sorted(added)) or "nenhum"
        removed_text = ", ".join(sorted(removed)) or "nenhum"
        change_text = f"\nIncluídos: {added_text} | Removidos: {removed_text}"

    ax.set_title(
        f"U{len(stage.assets)}: agrupamento por semelhança "
        f"({len(current)} representados de {len(stage.assets)} documentados)"
        f"{change_text}"
    )
    ax.legend(
        title="Grupos exploratórios",
        frameon=False,
        loc="best",
    )
    ax.grid(alpha=0.16)
    fig.tight_layout()

    if SAVE_FIGURES:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUT / f"{stage.stage_id + 2:02d}_U{len(stage.assets)}_agrupamento.png",
            dpi=220,
            bbox_inches="tight",
        )

    plt.show()
    return current


# %% 4 - Execução
def main() -> None:
    print("=" * 78)
    print("TCC - EVOLUÇÃO DO UNIVERSO E AGRUPAMENTO POR SEMELHANÇA")
    print(f"version={SCRIPT_VERSION}")
    print("modo=EXPLORATORIO_SEM_TREINO_SEM_BACKTEST")
    print("saida_principal=SPYDER_PLOTS")
    print("=" * 78)

    git_stages = discover_universe_stages()
    git_table = stage_table(git_stages)

    stages = (
        documented_research_stages()
        if USE_DOCUMENTED_RESEARCH_STAGES
        else git_stages
    )
    table = stage_table(stages)

    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 220)

    print("\nComposições literais encontradas apenas em engine/configuracao.py:")
    print(
        git_table[
            [
                "stage_id",
                "universe",
                "date",
                "asset_count",
                "added",
                "removed",
                "commit",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSequência documentada usada nos gráficos: U56 -> U59 -> U67. "
        "Ela foi preservada na documentação da pesquisa. Isso NÃO prova que "
        "U56 tenha sido o primeiro universo de toda a história do projeto."
    )
    print(
        table[
            [
                "stage_id",
                "universe",
                "asset_count",
                "added_count",
                "removed_count",
                "added",
                "removed",
                "source_path",
            ]
        ].to_string(index=False)
    )

    summary, excluded = build_asset_feature_summary()
    print(
        f"\nAtivos com representação válida: {len(summary)} | "
        f"exclusões estruturais atuais: {sorted(excluded)}"
    )

    context, available_by_stage = contextual_rows(stages, summary)
    table["represented_count"] = table["stage_id"].map(
        lambda sid: len(available_by_stage.get(int(sid), ()))
    )
    plot_stage_timeline(table)

    quality = cluster_quality(context)
    selected_k = int(quality.loc[quality["selected"], "k"].iloc[0])
    print("\nQualidade dos agrupamentos testados:")
    print(quality.to_string(index=False))
    print(f"\nQuantidade de grupos selecionada: K={selected_k}")
    plot_cluster_quality(quality)

    embedding = embed_and_cluster(context, selected_k)
    pca_explained = embedding.attrs["pca_explained"]

    selected = choose_stages(stages, available_by_stage)
    print(
        "\nEtapas selecionadas para os mapas: "
        + ", ".join(f"E{s.stage_id}({len(s.assets)})" for s in selected)
    )

    previous_assets: set[str] | None = None
    for stage in selected:
        previous_assets = plot_stage_map(
            stage,
            embedding,
            previous_assets,
            pca_explained,
        )

    if SAVE_TABLES:
        OUT.mkdir(parents=True, exist_ok=True)
        git_table.to_csv(
            OUT / "etapas_universo_descobertas_git.csv",
            index=False,
        )
        table.to_csv(
            OUT / "etapas_universo_documentadas.csv",
            index=False,
        )
        quality.to_csv(
            OUT / "qualidade_agrupamentos.csv",
            index=False,
            float_format="%.17g",
        )
        embedding.to_csv(
            OUT / "coordenadas_agrupamento.csv",
            index=False,
            float_format="%.17g",
        )
        summary.reset_index().to_csv(
            OUT / "resumo_atributos_por_ativo.csv",
            index=False,
            float_format="%.17g",
        )

    print(
        textwrap.dedent(
            """
            Interpretação:
            - proximidade no mapa = perfis semelhantes nos atributos resumidos;
            - proximidade no mapa = perfis semelhantes nos 52 atributos resumidos;
            - mudança de grupo = mudança da posição relativa do ativo no conjunto;
            - U56 -> U59 adiciona COLB, AMS e FOXF;
            - U59 -> U67 adiciona THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC;
            - DOC e CLMT permanecem documentados nos universos históricos, mas não
              recebem coordenadas porque a preparação final os exclui estruturalmente;
            - a visualização usa uma janela comum e é retrospectiva: não reproduz
              exatamente o estado dos dados existente no dia em que cada universo
              foi escolhido;
            - o gráfico é exploratório e não prova causalidade nem desempenho futuro.
            """
        ).strip()
    )
    print("\nConcluído. Consulte o painel Plots do Spyder.")


if __name__ == "__main__":
    main()
