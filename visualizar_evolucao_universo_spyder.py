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
from sklearn.preprocessing import StandardScaler

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS
from engine.rotacao import ROTATION_FEATURES, construir_quadro_rotacao
from reproducao.dados import SnapshotPaths
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuração da visualização
ROOT = Path(__file__).resolve().parent
DATA = SnapshotPaths.u67(ROOT)
OUT = ROOT / "output" / "universe_clustering"

SCRIPT_VERSION = "universe-clustering-visualization-v1.0.0"

# Procura listas distintas de ASSETS em todos os commits alcançáveis no clone.
# Isso evita assumir que o primeiro universo tinha 20, 25 ou qualquer outro
# número de ativos.
BUSCAR_EM_TODAS_AS_BRANCHES = True

# Arquivo histórico no qual a lista ASSETS é procurada.
CONFIG_PATHS = (
    "engine/configuracao.py",
)

# Janela usada para resumir os 52 atributos.
ANALYSIS_START = "2020-07-22"
ANALYSIS_END = "2026-10-06"

# Número de grupos apenas para visualização exploratória.
N_GROUPS = 4

# Se houver muitas listas históricas, mostra no máximo este número de mapas.
MAX_STAGES_TO_PLOT = 8

# Para escolher manualmente etapas depois da primeira inspeção, informe os
# identificadores exibidos no console, por exemplo [1, 3, 5, 8].
# None = seleção automática de etapas representativas.
SELECTED_STAGE_IDS: list[int] | None = None

SAVE_TABLES = True
SAVE_FIGURES = False

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
        return datetime.fromtimestamp(
            self.timestamp,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d")


# %% 1 - Descoberta do histórico real do universo
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
    ax.plot(x, table["asset_count"], marker="o", linewidth=1.5)

    for i, row in table.iterrows():
        label = (
            f"{int(row['asset_count'])} ativos\n"
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

    ax.set_xticks(x, [f"E{int(v)}" for v in table["stage_id"]])
    ax.set_xlabel("Etapas encontradas no histórico Git")
    ax.set_ylabel("Quantidade de ativos")
    ax.set_title(
        "Evolução das composições de universo encontradas no repositório"
    )
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
    _, _, _, _ = (None, None, None, None)

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
        if len(assets) < max(4, N_GROUPS):
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


def embed_and_cluster(context: pd.DataFrame) -> pd.DataFrame:
    matrix = context[ROTATION_FEATURES].to_numpy(dtype=float)
    matrix = StandardScaler().fit_transform(matrix)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    groups = KMeans(
        n_clusters=min(N_GROUPS, len(context)),
        random_state=42,
        n_init=20,
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
        added = current
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
        f"Etapa E{stage.stage_id}: agrupamento por semelhança "
        f"({len(current)} ativos)"
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
            OUT / f"etapa_{stage.stage_id:02d}_agrupamento.png",
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

    stages = discover_universe_stages()
    table = stage_table(stages)

    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 220)

    print("\nEtapas encontradas no histórico Git:")
    print(
        table[
            [
                "stage_id",
                "date",
                "asset_count",
                "added_count",
                "removed_count",
                "added",
                "removed",
                "commit",
            ]
        ].to_string(index=False)
    )

    print(
        "\nIMPORTANTE: a ordem acima é a ordem temporal das composições distintas "
        "encontradas no Git. Como o histórico pode conter branches de pesquisa, "
        "ela deve ser validada antes de ser descrita no texto científico."
    )

    plot_stage_timeline(table)

    summary, excluded = build_asset_feature_summary()
    print(
        f"\nAtivos com representação válida: {len(summary)} | "
        f"exclusões estruturais atuais: {sorted(excluded)}"
    )

    context, available_by_stage = contextual_rows(stages, summary)
    embedding = embed_and_cluster(context)
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
        table.to_csv(
            OUT / "etapas_universo_descobertas.csv",
            index=False,
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
            - mudança de grupo = mudança da posição relativa do ativo no conjunto;
            - inclusão/remoção = alteração da composição observada entre etapas;
            - o gráfico é exploratório e não prova causalidade nem desempenho futuro.
            """
        ).strip()
    )
    print("\nConcluído. Consulte o painel Plots do Spyder.")


if __name__ == "__main__":
    main()
