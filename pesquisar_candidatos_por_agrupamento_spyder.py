"""Busca exploratória de novos ativos por semelhança e complementaridade.

Objetivo
--------
Usar o mapa comportamental do universo U67 como FILTRO DE DESCOBERTA de novos
candidatos, sem usar capital, P&L ou resultado de backtest para escolhê-los.

A ideia difere da clusterização antiga de candidatos:
- a referência é o comportamento dos ativos já existentes no U67;
- são usados os mesmos 52 atributos técnicos do experimento;
- a PCA e o escalonamento são ajustados SOMENTE no U67;
- os candidatos são apenas projetados nesse espaço, sem reajustar a referência;
- dois tipos de candidato são separados:
  1. semelhantes: ocupam regiões já representadas pelo U67;
  2. complementares: ocupam regiões próximas, mas pouco representadas.

Esta etapa NÃO decide inclusão no universo.
Ela gera uma shortlist para uma etapa posterior de validação de substituição
(candidato, ativo atualmente mantido, data).

Execução no Spyder
------------------
1. Abra este arquivo.
2. Reinicie o kernel.
3. Execute com F5.

Os gráficos aparecem no painel Plots do Spyder e os CSVs são gravados em
output/candidate_cluster_search/.

A primeira execução consulta Alpaca para obter o catálogo e as barras dos
candidatos. As barras ficam em cache dentro de output/ para que execuções
seguintes não repitam o download.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS
from engine.rotacao import ROTATION_FEATURES, construir_quadro_rotacao
from reproducao.dados import (
    SnapshotPaths,
    _normalize_alpaca_frame,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuração
ROOT = Path(__file__).resolve().parent
U67 = SnapshotPaths.u67(ROOT)
OUT = ROOT / "output" / "candidate_cluster_search"
CACHE = OUT / "cache_bars"

SCRIPT_VERSION = "candidate-cluster-search-v1.0.1"
EXPECTED_BRANCH = "research/universe-clustering-visualization-v1"

ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")

# Busca inicial controlada. Use None somente se quiser varrer todo o catálogo.
CANDIDATE_POOL_SIZE: int | None = 500
CATALOG_SEED = "tcc-cluster-search-v1"

# O resumo usa apenas informação disponível até o cutoff do snapshot U67.
ANALYSIS_START = "2020-07-22"

# Critérios mínimos de continuidade antes da construção dos 52 atributos.
MIN_HISTORY_ROWS = 2400
MAX_HISTORY_GAP_DAYS = 10.0
MAX_START_DELAY_DAYS = 60
MAX_END_DELAY_DAYS = 10

# Download individual é intencional. Uma requisição multi-símbolo com limite
# agregado pode retornar somente parte dos tickers quando se pedem ~10 anos
# de barras diárias. A primeira execução v1.0.0 revelou exatamente esse
# problema: 395/500 candidatos apareceram falsamente como "no_bars".
ALPACA_REQUEST_SLEEP_SECONDS = 0.35
MIN_BAR_RESPONSE_SHARE = 0.90
K_VALUES = tuple(range(2, 7))

# Shortlist. O resultado continua exploratório e NÃO entra automaticamente no U67.
SIMILAR_COUNT = 12
COMPLEMENTARY_COUNT = 12

# Os limites da faixa complementar passam a ser derivados somente da geometria
# interna do U67: acima do percentil 75 da distância entre vizinhos do U67 e
# abaixo do percentil 95. Assim não escolhemos outliers só porque estão longe.
REFERENCE_COMPLEMENTARY_LOW_QUANTILE = 0.75
REFERENCE_COMPLEMENTARY_HIGH_QUANTILE = 0.95

SAVE_FIGURES = True
SHOW_PLOTS = True


# %% 1 - Utilidades
def _git(args: list[str]) -> str:
    import subprocess

    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return process.stdout.strip()


def _stable_key(symbol: str) -> str:
    return hashlib.sha256(
        f"{CATALOG_SEED}|{symbol}".encode("utf-8")
    ).hexdigest()


def _max_gap_days(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float("inf")
    diffs = pd.Series(index).sort_values().diff().dropna()
    if diffs.empty:
        return float("inf")
    return float(diffs.dt.total_seconds().div(86400.0).max())


def _save_figure(fig: plt.Figure, name: str) -> None:
    if SAVE_FIGURES:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUT / f"{name}.png",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _load_catalog(credentials) -> list[dict]:
    errors = []
    for endpoint in (
        "https://paper-api.alpaca.markets/v2/assets",
        "https://api.alpaca.markets/v2/assets",
    ):
        try:
            response = requests.get(
                endpoint,
                headers=credentials.headers,
                params={"status": "active", "asset_class": "us_equity"},
                timeout=60,
            )
        except Exception as exc:
            errors.append(str(exc))
            continue
        if response.status_code != 200:
            errors.append(
                f"{response.status_code}:{response.text[:120]}"
            )
            continue
        payload = response.json()
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError("Falha ao carregar catálogo Alpaca: " + " | ".join(errors))


def _instrument_warning(name: str) -> str:
    """Sinaliza instrumentos que exigem revisão antes da shortlist.

    Não exclui da projeção. Apenas impede que uma classe claramente diferente
    seja promovida automaticamente como candidato de shortlist.
    """
    text = str(name or "").upper()
    patterns = (
        (r"\bWARRANTS?\b", "warrant"),
        (r"\bUNITS?\b", "unit"),
        (r"\bPREFERRED\b", "preferred"),
        (r"\bSENIOR NOTES?\b", "senior_notes"),
        (r"\bRIGHTS?\b", "rights"),
        (r"\bETNS?\b", "etn"),
    )
    for pattern, label in patterns:
        if re.search(pattern, text):
            return label
    return ""


def _eligible_catalog(catalog: list[dict]) -> pd.DataFrame:
    excluded = set(U67_REQUESTED_ASSETS)
    rows = []
    seen = set()

    for item in catalog:
        symbol = str(item.get("symbol") or "").strip().upper()
        exchange = str(item.get("exchange") or "").strip().upper()
        status = str(item.get("status") or "").strip().lower()

        if (
            not symbol
            or symbol in seen
            or symbol in excluded
            or SYMBOL_RE.fullmatch(symbol) is None
            or exchange not in ALLOWED_EXCHANGES
            or (status and status != "active")
            or not bool(item.get("tradable", False))
            or not bool(item.get("marginable", False))
        ):
            continue

        seen.add(symbol)
        rows.append(
            {
                "asset": symbol,
                "name": str(item.get("name") or ""),
                "exchange": exchange,
                "fractionable": bool(item.get("fractionable", False)),
                "instrument_warning": _instrument_warning(
                    str(item.get("name") or "")
                ),
                "stable_key": _stable_key(symbol),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Nenhum candidato elegível no catálogo Alpaca.")

    frame = frame.sort_values(
        ["stable_key", "asset"],
        kind="stable",
    ).reset_index(drop=True)

    if CANDIDATE_POOL_SIZE is not None:
        frame = frame.head(int(CANDIDATE_POOL_SIZE)).copy()

    return frame


def _candidate_cache_path(symbol: str) -> Path:
    return CACHE / f"{symbol}.csv"


def _read_cached_candidate(symbol: str) -> pd.DataFrame | None:
    path = _candidate_cache_path(symbol)
    if not path.exists():
        return None

    frame = pd.read_csv(
        path,
        float_precision="round_trip",
    )
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        return None

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


def _download_candidate_frames(
    credentials,
    candidates: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Carrega cache e baixa cada símbolo ausente de forma independente.

    O download individual evita uma falha observada na v1.0.0: ao solicitar
    dezenas de símbolos com ~10 anos de barras e limit=10_000, a resposta
    multi-símbolo podia conter apenas parte do lote. Os símbolos ausentes eram
    então classificados incorretamente como "no_bars".
    """
    CACHE.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for symbol in candidates["asset"].astype(str):
        cached = _read_cached_candidate(symbol)
        if cached is not None and not cached.empty:
            frames[symbol] = cached
        else:
            missing.append(symbol)

    print(
        f"[cache] reutilizados={len(frames)} baixar_individualmente={len(missing)}",
        flush=True,
    )
    if not missing:
        return frames

    client = StockHistoricalDataClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
    )
    start = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
    api_end = (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    ).to_pydatetime()

    failures: list[str] = []

    for position, symbol in enumerate(missing, start=1):
        normalized = pd.DataFrame()
        last_error = None

        for attempt in range(1, 5):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=api_end,
                    adjustment=Adjustment.SPLIT,
                    feed=DataFeed.SIP,
                    limit=10_000,
                )
                raw = client.get_stock_bars(request).df
                normalized = _normalize_alpaca_frame(raw, symbol)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                wait = min(2 ** (attempt - 1), 8)
                print(
                    f"[alpaca] {symbol} retry={attempt}/4 "
                    f"wait={wait}s error={exc}",
                    flush=True,
                )
                time.sleep(wait)

        if last_error is not None:
            failures.append(symbol)
        elif not normalized.empty:
            target = _candidate_cache_path(symbol)
            normalized.to_csv(
                target,
                index=False,
                float_format="%.17g",
            )
            prepared = normalized.set_index("timestamp")
            prepared.index = pd.to_datetime(prepared.index, utc=True)
            frames[symbol] = prepared[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)

        if position % 20 == 0 or position == len(missing):
            print(
                f"[alpaca] individuais={position}/{len(missing)} "
                f"com_barras={len(frames)} falhas_http={len(failures)}",
                flush=True,
            )

        if ALPACA_REQUEST_SLEEP_SECONDS > 0:
            time.sleep(ALPACA_REQUEST_SLEEP_SECONDS)

    response_share = len(frames) / max(len(candidates), 1)
    print(
        f"[alpaca] cobertura_de_resposta={response_share:.2%}",
        flush=True,
    )
    if response_share < MIN_BAR_RESPONSE_SHARE:
        raise RuntimeError(
            "Cobertura de barras abaixo do mínimo esperado. "
            f"recebidos={len(frames)}/{len(candidates)} "
            f"share={response_share:.2%} mínimo={MIN_BAR_RESPONSE_SHARE:.2%}. "
            "Não interpretar a shortlist; verificar acesso Alpaca/cache."
        )

    return frames


def _history_eligible(
    frame: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> tuple[bool, dict]:
    if frame.empty:
        return False, {"reason": "empty"}

    index = pd.DatetimeIndex(frame.index)
    first = pd.Timestamp(index.min())
    last = pd.Timestamp(index.max())
    expected_start = pd.Timestamp(start_date, tz="UTC")
    expected_end = pd.Timestamp(end_date, tz="UTC")

    rows = len(frame)
    max_gap = _max_gap_days(index)

    reasons = []
    if rows < MIN_HISTORY_ROWS:
        reasons.append("rows")
    if first > expected_start + pd.Timedelta(days=MAX_START_DELAY_DAYS):
        reasons.append("late_start")
    if last < expected_end - pd.Timedelta(days=MAX_END_DELAY_DAYS):
        reasons.append("early_end")
    if max_gap > MAX_HISTORY_GAP_DAYS:
        reasons.append("gap")

    return (
        not reasons,
        {
            "rows": rows,
            "first": str(first),
            "last": str(last),
            "max_gap_days": max_gap,
            "reason": ",".join(reasons),
        },
    )


def _summarize_features(
    symbol: str,
    frame: pd.DataFrame,
    *,
    analysis_start: str,
    analysis_end: str,
) -> dict | None:
    try:
        model_frame = construir_quadro_rotacao(frame, CONFIG)
    except Exception as exc:
        print(f"[features] {symbol} falhou: {exc}", flush=True)
        return None

    start = pd.Timestamp(analysis_start, tz="UTC")
    end = pd.Timestamp(analysis_end, tz="UTC")
    model_frame = model_frame.loc[
        (model_frame.index >= start)
        & (model_frame.index <= end)
    ]

    if model_frame.empty:
        return None

    values = model_frame[ROTATION_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    medians = values.median(axis=0, skipna=True)

    if medians.isna().any():
        return None

    row = {"asset": symbol}
    for feature in ROTATION_FEATURES:
        row[feature] = float(medians[feature])
    return row


# %% 2 - Referência U67
def _build_u67_summary(
    *,
    analysis_end: str,
) -> tuple[pd.DataFrame, set[str]]:
    frames, exclusions, _, _ = prepare_model_frames(
        U67,
        assets=U67_REQUESTED_ASSETS,
        csv_float_precision="round_trip",
    )

    rows = []
    for symbol, frame in frames.items():
        row = _summarize_features(
            symbol,
            frame,
            analysis_start=ANALYSIS_START,
            analysis_end=analysis_end,
        )
        if row is not None:
            rows.append(row)

    summary = pd.DataFrame(rows).set_index("asset").sort_index()
    excluded = {
        str(item.get("symbol") or "").strip().upper()
        for item in exclusions
    }

    if len(summary) < 50:
        raise RuntimeError(
            f"Referência U67 muito pequena após preparação: {len(summary)}"
        )
    return summary, excluded


def _select_k(reference_scaled: np.ndarray) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in K_VALUES:
        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50,
        ).fit_predict(reference_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": float(
                    silhouette_score(reference_scaled, labels)
                ),
            }
        )

    quality = pd.DataFrame(rows)
    best = quality.sort_values(
        ["silhouette", "k"],
        ascending=[False, True],
    ).iloc[0]
    quality["selected"] = quality["k"].eq(int(best["k"]))
    return int(best["k"]), quality


def _reference_neighbor_distribution(
    reference_scaled: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Retorna distância mediana e razões internas de vizinhança do U67."""
    diff = (
        reference_scaled[:, None, :]
        - reference_scaled[None, :, :]
    )
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    median = float(np.median(nearest))
    if not np.isfinite(median) or median <= 0:
        raise RuntimeError("Escala de vizinhança U67 inválida.")
    return median, nearest / median


# %% 3 - Projeção dos candidatos no espaço do U67
def _project_candidates(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
    tuple[float, float],
    pd.DataFrame,
]:
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(
        reference[ROTATION_FEATURES]
    )
    candidate_scaled = scaler.transform(
        candidates[ROTATION_FEATURES]
    )

    selected_k, quality = _select_k(reference_scaled)

    clustering = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=50,
    ).fit(reference_scaled)

    pca = PCA(n_components=2, random_state=42)
    reference_xy = pca.fit_transform(reference_scaled)
    candidate_xy = pca.transform(candidate_scaled)

    cluster_ids = clustering.predict(candidate_scaled)

    diff = (
        candidate_scaled[:, None, :]
        - reference_scaled[None, :, :]
    )
    distances = np.linalg.norm(diff, axis=2)
    nearest_index = distances.argmin(axis=1)
    nearest_distance = distances[
        np.arange(len(candidate_scaled)),
        nearest_index,
    ]

    typical_nn, reference_nn_ratios = _reference_neighbor_distribution(
        reference_scaled
    )
    complementary_low = float(
        np.quantile(
            reference_nn_ratios,
            REFERENCE_COMPLEMENTARY_LOW_QUANTILE,
        )
    )
    complementary_high = float(
        np.quantile(
            reference_nn_ratios,
            REFERENCE_COMPLEMENTARY_HIGH_QUANTILE,
        )
    )

    centroid_distance = np.linalg.norm(
        candidate_scaled
        - clustering.cluster_centers_[cluster_ids],
        axis=1,
    )

    out = candidates[["asset"]].copy()
    out["cluster"] = cluster_ids + 1
    out["nearest_u67_asset"] = reference.index.to_numpy()[nearest_index]
    out["nearest_u67_distance"] = nearest_distance
    out["novelty_ratio"] = nearest_distance / typical_nn
    out["cluster_centroid_distance"] = centroid_distance
    out["reference_complementary_low"] = complementary_low
    out["reference_complementary_high"] = complementary_high
    out["pc1"] = candidate_xy[:, 0]
    out["pc2"] = candidate_xy[:, 1]

    ref = pd.DataFrame(
        {
            "asset": reference.index,
            "cluster": clustering.labels_ + 1,
            "pc1": reference_xy[:, 0],
            "pc2": reference_xy[:, 1],
        }
    )

    explained = tuple(
        float(value)
        for value in pca.explained_variance_ratio_
    )
    return out, ref, selected_k, explained, quality


def _shortlist(projected: pd.DataFrame) -> pd.DataFrame:
    """Seleciona candidatos sem usar resultados financeiros.

    Semelhantes = menores distâncias ao U67.
    Complementares = região entre P75 e P95 da própria distância de vizinhança
    observada dentro do U67. Isso procura diversidade ainda compatível com o
    suporte da referência, em vez de premiar outliers extremos.
    """
    clean = projected.loc[
        projected["instrument_warning"].fillna("").eq("")
    ].copy()

    similar = (
        clean.sort_values(
            ["novelty_ratio", "cluster_centroid_distance", "asset"],
            ascending=[True, True, True],
        )
        .head(SIMILAR_COUNT)
        .copy()
    )
    similar["search_role"] = "semelhante"

    low = float(clean["reference_complementary_low"].iloc[0])
    high = float(clean["reference_complementary_high"].iloc[0])
    target = (low + high) / 2.0

    complementary = clean.loc[
        clean["novelty_ratio"].between(
            low,
            high,
            inclusive="both",
        )
    ].copy()
    complementary["distance_to_complementary_center"] = (
        complementary["novelty_ratio"] - target
    ).abs()
    complementary = (
        complementary.sort_values(
            [
                "distance_to_complementary_center",
                "cluster_centroid_distance",
                "asset",
            ],
            ascending=[True, True, True],
        )
        .head(COMPLEMENTARY_COUNT)
        .copy()
    )
    complementary["search_role"] = "complementar"

    result = pd.concat(
        [similar, complementary],
        ignore_index=True,
    )
    result = result.drop_duplicates("asset", keep="first")
    return result


# %% 4 - Gráficos
def _plot_map(
    reference_xy: pd.DataFrame,
    projected: pd.DataFrame,
    shortlist: pd.DataFrame,
    explained: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.5))

    for cluster_id in sorted(reference_xy["cluster"].unique()):
        part = reference_xy.loc[
            reference_xy["cluster"] == cluster_id
        ]
        ax.scatter(
            part["pc1"],
            part["pc2"],
            s=55,
            alpha=0.85,
            label=f"U67 grupo {int(cluster_id)}",
        )

    ax.scatter(
        projected["pc1"],
        projected["pc2"],
        s=18,
        alpha=0.18,
        marker="x",
        label="candidatos avaliados",
    )

    for row in shortlist.itertuples(index=False):
        marker = "s" if row.search_role == "semelhante" else "^"
        ax.scatter(
            [row.pc1],
            [row.pc2],
            s=95,
            marker=marker,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        ax.annotate(
            f"{row.asset} ({row.search_role})",
            (row.pc1, row.pc2),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            zorder=5,
        )

    ax.set_xlabel(
        f"Componente 1 ({100 * explained[0]:.1f}% da variação do U67)"
    )
    ax.set_ylabel(
        f"Componente 2 ({100 * explained[1]:.1f}% da variação do U67)"
    )
    ax.set_title(
        "Candidatos projetados no espaço comportamental do U67"
    )
    ax.text(
        0.0,
        1.01,
        (
            "A referência, a escala e a PCA são ajustadas somente no U67. "
            "Os candidatos não alteram o mapa de referência."
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    _save_figure(fig, "01_mapa_candidatos_u67")


def _plot_novelty(projected: pd.DataFrame, shortlist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(
        projected["novelty_ratio"],
        bins=30,
        alpha=0.65,
    )
    ax.axvline(
        COMPLEMENTARY_MIN_RATIO,
        linestyle="--",
        linewidth=1,
        label="início faixa complementar",
    )
    ax.axvline(
        COMPLEMENTARY_MAX_RATIO,
        linestyle="--",
        linewidth=1,
        label="limite de suporte",
    )
    ax.set_xlabel(
        "Distância ao ativo U67 mais próximo / distância típica entre ativos do U67"
    )
    ax.set_ylabel("Quantidade de candidatos")
    ax.set_title("Distribuição da distância comportamental dos candidatos")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    _save_figure(fig, "02_distribuicao_distancia_comportamental")


def _plot_shortlist(shortlist: pd.DataFrame) -> None:
    if shortlist.empty:
        return

    ordered = shortlist.sort_values(
        ["search_role", "novelty_ratio", "asset"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    y = np.arange(len(ordered))
    ax.barh(
        y,
        ordered["novelty_ratio"],
        alpha=0.75,
    )
    ax.set_yticks(
        y,
        [
            f"{row.asset} | {row.search_role} | perto de {row.nearest_u67_asset}"
            for row in ordered.itertuples(index=False)
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Distância relativa ao U67")
    ax.set_title(
        "Shortlist para validação posterior de substituição"
    )
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    _save_figure(fig, "03_shortlist_cluster_search")


# %% 5 - Execução
def main() -> None:
    try:
        branch = _git(["branch", "--show-current"])
        commit = _git(["rev-parse", "--short", "HEAD"])
    except Exception:
        branch = "sem-git"
        commit = "sem-git"

    print("=" * 78)
    print("TCC - BUSCA DE CANDIDATOS POR AGRUPAMENTO COMPORTAMENTAL")
    print(f"version={SCRIPT_VERSION}")
    print(f"branch={branch}")
    print(f"commit={commit}")
    print(f"arquivo={Path(__file__).resolve()}")
    print("=" * 78)

    if branch not in (EXPECTED_BRANCH, "sem-git"):
        raise RuntimeError(
            f"Branch incorreta: esperado={EXPECTED_BRANCH} observado={branch}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = validate_snapshot(U67)
    start_date = str((manifest.get("bars") or {}).get("start") or "2016-01-01")
    analysis_end = str(
        (manifest.get("bars") or {}).get("bar_snapshot_as_of_end")
        or "2026-10-06"
    )

    print(
        f"[reference] start={start_date} analysis_end={analysis_end}",
        flush=True,
    )

    reference, excluded = _build_u67_summary(
        analysis_end=analysis_end,
    )
    print(
        f"[reference] representados={len(reference)} "
        f"exclusões_estruturais={sorted(excluded)}",
        flush=True,
    )

    credentials = load_alpaca_credentials(ROOT)
    catalog = _eligible_catalog(_load_catalog(credentials))
    catalog.to_csv(
        OUT / "candidate_catalog_sample.csv",
        index=False,
    )
    print(
        f"[catalog] candidatos_amostrados={len(catalog)} "
        f"limite={CANDIDATE_POOL_SIZE}",
        flush=True,
    )

    raw_frames = _download_candidate_frames(
        credentials,
        catalog,
        start_date=start_date,
        end_date=analysis_end,
    )

    quality_rows = []
    candidate_rows = []

    for position, row in enumerate(
        catalog.itertuples(index=False),
        start=1,
    ):
        symbol = row.asset
        frame = raw_frames.get(symbol)
        if frame is None or frame.empty:
            quality_rows.append(
                {
                    "asset": symbol,
                    "eligible": False,
                    "reason": "no_bars",
                }
            )
            continue

        eligible, quality = _history_eligible(
            frame,
            start_date=start_date,
            end_date=analysis_end,
        )
        quality_rows.append(
            {
                "asset": symbol,
                "eligible": bool(eligible),
                **quality,
            }
        )
        if not eligible:
            continue

        summary = _summarize_features(
            symbol,
            frame,
            analysis_start=ANALYSIS_START,
            analysis_end=analysis_end,
        )
        if summary is not None:
            candidate_rows.append(summary)

        if position % 50 == 0:
            print(
                f"[features] {position}/{len(catalog)} "
                f"válidos={len(candidate_rows)}",
                flush=True,
            )

    quality_frame = pd.DataFrame(quality_rows)
    quality_frame.to_csv(
        OUT / "candidate_history_quality.csv",
        index=False,
    )

    if len(candidate_rows) < 20:
        raise RuntimeError(
            f"Apenas {len(candidate_rows)} candidatos com 52 atributos válidos."
        )

    candidates = (
        pd.DataFrame(candidate_rows)
        .set_index("asset")
        .sort_index()
    )
    candidates.to_csv(
        OUT / "candidate_feature_summary.csv",
        float_format="%.17g",
    )

    projected, reference_xy, selected_k, explained, quality = (
        _project_candidates(reference, candidates.reset_index())
    )

    metadata = catalog[
        [
            "asset",
            "name",
            "exchange",
            "fractionable",
            "instrument_warning",
        ]
    ]
    projected = projected.merge(
        metadata,
        on="asset",
        how="left",
        validate="one_to_one",
    )

    shortlist = _shortlist(projected)

    quality.to_csv(
        OUT / "cluster_quality_u67.csv",
        index=False,
        float_format="%.17g",
    )
    reference_xy.to_csv(
        OUT / "u67_cluster_reference.csv",
        index=False,
        float_format="%.17g",
    )
    projected.to_csv(
        OUT / "candidate_projection.csv",
        index=False,
        float_format="%.17g",
    )
    shortlist.to_csv(
        OUT / "candidatos_cluster_shortlist.csv",
        index=False,
        float_format="%.17g",
    )

    _plot_map(
        reference_xy,
        projected,
        shortlist,
        explained,
    )
    _plot_novelty(
        projected,
        shortlist,
    )
    _plot_shortlist(
        shortlist,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "branch": branch,
        "commit": commit,
        "reference_requested_assets": len(U67_REQUESTED_ASSETS),
        "reference_represented_assets": len(reference),
        "reference_structural_exclusions": sorted(excluded),
        "candidate_catalog_sample": len(catalog),
        "candidate_frames_downloaded_or_cached": len(raw_frames),
        "candidate_bar_response_share": len(raw_frames) / max(len(catalog), 1),
        "candidate_feature_valid": len(candidates),
        "selected_k": selected_k,
        "pca_explained_pc1_pc2": list(explained),
        "similar_count": int(
            (shortlist["search_role"] == "semelhante").sum()
        ),
        "complementary_count": int(
            (shortlist["search_role"] == "complementar").sum()
        ),
        "reference_complementary_low": float(
            projected["reference_complementary_low"].iloc[0]
        ),
        "reference_complementary_high": float(
            projected["reference_complementary_high"].iloc[0]
        ),
        "financial_outcomes_used_for_selection": False,
        "backtest_executed": False,
        "models_trained": False,
        "next_step": (
            "validate shortlist at substitution-event level before considering "
            "any universe inclusion"
        ),
    }
    (OUT / "candidate_cluster_search.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nShortlist:")
    columns = [
        "asset",
        "search_role",
        "cluster",
        "nearest_u67_asset",
        "novelty_ratio",
        "cluster_centroid_distance",
        "exchange",
    ]
    print(
        shortlist[columns]
        .sort_values(
            ["search_role", "novelty_ratio"],
            ascending=[True, True],
        )
        .to_string(index=False)
    )

    print(
        "\nIMPORTANTE: esta shortlist é somente um filtro de descoberta. "
        "Nenhum ativo deve ser incorporado ao universo com base neste resultado."
    )
    print(
        "Próxima etapa: medir, para cada candidato, a qualidade da substituição "
        "do ativo atualmente mantido usando apenas informação disponível na data."
    )
    print(f"\nSaídas: {OUT}")


if __name__ == "__main__":
    main()
