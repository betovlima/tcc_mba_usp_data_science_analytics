"""Pesquisa incremental U67+n com validacao da decisao e do universo dinamico.

A versao 1.4.0 parte dos checkpoints reproduzidos pela assurance v1.3.0 e
avalia se o LightGBM e util para a decisao relativa entre ativos, mesmo quando
MAE/RMSE absolutos mostram fraca generalizacao. Tambem mede o risco de
selection overfitting da busca de subconjuntos com Deflated Sharpe Ratio e
Probability of Backtest Overfitting.

A lista de ativos permanece um estado dinamico da estrategia. O piloto temporal
registra transicoes ADICIONAR, MANTER e REMOVER entre universos escolhidos
somente com folds anteriores. A redescoberta diaria completa dos candidatos
fica para a etapa posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from statistics import NormalDist
from typing import Any
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import requests
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

from engine.configuracao import (
    CONFIG,
    START_DATE,
    U67_EXPECTED_EFFECTIVE_COUNT,
    U67_EXPECTED_EXCLUSIONS,
    U67_EXPECTED_REQUESTED_COUNT,
    U67_REQUESTED_ASSETS,
    construir_configuracao_controle,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    diagnosticos_fora_amostra,
    selecionar_switch_margin,
)
from engine.rotacao import (
    ROTATION_FEATURES,
    _crescimento_politica_simples,
    _politica_agendada,
    _precalcular_utilidades_modelo,
    _politica_utilidade,
    _simular_exato,
    construir_quadro_rotacao,
)
from reproducao.artefatos import criar_pacote_analise, sinal_sonoro_conclusao
from reproducao.dados import (
    SnapshotPaths,
    _normalize_alpaca_frame,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import (
    load_actions_file,
    load_raw_bar_file,
    prepare_model_frames,
    split_normalize,
    structural_identity_issue,
)


# %% 0 - Parametros da campanha
ROOT = Path(__file__).resolve().parents[1]
RUN_MODE = "decision_validation"
BASE_DATA = SnapshotPaths.u67(ROOT)
CACHE_ROOT = ROOT / "dados" / ".u67_expansion_cache"
FAST_CACHE = CACHE_ROOT / "fast_split_20261006"
FULL_DATA = SnapshotPaths.from_root(CACHE_ROOT / "full_raw_20261006")
OUTPUT_DIR_BY_MODE = {
    "decision_validation": "u67_decision_validation",
    "assurance_validation": "u67_assurance_validation",
    "subset_optimization": "u67_subset_optimization",
    "direct_marginal": "u67_asset_expansion",
}
OUT = ROOT / "output" / OUTPUT_DIR_BY_MODE[RUN_MODE]
CANDIDATE_SAMPLE_MANIFEST = ROOT / "pesquisa_u67_amostra_500.csv"

SCRIPT_VERSION = "u67-asset-expansion-v1.4.1"
EXECUTION_SCHEMA = "u67-asset-expansion-v5"
ANALYSIS_END_DATE = "2026-10-06"
EXPECTED_U67_ENDING_CAPITAL = 76_927_051.38897176
EXPECTED_ALL_POSITIVE_ENDING_CAPITAL = 290_727_954.41339606
EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL = 587_165_941.1330212
BASELINE_ABS_TOL = 0.01
SUBSET_IMPROVEMENT_TOL = 0.01
SOURCE_DIRECT_TESTED_COUNT = 146
SOURCE_SUBSET_UNIQUE_SCENARIOS = 180
PBO_BLOCKS = 10
PBO_SPLIT_COUNT = math.comb(PBO_BLOCKS, PBO_BLOCKS // 2)
SUBSET_SOURCE_CANDIDATES = (
    "CRT",
    "NBR",
    "CAPR",
    "PNBK",
    "PENN",
    "JBL",
    "LEU",
    "JBSS",
    "STAG",
    "MCHB",
    "EML",
    "MAR",
    "LAD",
    "TOWN",
    "ACIW",
)
DISCOVERY_SELECTED_CANDIDATES = (
    "CRT",
    "NBR",
    "CAPR",
    "PNBK",
    "PENN",
    "JBL",
    "STAG",
    "MCHB",
    "LAD",
    "TOWN",
)

CATALOG_SAMPLE_SEED = "u67-asset-expansion-v1"
CANDIDATE_SCAN_BUDGET = 500
RANKED_REPLAY_LIMIT = 24
FAST_HISTORY_DAYS = 900
FAST_REQUEST_SLEEP_SECONDS = 0.25

ALLOWED_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"})
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
MIN_LATEST_PRICE = 5.0
MIN_MEDIAN_DOLLAR_VOLUME = 10_000_000.0
MIN_NONZERO_VOLUME_RATIO = 0.98
MIN_FAST_FEATURE_SESSIONS = 126
HISTORY_START_TOLERANCE_DAYS = 10

RANKER_FEATURE_COLUMNS = (
    "return_5",
    "return_10",
    "return_20",
    "return_40",
    "return_60",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "drawdown_20",
    "drawdown_60",
    "position_60",
    "trend_efficiency_20",
    "volume_ratio_20",
    "dollar_volume_log",
)
RANKER_TARGET_HORIZON = 20
RANKER_VALIDATION_FOLDS = 4
RANKER_INITIAL_TRAIN_FRACTION = 0.50
RANKER_RANDOM_BASELINE_REPEATS = 24


@dataclass(frozen=True)
class BaselineFoldState:
    fold_id: int
    train_dates: pd.DatetimeIndex
    calibration_dates: pd.DatetimeIndex
    final_fit_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex
    calibration_cache: dict[pd.Timestamp, np.ndarray]
    decision_cache: dict[pd.Timestamp, np.ndarray]
    selected_margin: float
    effective_margin: float
    fit_diagnostics: dict[str, Any] | None = None
    oos_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateFoldCache:
    fold_id: int
    calibration_cache: dict[pd.Timestamp, np.ndarray]
    decision_cache: dict[pd.Timestamp, np.ndarray]
    fit_diagnostics: dict[str, Any] | None = None
    oos_diagnostics: dict[str, Any] | None = None


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(snapshot_sha: str, symbol: str) -> str:
    material = f"{CATALOG_SAMPLE_SEED}|{snapshot_sha}|{ANALYSIS_END_DATE}|{symbol}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Converte escalares nao finitos para null antes de serializar JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _capital_delta(candidate: float | None, baseline: float | None) -> tuple[float | None, float | None]:
    if candidate is None or baseline in (None, 0.0):
        return None, None
    delta = float(candidate - baseline)
    return delta, float(candidate / baseline - 1.0)


def _numeric_delta(candidate: Any, baseline: Any) -> float | None:
    left = _finite(candidate)
    right = _finite(baseline)
    if left is None or right is None:
        return None
    return float(left - right)


# %% 1 - Learning-to-Rank adaptado da busca de ativos operacional
# A arquitetura abaixo replica a ideia central da busca de ativos operacional:
# LambdaMART, atributos de retorno/risco/liquidez e validacao cronologica purgada.
def _ranker_clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    result = frame.copy().sort_index()
    result = result[~result.index.duplicated(keep="last")]
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = result[result["volume"] >= 0]
    return result


def _ranker_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source = _ranker_clean_frame(frame)
    if source.empty:
        return pd.DataFrame(columns=list(RANKER_FEATURE_COLUMNS))
    close = source["close"].astype(float)
    volume = source["volume"].astype(float)
    ret1 = close.pct_change()
    features = pd.DataFrame(index=source.index)
    for horizon in (5, 10, 20, 40, 60):
        features[f"return_{horizon}"] = close.pct_change(horizon)
    for horizon in (10, 20, 60):
        features[f"volatility_{horizon}"] = ret1.rolling(horizon).std(ddof=0) * math.sqrt(252.0)
    high20 = close.rolling(20).max()
    high60 = close.rolling(60).max()
    low60 = close.rolling(60).min()
    features["drawdown_20"] = close / high20 - 1.0
    features["drawdown_60"] = close / high60 - 1.0
    spread60 = (high60 - low60).replace(0.0, np.nan)
    features["position_60"] = (close - low60) / spread60
    gross20 = ret1.abs().rolling(20).sum().replace(0.0, np.nan)
    features["trend_efficiency_20"] = close.pct_change(20).abs() / gross20
    median_volume20 = volume.rolling(20).median().replace(0.0, np.nan)
    features["volume_ratio_20"] = volume / median_volume20
    features["dollar_volume_log"] = np.log1p((close * volume).clip(lower=0.0))
    return features.replace([np.inf, -np.inf], np.nan)


def _ranker_future_utility(frame: pd.DataFrame) -> pd.Series:
    source = _ranker_clean_frame(frame)
    close = source["close"].astype(float)
    forward_return = close.shift(-RANKER_TARGET_HORIZON) / close - 1.0
    path = pd.concat(
        [
            (close.shift(-step) / close - 1.0).rename(step)
            for step in range(1, RANKER_TARGET_HORIZON + 1)
        ],
        axis=1,
    )
    adverse = path.min(axis=1)
    return forward_return + 0.45 * adverse.clip(upper=0.0)


def _ranker_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        features = _ranker_feature_frame(frame)
        if features.empty:
            continue
        combined = features.copy()
        combined["utility"] = _ranker_future_utility(frame).reindex(features.index)
        combined["symbol"] = str(symbol).upper()
        combined["date"] = pd.to_datetime(combined.index, utc=True).normalize()
        rows.append(combined.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.dropna(subset=[*RANKER_FEATURE_COLUMNS, "utility", "date"])
    if dataset.empty:
        return dataset
    group_count = dataset.groupby("date")["symbol"].transform("count")
    dataset = dataset.loc[group_count >= 3].copy()
    percentile = dataset.groupby("date")["utility"].rank(method="average", pct=True)
    dataset["relevance"] = np.minimum(np.floor(percentile * 5.0), 4).astype(int)
    return dataset.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _ranker_group_sizes(dataset: pd.DataFrame) -> list[int]:
    return dataset.groupby("date", sort=False).size().astype(int).tolist()


def _average_ndcg(dataset: pd.DataFrame, predictions: np.ndarray, k: int = 5) -> float | None:
    if dataset.empty or len(dataset) != len(predictions):
        return None
    values: list[float] = []
    offset = 0
    for _, group in dataset.groupby("date", sort=False):
        size = len(group)
        truth = group["relevance"].to_numpy(dtype=float)
        pred = np.asarray(predictions[offset : offset + size], dtype=float)
        offset += size
        if size < 2:
            continue
        values.append(float(ndcg_score([truth], [pred], k=min(k, size))))
    return float(np.mean(values)) if values else None


def _new_ranker(seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=int(seed),
        n_jobs=1,
        verbosity=-1,
    )


def _random_ndcg(dataset: pd.DataFrame, seed: int) -> float | None:
    rng = np.random.default_rng(int(seed))
    scores = []
    for _ in range(RANKER_RANDOM_BASELINE_REPEATS):
        score = _average_ndcg(dataset, rng.random(len(dataset)), 5)
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else None


def _train_ranker(frames: dict[str, pd.DataFrame], seed: int) -> tuple[LGBMRanker, dict[str, Any]]:
    dataset = _ranker_dataset(frames)
    unique_dates = list(pd.unique(dataset["date"])) if not dataset.empty else []
    if len(unique_dates) < 360 or len(dataset) < 2_000:
        raise RuntimeError("U67 insuficiente para treinar o Learning-to-Rank.")

    initial = max(180, int(len(unique_dates) * RANKER_INITIAL_TRAIN_FRACTION))
    boundaries = np.linspace(initial, len(unique_dates), RANKER_VALIDATION_FOLDS + 1, dtype=int)
    validation_rows = []
    for fold_index in range(RANKER_VALIDATION_FOLDS):
        validation_start = int(boundaries[fold_index])
        validation_end = int(boundaries[fold_index + 1])
        purge_end = max(0, validation_start - RANKER_TARGET_HORIZON)
        train_dates = unique_dates[:purge_end]
        valid_dates = unique_dates[validation_start:validation_end]
        train = dataset[dataset["date"].isin(set(train_dates))].copy()
        valid = dataset[dataset["date"].isin(set(valid_dates))].copy()
        if len(train_dates) < 180 or len(valid_dates) < 20 or train.empty or valid.empty:
            raise RuntimeError("Nao foi possivel construir os folds cronologicos do Learning-to-Rank.")
        fold_seed = int(seed) + fold_index + 1
        model = _new_ranker(fold_seed)
        model.fit(
            train[list(RANKER_FEATURE_COLUMNS)],
            train["relevance"],
            group=_ranker_group_sizes(train),
        )
        train_ndcg = _average_ndcg(train, model.predict(train[list(RANKER_FEATURE_COLUMNS)]), 5)
        valid_ndcg = _average_ndcg(valid, model.predict(valid[list(RANKER_FEATURE_COLUMNS)]), 5)
        momentum_ndcg = _average_ndcg(valid, valid["return_20"].to_numpy(dtype=float), 5)
        random_ndcg = _random_ndcg(valid, fold_seed + 10_000)
        validation_rows.append(
            {
                "fold": fold_index + 1,
                "train_start": pd.Timestamp(train_dates[0]).date().isoformat(),
                "train_end": pd.Timestamp(train_dates[-1]).date().isoformat(),
                "validation_start": pd.Timestamp(valid_dates[0]).date().isoformat(),
                "validation_end": pd.Timestamp(valid_dates[-1]).date().isoformat(),
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "train_ndcg_at_5": train_ndcg,
                "ranker_ndcg_at_5": valid_ndcg,
                "momentum_ndcg_at_5": momentum_ndcg,
                "random_ndcg_at_5": random_ndcg,
            }
        )
        print(
            f"[ranker] fold={fold_index + 1}/{RANKER_VALIDATION_FOLDS} "
            f"ndcg={valid_ndcg} momentum={momentum_ndcg} random={random_ndcg}",
            flush=True,
        )

    final_model = _new_ranker(seed)
    final_model.fit(
        dataset[list(RANKER_FEATURE_COLUMNS)],
        dataset["relevance"],
        group=_ranker_group_sizes(dataset),
    )
    valid_scores = [row["ranker_ndcg_at_5"] for row in validation_rows if row["ranker_ndcg_at_5"] is not None]
    diagnostics = {
        "family": "lightgbm_lambdamart",
        "objective": "lambdarank",
        "source": "operational asset-discovery architecture adapted to frozen U67 research",
        "target_horizon_sessions": RANKER_TARGET_HORIZON,
        "features": list(RANKER_FEATURE_COLUMNS),
        "validation_method": "purged_expanding_walk_forward",
        "validation_purge_sessions": RANKER_TARGET_HORIZON,
        "validation_folds": validation_rows,
        "median_oos_ndcg_at_5": float(np.median(valid_scores)) if valid_scores else None,
        "refit_rows": int(len(dataset)),
        "refit_sessions": int(dataset["date"].nunique()),
    }
    return final_model, diagnostics


def _latest_ranker_features(frame: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    features = _ranker_feature_frame(frame).dropna(subset=list(RANKER_FEATURE_COLUMNS))
    if features.empty:
        raise RuntimeError("insufficient_ranker_history")
    stamp = pd.Timestamp(features.index[-1])
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return features.iloc[-1], stamp


def _market_quality(frame: pd.DataFrame) -> dict[str, float]:
    source = _ranker_clean_frame(frame)
    if len(source) < MIN_FAST_FEATURE_SESSIONS:
        raise RuntimeError("insufficient_fast_history")
    recent = source.tail(min(63, len(source)))
    close = recent["close"].astype(float)
    volume = recent["volume"].astype(float)
    return {
        "latest_close": float(close.iloc[-1]),
        "median_dollar_volume": float((close * volume).median()),
        "nonzero_volume_ratio": float((volume > 0).mean()),
    }


def _baseline_recent_returns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    for symbol, frame in frames.items():
        series.append(pd.to_numeric(frame["close"], errors="coerce").pct_change().tail(90).rename(symbol))
    return pd.concat(series, axis=1) if series else pd.DataFrame()


def _redundancy(candidate: pd.DataFrame, baseline_returns: pd.DataFrame) -> float | None:
    if baseline_returns.empty or candidate is None or candidate.empty:
        return None
    candidate_returns = pd.to_numeric(candidate["close"], errors="coerce").pct_change().rename("candidate")
    joined = baseline_returns.join(candidate_returns, how="inner").tail(60)
    if len(joined) < 30:
        return None
    corr = joined.drop(columns=["candidate"]).corrwith(joined["candidate"]).abs().dropna()
    return float(corr.max()) if not corr.empty else None


# %% 2 - Catalogo e fast scan Alpaca
def _load_catalog(credentials: Any) -> list[dict[str, Any]]:
    endpoints = (
        "https://paper-api.alpaca.markets/v2/assets",
        "https://api.alpaca.markets/v2/assets",
    )
    errors = []
    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint,
                params={"status": "active", "asset_class": "us_equity"},
                headers=credentials.headers,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("Falha ao consultar catalogo Alpaca: " + " | ".join(errors))


def _eligible_catalog(catalog: list[dict[str, Any]], snapshot_sha: str) -> pd.DataFrame:
    excluded = set(U67_REQUESTED_ASSETS) | set(U67_EXPECTED_EXCLUSIONS)
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
            or SYMBOL_PATTERN.fullmatch(symbol) is None
            or exchange not in ALLOWED_EXCHANGES
            or (status and status != "active")
            or not bool(item.get("tradable", False))
        ):
            continue
        seen.add(symbol)
        rows.append(
            {
                "asset": symbol,
                "name": str(item.get("name") or ""),
                "exchange": exchange,
                "asset_id": str(item.get("id") or ""),
                "cusip": str(item.get("cusip") or ""),
                "fractionable": bool(item.get("fractionable", False)),
                "stable_key": _stable_key(snapshot_sha, symbol),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Catalogo Alpaca nao produziu candidatos externos elegiveis.")
    return frame.sort_values(["stable_key", "asset"], kind="stable").reset_index(drop=True)


def _load_candidate_sample(catalog: pd.DataFrame) -> pd.DataFrame:
    if not CANDIDATE_SAMPLE_MANIFEST.exists():
        raise RuntimeError(
            f"Amostra congelada ausente: {CANDIDATE_SAMPLE_MANIFEST.name}"
        )
    manifest = pd.read_csv(CANDIDATE_SAMPLE_MANIFEST)
    if "asset" not in manifest.columns:
        raise RuntimeError("Amostra congelada deve conter a coluna asset.")
    manifest = manifest[["asset"]].copy()
    manifest["asset"] = manifest["asset"].astype(str).str.strip().str.upper()
    manifest = manifest.loc[manifest["asset"].ne("")].reset_index(drop=True)
    if len(manifest) != CANDIDATE_SCAN_BUDGET:
        raise RuntimeError(
            "Amostra congelada divergente: "
            f"observado={len(manifest)} esperado={CANDIDATE_SCAN_BUDGET}"
        )
    if manifest["asset"].duplicated().any():
        duplicated = sorted(manifest.loc[manifest["asset"].duplicated(), "asset"].unique())
        raise RuntimeError(f"Amostra congelada contem duplicatas: {duplicated[:10]}")
    manifest["sample_order"] = np.arange(1, len(manifest) + 1)

    metadata = catalog.drop_duplicates("asset", keep="first").copy()
    sample = manifest.merge(metadata, on="asset", how="left", validate="one_to_one")
    sample["catalog_currently_eligible"] = sample["stable_key"].notna()
    return sample


def _fast_cache_path(symbol: str) -> Path:
    return FAST_CACHE / f"{symbol}.csv"


def _read_fast_cache(symbol: str) -> pd.DataFrame | None:
    path = _fast_cache_path(symbol)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            return None
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        last = pd.Timestamp(frame.index.max())
        cutoff = pd.Timestamp(ANALYSIS_END_DATE, tz="UTC")
        if last.normalize() < cutoff.normalize():
            return None
        return frame[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception:
        return None


def _download_fast_frame(client: StockHistoricalDataClient, symbol: str) -> pd.DataFrame:
    cached = _read_fast_cache(symbol)
    if cached is not None and not cached.empty:
        return cached
    FAST_CACHE.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(ANALYSIS_END_DATE, tz="UTC")
    start = (cutoff - pd.Timedelta(days=FAST_HISTORY_DAYS)).to_pydatetime()
    end = (cutoff + pd.Timedelta(days=1)).to_pydatetime()
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment=Adjustment.SPLIT,
                feed=DataFeed.SIP,
                limit=10_000,
            )
            normalized = _normalize_alpaca_frame(client.get_stock_bars(request).df, symbol)
            if normalized.empty:
                raise RuntimeError("no_bars")
            normalized.to_csv(_fast_cache_path(symbol), index=False, float_format="%.17g")
            prepared = normalized.set_index("timestamp")
            prepared.index = pd.to_datetime(prepared.index, utc=True)
            return prepared[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                wait = min(2 ** (attempt - 1), 8)
                time.sleep(wait)
    raise RuntimeError(f"fast_scan_failed: {last_error}")


def _scan_and_rank_candidates(
    credentials: Any,
    catalog: pd.DataFrame,
    ranker: LGBMRanker,
    baseline_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = catalog.head(min(CANDIDATE_SCAN_BUDGET, len(catalog))).copy()
    client = StockHistoricalDataClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
    )
    screening_rows = []
    ranked_rows = []
    for position, row in enumerate(sample.itertuples(index=False), start=1):
        symbol = str(row.asset)
        screening: dict[str, Any] = {
            "asset": symbol,
            "name": row.name,
            "exchange": row.exchange,
            "status": "rejected",
            "reason": None,
        }
        try:
            frame = _download_fast_frame(client, symbol)
            quality = _market_quality(frame)
            screening.update(quality)
            if quality["latest_close"] < MIN_LATEST_PRICE:
                raise RuntimeError("price_below_minimum")
            if quality["median_dollar_volume"] < MIN_MEDIAN_DOLLAR_VOLUME:
                raise RuntimeError("dollar_volume_below_minimum")
            if quality["nonzero_volume_ratio"] < MIN_NONZERO_VOLUME_RATIO:
                raise RuntimeError("nonzero_volume_ratio_below_minimum")
            feature_row, feature_at = _latest_ranker_features(frame)
            vector = pd.DataFrame(
                [[float(feature_row[column]) for column in RANKER_FEATURE_COLUMNS]],
                columns=list(RANKER_FEATURE_COLUMNS),
            )
            raw_score = float(ranker.predict(vector)[0])
            screening.update({"status": "passed", "reason": None, "raw_score": raw_score})
            ranked_rows.append(
                {
                    "asset": symbol,
                    "name": row.name,
                    "exchange": row.exchange,
                    "asset_id": row.asset_id,
                    "cusip": row.cusip,
                    "raw_score": raw_score,
                    "feature_at": feature_at.isoformat(),
                    **quality,
                    "return_20": float(feature_row["return_20"]),
                    "return_60": float(feature_row["return_60"]),
                    "volatility_20": float(feature_row["volatility_20"]),
                    "drawdown_60": float(feature_row["drawdown_60"]),
                    "trend_efficiency_20": float(feature_row["trend_efficiency_20"]),
                    "max_baseline_correlation_60": _redundancy(frame, baseline_returns),
                }
            )
        except RuntimeError as exc:
            screening["reason"] = str(exc)[:240]
        except Exception as exc:
            screening["reason"] = f"technical_failure: {str(exc)[:200]}"
        screening_rows.append(screening)
        if position % 20 == 0 or position == len(sample):
            print(
                f"[fast-scan] {position}/{len(sample)} passed={len(ranked_rows)}",
                flush=True,
            )
        if FAST_REQUEST_SLEEP_SECONDS > 0:
            time.sleep(FAST_REQUEST_SLEEP_SECONDS)

    screening = pd.DataFrame(screening_rows)
    ranked = pd.DataFrame(ranked_rows)
    if ranked.empty:
        return screening, ranked
    ranked = ranked.sort_values(["raw_score", "asset"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["rank_score"] = 1.0 if len(ranked) == 1 else 1.0 - (ranked["rank"] - 1) / (len(ranked) - 1)
    ranked["replay_selected"] = ranked["rank"] <= min(RANKED_REPLAY_LIMIT, len(ranked))
    return screening, ranked


# %% 3 - Historico RAW completo, Corporate Actions e continuidade
def _full_raw_cache_valid(symbol: str) -> bool:
    path = FULL_DATA.raw_bars / f"{symbol}.csv"
    if not path.exists():
        return False
    try:
        frame = load_raw_bar_file(path, float_precision="round_trip")
    except Exception:
        return False
    if frame.empty:
        return False
    last = pd.Timestamp(frame.index.max()).tz_convert("UTC")
    cutoff = pd.Timestamp(ANALYSIS_END_DATE, tz="UTC")
    # Historico curto e uma propriedade a ser auditada, nao motivo para baixar
    # novamente o mesmo arquivo em toda execucao.
    return last.normalize() >= cutoff.normalize()


def _prepare_full_candidate_files(credentials: Any, symbols: list[str]) -> pd.DataFrame:
    """Baixa RAW por candidato sem abortar toda a campanha por um unico ativo."""
    FULL_DATA.ensure()
    rows: list[dict[str, Any]] = []
    raw_ready: list[str] = []
    for position, symbol in enumerate(symbols, start=1):
        row: dict[str, Any] = {
            "asset": symbol,
            "status": "pending",
            "reason": None,
        }
        if _full_raw_cache_valid(symbol):
            row["status"] = "cached"
            raw_ready.append(symbol)
        else:
            last_error: Exception | None = None
            for attempt in range(1, 5):
                try:
                    download_raw_bars(
                        credentials,
                        FULL_DATA,
                        assets=(symbol,),
                        bar_snapshot_as_of_end=ANALYSIS_END_DATE,
                        analysis_end_date=ANALYSIS_END_DATE,
                    )
                    row["status"] = "downloaded"
                    raw_ready.append(symbol)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 4:
                        time.sleep(min(2 ** (attempt - 1), 8))
            if last_error is not None:
                row["status"] = "failed"
                row["reason"] = str(last_error)[:500]
        rows.append(row)
        print(
            f"[full-data] {position}/{len(symbols)} {symbol} status={row['status']}",
            flush=True,
        )

    # A consulta e paginada/chunked pelo helper oficial. Somente ativos com RAW
    # disponivel participam da auditoria estrutural subsequente.
    if raw_ready:
        download_corporate_actions(
            credentials,
            FULL_DATA,
            assets=tuple(raw_ready),
            query_end=ANALYSIS_END_DATE,
        )
    return pd.DataFrame(rows)


def _generic_identity_issue(symbol: str, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    known = structural_identity_issue(symbol, actions)
    if known is not None:
        return known
    normalized = symbol.strip().upper()
    for action in actions:
        action_type = str(action.get("action_type") or "").strip().lower()
        old_symbol = str(action.get("old_symbol") or "").strip().upper()
        new_symbol = str(action.get("new_symbol") or "").strip().upper()
        old_cusip = str(action.get("old_cusip") or "").strip().upper()
        new_cusip = str(action.get("new_cusip") or "").strip().upper()
        source_symbol = str(action.get("source_symbol") or "").strip().upper()
        event_symbol = str(action.get("symbol") or "").strip().upper()
        acquiree = str(action.get("acquiree_symbol") or "").strip().upper()
        acquirer = str(action.get("acquirer_symbol") or "").strip().upper()
        aliases = {value for value in (event_symbol, source_symbol, old_symbol, new_symbol, acquiree, acquirer) if value}
        if normalized not in aliases:
            continue
        breaks = False
        if action_type in {"cash_merger", "stock_merger", "stock_and_cash_merger"}:
            breaks = bool(acquiree == normalized and acquirer and acquirer != normalized)
        elif action_type in {"redemption", "worthless_removal"}:
            breaks = normalized in {event_symbol, source_symbol, old_symbol}
        elif action_type == "name_change":
            breaks = bool(
                (old_symbol == normalized and new_symbol and old_symbol != new_symbol)
                or (old_cusip and new_cusip and old_cusip != new_cusip)
            )
        elif action_type == "reverse_split" and old_symbol == normalized and new_symbol and old_symbol != new_symbol:
            breaks = True
        if breaks:
            return {
                "symbol": normalized,
                "reason": "structural_identity_change",
                "action_type": action_type,
                "process_date": action.get("process_date"),
                "effective_date": action.get("effective_date"),
                "old_symbol": old_symbol or None,
                "new_symbol": new_symbol or None,
                "old_cusip": old_cusip or None,
                "new_cusip": new_cusip or None,
            }
    return None


def _continuity_audit(
    symbol: str,
    normalized_raw: pd.DataFrame,
    required_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(pd.to_datetime(normalized_raw.index, utc=True)).normalize().unique().sort_values()
    required = pd.DatetimeIndex(pd.to_datetime(required_sessions, utc=True)).normalize().unique().sort_values()
    if index.empty or required.empty:
        return {"asset": symbol, "status": "rejected", "reason": "empty_history"}
    first = pd.Timestamp(index.min())
    last = pd.Timestamp(index.max())
    latest_start = pd.Timestamp(START_DATE, tz="UTC") + pd.Timedelta(days=HISTORY_START_TOLERANCE_DAYS)
    missing = required.difference(index)
    positions = sorted(int(required.get_loc(item)) for item in missing if item in required)
    longest = 0
    current = 0
    previous = None
    for pos in positions:
        if previous is not None and pos == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = pos
    reason = None
    if first.normalize() > latest_start.normalize():
        reason = "insufficient_history_start"
    elif len(missing):
        reason = "discontinuous_history"
    return {
        "asset": symbol,
        "status": "passed" if reason is None else "rejected",
        "reason": reason,
        "history_actual_start": first.isoformat(),
        "history_actual_end": last.isoformat(),
        "required_session_count": int(len(required)),
        "missing_required_sessions": int(len(missing)),
        "longest_missing_run": int(longest),
    }


def _prepare_candidate_frames(
    symbols: list[str],
    common_dates: pd.DatetimeIndex,
    all_decision_dates: pd.DatetimeIndex,
    config: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, str]]]:
    frames: dict[str, pd.DataFrame] = {}
    audits = []
    hashes: dict[str, dict[str, str]] = {}
    for position, symbol in enumerate(symbols, start=1):
        raw_path = FULL_DATA.raw_bars / f"{symbol}.csv"
        action_path = FULL_DATA.corporate_actions / f"{symbol}.csv"
        row: dict[str, Any] = {"asset": symbol, "status": "rejected", "reason": None}
        try:
            raw = load_raw_bar_file(raw_path, float_precision="round_trip")
            actions = load_actions_file(action_path)
            issue = _generic_identity_issue(symbol, actions)
            if issue is not None:
                row.update(
                    {
                        "reason": "structural_identity_change",
                        "action_type": issue.get("action_type"),
                        "identity_detail": json.dumps(issue, ensure_ascii=False, default=str),
                    }
                )
                audits.append(row)
                continue
            normalized, applied = split_normalize(raw, actions)
            continuity = _continuity_audit(symbol, normalized, common_dates)
            row.update(continuity)
            row["corporate_actions"] = int(len(actions))
            row["splits_applied"] = int(len(applied))
            if continuity["status"] != "passed":
                audits.append(row)
                continue
            prepared = construir_quadro_rotacao(normalized, config).reindex(common_dates)
            execution = prepared.reindex(all_decision_dates[1:])
            prices = execution[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
            if not np.isfinite(prices).all() or (execution[["open", "high", "low", "close"]] <= 0).any().any():
                row.update({"status": "rejected", "reason": "incomplete_execution_prices"})
                audits.append(row)
                continue
            row.update({"status": "passed", "reason": None, "model_rows": int(len(prepared))})
            frames[symbol] = prepared
            hashes[symbol] = {
                "raw_sha256": _sha256_file(raw_path),
                "corporate_actions_sha256": _sha256_file(action_path),
            }
        except Exception as exc:
            row.update({"status": "failed", "reason": str(exc)[:500]})
        audits.append(row)
        print(
            f"[continuity] {position}/{len(symbols)} {symbol} status={row['status']} reason={row.get('reason')}",
            flush=True,
        )
    return frames, pd.DataFrame(audits), hashes


# %% 4 - Baseline U67 e caches reutilizaveis
def _scenario_metrics(result: Any, folds: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    metrics = summarize_metrics(result, folds, initial_capital)
    raw = result.metrics or {}
    metrics.update(
        {
            "capital_rotations": int(raw.get("capital_rotations") or 0),
            "simulated_buys": int(raw.get("simulated_buys") or 0),
            "simulated_sells": int(raw.get("simulated_sells") or 0),
            "cash_days": int(raw.get("cash_days") or 0),
            "market_exposure": float(raw.get("market_exposure") or 0.0),
            "total_transaction_fees": float(raw.get("total_transaction_fees") or 0.0),
        }
    )
    return metrics


def _train_baseline_states(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    folds: list[dict[str, Any]],
    config: Any,
) -> tuple[list[BaselineFoldState], dict[int, Any], list[dict[str, Any]]]:
    states = []
    policies = {}
    calibration_rows = []
    candidate_margins = tuple(float(value) for value in config.rotation_switch_margin_candidates)
    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]) : int(fold["calibration_end_index"])
        ]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        decision_dates = pd.DatetimeIndex(fold["decision_dates"])
        print(f"[baseline-train] fold={fold_id} {position}/{len(folds)} calibration", flush=True)
        calibration_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config,
            phase=f"u67_expansion_baseline_fold_{fold_id}_calibration",
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_models, frames, symbols, calibration_dates, config
        )
        scores = []
        for margin in candidate_margins:
            policy = _politica_utilidade(
                {}, frames, symbols, config, margin, utility_cache=calibration_cache
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, calibration_dates, config
            )
            scores.append((margin, float(score)))
        selection = selecionar_switch_margin(scores)
        selected_margin = float(selection["selected_candidate_margin"])
        effective_margin = max(float(config.rotation_switch_margin), selected_margin)
        for margin, score in scores:
            calibration_rows.append(
                {
                    "scenario": "U67_BASELINE",
                    "fold_id": fold_id,
                    "candidate_margin": margin,
                    "calibration_score": score,
                    "selected": bool(abs(margin - selected_margin) <= 1e-12),
                }
            )
        print(f"[baseline-train] fold={fold_id} final", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"u67_expansion_baseline_fold_{fold_id}_final",
        )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, decision_dates, config
        )
        fit_diagnostics = {
            symbol: dict(getattr(model, "_fit_diagnostics", {}) or {})
            for symbol, model in final_models.items()
        }
        oos_diagnostics = diagnosticos_fora_amostra(
            final_models,
            frames,
            symbols,
            decision_dates,
            decision_cache,
            fold_id=fold_id,
        )
        policies[fold_id] = _politica_utilidade(
            {},
            frames,
            symbols,
            config,
            effective_margin,
            fold_id=fold_id,
            calibrated_switch_margin=selected_margin,
            utility_cache=decision_cache,
        )
        states.append(
            BaselineFoldState(
                fold_id=fold_id,
                train_dates=train_dates,
                calibration_dates=calibration_dates,
                final_fit_dates=final_fit_dates,
                decision_dates=decision_dates,
                calibration_cache=calibration_cache,
                decision_cache=decision_cache,
                selected_margin=selected_margin,
                effective_margin=effective_margin,
                fit_diagnostics=fit_diagnostics,
                oos_diagnostics=oos_diagnostics,
            )
        )
        del calibration_models, final_models
    return states, policies, calibration_rows


def _train_candidate_caches(
    symbol: str,
    frame: pd.DataFrame,
    states: list[BaselineFoldState],
    config: Any,
) -> dict[int, CandidateFoldCache]:
    output = {}
    for position, state in enumerate(states, start=1):
        training = frame.reindex(state.train_dates)
        valid_train = training[[*ROTATION_FEATURES, "forward_risk_adjusted_utility"]].dropna()
        if len(valid_train) < int(config.rotation_minimum_training_rows):
            raise RuntimeError(
                f"{symbol}: insufficient_model_training_rows fold={state.fold_id} rows={len(valid_train)}"
            )
        one_frame = {symbol: frame}
        print(
            f"[candidate-train] {symbol} fold={state.fold_id} {position}/{len(states)} calibration",
            flush=True,
        )
        calibration_model = _ajustar_modelos_lightgbm(
            one_frame,
            [symbol],
            state.train_dates,
            config,
            phase=f"u67_expansion_{symbol}_fold_{state.fold_id}_calibration",
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_model,
            one_frame,
            [symbol],
            state.calibration_dates,
            config,
        )
        print(f"[candidate-train] {symbol} fold={state.fold_id} final", flush=True)
        final_model = _ajustar_modelos_lightgbm(
            one_frame,
            [symbol],
            state.final_fit_dates,
            config,
            phase=f"u67_expansion_{symbol}_fold_{state.fold_id}_final",
        )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_model,
            one_frame,
            [symbol],
            state.decision_dates,
            config,
        )
        fit_diagnostics = dict(
            getattr(final_model.get(symbol), "_fit_diagnostics", {}) or {}
        )
        oos_diagnostics = diagnosticos_fora_amostra(
            final_model,
            one_frame,
            [symbol],
            state.decision_dates,
            decision_cache,
            fold_id=state.fold_id,
        )
        output[state.fold_id] = CandidateFoldCache(
            fold_id=state.fold_id,
            calibration_cache=calibration_cache,
            decision_cache=decision_cache,
            fit_diagnostics=fit_diagnostics,
            oos_diagnostics=oos_diagnostics,
        )
        del calibration_model, final_model
    return output


def _merge_utility_caches(
    baseline_cache: dict[pd.Timestamp, np.ndarray],
    baseline_symbols: list[str],
    candidate_caches: dict[str, dict[pd.Timestamp, np.ndarray]],
    combined_symbols: list[str],
) -> dict[pd.Timestamp, np.ndarray]:
    """Combina caches mantendo a posicao 0 reservada para CASH.

    Cada cache de candidato foi treinado isoladamente e possui vetor [CASH, score].
    Isso e exato porque o Control atual ajusta um LightGBM independente por ativo.
    """
    combined_position = {symbol: index + 1 for index, symbol in enumerate(combined_symbols)}
    base_position = {symbol: index + 1 for index, symbol in enumerate(baseline_symbols)}
    output: dict[pd.Timestamp, np.ndarray] = {}
    for timestamp, base_values in baseline_cache.items():
        values = np.full(len(combined_symbols) + 1, float("-inf"), dtype=np.float64)
        values[0] = 0.0
        for symbol, source_position in base_position.items():
            values[combined_position[symbol]] = float(base_values[source_position])
        for symbol, cache in candidate_caches.items():
            candidate_values = cache.get(pd.Timestamp(timestamp))
            if candidate_values is not None and len(candidate_values) >= 2:
                values[combined_position[symbol]] = float(candidate_values[1])
        output[pd.Timestamp(timestamp)] = values
    return output


def _evaluate_scenario(
    scenario: str,
    extra_symbols: list[str],
    baseline_frames: dict[str, pd.DataFrame],
    candidate_frames: dict[str, pd.DataFrame],
    baseline_symbols: list[str],
    states: list[BaselineFoldState],
    candidate_fold_caches: dict[str, dict[int, CandidateFoldCache]],
    folds: list[dict[str, Any]],
    all_decision_dates: pd.DatetimeIndex,
    decision_to_fold: dict[pd.Timestamp, int],
    decision_metadata: dict[pd.Timestamp, dict[str, Any]],
    config_base: Any,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    extras = sorted(dict.fromkeys(extra_symbols))
    combined_symbols = sorted([*baseline_symbols, *extras])
    combined_frames = dict(baseline_frames)
    combined_frames.update({symbol: candidate_frames[symbol] for symbol in extras})
    config = construir_configuracao_controle(config_base, assets=combined_symbols)
    policies = {}
    calibration_rows = []
    margins = tuple(float(value) for value in config.rotation_switch_margin_candidates)

    for state in states:
        extra_calibration = {
            symbol: candidate_fold_caches[symbol][state.fold_id].calibration_cache
            for symbol in extras
        }
        calibration_cache = _merge_utility_caches(
            state.calibration_cache,
            baseline_symbols,
            extra_calibration,
            combined_symbols,
        )
        scores = []
        for margin in margins:
            policy = _politica_utilidade(
                {}, combined_frames, combined_symbols, config, margin, utility_cache=calibration_cache
            )
            score = _crescimento_politica_simples(
                policy, combined_frames, combined_symbols, state.calibration_dates, config
            )
            scores.append((margin, float(score)))
        selection = selecionar_switch_margin(scores)
        selected_margin = float(selection["selected_candidate_margin"])
        effective_margin = max(float(config.rotation_switch_margin), selected_margin)
        for margin, score in scores:
            calibration_rows.append(
                {
                    "scenario": scenario,
                    "fold_id": state.fold_id,
                    "candidate_margin": margin,
                    "calibration_score": score,
                    "selected": bool(abs(margin - selected_margin) <= 1e-12),
                }
            )
        extra_decision = {
            symbol: candidate_fold_caches[symbol][state.fold_id].decision_cache
            for symbol in extras
        }
        decision_cache = _merge_utility_caches(
            state.decision_cache,
            baseline_symbols,
            extra_decision,
            combined_symbols,
        )
        policies[state.fold_id] = _politica_utilidade(
            {},
            combined_frames,
            combined_symbols,
            config,
            effective_margin,
            fold_id=state.fold_id,
            calibrated_switch_margin=selected_margin,
            utility_cache=decision_cache,
        )

    result = _simular_exato(
        "tcc_u67_asset_expansion",
        _politica_agendada(policies, decision_to_fold),
        combined_frames,
        combined_symbols,
        all_decision_dates,
        config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"TCC U67 + {','.join(extras)}" if extras else "TCC U67 baseline",
        method_line="- U67 fixo; candidatos externos adicionados sem remover ativos originais.",
    )
    metrics = _scenario_metrics(result, folds, float(config.initial_capital))
    return result, metrics, calibration_rows


# %% 5 - Comparacoes e artefatos
def _decision_changes(
    scenario: str,
    baseline_result: Any,
    candidate_result: Any,
    candidate_symbols: set[str],
) -> pd.DataFrame:
    base = baseline_result.predictions[["selected_asset", "strategy_equity"]].rename(
        columns={"selected_asset": "baseline_asset", "strategy_equity": "baseline_equity"}
    )
    candidate = candidate_result.predictions[["selected_asset", "strategy_equity"]].rename(
        columns={"selected_asset": "scenario_asset", "strategy_equity": "scenario_equity"}
    )
    joined = base.join(candidate, how="inner")
    changed = joined["baseline_asset"] != joined["scenario_asset"]
    out = joined.loc[changed].reset_index()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "scenario", "timestamp", "baseline_asset", "scenario_asset",
                "baseline_equity", "scenario_equity", "scenario_selected_new_asset",
            ]
        )
    out.insert(0, "scenario", scenario)
    out["scenario_selected_new_asset"] = out["scenario_asset"].isin(candidate_symbols)
    return out


def _new_asset_trades(scenario: str, result: Any, candidate_symbols: set[str]) -> pd.DataFrame:
    trades = result.trades.copy()
    if trades.empty or "asset" not in trades.columns:
        return pd.DataFrame()
    subset = trades.loc[trades["asset"].astype(str).isin(candidate_symbols)].copy()
    if not subset.empty:
        subset.insert(0, "scenario", scenario)
    return subset


def _forward_backward_subset_search(
    symbols: list[str],
    score_assets: Any,
    baseline_capital: float,
    improvement_tol: float = SUBSET_IMPROVEMENT_TOL,
) -> dict[str, Any]:
    """Busca deterministica forward + backward usando capital final como score."""
    ordered = list(dict.fromkeys(str(symbol) for symbol in symbols))
    order = {symbol: index for index, symbol in enumerate(ordered)}
    trace: list[dict[str, Any]] = []

    def canonical(values: list[str] | tuple[str, ...]) -> list[str]:
        chosen = set(values)
        return [symbol for symbol in ordered if symbol in chosen]

    def best_trial(
        phase: str,
        step: int,
        current_assets: list[str],
        candidates: list[tuple[str, list[str]]],
        current_capital: float,
    ) -> tuple[str | None, list[str], float, bool]:
        evaluated: list[tuple[str, list[str], float]] = []
        for action_symbol, assets in candidates:
            normalized = canonical(assets)
            capital = float(score_assets(normalized))
            evaluated.append((action_symbol, normalized, capital))
        if not evaluated:
            return None, current_assets, current_capital, False
        evaluated.sort(key=lambda item: (-item[2], order[item[0]]))
        best_symbol, best_assets, best_capital = evaluated[0]
        accepted = best_capital > current_capital + improvement_tol
        for action_symbol, assets, capital in evaluated:
            trace.append(
                {
                    "phase": phase,
                    "step": step,
                    "action_symbol": action_symbol,
                    "assets": ",".join(assets),
                    "asset_count": len(assets),
                    "ending_capital": capital,
                    "current_capital_before": current_capital,
                    "best_trial": action_symbol == best_symbol,
                    "accepted": bool(accepted and action_symbol == best_symbol),
                }
            )
        return best_symbol, best_assets, best_capital, accepted

    forward_assets: list[str] = []
    forward_capital = float(baseline_capital)
    remaining = ordered.copy()
    step = 0
    while remaining:
        step += 1
        trials = [
            (symbol, [*forward_assets, symbol])
            for symbol in remaining
        ]
        symbol, assets, capital, accepted = best_trial(
            "forward",
            step,
            forward_assets,
            trials,
            forward_capital,
        )
        if not accepted or symbol is None:
            break
        forward_assets = assets
        forward_capital = capital
        remaining = [item for item in remaining if item != symbol]

    def backward(start_assets: list[str], phase: str) -> tuple[list[str], float]:
        current_assets = canonical(start_assets)
        current_capital = (
            float(score_assets(current_assets))
            if current_assets
            else float(baseline_capital)
        )
        step = 0
        while len(current_assets) > 1:
            step += 1
            trials = [
                (
                    symbol,
                    [item for item in current_assets if item != symbol],
                )
                for symbol in current_assets
            ]
            symbol, assets, capital, accepted = best_trial(
                phase,
                step,
                current_assets,
                trials,
                current_capital,
            )
            if not accepted or symbol is None:
                break
            current_assets = assets
            current_capital = capital
        return current_assets, current_capital

    all_assets = canonical(ordered)
    all_capital = float(score_assets(all_assets))
    backward_all_assets, backward_all_capital = backward(
        all_assets,
        "backward_from_all",
    )
    backward_forward_assets, backward_forward_capital = backward(
        forward_assets,
        "backward_from_forward",
    )

    solutions = [
        ("all_positive", all_assets, all_capital),
        ("forward", forward_assets, forward_capital),
        ("backward_from_all", backward_all_assets, backward_all_capital),
        (
            "backward_from_forward",
            backward_forward_assets,
            backward_forward_capital,
        ),
    ]
    solutions.sort(key=lambda item: (-item[2], len(item[1]), item[0]))
    selected_label, selected_assets, selected_capital = solutions[0]
    return {
        "forward": {
            "assets": forward_assets,
            "ending_capital": forward_capital,
        },
        "backward_from_all": {
            "assets": backward_all_assets,
            "ending_capital": backward_all_capital,
        },
        "backward_from_forward": {
            "assets": backward_forward_assets,
            "ending_capital": backward_forward_capital,
        },
        "all_positive": {
            "assets": all_assets,
            "ending_capital": all_capital,
        },
        "selected": {
            "source": selected_label,
            "assets": selected_assets,
            "ending_capital": selected_capital,
        },
        "trace": trace,
    }


def _run_subset_optimization(
    *,
    snapshot: dict[str, Any],
    baseline_payload: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_result: Any,
    baseline_frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    baseline_symbols: list[str],
    folds: list[dict[str, Any]],
    all_decision_dates: pd.DatetimeIndex,
    decision_to_fold: dict[pd.Timestamp, int],
    decision_metadata: Any,
    config_u67: Any,
    states: list[BaselineFoldState],
    calibration_rows: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    symbols = list(SUBSET_SOURCE_CANDIDATES)
    credentials = load_alpaca_credentials(ROOT)
    download_audit = _prepare_full_candidate_files(credentials, symbols)
    download_audit.to_csv(OUT / "candidate_download.csv", index=False)
    failed_downloads = download_audit.loc[
        download_audit["status"].eq("failed"),
        "asset",
    ].astype(str).tolist()
    if failed_downloads:
        raise RuntimeError(
            "Falha ao reproduzir dados dos 15 positivos: "
            + ", ".join(failed_downloads)
        )

    candidate_frames, continuity, candidate_hashes = _prepare_candidate_frames(
        symbols,
        common_dates,
        all_decision_dates,
        config_u67,
    )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    missing_structural = [
        symbol for symbol in symbols if symbol not in candidate_frames
    ]
    if missing_structural:
        raise RuntimeError(
            "Checkpoint dos 15 positivos nao pode ser reproduzido; "
            "falha estrutural em: "
            + ", ".join(missing_structural)
        )

    candidate_fold_caches: dict[str, dict[int, CandidateFoldCache]] = {}
    for position, symbol in enumerate(symbols, start=1):
        candidate_fold_caches[symbol] = _train_candidate_caches(
            symbol,
            candidate_frames[symbol],
            states,
            config_u67,
        )
        print(
            f"[subset-train] {position}/{len(symbols)} {symbol}",
            flush=True,
        )

    scenario_metrics_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    scenario_rows: list[dict[str, Any]] = []
    subset_calibration_rows: list[dict[str, Any]] = []

    def canonical(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        chosen = set(values)
        return tuple(symbol for symbol in symbols if symbol in chosen)

    def score_assets(values: list[str] | tuple[str, ...]) -> float:
        key = canonical(values)
        if not key:
            return float(baseline_metrics["ending_capital"])
        cached = scenario_metrics_cache.get(key)
        if cached is not None:
            return float(cached["ending_capital"])
        scenario_name = "U67+SUBSET_" + "_".join(key)
        _, metrics, rows = _evaluate_scenario(
            scenario_name,
            list(key),
            baseline_frames,
            candidate_frames,
            baseline_symbols,
            states,
            candidate_fold_caches,
            folds,
            all_decision_dates,
            decision_to_fold,
            decision_metadata,
            config_u67,
        )
        scenario_metrics_cache[key] = metrics
        subset_calibration_rows.extend(rows)
        scenario_rows.append(
            {
                "assets": ",".join(key),
                "asset_count": len(key),
                "ending_capital": metrics.get("ending_capital"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "maximum_drawdown": metrics.get("maximum_drawdown"),
                "worst_fold_return": metrics.get("worst_fold_return"),
                "capital_rotations": metrics.get("capital_rotations"),
            }
        )
        pd.DataFrame(scenario_rows).to_csv(
            OUT / "subset_scenarios_partial.csv",
            index=False,
        )
        print(
            f"[subset-score] n={len(key):02d} "
            f"capital=US$ {metrics['ending_capital']:,.2f} "
            f"assets={','.join(key)}",
            flush=True,
        )
        return float(metrics["ending_capital"])

    all_positive_capital = score_assets(symbols)
    all_positive_exact = math.isclose(
        all_positive_capital,
        EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    )
    if not all_positive_exact:
        raise RuntimeError(
            "Checkpoint U67+15 nao foi reproduzido. "
            f"observado={all_positive_capital:.8f} "
            f"esperado={EXPECTED_ALL_POSITIVE_ENDING_CAPITAL:.8f}"
        )
    print(
        "[subset-checkpoint] U67+15 reproduced_exact_to_cent=True",
        flush=True,
    )

    search = _forward_backward_subset_search(
        symbols,
        score_assets,
        float(baseline_metrics["ending_capital"]),
        SUBSET_IMPROVEMENT_TOL,
    )
    pd.DataFrame(search["trace"]).to_csv(
        OUT / "subset_search.csv",
        index=False,
    )
    pd.DataFrame(scenario_rows).sort_values(
        ["ending_capital", "asset_count"],
        ascending=[False, True],
        kind="stable",
    ).to_csv(OUT / "subset_scenarios.csv", index=False)

    selected_assets = list(search["selected"]["assets"])
    selected_score = float(search["selected"]["ending_capital"])
    final_result, final_metrics, final_rows = _evaluate_scenario(
        "U67+SUBSET_SELECTED",
        selected_assets,
        baseline_frames,
        candidate_frames,
        baseline_symbols,
        states,
        candidate_fold_caches,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
        config_u67,
    )
    calibration_rows.extend(subset_calibration_rows)
    calibration_rows.extend(final_rows)
    if not math.isclose(
        float(final_metrics["ending_capital"]),
        selected_score,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    ):
        raise RuntimeError("Replay final do subconjunto divergiu do score da busca.")

    final_changes = _decision_changes(
        "U67+SUBSET_SELECTED",
        baseline_result,
        final_result,
        set(selected_assets),
    )
    final_trades = _new_asset_trades(
        "U67+SUBSET_SELECTED",
        final_result,
        set(selected_assets),
    )
    final_changes.to_csv(OUT / "decision_changes.csv", index=False)
    final_trades.to_csv(OUT / "trades_new_assets.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        OUT / "scenario_fold_calibration.csv",
        index=False,
    )

    delta, rate = _capital_delta(
        float(final_metrics["ending_capital"]),
        float(baseline_metrics["ending_capital"]),
    )
    delta_vs_all, rate_vs_all = _capital_delta(
        float(final_metrics["ending_capital"]),
        EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
    )
    result_payload = {
        "status": "completed",
        "source_positive_candidates": symbols,
        "source_all_positive_checkpoint": {
            "expected_ending_capital": EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
            "observed_ending_capital": all_positive_capital,
            "exact_to_cent": all_positive_exact,
        },
        "search": {
            key: value
            for key, value in search.items()
            if key != "trace"
        },
        "selected": {
            "assets": selected_assets,
            "asset_count": len(selected_assets),
            "metrics": final_metrics,
            "ending_capital_delta_vs_u67": delta,
            "ending_capital_delta_rate_vs_u67": rate,
            "ending_capital_delta_vs_all_15": delta_vs_all,
            "ending_capital_delta_rate_vs_all_15": rate_vs_all,
            "decision_sessions_changed": int(len(final_changes)),
            "new_asset_trade_rows": int(len(final_trades)),
        },
    }
    _write_json(OUT / "selected_subset_replay.json", result_payload)

    payload = {
        "execution_schema": EXECUTION_SCHEMA,
        "version": SCRIPT_VERSION,
        "status": "completed",
        "purpose": "u67_positive_subset_optimization",
        "run_mode": RUN_MODE,
        "baseline": baseline_payload,
        "source_campaign": {
            "positive_candidate_count": len(symbols),
            "positive_candidates": symbols,
            "expected_all_positive_ending_capital": (
                EXPECTED_ALL_POSITIVE_ENDING_CAPITAL
            ),
        },
        "optimization": result_payload,
        "data": {
            "baseline_snapshot_sha256": snapshot.get("snapshot_sha256"),
            "candidate_financial_replay_feed": "sip",
            "candidate_financial_replay_timeframe": "1Day",
            "candidate_financial_replay_adjustment": "raw",
            "candidate_financial_replay_start": START_DATE,
            "candidate_financial_replay_end": ANALYSIS_END_DATE,
            "candidate_file_hashes": candidate_hashes,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(OUT / "pesquisa_u67_subset_optimization.json", payload)

    package = criar_pacote_analise(
        OUT,
        comparison_file="pesquisa_u67_subset_optimization.json",
        execution_schema=EXECUTION_SCHEMA,
        archive_name="pacote_analise_u67_subset_optimization.zip",
    )
    print("=" * 78, flush=True)
    print(
        f"[final-subset] U67=US$ {baseline_metrics['ending_capital']:,.2f} "
        f"U67+15=US$ {all_positive_capital:,.2f}",
        flush=True,
    )
    print(
        f"[final-subset] selected={selected_assets} "
        f"capital=US$ {final_metrics['ending_capital']:,.2f}",
        flush=True,
    )
    print(f"[final-subset] package={package}", flush=True)
    print("=" * 78, flush=True)
    sinal_sonoro_conclusao()
    return payload



def _regression_error(actual: Any, predicted: Any) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not bool(valid.any()):
        return {"rows": 0, "mae": None, "rmse": None}
    residual = predicted_values[valid] - actual_values[valid]
    return {
        "rows": int(valid.sum()),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _naive_mean_errors(
    frame: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
    target_column: str = "forward_risk_adjusted_utility",
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = pd.to_numeric(
        frame.reindex(train_dates)[target_column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    finite_train = train[np.isfinite(train)]
    if len(finite_train) == 0:
        empty = {"rows": 0, "mae": None, "rmse": None}
        return empty, empty
    mean_target = float(np.mean(finite_train))
    train_error = _regression_error(
        train,
        np.full(len(train), mean_target, dtype=np.float64),
    )
    oos = pd.to_numeric(
        frame.reindex(decision_dates)[target_column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    oos_error = _regression_error(
        oos,
        np.full(len(oos), mean_target, dtype=np.float64),
    )
    return train_error, oos_error


def _fit_quality_row(
    *,
    universe_role: str,
    symbol: str,
    fold_id: int,
    frame: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
    fit_diagnostics: dict[str, Any] | None,
    oos_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    fit = dict(fit_diagnostics or {})
    train = dict(fit.get("train") or {})
    oos = dict(oos_diagnostics or {})
    naive_train, naive_oos = _naive_mean_errors(
        frame,
        train_dates,
        decision_dates,
    )
    train_rmse = _finite(train.get("rmse"))
    oos_rmse = _finite(oos.get("rmse"))
    naive_train_rmse = _finite(naive_train.get("rmse"))
    naive_oos_rmse = _finite(naive_oos.get("rmse"))

    train_skill = (
        1.0 - train_rmse / naive_train_rmse
        if train_rmse is not None
        and naive_train_rmse is not None
        and naive_train_rmse > 0
        else None
    )
    oos_skill = (
        1.0 - oos_rmse / naive_oos_rmse
        if oos_rmse is not None
        and naive_oos_rmse is not None
        and naive_oos_rmse > 0
        else None
    )
    generalization_gap = (
        oos_rmse / train_rmse - 1.0
        if oos_rmse is not None and train_rmse is not None and train_rmse > 0
        else None
    )
    overfit_warning = bool(
        train_skill is not None
        and oos_skill is not None
        and train_skill > 0.0
        and oos_skill <= 0.0
    )
    underfit_warning = bool(
        train_skill is not None
        and oos_skill is not None
        and train_skill <= 0.0
        and oos_skill <= 0.0
    )
    return {
        "universe_role": universe_role,
        "asset": symbol,
        "fold_id": int(fold_id),
        "train_rows": int(train.get("rows") or 0),
        "oos_rows": int(oos.get("rows") or 0),
        "train_mae": _finite(train.get("mae")),
        "train_rmse": train_rmse,
        "oos_mae": _finite(oos.get("mae")),
        "oos_rmse": oos_rmse,
        "naive_train_mae": _finite(naive_train.get("mae")),
        "naive_train_rmse": naive_train_rmse,
        "naive_oos_mae": _finite(naive_oos.get("mae")),
        "naive_oos_rmse": naive_oos_rmse,
        "train_skill_vs_naive": train_skill,
        "oos_skill_vs_naive": oos_skill,
        "generalization_gap_rmse_ratio": generalization_gap,
        "overfit_warning": overfit_warning,
        "underfit_warning": underfit_warning,
    }


def _fold_metric(metrics: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for row in metrics.get("folds") or []:
        if int(row.get("fold_id")) == int(fold_id):
            return dict(row)
    raise KeyError(f"fold_id={fold_id} ausente nas metricas")


def _compound_fold_capital(
    metrics: dict[str, Any],
    fold_ids: list[int] | tuple[int, ...],
    initial_capital: float = 10_000.0,
) -> float:
    capital = float(initial_capital)
    for fold_id in fold_ids:
        row = _fold_metric(metrics, int(fold_id))
        capital *= 1.0 + float(row["strategy_return"])
    return float(capital)


def _jaccard(left: list[str] | tuple[str, ...], right: list[str] | tuple[str, ...]) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _run_assurance_validation(
    *,
    snapshot: dict[str, Any],
    baseline_payload: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_result: Any,
    baseline_frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    baseline_symbols: list[str],
    folds: list[dict[str, Any]],
    all_decision_dates: pd.DatetimeIndex,
    decision_to_fold: dict[pd.Timestamp, int],
    decision_metadata: Any,
    config_u67: Any,
    states: list[BaselineFoldState],
    calibration_rows: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    symbols = list(SUBSET_SOURCE_CANDIDATES)
    discovery_selected = list(DISCOVERY_SELECTED_CANDIDATES)
    credentials = load_alpaca_credentials(ROOT)

    download_audit = _prepare_full_candidate_files(credentials, symbols)
    download_audit.to_csv(OUT / "candidate_download.csv", index=False)
    failed = download_audit.loc[
        download_audit["status"].eq("failed"),
        "asset",
    ].astype(str).tolist()
    if failed:
        raise RuntimeError(
            "Falha ao recuperar candidatos para assurance: " + ", ".join(failed)
        )

    candidate_frames, continuity, candidate_hashes = _prepare_candidate_frames(
        symbols,
        common_dates,
        all_decision_dates,
        config_u67,
    )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    missing = [symbol for symbol in symbols if symbol not in candidate_frames]
    if missing:
        raise RuntimeError(
            "Assurance nao reproduziu a elegibilidade dos 15 positivos: "
            + ", ".join(missing)
        )

    candidate_fold_caches: dict[str, dict[int, CandidateFoldCache]] = {}
    for position, symbol in enumerate(symbols, start=1):
        candidate_fold_caches[symbol] = _train_candidate_caches(
            symbol,
            candidate_frames[symbol],
            states,
            config_u67,
        )
        print(
            f"[assurance-train] {position}/{len(symbols)} {symbol}",
            flush=True,
        )

    model_rows: list[dict[str, Any]] = []
    for state in states:
        baseline_oos_by_asset = dict(
            (state.oos_diagnostics or {}).get("by_asset") or {}
        )
        for symbol in baseline_symbols:
            model_rows.append(
                _fit_quality_row(
                    universe_role="u67_baseline",
                    symbol=symbol,
                    fold_id=state.fold_id,
                    frame=baseline_frames[symbol],
                    train_dates=state.final_fit_dates,
                    decision_dates=state.decision_dates,
                    fit_diagnostics=dict(
                        (state.fit_diagnostics or {}).get(symbol) or {}
                    ),
                    oos_diagnostics=dict(
                        baseline_oos_by_asset.get(symbol) or {}
                    ),
                )
            )
    for symbol in symbols:
        for state in states:
            cache = candidate_fold_caches[symbol][state.fold_id]
            candidate_oos = dict(
                (cache.oos_diagnostics or {}).get("by_asset") or {}
            )
            model_rows.append(
                _fit_quality_row(
                    universe_role=(
                        "discovery_selected"
                        if symbol in discovery_selected
                        else "positive_not_selected"
                    ),
                    symbol=symbol,
                    fold_id=state.fold_id,
                    frame=candidate_frames[symbol],
                    train_dates=state.final_fit_dates,
                    decision_dates=state.decision_dates,
                    fit_diagnostics=cache.fit_diagnostics,
                    oos_diagnostics=dict(candidate_oos.get(symbol) or {}),
                )
            )
    model_diagnostics = pd.DataFrame(model_rows)
    model_diagnostics.to_csv(OUT / "model_generalization.csv", index=False)

    candidate_model_diag = model_diagnostics.loc[
        model_diagnostics["asset"].isin(symbols)
    ].copy()
    model_summary = {
        "candidate_asset_fold_rows": int(len(candidate_model_diag)),
        "overfit_warning_rows": int(candidate_model_diag["overfit_warning"].sum()),
        "underfit_warning_rows": int(candidate_model_diag["underfit_warning"].sum()),
        "positive_oos_skill_rows": int(
            (candidate_model_diag["oos_skill_vs_naive"] > 0.0).sum()
        ),
        "nonpositive_oos_skill_rows": int(
            (candidate_model_diag["oos_skill_vs_naive"] <= 0.0).sum()
        ),
        "classification_rule": {
            "overfit_warning": (
                "train skill > 0 versus naive and OOS skill <= 0"
            ),
            "underfit_warning": (
                "train skill <= 0 and OOS skill <= 0 versus naive"
            ),
            "no_arbitrary_gap_threshold": True,
        },
    }

    scenario_metrics_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    assurance_calibration_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    def canonical(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        chosen = set(values)
        return tuple(symbol for symbol in symbols if symbol in chosen)

    def evaluate_subset(values: list[str] | tuple[str, ...]) -> dict[str, Any]:
        key = canonical(values)
        if not key:
            return baseline_metrics
        if key in scenario_metrics_cache:
            return scenario_metrics_cache[key]
        _, metrics, rows = _evaluate_scenario(
            "U67+ASSURANCE_" + "_".join(key),
            list(key),
            baseline_frames,
            candidate_frames,
            baseline_symbols,
            states,
            candidate_fold_caches,
            folds,
            all_decision_dates,
            decision_to_fold,
            decision_metadata,
            config_u67,
        )
        scenario_metrics_cache[key] = metrics
        assurance_calibration_rows.extend(rows)
        scenario_rows.append(
            {
                "assets": ",".join(key),
                "asset_count": len(key),
                "ending_capital": metrics.get("ending_capital"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "maximum_drawdown": metrics.get("maximum_drawdown"),
                "worst_fold_return": metrics.get("worst_fold_return"),
            }
        )
        pd.DataFrame(scenario_rows).to_csv(
            OUT / "assurance_scenarios_partial.csv",
            index=False,
        )
        return metrics

    all_positive_metrics = evaluate_subset(symbols)
    selected_metrics = evaluate_subset(discovery_selected)
    all_positive_exact = math.isclose(
        float(all_positive_metrics["ending_capital"]),
        EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    )
    selected_exact = math.isclose(
        float(selected_metrics["ending_capital"]),
        EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    )
    if not all_positive_exact or not selected_exact:
        raise RuntimeError(
            "Checkpoint de descoberta divergiu: "
            f"all15={all_positive_metrics['ending_capital']:.8f} "
            f"selected10={selected_metrics['ending_capital']:.8f}"
        )

    temporal_rows: list[dict[str, Any]] = []
    temporal_trace_frames: list[pd.DataFrame] = []
    previous_assets: list[str] | None = None
    for stage, selection_folds, holdout_fold in (
        ("past_fold_1_to_future_fold_2", [1], 2),
        ("past_folds_1_2_to_future_fold_3", [1, 2], 3),
    ):
        def temporal_score(values: list[str] | tuple[str, ...]) -> float:
            return _compound_fold_capital(
                evaluate_subset(values),
                selection_folds,
                initial_capital=float(config_u67.initial_capital),
            )

        baseline_selection_score = _compound_fold_capital(
            baseline_metrics,
            selection_folds,
            initial_capital=float(config_u67.initial_capital),
        )
        search = _forward_backward_subset_search(
            symbols,
            temporal_score,
            baseline_selection_score,
            SUBSET_IMPROVEMENT_TOL,
        )
        chosen = list(search["selected"]["assets"])
        chosen_metrics = evaluate_subset(chosen)
        heldout = _fold_metric(chosen_metrics, holdout_fold)
        baseline_holdout = _fold_metric(baseline_metrics, holdout_fold)
        discovery_holdout = _fold_metric(selected_metrics, holdout_fold)
        temporal_rows.append(
            {
                "stage": stage,
                "selection_folds": ",".join(str(value) for value in selection_folds),
                "holdout_fold": holdout_fold,
                "selected_assets": ",".join(chosen),
                "selected_asset_count": len(chosen),
                "selection_score": search["selected"]["ending_capital"],
                "holdout_return": heldout["strategy_return"],
                "u67_holdout_return": baseline_holdout["strategy_return"],
                "holdout_excess_vs_u67": (
                    float(heldout["strategy_return"])
                    - float(baseline_holdout["strategy_return"])
                ),
                "discovery_10_holdout_return": discovery_holdout["strategy_return"],
                "jaccard_vs_discovery_10": _jaccard(
                    chosen,
                    discovery_selected,
                ),
                "jaccard_vs_previous_stage": (
                    _jaccard(chosen, previous_assets)
                    if previous_assets is not None
                    else None
                ),
            }
        )
        trace = pd.DataFrame(search["trace"])
        if not trace.empty:
            trace.insert(0, "stage", stage)
            temporal_trace_frames.append(trace)
        previous_assets = chosen

    temporal_validation = pd.DataFrame(temporal_rows)
    temporal_validation.to_csv(
        OUT / "temporal_universe_validation.csv",
        index=False,
    )
    temporal_trace = (
        pd.concat(temporal_trace_frames, ignore_index=True)
        if temporal_trace_frames
        else pd.DataFrame()
    )
    temporal_trace.to_csv(OUT / "temporal_universe_search.csv", index=False)

    loo_rows = []
    selected_capital = float(selected_metrics["ending_capital"])
    for symbol in discovery_selected:
        reduced = [item for item in discovery_selected if item != symbol]
        metrics = evaluate_subset(reduced)
        loo_rows.append(
            {
                "removed_asset": symbol,
                "remaining_assets": ",".join(reduced),
                "ending_capital": metrics["ending_capital"],
                "delta_vs_selected_10": (
                    float(metrics["ending_capital"]) - selected_capital
                ),
                "relative_delta_vs_selected_10": (
                    float(metrics["ending_capital"]) / selected_capital - 1.0
                ),
                "sharpe": metrics["sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "worst_fold_return": metrics["worst_fold_return"],
            }
        )
    pd.DataFrame(loo_rows).sort_values(
        "delta_vs_selected_10",
        ascending=False,
        kind="stable",
    ).to_csv(OUT / "leave_one_asset_out.csv", index=False)

    stress_profiles = [
        ("base", {}),
        ("slippage_5bps", {"slippage_bps": 5.0}),
        ("slippage_10bps", {"slippage_bps": 10.0}),
        (
            "regulatory_fees_2x",
            {
                "sec_fee_rate": float(config_u67.sec_fee_rate) * 2.0,
                "taf_fee_per_share": float(config_u67.taf_fee_per_share) * 2.0,
                "taf_fee_cap": float(config_u67.taf_fee_cap) * 2.0,
                "cat_fee_per_share": float(config_u67.cat_fee_per_share) * 2.0,
            },
        ),
        (
            "regulatory_fees_3x",
            {
                "sec_fee_rate": float(config_u67.sec_fee_rate) * 3.0,
                "taf_fee_per_share": float(config_u67.taf_fee_per_share) * 3.0,
                "taf_fee_cap": float(config_u67.taf_fee_cap) * 3.0,
                "cat_fee_per_share": float(config_u67.cat_fee_per_share) * 3.0,
            },
        ),
    ]
    stress_rows = []
    for name, update in stress_profiles:
        if name == "base":
            metrics = selected_metrics
        else:
            stress_config = config_u67.copiar_modelo(update=update)
            _, metrics, rows = _evaluate_scenario(
                "U67+ASSURANCE_STRESS_" + name,
                discovery_selected,
                baseline_frames,
                candidate_frames,
                baseline_symbols,
                states,
                candidate_fold_caches,
                folds,
                all_decision_dates,
                decision_to_fold,
                decision_metadata,
                stress_config,
            )
            assurance_calibration_rows.extend(rows)
        stress_rows.append(
            {
                "profile": name,
                "ending_capital": metrics["ending_capital"],
                "delta_vs_base": (
                    float(metrics["ending_capital"]) - selected_capital
                ),
                "relative_delta_vs_base": (
                    float(metrics["ending_capital"]) / selected_capital - 1.0
                ),
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "worst_fold_return": metrics["worst_fold_return"],
            }
        )
    pd.DataFrame(stress_rows).to_csv(OUT / "cost_stress.csv", index=False)

    reproducibility_result, reproducibility_metrics, reproducibility_rows = (
        _evaluate_scenario(
            "U67+ASSURANCE_REPRODUCIBILITY",
            discovery_selected,
            baseline_frames,
            candidate_frames,
            baseline_symbols,
            states,
            candidate_fold_caches,
            folds,
            all_decision_dates,
            decision_to_fold,
            decision_metadata,
            config_u67,
        )
    )
    assurance_calibration_rows.extend(reproducibility_rows)
    selected_result, _, selected_rows = _evaluate_scenario(
        "U67+ASSURANCE_SELECTED_REFERENCE",
        discovery_selected,
        baseline_frames,
        candidate_frames,
        baseline_symbols,
        states,
        candidate_fold_caches,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
        config_u67,
    )
    assurance_calibration_rows.extend(selected_rows)
    left = selected_result.predictions[["selected_asset", "strategy_equity"]]
    right = reproducibility_result.predictions[
        ["selected_asset", "strategy_equity"]
    ]
    aligned = left.join(
        right,
        how="inner",
        lsuffix="_reference",
        rsuffix="_rerun",
    )
    max_equity_abs_diff = float(
        (
            aligned["strategy_equity_reference"]
            - aligned["strategy_equity_rerun"]
        ).abs().max()
    )
    same_assets = bool(
        (
            aligned["selected_asset_reference"]
            == aligned["selected_asset_rerun"]
        ).all()
    )
    reproducibility = {
        "ending_capital_reference": selected_capital,
        "ending_capital_rerun": reproducibility_metrics["ending_capital"],
        "exact_to_cent": math.isclose(
            float(reproducibility_metrics["ending_capital"]),
            selected_capital,
            rel_tol=0.0,
            abs_tol=BASELINE_ABS_TOL,
        ),
        "selected_asset_sequence_identical": same_assets,
        "max_equity_absolute_difference": max_equity_abs_diff,
    }
    _write_json(OUT / "reproducibility.json", reproducibility)

    pd.DataFrame(scenario_rows).drop_duplicates(
        subset=["assets"],
        keep="last",
    ).to_csv(OUT / "assurance_scenarios.csv", index=False)
    pd.DataFrame(
        [*calibration_rows, *assurance_calibration_rows]
    ).to_csv(OUT / "scenario_fold_calibration.csv", index=False)

    payload = {
        "execution_schema": EXECUTION_SCHEMA,
        "version": SCRIPT_VERSION,
        "status": "completed",
        "purpose": "u67_assurance_validation",
        "run_mode": RUN_MODE,
        "baseline": baseline_payload,
        "checkpoints": {
            "all_15_exact_to_cent": all_positive_exact,
            "selected_10_exact_to_cent": selected_exact,
            "all_15_ending_capital": all_positive_metrics["ending_capital"],
            "selected_10_ending_capital": selected_metrics["ending_capital"],
        },
        "model_generalization": model_summary,
        "temporal_universe_validation": temporal_rows,
        "leave_one_asset_out_count": len(loo_rows),
        "cost_stress_profiles": [row["profile"] for row in stress_rows],
        "reproducibility": reproducibility,
        "selection_bias_context": {
            "source_direct_financially_tested_candidates": (
                SOURCE_DIRECT_TESTED_COUNT
            ),
            "source_subset_unique_scenarios": SOURCE_SUBSET_UNIQUE_SCENARIOS,
            "candidate_pool_for_temporal_validation": symbols,
            "candidate_pool_was_discovered_using_full_controlled_window": True,
            "interpretation": (
                "Temporal test validates subset selection with a fixed discovery "
                "pool; it is not yet a fully nested rediscovery of the 500 assets."
            ),
        },
        "dynamic_universe_principle": {
            "asset_list_is_permanent": False,
            "future_target": (
                "estimate marginal value of adding, keeping, or removing an asset "
                "using information available at the decision time"
            ),
            "current_phase": "controlled_historical_window_assurance",
        },
        "data": {
            "baseline_snapshot_sha256": snapshot.get("snapshot_sha256"),
            "candidate_file_hashes": candidate_hashes,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(OUT / "pesquisa_u67_assurance_validation.json", payload)

    package = criar_pacote_analise(
        OUT,
        comparison_file="pesquisa_u67_assurance_validation.json",
        execution_schema=EXECUTION_SCHEMA,
        archive_name="pacote_analise_u67_assurance_validation.zip",
    )
    print("=" * 78, flush=True)
    print(
        f"[final-assurance] selected10=US$ {selected_capital:,.2f} "
        f"overfit_warnings={model_summary['overfit_warning_rows']} "
        f"underfit_warnings={model_summary['underfit_warning_rows']}",
        flush=True,
    )
    print(
        f"[final-assurance] reproducible={reproducibility['exact_to_cent']} "
        f"same_assets={same_assets}",
        flush=True,
    )
    print(f"[final-assurance] package={package}", flush=True)
    print("=" * 78, flush=True)
    sinal_sonoro_conclusao()
    return payload



def _daily_returns(result: Any) -> pd.Series:
    equity = pd.to_numeric(
        result.predictions["strategy_equity"],
        errors="coerce",
    ).astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns.name = "strategy_return"
    return returns


def _daily_sharpe(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan")
    std = float(np.std(array, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(np.mean(array) / std)


def _safe_rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or len(right) != len(left):
        return None
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if float(np.std(left_rank)) <= 0.0 or float(np.std(right_rank)) <= 0.0:
        return None
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else None


def _rank_diagnostic_sessions(
    *,
    scope: str,
    symbols: list[str],
    states: list[BaselineFoldState],
    baseline_symbols: list[str],
    baseline_frames: dict[str, pd.DataFrame],
    candidate_frames: dict[str, pd.DataFrame],
    candidate_fold_caches: dict[str, dict[int, CandidateFoldCache]],
    strategy_result: Any | None = None,
) -> pd.DataFrame:
    """Avalia ranking previsto versus utility realizada, sem usar futuro na decisao."""
    baseline_position = {
        symbol: index + 1
        for index, symbol in enumerate(baseline_symbols)
    }
    rows: list[dict[str, Any]] = []
    strategy_predictions = (
        strategy_result.predictions
        if strategy_result is not None
        else pd.DataFrame()
    )

    for state in states:
        for timestamp in state.decision_dates:
            date = pd.Timestamp(timestamp)
            predicted: list[float] = []
            realized: list[float] = []
            valid_symbols: list[str] = []
            for symbol in symbols:
                if symbol in baseline_position:
                    values = state.decision_cache.get(date)
                    score = (
                        float(values[baseline_position[symbol]])
                        if values is not None
                        else float("nan")
                    )
                    frame = baseline_frames[symbol]
                else:
                    fold_cache = candidate_fold_caches.get(symbol, {}).get(
                        state.fold_id
                    )
                    values = (
                        fold_cache.decision_cache.get(date)
                        if fold_cache is not None
                        else None
                    )
                    score = (
                        float(values[1])
                        if values is not None and len(values) >= 2
                        else float("nan")
                    )
                    frame = candidate_frames[symbol]
                try:
                    target = float(
                        frame.loc[date, "forward_risk_adjusted_utility"]
                    )
                except (KeyError, TypeError, ValueError):
                    target = float("nan")
                if np.isfinite(score) and np.isfinite(target):
                    predicted.append(score)
                    realized.append(target)
                    valid_symbols.append(symbol)

            if len(valid_symbols) < 3:
                continue

            pred = np.asarray(predicted, dtype=np.float64)
            actual = np.asarray(realized, dtype=np.float64)
            n = len(valid_symbols)
            predicted_order = sorted(
                range(n),
                key=lambda idx: (-float(pred[idx]), valid_symbols[idx]),
            )
            actual_order = sorted(
                range(n),
                key=lambda idx: (-float(actual[idx]), valid_symbols[idx]),
            )
            predicted_top1_index = predicted_order[0]
            predicted_top1 = valid_symbols[predicted_top1_index]
            actual_top1 = valid_symbols[actual_order[0]]
            actual_rank_pct = (
                pd.Series(actual)
                .rank(method="average", pct=True)
                .to_numpy(dtype=float)
            )
            relevance = (
                pd.Series(actual)
                .rank(method="average")
                .to_numpy(dtype=float)
                - 1.0
            )
            top3_n = min(3, n)
            top5_n = min(5, n)
            actual_top3 = set(actual_order[:top3_n])
            actual_top5 = set(actual_order[:top5_n])
            predicted_top3 = set(predicted_order[:top3_n])
            ndcg5 = float(
                ndcg_score(
                    relevance.reshape(1, -1),
                    pred.reshape(1, -1),
                    k=top5_n,
                )
            )
            ndcg10 = float(
                ndcg_score(
                    relevance.reshape(1, -1),
                    pred.reshape(1, -1),
                    k=min(10, n),
                )
            )
            top1_actual = float(actual[predicted_top1_index])
            oracle_actual = float(actual[actual_order[0]])
            sign_accuracy = float(
                np.mean((pred > 0.0) == (actual > 0.0))
            )

            selected_asset = None
            selected_realized_utility = None
            selected_actual_percentile = None
            selected_in_actual_top5 = None
            selected_equals_predicted_top1 = None
            if (
                strategy_result is not None
                and date in strategy_predictions.index
                and "selected_asset" in strategy_predictions
            ):
                raw_selected = strategy_predictions.loc[date, "selected_asset"]
                selected_asset = str(raw_selected)
                if selected_asset in valid_symbols:
                    selected_index = valid_symbols.index(selected_asset)
                    selected_realized_utility = float(actual[selected_index])
                    selected_actual_percentile = float(
                        actual_rank_pct[selected_index]
                    )
                    selected_in_actual_top5 = bool(
                        selected_index in actual_top5
                    )
                    selected_equals_predicted_top1 = bool(
                        selected_asset == predicted_top1
                    )

            rows.append(
                {
                    "scope": scope,
                    "fold_id": state.fold_id,
                    "timestamp": date,
                    "valid_assets": n,
                    "spearman_ic": _safe_rank_correlation(pred, actual),
                    "ndcg_at_5": ndcg5,
                    "ndcg_at_10": ndcg10,
                    "sign_accuracy": sign_accuracy,
                    "predicted_top1": predicted_top1,
                    "realized_top1": actual_top1,
                    "top1_exact_hit": bool(
                        predicted_top1_index == actual_order[0]
                    ),
                    "top1_in_realized_top3": bool(
                        predicted_top1_index in actual_top3
                    ),
                    "top1_in_realized_top5": bool(
                        predicted_top1_index in actual_top5
                    ),
                    "top1_realized_percentile": float(
                        actual_rank_pct[predicted_top1_index]
                    ),
                    "top3_overlap_rate": float(
                        len(predicted_top3 & actual_top3) / top3_n
                    ),
                    "top1_realized_utility": top1_actual,
                    "oracle_realized_utility": oracle_actual,
                    "top1_regret": float(oracle_actual - top1_actual),
                    "strategy_selected_asset": selected_asset,
                    "strategy_selected_realized_utility": (
                        selected_realized_utility
                    ),
                    "strategy_selected_realized_percentile": (
                        selected_actual_percentile
                    ),
                    "strategy_selected_in_realized_top5": (
                        selected_in_actual_top5
                    ),
                    "strategy_selected_equals_predicted_top1": (
                        selected_equals_predicted_top1
                    ),
                }
            )
    return pd.DataFrame(rows)


def _rank_diagnostic_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = (
        "spearman_ic",
        "ndcg_at_5",
        "ndcg_at_10",
        "sign_accuracy",
        "top1_exact_hit",
        "top1_in_realized_top3",
        "top1_in_realized_top5",
        "top1_realized_percentile",
        "top3_overlap_rate",
        "top1_regret",
        "strategy_selected_realized_percentile",
        "strategy_selected_in_realized_top5",
        "strategy_selected_equals_predicted_top1",
    )
    for scope, scope_frame in sessions.groupby("scope", sort=False):
        groups: list[tuple[str, pd.DataFrame]] = [("ALL", scope_frame)]
        groups.extend(
            (
                str(int(fold_id)),
                fold_frame,
            )
            for fold_id, fold_frame in scope_frame.groupby(
                "fold_id",
                sort=True,
            )
        )
        for fold_label, frame in groups:
            row: dict[str, Any] = {
                "scope": scope,
                "fold_id": fold_label,
                "sessions": int(len(frame)),
                "median_valid_assets": float(
                    frame["valid_assets"].median()
                ),
            }
            for column in numeric:
                if column not in frame:
                    continue
                values = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )
                row[f"{column}_mean"] = (
                    float(values.mean())
                    if values.notna().any()
                    else None
                )
                row[f"{column}_median"] = (
                    float(values.median())
                    if values.notna().any()
                    else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _normal_inv_cdf(probability: float) -> float:
    bounded = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(NormalDist().inv_cdf(bounded))


def _effective_trial_count(returns_matrix: pd.DataFrame) -> float:
    matrix = returns_matrix.to_numpy(dtype=np.float64)
    if matrix.shape[1] <= 1:
        return 1.0
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    squared = float(np.square(eigenvalues).sum())
    if total <= 0.0 or squared <= 0.0:
        return 1.0
    return float((total * total) / squared)


def _expected_max_sharpe(
    trial_sharpes_daily: np.ndarray,
    n_trials: float,
) -> float:
    values = np.asarray(trial_sharpes_daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    variance = float(np.var(values, ddof=1))
    if variance <= 0.0 or n_trials <= 1.0:
        return 0.0
    gamma = 0.5772156649015329
    n = max(float(n_trials), 1.000001)
    first = _normal_inv_cdf(1.0 - 1.0 / n)
    second = _normal_inv_cdf(1.0 - 1.0 / (n * math.e))
    return float(
        math.sqrt(variance)
        * ((1.0 - gamma) * first + gamma * second)
    )


def _deflated_sharpe_ratio(
    selected_returns: pd.Series,
    trial_sharpes_daily: np.ndarray,
    n_trials: float,
) -> dict[str, Any]:
    values = pd.to_numeric(
        selected_returns,
        errors="coerce",
    ).dropna().astype(float)
    sr = _daily_sharpe(values.to_numpy(dtype=np.float64))
    if not np.isfinite(sr) or len(values) < 3:
        return {
            "observations": int(len(values)),
            "n_trials": float(n_trials),
            "probabilistic_sharpe_ratio": None,
            "deflated_sharpe_ratio": None,
        }
    skew = float(values.skew())
    kurtosis = float(values.kurt()) + 3.0
    sr0 = _expected_max_sharpe(trial_sharpes_daily, n_trials)
    variance_term = (
        1.0
        - skew * sr
        + ((kurtosis - 1.0) / 4.0) * sr * sr
    )
    variance_term = max(float(variance_term), 1e-12)
    scale = math.sqrt(len(values) - 1.0) / math.sqrt(variance_term)
    psr_z = float(sr * scale)
    dsr_z = float((sr - sr0) * scale)
    return {
        "observations": int(len(values)),
        "n_trials": float(n_trials),
        "daily_sharpe": float(sr),
        "annualized_sharpe": float(sr * math.sqrt(252.0)),
        "skewness": skew,
        "kurtosis": kurtosis,
        "trial_sharpe_variance_daily": float(
            np.var(
                np.asarray(trial_sharpes_daily, dtype=np.float64)[
                    np.isfinite(
                        np.asarray(
                            trial_sharpes_daily,
                            dtype=np.float64,
                        )
                    )
                ],
                ddof=1,
            )
        ),
        "expected_max_daily_sharpe": float(sr0),
        "expected_max_annualized_sharpe": float(
            sr0 * math.sqrt(252.0)
        ),
        "probabilistic_sharpe_ratio": float(
            NormalDist().cdf(psr_z)
        ),
        "deflated_sharpe_ratio": float(
            NormalDist().cdf(dsr_z)
        ),
    }


def _score_matrix_columns(
    values: np.ndarray,
    metric: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if metric == "log_growth":
        clipped = np.clip(matrix, -0.999999999, None)
        return np.nansum(np.log1p(clipped), axis=0)
    if metric == "sharpe":
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0, ddof=1)
        return np.divide(
            means,
            stds,
            out=np.full_like(means, np.nan, dtype=float),
            where=stds > 0.0,
        )
    raise ValueError(f"Metrica PBO desconhecida: {metric}")


def _pbo_cscv(
    returns_matrix: pd.DataFrame,
    *,
    blocks: int = PBO_BLOCKS,
    metric: str = "log_growth",
) -> tuple[dict[str, Any], pd.DataFrame]:
    if blocks < 2 or blocks % 2 != 0:
        raise ValueError("PBO exige quantidade par de blocos.")
    clean = returns_matrix.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(axis=0, how="any")
    matrix = clean.to_numpy(dtype=np.float64)
    if len(matrix) < blocks:
        raise ValueError("Observacoes insuficientes para PBO.")
    block_indices = np.array_split(np.arange(len(matrix)), blocks)
    all_blocks = tuple(range(blocks))
    rows: list[dict[str, Any]] = []

    for split_id, train_blocks in enumerate(
        combinations(all_blocks, blocks // 2),
        start=1,
    ):
        train_set = set(train_blocks)
        test_blocks = tuple(
            block for block in all_blocks if block not in train_set
        )
        train_idx = np.concatenate(
            [block_indices[block] for block in train_blocks]
        )
        test_idx = np.concatenate(
            [block_indices[block] for block in test_blocks]
        )
        is_scores = _score_matrix_columns(
            matrix[train_idx],
            metric,
        )
        valid_is = np.where(np.isfinite(is_scores), is_scores, -np.inf)
        selected_index = int(np.argmax(valid_is))
        oos_scores = _score_matrix_columns(
            matrix[test_idx],
            metric,
        )
        finite = np.isfinite(oos_scores)
        if not finite[selected_index]:
            continue
        selected_oos = float(oos_scores[selected_index])
        comparison = oos_scores[finite]
        below = int(np.sum(comparison < selected_oos))
        equal = int(np.sum(comparison == selected_oos))
        rank = 1.0 + below + 0.5 * max(equal - 1, 0)
        omega = float(rank / (len(comparison) + 1.0))
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)
        logit = float(math.log(omega / (1.0 - omega)))
        rows.append(
            {
                "metric": metric,
                "split_id": split_id,
                "train_blocks": ",".join(
                    str(value) for value in train_blocks
                ),
                "test_blocks": ",".join(
                    str(value) for value in test_blocks
                ),
                "selected_scenario": clean.columns[selected_index],
                "is_score": float(is_scores[selected_index]),
                "oos_score": selected_oos,
                "oos_rank_percentile": omega,
                "logit_oos_rank": logit,
                "below_median_oos": bool(logit <= 0.0),
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "metric": metric,
        "blocks": int(blocks),
        "expected_splits": int(math.comb(blocks, blocks // 2)),
        "completed_splits": int(len(frame)),
        "scenario_count": int(returns_matrix.shape[1]),
        "observation_count": int(len(clean)),
        "pbo": (
            float(frame["below_median_oos"].mean())
            if not frame.empty
            else None
        ),
        "median_oos_rank_percentile": (
            float(frame["oos_rank_percentile"].median())
            if not frame.empty
            else None
        ),
        "median_logit_oos_rank": (
            float(frame["logit_oos_rank"].median())
            if not frame.empty
            else None
        ),
    }
    return summary, frame


def _dynamic_transition_rows(
    stages: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous: set[str] = set()
    for stage in stages:
        current = set(stage["selected_assets"])
        universe = sorted(previous | current)
        for symbol in universe:
            if symbol in current and symbol in previous:
                action = "MANTER"
            elif symbol in current:
                action = "ADICIONAR"
            else:
                action = "REMOVER"
            rows.append(
                {
                    "stage": stage["stage"],
                    "holdout_fold": stage["holdout_fold"],
                    "asset": symbol,
                    "action": action,
                    "selected_before": symbol in previous,
                    "selected_after": symbol in current,
                }
            )
        previous = current
    return pd.DataFrame(rows)


def _run_decision_validation(
    *,
    snapshot: dict[str, Any],
    baseline_payload: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_result: Any,
    baseline_frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    baseline_symbols: list[str],
    folds: list[dict[str, Any]],
    all_decision_dates: pd.DatetimeIndex,
    decision_to_fold: dict[pd.Timestamp, int],
    decision_metadata: Any,
    config_u67: Any,
    states: list[BaselineFoldState],
    calibration_rows: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    symbols = list(SUBSET_SOURCE_CANDIDATES)
    discovery_selected = list(DISCOVERY_SELECTED_CANDIDATES)
    credentials = load_alpaca_credentials(ROOT)

    download_audit = _prepare_full_candidate_files(credentials, symbols)
    download_audit.to_csv(OUT / "candidate_download.csv", index=False)
    failed = download_audit.loc[
        download_audit["status"].eq("failed"),
        "asset",
    ].astype(str).tolist()
    if failed:
        raise RuntimeError(
            "Falha ao recuperar candidatos para decision validation: "
            + ", ".join(failed)
        )

    candidate_frames, continuity, candidate_hashes = _prepare_candidate_frames(
        symbols,
        common_dates,
        all_decision_dates,
        config_u67,
    )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    missing = [symbol for symbol in symbols if symbol not in candidate_frames]
    if missing:
        raise RuntimeError(
            "Decision validation nao reproduziu os 15 candidatos: "
            + ", ".join(missing)
        )

    candidate_fold_caches: dict[str, dict[int, CandidateFoldCache]] = {}
    for position, symbol in enumerate(symbols, start=1):
        candidate_fold_caches[symbol] = _train_candidate_caches(
            symbol,
            candidate_frames[symbol],
            states,
            config_u67,
        )
        print(
            f"[decision-train] {position}/{len(symbols)} {symbol}",
            flush=True,
        )

    scenario_metrics_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    scenario_returns_cache: dict[tuple[str, ...], pd.Series] = {}
    retained_results: dict[tuple[str, ...], Any] = {}
    scenario_calibration_rows: list[dict[str, Any]] = []
    evaluation_order: list[tuple[str, ...]] = []
    retain_keys = {
        tuple(symbols),
        tuple(
            symbol
            for symbol in symbols
            if symbol in set(discovery_selected)
        ),
    }

    def canonical(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        chosen = set(values)
        return tuple(symbol for symbol in symbols if symbol in chosen)

    def evaluate_subset(
        values: list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        key = canonical(values)
        if not key:
            return baseline_metrics
        if key in scenario_metrics_cache:
            return scenario_metrics_cache[key]
        result, metrics, rows = _evaluate_scenario(
            "U67+DECISION_" + "_".join(key),
            list(key),
            baseline_frames,
            candidate_frames,
            baseline_symbols,
            states,
            candidate_fold_caches,
            folds,
            all_decision_dates,
            decision_to_fold,
            decision_metadata,
            config_u67,
        )
        scenario_metrics_cache[key] = metrics
        scenario_returns_cache[key] = _daily_returns(result)
        if key in retain_keys:
            retained_results[key] = result
        evaluation_order.append(key)
        scenario_calibration_rows.extend(rows)
        print(
            f"[decision-scenario] {len(scenario_metrics_cache):03d} "
            f"n={len(key):02d} capital=US$ "
            f"{metrics['ending_capital']:,.2f}",
            flush=True,
        )
        return metrics

    def full_window_score(values: list[str] | tuple[str, ...]) -> float:
        return float(evaluate_subset(values)["ending_capital"])

    selected_key = canonical(discovery_selected)
    all15_metrics = evaluate_subset(symbols)
    selected_metrics = evaluate_subset(discovery_selected)
    selected_result = retained_results[selected_key]
    if not math.isclose(
        float(all15_metrics["ending_capital"]),
        EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    ):
        raise RuntimeError("Checkpoint U67+15 divergiu na v1.4.")
    if not math.isclose(
        float(selected_metrics["ending_capital"]),
        EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    ):
        raise RuntimeError("Checkpoint U67+10 divergiu na v1.4.")

    full_search = _forward_backward_subset_search(
        symbols,
        full_window_score,
        float(baseline_metrics["ending_capital"]),
        SUBSET_IMPROVEMENT_TOL,
    )
    full_search_keys = list(evaluation_order)
    if len(full_search_keys) != SOURCE_SUBSET_UNIQUE_SCENARIOS:
        raise RuntimeError(
            "Paisagem da busca divergente: "
            f"observado={len(full_search_keys)} "
            f"esperado={SOURCE_SUBSET_UNIQUE_SCENARIOS}"
        )
    if list(full_search["selected"]["assets"]) != discovery_selected:
        raise RuntimeError(
            "Subconjunto selecionado divergiu do Challenger congelado."
        )
    pd.DataFrame(full_search["trace"]).to_csv(
        OUT / "full_window_subset_search.csv",
        index=False,
    )

    scenario_columns: list[str] = []
    scenario_returns: list[pd.Series] = []
    scenario_rows: list[dict[str, Any]] = []
    for position, key in enumerate(full_search_keys, start=1):
        metrics = scenario_metrics_cache[key]
        scenario_id = f"S{position:03d}"
        scenario_columns.append(scenario_id)
        series = scenario_returns_cache[key].rename(scenario_id)
        scenario_returns.append(series)
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "assets": ",".join(key),
                "asset_count": len(key),
                "ending_capital": metrics["ending_capital"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "worst_fold_return": metrics["worst_fold_return"],
                "daily_sharpe": _daily_sharpe(
                    series.to_numpy(dtype=np.float64)
                ),
            }
        )
    multiple_testing_scenarios = pd.DataFrame(scenario_rows)
    multiple_testing_scenarios.to_csv(
        OUT / "multiple_testing_scenarios.csv",
        index=False,
    )
    returns_matrix = pd.concat(scenario_returns, axis=1).dropna(how="any")
    returns_matrix.to_csv(
        OUT / "multiple_testing_daily_returns.csv",
        index_label="timestamp",
    )

    trial_sharpes = multiple_testing_scenarios[
        "daily_sharpe"
    ].to_numpy(dtype=np.float64)
    effective_trials = _effective_trial_count(returns_matrix)
    selected_returns = _daily_returns(selected_result)
    dsr_raw = _deflated_sharpe_ratio(
        selected_returns,
        trial_sharpes,
        float(SOURCE_SUBSET_UNIQUE_SCENARIOS),
    )
    dsr_effective = _deflated_sharpe_ratio(
        selected_returns,
        trial_sharpes,
        effective_trials,
    )
    dsr_conservative = _deflated_sharpe_ratio(
        selected_returns,
        trial_sharpes,
        float(
            SOURCE_DIRECT_TESTED_COUNT
            + SOURCE_SUBSET_UNIQUE_SCENARIOS
        ),
    )
    multiple_testing_summary = {
        "subset_scenario_count": SOURCE_SUBSET_UNIQUE_SCENARIOS,
        "direct_candidate_count": SOURCE_DIRECT_TESTED_COUNT,
        "effective_independent_trial_count_from_subset_returns": (
            effective_trials
        ),
        "dsr_subset_raw_trials": dsr_raw,
        "dsr_subset_effective_trials": dsr_effective,
        "dsr_conservative_326_trials": dsr_conservative,
        "conservative_326_note": (
            "Sensitivity calculation: 146 marginal candidate tests plus 180 "
            "subset scenarios. These trials are correlated and not identical "
            "strategy families, so this is not an independent-trial estimate."
        ),
    }

    pbo_growth_summary, pbo_growth = _pbo_cscv(
        returns_matrix,
        blocks=PBO_BLOCKS,
        metric="log_growth",
    )
    pbo_sharpe_summary, pbo_sharpe = _pbo_cscv(
        returns_matrix,
        blocks=PBO_BLOCKS,
        metric="sharpe",
    )
    pbo_growth.to_csv(OUT / "pbo_log_growth.csv", index=False)
    pbo_sharpe.to_csv(OUT / "pbo_sharpe.csv", index=False)
    multiple_testing_summary["pbo_log_growth"] = pbo_growth_summary
    multiple_testing_summary["pbo_sharpe"] = pbo_sharpe_summary
    _write_json(
        OUT / "multiple_testing_summary.json",
        multiple_testing_summary,
    )

    rank_frames = [
        _rank_diagnostic_sessions(
            scope="u67_baseline",
            symbols=baseline_symbols,
            states=states,
            baseline_symbols=baseline_symbols,
            baseline_frames=baseline_frames,
            candidate_frames=candidate_frames,
            candidate_fold_caches=candidate_fold_caches,
            strategy_result=baseline_result,
        ),
        _rank_diagnostic_sessions(
            scope="challenger_u67_plus_10",
            symbols=sorted(
                [*baseline_symbols, *discovery_selected]
            ),
            states=states,
            baseline_symbols=baseline_symbols,
            baseline_frames=baseline_frames,
            candidate_frames=candidate_frames,
            candidate_fold_caches=candidate_fold_caches,
            strategy_result=selected_result,
        ),
        _rank_diagnostic_sessions(
            scope="candidate_pool_15",
            symbols=symbols,
            states=states,
            baseline_symbols=baseline_symbols,
            baseline_frames=baseline_frames,
            candidate_frames=candidate_frames,
            candidate_fold_caches=candidate_fold_caches,
            strategy_result=None,
        ),
    ]
    rank_sessions = pd.concat(rank_frames, ignore_index=True)
    rank_sessions.to_csv(
        OUT / "decision_rank_sessions.csv",
        index=False,
    )
    rank_summary = _rank_diagnostic_summary(rank_sessions)
    rank_summary.to_csv(
        OUT / "decision_rank_summary.csv",
        index=False,
    )

    temporal_rows: list[dict[str, Any]] = []
    temporal_trace_frames: list[pd.DataFrame] = []
    for stage, selection_folds, holdout_fold in (
        ("past_fold_1_to_future_fold_2", [1], 2),
        ("past_folds_1_2_to_future_fold_3", [1, 2], 3),
    ):
        def temporal_score(values: list[str] | tuple[str, ...]) -> float:
            return _compound_fold_capital(
                evaluate_subset(values),
                selection_folds,
                initial_capital=float(config_u67.initial_capital),
            )

        baseline_selection = _compound_fold_capital(
            baseline_metrics,
            selection_folds,
            initial_capital=float(config_u67.initial_capital),
        )
        search = _forward_backward_subset_search(
            symbols,
            temporal_score,
            baseline_selection,
            SUBSET_IMPROVEMENT_TOL,
        )
        chosen = list(search["selected"]["assets"])
        chosen_metrics = evaluate_subset(chosen)
        heldout = _fold_metric(chosen_metrics, holdout_fold)
        baseline_holdout = _fold_metric(
            baseline_metrics,
            holdout_fold,
        )
        temporal_rows.append(
            {
                "stage": stage,
                "selection_folds": ",".join(
                    str(value) for value in selection_folds
                ),
                "holdout_fold": holdout_fold,
                "selected_assets": chosen,
                "selected_asset_count": len(chosen),
                "selection_score": search["selected"]["ending_capital"],
                "holdout_return": heldout["strategy_return"],
                "u67_holdout_return": baseline_holdout["strategy_return"],
                "holdout_excess_vs_u67": (
                    float(heldout["strategy_return"])
                    - float(baseline_holdout["strategy_return"])
                ),
                "jaccard_vs_discovery_10": _jaccard(
                    chosen,
                    discovery_selected,
                ),
            }
        )
        trace = pd.DataFrame(search["trace"])
        if not trace.empty:
            trace.insert(0, "stage", stage)
            temporal_trace_frames.append(trace)

    temporal_output = pd.DataFrame(
        [
            {
                **row,
                "selected_assets": ",".join(row["selected_assets"]),
            }
            for row in temporal_rows
        ]
    )
    temporal_output.to_csv(
        OUT / "dynamic_universe_temporal_validation.csv",
        index=False,
    )
    transition_frame = _dynamic_transition_rows(temporal_rows)
    transition_frame.to_csv(
        OUT / "dynamic_universe_transitions.csv",
        index=False,
    )
    (
        pd.concat(temporal_trace_frames, ignore_index=True)
        if temporal_trace_frames
        else pd.DataFrame()
    ).to_csv(
        OUT / "dynamic_universe_search.csv",
        index=False,
    )

    pd.DataFrame(
        [*calibration_rows, *scenario_calibration_rows]
    ).to_csv(
        OUT / "scenario_fold_calibration.csv",
        index=False,
    )

    rank_overall = rank_summary.loc[
        rank_summary["fold_id"].eq("ALL")
    ].to_dict(orient="records")
    payload = {
        "execution_schema": EXECUTION_SCHEMA,
        "version": SCRIPT_VERSION,
        "status": "completed",
        "purpose": "u67_decision_and_selection_validation",
        "run_mode": RUN_MODE,
        "baseline": baseline_payload,
        "checkpoints": {
            "u67_exact_to_cent": baseline_payload["exact_to_cent"],
            "all_15_exact_to_cent": True,
            "selected_10_exact_to_cent": True,
            "all_15_ending_capital": all15_metrics["ending_capital"],
            "selected_10_ending_capital": (
                selected_metrics["ending_capital"]
            ),
            "full_window_subset_scenarios_reproduced": len(
                full_search_keys
            ),
        },
        "decision_ranking": {
            "overall": rank_overall,
            "interpretation": (
                "Ranking diagnostics compare predictions available at the "
                "decision session with forward utility realized afterward. "
                "They are evaluation diagnostics and are not used to make "
                "historical decisions."
            ),
        },
        "multiple_testing": multiple_testing_summary,
        "dynamic_universe": {
            "asset_list_is_permanent": False,
            "stages": temporal_rows,
            "transition_actions": (
                transition_frame.groupby(
                    ["stage", "action"],
                    sort=False,
                ).size().reset_index(name="count").to_dict(
                    orient="records"
                )
            ),
            "current_scope": (
                "Fold-boundary pilot over the fixed 15-candidate discovery "
                "pool. It does not yet rediscover the external universe daily."
            ),
        },
        "data": {
            "baseline_snapshot_sha256": snapshot.get("snapshot_sha256"),
            "candidate_file_hashes": candidate_hashes,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(
        OUT / "pesquisa_u67_decision_validation.json",
        payload,
    )

    package = criar_pacote_analise(
        OUT,
        comparison_file="pesquisa_u67_decision_validation.json",
        execution_schema=EXECUTION_SCHEMA,
        archive_name="pacote_analise_u67_decision_validation.zip",
    )
    print("=" * 78, flush=True)
    print(
        f"[final-decision] scenarios={len(full_search_keys)} "
        f"effective_trials={effective_trials:.2f} "
        f"PBO_growth={pbo_growth_summary['pbo']:.4f} "
        f"PBO_sharpe={pbo_sharpe_summary['pbo']:.4f}",
        flush=True,
    )
    print(
        "[final-decision] dynamic_stages="
        + " | ".join(
            f"{row['stage']}:{','.join(row['selected_assets'])}"
            for row in temporal_rows
        ),
        flush=True,
    )
    print(f"[final-decision] package={package}", flush=True)
    print("=" * 78, flush=True)
    sinal_sonoro_conclusao()
    return payload


def _marginal_row(
    symbol: str,
    sample_row: dict[str, Any],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    changes: pd.DataFrame,
    new_trades: pd.DataFrame,
) -> dict[str, Any]:
    delta, delta_rate = _capital_delta(metrics.get("ending_capital"), baseline_metrics.get("ending_capital"))
    return {
        "asset": symbol,
        "sample_order": sample_row.get("sample_order"),
        "catalog_currently_eligible": sample_row.get("catalog_currently_eligible"),
        "ending_capital": metrics.get("ending_capital"),
        "ending_capital_delta": delta,
        "ending_capital_delta_rate": delta_rate,
        "cagr": metrics.get("cagr"),
        "cagr_delta": _numeric_delta(metrics.get("cagr"), baseline_metrics.get("cagr")),
        "sharpe": metrics.get("sharpe"),
        "sharpe_delta": _numeric_delta(metrics.get("sharpe"), baseline_metrics.get("sharpe")),
        "maximum_drawdown": metrics.get("maximum_drawdown"),
        "maximum_drawdown_delta": _numeric_delta(metrics.get("maximum_drawdown"), baseline_metrics.get("maximum_drawdown")),
        "worst_fold_return": metrics.get("worst_fold_return"),
        "worst_fold_delta": _numeric_delta(metrics.get("worst_fold_return"), baseline_metrics.get("worst_fold_return")),
        "capital_rotations": metrics.get("capital_rotations"),
        "capital_rotations_delta": int(metrics.get("capital_rotations") or 0) - int(baseline_metrics.get("capital_rotations") or 0),
        "decision_sessions_changed": int(len(changes)),
        "new_asset_trade_rows": int(len(new_trades)),
        "positive_capital_delta": bool(delta is not None and delta > 0.0),
    }


def main() -> dict[str, Any]:
    started = time.perf_counter()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("TCC MBA USP - PESQUISA INCREMENTAL U67+n", flush=True)
    print(f"version={SCRIPT_VERSION} schema={EXECUTION_SCHEMA}", flush=True)
    print("baseline=U67_FROZEN_GIT_SNAPSHOT", flush=True)
    print("policy=append_only_no_reconstruction_no_replacement", flush=True)
    print(f"analysis_end={ANALYSIS_END_DATE}", flush=True)
    print("=" * 78, flush=True)

    # Baseline congelado. Nenhum download ou escrita ocorre em dados/u67.
    snapshot = validate_snapshot(BASE_DATA)
    manifest_assets = tuple(snapshot.get("assets") or ())
    if len(manifest_assets) != U67_EXPECTED_REQUESTED_COUNT or manifest_assets != tuple(U67_REQUESTED_ASSETS):
        raise RuntimeError("Snapshot U67 nao corresponde ao contrato oficial de 67 ativos.")
    baseline_raw, exclusions, diagnostics, data_audit = prepare_model_frames(
        BASE_DATA,
        assets=U67_REQUESTED_ASSETS,
        csv_float_precision="round_trip",
    )
    excluded = frozenset(str(item.get("symbol") or "").strip().upper() for item in exclusions)
    if excluded != U67_EXPECTED_EXCLUSIONS or len(baseline_raw) != U67_EXPECTED_EFFECTIVE_COUNT:
        raise RuntimeError(
            f"Contrato U67 efetivo divergente: count={len(baseline_raw)} exclusions={sorted(excluded)}"
        )
    baseline_order = tuple(symbol for symbol in U67_REQUESTED_ASSETS if symbol in baseline_raw)
    config_u67 = build_control_config({symbol: baseline_raw[symbol] for symbol in baseline_order}, CONFIG)
    config_u67 = config_u67.copiar_modelo(update={"analysis_end_date": ANALYSIS_END_DATE})
    (
        baseline_frames,
        common_dates,
        calendar_source,
        baseline_symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _construir_contexto_execucao(
        {symbol: baseline_raw[symbol] for symbol in baseline_order},
        config_u67,
    )
    if len(baseline_symbols) != U67_EXPECTED_EFFECTIVE_COUNT:
        raise RuntimeError("Contexto U67 nao preservou os 65 ativos efetivos.")

    states, baseline_policies, calibration_rows = _train_baseline_states(
        baseline_frames, common_dates, baseline_symbols, folds, config_u67
    )
    baseline_result = _simular_exato(
        "tcc_u67_asset_expansion_baseline",
        _politica_agendada(baseline_policies, decision_to_fold),
        baseline_frames,
        baseline_symbols,
        all_decision_dates,
        config_u67,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="TCC U67 baseline",
        method_line="- Snapshot U67 congelado; baseline fixo da pesquisa de expansao.",
    )
    baseline_metrics = _scenario_metrics(baseline_result, folds, float(config_u67.initial_capital))
    baseline_delta = float(baseline_metrics["ending_capital"] - EXPECTED_U67_ENDING_CAPITAL)
    baseline_exact = math.isclose(
        float(baseline_metrics["ending_capital"]),
        EXPECTED_U67_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    )
    baseline_payload = {
        "metrics": baseline_metrics,
        "expected_ending_capital": EXPECTED_U67_ENDING_CAPITAL,
        "delta_from_expected": baseline_delta,
        "exact_to_cent": baseline_exact,
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "requested_assets": len(U67_REQUESTED_ASSETS),
        "effective_assets": len(baseline_symbols),
        "excluded_assets": sorted(excluded),
        "calendar_source": calendar_source,
        "common_dates": len(common_dates),
        "decision_sessions": len(all_decision_dates) - 1,
    }
    _write_json(OUT / "u67_baseline_metrics.json", baseline_payload)
    if not baseline_exact:
        raise RuntimeError(
            "Baseline U67 nao reproduziu o checkpoint oficial. "
            f"observado={baseline_metrics['ending_capital']:.8f} "
            f"esperado={EXPECTED_U67_ENDING_CAPITAL:.8f}"
        )
    print(
        f"[baseline] capital=US$ {baseline_metrics['ending_capital']:,.2f} exact_to_cent=True",
        flush=True,
    )

    if RUN_MODE == "decision_validation":
        return _run_decision_validation(
            snapshot=snapshot,
            baseline_payload=baseline_payload,
            baseline_metrics=baseline_metrics,
            baseline_result=baseline_result,
            baseline_frames=baseline_frames,
            common_dates=common_dates,
            baseline_symbols=baseline_symbols,
            folds=folds,
            all_decision_dates=all_decision_dates,
            decision_to_fold=decision_to_fold,
            decision_metadata=decision_metadata,
            config_u67=config_u67,
            states=states,
            calibration_rows=calibration_rows,
            started=started,
        )

    if RUN_MODE == "assurance_validation":
        return _run_assurance_validation(
            snapshot=snapshot,
            baseline_payload=baseline_payload,
            baseline_metrics=baseline_metrics,
            baseline_result=baseline_result,
            baseline_frames=baseline_frames,
            common_dates=common_dates,
            baseline_symbols=baseline_symbols,
            folds=folds,
            all_decision_dates=all_decision_dates,
            decision_to_fold=decision_to_fold,
            decision_metadata=decision_metadata,
            config_u67=config_u67,
            states=states,
            calibration_rows=calibration_rows,
            started=started,
        )

    if RUN_MODE == "subset_optimization":
        return _run_subset_optimization(
            snapshot=snapshot,
            baseline_payload=baseline_payload,
            baseline_metrics=baseline_metrics,
            baseline_result=baseline_result,
            baseline_frames=baseline_frames,
            common_dates=common_dates,
            baseline_symbols=baseline_symbols,
            folds=folds,
            all_decision_dates=all_decision_dates,
            decision_to_fold=decision_to_fold,
            decision_metadata=decision_metadata,
            config_u67=config_u67,
            states=states,
            calibration_rows=calibration_rows,
            started=started,
        )
    if RUN_MODE != "direct_marginal":
        raise RuntimeError(f"RUN_MODE desconhecido: {RUN_MODE}")

    # O modo direct_marginal reutiliza exatamente os 500 simbolos da campanha inicial.
    # Nenhum filtro de preco/liquidez/volume ou ranking decide quem recebe replay.
    credentials = load_alpaca_credentials(ROOT)
    catalog = _eligible_catalog(
        _load_catalog(credentials),
        str(snapshot.get("snapshot_sha256") or ""),
    )
    catalog.to_csv(OUT / "candidate_catalog.csv", index=False)
    sample = _load_candidate_sample(catalog)
    sample.to_csv(OUT / "candidate_sample.csv", index=False)
    selected_symbols = list(sample["asset"].astype(str))
    print(
        f"[direct-search] catalog={len(catalog)} sample={len(selected_symbols)} "
        "screening_gate=False ranking_gate=False",
        flush=True,
    )

    download_audit = _prepare_full_candidate_files(credentials, selected_symbols)
    download_audit.to_csv(OUT / "candidate_download.csv", index=False)
    download_lookup = {
        str(row.asset): row._asdict()
        for row in download_audit.itertuples(index=False)
    }
    sample_lookup = {
        str(row.asset): row._asdict()
        for row in sample.itertuples(index=False)
    }

    # Processamento sequencial: candidatos negativos nao permanecem em memoria.
    # Somente positivos conservam frame/cache para a validacao conjunta posterior.
    candidate_frames: dict[str, pd.DataFrame] = {}
    candidate_fold_caches: dict[str, dict[int, CandidateFoldCache]] = {}
    candidate_hashes: dict[str, dict[str, str]] = {}
    usable_symbols: list[str] = []
    model_failures: dict[str, str] = {}
    continuity_frames: list[pd.DataFrame] = []
    marginal_rows: list[dict[str, Any]] = []
    decision_change_frames: list[pd.DataFrame] = []
    new_trade_frames: list[pd.DataFrame] = []
    individual_results: dict[str, tuple[Any, dict[str, Any]]] = {}

    for position, symbol in enumerate(selected_symbols, start=1):
        download_row = download_lookup.get(symbol, {})
        if download_row.get("status") == "failed":
            continuity_frames.append(
                pd.DataFrame(
                    [
                        {
                            "asset": symbol,
                            "status": "failed",
                            "reason": "download_failed: " + str(download_row.get("reason") or ""),
                        }
                    ]
                )
            )
            print(
                f"[direct] {position}/{len(selected_symbols)} {symbol} skipped=download_failed",
                flush=True,
            )
            continue

        frames_one, audit_one, hashes_one = _prepare_candidate_frames(
            [symbol],
            common_dates,
            all_decision_dates,
            config_u67,
        )
        continuity_frames.append(audit_one)
        candidate_hashes.update(hashes_one)
        if symbol not in frames_one:
            print(
                f"[direct] {position}/{len(selected_symbols)} {symbol} skipped=structural_audit",
                flush=True,
            )
            continue

        frame = frames_one[symbol]
        try:
            caches = _train_candidate_caches(symbol, frame, states, config_u67)
        except Exception as exc:
            model_failures[symbol] = str(exc)
            print(
                f"[direct] {position}/{len(selected_symbols)} {symbol} model_failed={exc}",
                flush=True,
            )
            continue

        usable_symbols.append(symbol)
        result, metrics, rows = _evaluate_scenario(
            f"U67+{symbol}",
            [symbol],
            baseline_frames,
            {symbol: frame},
            baseline_symbols,
            states,
            {symbol: caches},
            folds,
            all_decision_dates,
            decision_to_fold,
            decision_metadata,
            config_u67,
        )
        calibration_rows.extend(rows)
        changes = _decision_changes(f"U67+{symbol}", baseline_result, result, {symbol})
        trades = _new_asset_trades(f"U67+{symbol}", result, {symbol})
        if not changes.empty:
            decision_change_frames.append(changes)
        if not trades.empty:
            new_trade_frames.append(trades)

        marginal_row = _marginal_row(
            symbol,
            sample_lookup.get(symbol, {}),
            metrics,
            baseline_metrics,
            changes,
            trades,
        )
        marginal_rows.append(marginal_row)
        delta, rate = _capital_delta(
            metrics["ending_capital"],
            baseline_metrics["ending_capital"],
        )
        if delta is not None and delta > 0.0:
            candidate_frames[symbol] = frame
            candidate_fold_caches[symbol] = caches
            individual_results[symbol] = (result, metrics)

        # Checkpoint legivel mesmo se uma execucao longa for interrompida.
        pd.DataFrame(marginal_rows).to_csv(
            OUT / "marginal_replay_partial.csv",
            index=False,
        )
        pd.concat(continuity_frames, ignore_index=True).to_csv(
            OUT / "continuity_audit_partial.csv",
            index=False,
        )
        print(
            f"[marginal] {position}/{len(selected_symbols)} {symbol} "
            f"capital=US$ {metrics['ending_capital']:,.2f} "
            f"delta=US$ {delta:+,.2f} ({rate:+.4%})",
            flush=True,
        )

    continuity = (
        pd.concat(continuity_frames, ignore_index=True)
        if continuity_frames
        else pd.DataFrame()
    )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    if not marginal_rows:
        raise RuntimeError(
            "Nenhum candidato da amostra congelada chegou ao replay financeiro."
        )

    marginal = pd.DataFrame(marginal_rows).sort_values(
        ["ending_capital_delta", "sample_order"],
        ascending=[False, True],
        kind="stable",
    )
    marginal.to_csv(OUT / "marginal_replay.csv", index=False)
    positives = marginal.loc[marginal["positive_capital_delta"]].copy()
    positives.to_csv(OUT / "positive_candidates.csv", index=False)

    combined_payload: dict[str, Any] = {
        "status": "no_positive_candidates",
        "baseline_ending_capital": baseline_metrics["ending_capital"],
        "positive_candidates": list(positives["asset"].astype(str)),
        "all_positive": None,
        "greedy_subset": None,
        "selected_expansion": [],
    }
    selected_expansion: list[str] = []
    if not positives.empty:
        positive_symbols = list(positives["asset"].astype(str))
        result_all, metrics_all, rows = _evaluate_scenario(
            "U67+ALL_POSITIVES",
            positive_symbols,
            baseline_frames,
            candidate_frames,
            baseline_symbols,
            states,
            candidate_fold_caches,
            folds,
            all_decision_dates,
            decision_to_fold,
            decision_metadata,
            config_u67,
        )
        calibration_rows.extend(rows)
        delta_all, rate_all = _capital_delta(metrics_all["ending_capital"], baseline_metrics["ending_capital"])
        changes_all = _decision_changes(
            "U67+ALL_POSITIVES", baseline_result, result_all, set(positive_symbols)
        )
        trades_all = _new_asset_trades("U67+ALL_POSITIVES", result_all, set(positive_symbols))
        if not changes_all.empty:
            decision_change_frames.append(changes_all)
        if not trades_all.empty:
            new_trade_frames.append(trades_all)
        combined_payload["all_positive"] = {
            "assets": positive_symbols,
            "metrics": metrics_all,
            "ending_capital_delta": delta_all,
            "ending_capital_delta_rate": rate_all,
            "decision_sessions_changed": int(len(changes_all)),
            "new_asset_trade_rows": int(len(trades_all)),
        }
        if delta_all is not None and delta_all > 0.0:
            combined_payload["status"] = "all_positive_candidates_improved_baseline"
            selected_expansion = positive_symbols
        else:
            # Subconjunto guloso: candidatos em ordem de delta marginal individual.
            # Como os modelos por ativo ja estao em cache, esta etapa nao refaz treino.
            greedy_assets: list[str] = []
            greedy_capital = float(baseline_metrics["ending_capital"])
            greedy_metrics: dict[str, Any] | None = None
            greedy_best_result: Any | None = None
            greedy_steps = []
            for symbol in positive_symbols:
                trial = [*greedy_assets, symbol]
                if len(trial) == 1:
                    trial_result, trial_metrics = individual_results[symbol]
                    trial_rows = []
                else:
                    trial_result, trial_metrics, trial_rows = _evaluate_scenario(
                        "U67+GREEDY_" + "_".join(trial),
                        trial,
                        baseline_frames,
                        candidate_frames,
                        baseline_symbols,
                        states,
                        candidate_fold_caches,
                        folds,
                        all_decision_dates,
                        decision_to_fold,
                        decision_metadata,
                        config_u67,
                    )
                    calibration_rows.extend(trial_rows)
                accepted = float(trial_metrics["ending_capital"]) > greedy_capital
                greedy_steps.append(
                    {
                        "candidate": symbol,
                        "trial_assets": trial,
                        "trial_ending_capital": trial_metrics["ending_capital"],
                        "accepted": accepted,
                    }
                )
                if accepted:
                    greedy_assets = trial
                    greedy_capital = float(trial_metrics["ending_capital"])
                    greedy_metrics = trial_metrics
                    greedy_best_result = trial_result
            delta_greedy, rate_greedy = _capital_delta(greedy_capital, baseline_metrics["ending_capital"])
            combined_payload["greedy_subset"] = {
                "assets": greedy_assets,
                "ending_capital": greedy_capital,
                "ending_capital_delta": delta_greedy,
                "ending_capital_delta_rate": rate_greedy,
                "metrics": greedy_metrics,
                "steps": greedy_steps,
            }
            if greedy_assets and delta_greedy is not None and delta_greedy > 0.0:
                combined_payload["status"] = "greedy_positive_subset_improved_baseline"
                selected_expansion = greedy_assets
                if greedy_best_result is not None:
                    changes_greedy = _decision_changes(
                        "U67+GREEDY_SELECTED", baseline_result, greedy_best_result, set(greedy_assets)
                    )
                    trades_greedy = _new_asset_trades(
                        "U67+GREEDY_SELECTED", greedy_best_result, set(greedy_assets)
                    )
                    if not changes_greedy.empty:
                        decision_change_frames.append(changes_greedy)
                    if not trades_greedy.empty:
                        new_trade_frames.append(trades_greedy)
            else:
                combined_payload["status"] = "positive_individuals_failed_joint_validation"
    combined_payload["selected_expansion"] = selected_expansion
    _write_json(OUT / "combined_positive_replay.json", combined_payload)

    decision_changes = (
        pd.concat(decision_change_frames, ignore_index=True)
        if decision_change_frames
        else pd.DataFrame()
    )
    decision_changes.to_csv(OUT / "decision_changes.csv", index=False)
    trades_new = (
        pd.concat(new_trade_frames, ignore_index=True)
        if new_trade_frames
        else pd.DataFrame()
    )
    trades_new.to_csv(OUT / "trades_new_assets.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "scenario_fold_calibration.csv", index=False)

    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": _package_version("lightgbm"),
        "scikit_learn": _package_version("scikit-learn"),
    }
    payload = {
        "execution_schema": EXECUTION_SCHEMA,
        "version": SCRIPT_VERSION,
        "status": "completed",
        "purpose": "incremental_u67_asset_expansion",
        "policy": {
            "baseline_reconstructed": False,
            "u67_assets_removed": False,
            "u67_assets_replaced": False,
            "selection_rule": "positive_full_strategy_ending_capital_delta_then_joint_validation",
            "risk_metrics_are_reported_not_hard_gated": True,
        },
        "baseline": baseline_payload,
        "ranker": {
            "used_for_selection": False,
            "status": "disabled_in_direct_marginal_v1_1_0",
        },
        "candidate_search": {
            "active_alpaca_catalog_observed": True,
            "survivorship_bias_fully_eliminated": False,
            "exploratory_asset_discovery": True,
            "selection_bias_possible": True,
            "sample_reused_from_initial_campaign": True,
            "sample_manifest": CANDIDATE_SAMPLE_MANIFEST.name,
            "sample_manifest_sha256": _sha256_file(CANDIDATE_SAMPLE_MANIFEST),
            "catalog_external_count": int(len(catalog)),
            "sample_size": int(len(selected_symbols)),
            "screening_gate_enabled": False,
            "ranking_gate_enabled": False,
            "direct_marginal_replay": True,
            "download_failures": int((download_audit["status"] == "failed").sum()),
            "structural_audit_passed": int(
                (continuity["status"] == "passed").sum()
                if not continuity.empty and "status" in continuity.columns
                else 0
            ),
            "financial_usable_candidates": usable_symbols,
            "pre_replay_checks": [
                "raw_history_available",
                "corporate_actions_available",
                "no_structural_identity_change",
                "complete_u67_calendar_continuity",
                "complete_execution_prices",
            ],
            "performance_filters_before_replay": [],
        },
        "data": {
            "baseline_snapshot_sha256": snapshot.get("snapshot_sha256"),
            "candidate_financial_replay_feed": "sip",
            "candidate_financial_replay_timeframe": "1Day",
            "candidate_financial_replay_adjustment": "raw",
            "candidate_financial_replay_start": START_DATE,
            "candidate_financial_replay_end": ANALYSIS_END_DATE,
            "candidate_file_hashes": candidate_hashes,
            "baseline_data_audit": data_audit,
            "baseline_diagnostics": diagnostics,
        },
        "marginal": {
            "tested_count": int(len(marginal)),
            "positive_count": int(len(positives)),
            "positive_candidates": list(positives["asset"].astype(str)),
        },
        "combined": combined_payload,
        "model_failures": model_failures,
        "runtime": runtime,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(OUT / "pesquisa_u67_asset_expansion.json", payload)

    package = criar_pacote_analise(
        OUT,
        comparison_file="pesquisa_u67_asset_expansion.json",
        execution_schema=EXECUTION_SCHEMA,
        archive_name="pacote_analise_u67_asset_expansion.zip",
    )
    print("=" * 78, flush=True)
    print(
        f"[final] baseline=US$ {baseline_metrics['ending_capital']:,.2f} "
        f"positivos={len(positives)} selected_expansion={selected_expansion}",
        flush=True,
    )
    print(f"[final] package={package}", flush=True)
    print("=" * 78, flush=True)
    sinal_sonoro_conclusao()
    return payload


if __name__ == "__main__":
    main()
