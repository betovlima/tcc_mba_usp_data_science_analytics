"""Pesquisa de expansao aleatoria U56 -> U76 tratando cada ativo como objeto.

A campanha adiciona 20 ativos escolhidos por amostragem pseudoaleatoria
reprodutivel e mede:
- efeito do grupo de 20 no capital final;
- contribuicao marginal leave-one-out de cada um dos 76 objetos;
- propriedades de mercado, score, ranking, selecao e calibracao de cada objeto;
- relacao entre propriedades e contribuicao positiva/negativa ao lucro.

Os 20 ativos novos usam um snapshot separado e congelavel. O snapshot original
de 56 tickers nao e alterado.
"""

# %% 0 - Imports e configuracao
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd

from engine.configuracao import (
    ANALYSIS_END_DATE,
    BAR_SNAPSHOT_AS_OF_END,
    CONFIG,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    _selecionar_switch_margin_fold,
)
from engine.rotacao import (
    _benchmark_pesos_iguais,
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
    preparar_painel_rotacao,
)
from pesquisas.directional_change_lightgbm import (
    EXPECTED_EXECUTION_SCHEMA,
    RESEARCH_VERSION,
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.experimento import build_variant_configs, summarize_metrics
from reproducao.preparacao import prepare_model_frames


RAIZ_PROJETO = Path(__file__).resolve().parent
SNAPSHOT_BASE = SnapshotPaths.research(RAIZ_PROJETO)
SNAPSHOT_EXTENSAO = SnapshotPaths.from_root(
    RAIZ_PROJETO / "dados" / "pesquisa_expansao_76"
)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"
SCRIPT_RESEARCH_VERSION = "1.13.2-dev.1"
EXECUTION_SCHEMA = "object-universe-expansion-76-v2"

RANDOM_SELECTION_SEED = 20261004
RANDOM_SELECTION_CATALOG_DATE = "2026-10-04"
RANDOM_SELECTION_RULE = (
    "Alpaca active/tradable/marginable US equities; exchanges NYSE/NASDAQ/"
    "AMEX/ARCA/BATS; simple ticker; excludes original 56; deterministic "
    "xorshift32 seed=20261004; requires continuous daily SIP RAW history "
    "covering 2016-01 through 2026-09 with >=2600 rows and max gap <=10 days."
)

RANDOM_ASSET_OBJECTS = (
    {"symbol": "FAF", "name": "First American Financial Corporation", "exchange": "NYSE"},
    {"symbol": "IJR", "name": "iShares Core S&P Small-Cap ETF", "exchange": "ARCA"},
    {"symbol": "GAB", "name": "The Gabelli Equity Trust Inc.", "exchange": "NYSE"},
    {"symbol": "ELS", "name": "Equity Lifestyle Properties, Inc.", "exchange": "NYSE"},
    {"symbol": "AEIS", "name": "Advanced Energy Industries, Inc.", "exchange": "NASDAQ"},
    {"symbol": "VWOB", "name": "Vanguard Emerging Markets Government Bond ETF", "exchange": "NASDAQ"},
    {"symbol": "BDJ", "name": "BlackRock Enhanced Equity Dividend Trust", "exchange": "NYSE"},
    {"symbol": "DGX", "name": "Quest Diagnostics Inc.", "exchange": "NYSE"},
    {"symbol": "ESP", "name": "Espey Mfg. & Electronics Corp", "exchange": "AMEX"},
    {"symbol": "BWZ", "name": "SPDR Bloomberg Short Term International Treasury Bond ETF", "exchange": "ARCA"},
    {"symbol": "PSF", "name": "Cohen & Steers Select Preferred and Income Fund", "exchange": "NYSE"},
    {"symbol": "DBA", "name": "Invesco DB Agriculture Fund", "exchange": "ARCA"},
    {"symbol": "HEEM", "name": "iShares Currency Hedged MSCI Emerging Markets", "exchange": "BATS"},
    {"symbol": "NPKI", "name": "NPK International Inc.", "exchange": "NYSE"},
    {"symbol": "MHK", "name": "Mohawk Industries, Inc.", "exchange": "NYSE"},
    {"symbol": "BLKB", "name": "Blackbaud, Inc.", "exchange": "NASDAQ"},
    {"symbol": "ARCO", "name": "Arcos Dorados Holdings Inc.", "exchange": "NYSE"},
    {"symbol": "AGM", "name": "Federal Agricultural Mortgage Corporation", "exchange": "NYSE"},
    {"symbol": "NWFL", "name": "Norwood Financial Corp.", "exchange": "NASDAQ"},
    {"symbol": "SKOR", "name": "FlexShares Credit-Scored US Corporate Bond ETF", "exchange": "NASDAQ"},
)
RANDOM_ASSETS = tuple(item["symbol"] for item in RANDOM_ASSET_OBJECTS)
RANDOM_METADATA = {
    item["symbol"]: dict(item)
    for item in RANDOM_ASSET_OBJECTS
}


# %% 1 - Guards e snapshot base
if EXECUTION_SCHEMA != EXPECTED_EXECUTION_SCHEMA:
    raise RuntimeError(
        "Script e modulo de pesquisa incompatíveis antes do replay: "
        f"script={EXECUTION_SCHEMA!r} "
        f"modulo={EXPECTED_EXECUTION_SCHEMA!r}. "
        "Atualize a branch e reinicie o kernel do Spyder."
    )

if SCRIPT_RESEARCH_VERSION != RESEARCH_VERSION:
    raise RuntimeError(
        "Versao do runner e modulo de pesquisa incompatíveis antes do replay: "
        f"runner={SCRIPT_RESEARCH_VERSION!r} "
        f"modulo={RESEARCH_VERSION!r}. "
        "Restaure pesquisar_directional_change_spyder.py da branch remota "
        "e reinicie o kernel do Spyder."
    )

if len(RANDOM_ASSETS) != 20 or len(set(RANDOM_ASSETS)) != 20:
    raise RuntimeError("A expansao precisa conter exatamente 20 ativos unicos.")

print("=" * 78, flush=True)
print("TCC - Object Universe Expansion U56 -> U76", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print(f"versao_runner={SCRIPT_RESEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print(f"script_path={Path(__file__).resolve()}", flush=True)
print(f"random_seed={RANDOM_SELECTION_SEED}", flush=True)
print("random_assets=" + ",".join(RANDOM_ASSETS), flush=True)
print("=" * 78, flush=True)

manifesto_base = validate_snapshot(SNAPSHOT_BASE)


# %% 2 - Snapshot congelado dos 20 ativos adicionais

def _extension_manifest_is_usable() -> bool:
    if not SNAPSHOT_EXTENSAO.manifest.exists():
        return False
    try:
        payload = validate_snapshot(SNAPSHOT_EXTENSAO)
    except Exception:
        return False
    return (
        tuple(payload.get("assets") or ()) == RANDOM_ASSETS
        and str(payload.get("parent_snapshot_sha256") or "")
        == str(manifesto_base.get("snapshot_sha256") or "")
        and str((payload.get("bars") or {}).get("bar_snapshot_as_of_end") or "")
        == str(BAR_SNAPSHOT_AS_OF_END)
    )


if not _extension_manifest_is_usable():
    print(
        "[extension] snapshot ausente/incompativel; "
        "baixando somente os 20 ativos novos da Alpaca",
        flush=True,
    )
    SNAPSHOT_EXTENSAO.clear_generated()
    credenciais = load_alpaca_credentials(RAIZ_PROJETO)
    arquivos_barras = download_raw_bars(
        credenciais,
        SNAPSHOT_EXTENSAO,
        assets=RANDOM_ASSETS,
        replace=True,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
    )
    arquivos_eventos = download_corporate_actions(
        credenciais,
        SNAPSHOT_EXTENSAO,
        assets=RANDOM_ASSETS,
        replace=True,
        query_end=ANALYSIS_END_DATE,
    )
    build_snapshot_manifest(
        SNAPSHOT_EXTENSAO,
        arquivos_barras,
        arquivos_eventos,
        credentials=credenciais,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
        assets=RANDOM_ASSETS,
        snapshot_name="tcc-random-extension-20-v1",
        parent_snapshot_sha256=str(
            manifesto_base.get("snapshot_sha256") or ""
        ),
    )

manifesto_extensao = validate_snapshot(SNAPSHOT_EXTENSAO)


# %% 3 - Preparacao U56 original e extensao U20
frames_u56_raw, exclusoes_u56, diagnosticos_u56, auditoria_u56 = (
    prepare_model_frames(
        SNAPSHOT_BASE,
        assets=CONFIG.assets,
        comparar_snapshot_referencia=False,
        allow_structural_assets=frozenset({"CLMT", "DOC"}),
    )
)
if len(frames_u56_raw) != 56:
    raise RuntimeError(
        f"U56 diagnostico deveria ter 56 ativos; obtidos {len(frames_u56_raw)}."
    )

frames_random, exclusoes_random, diagnosticos_random, auditoria_random = (
    prepare_model_frames(
        SNAPSHOT_EXTENSAO,
        assets=RANDOM_ASSETS,
        comparar_snapshot_referencia=False,
    )
)
if exclusoes_random:
    raise RuntimeError(
        "Um dos 20 ativos aleatorios apresentou quebra estrutural: "
        + ",".join(str(row.get("symbol")) for row in exclusoes_random)
    )
if len(frames_random) != 20:
    raise RuntimeError(
        f"A extensao deveria manter 20 ativos; obtidos {len(frames_random)}."
    )

for symbol, frame in frames_random.items():
    dates = pd.DatetimeIndex(frame.index)
    if len(frame) < 2600:
        raise RuntimeError(
            f"{symbol}: historico insuficiente para a coorte congelada "
            f"({len(frame)} < 2600)."
        )
    gaps = dates.to_series().diff().dt.total_seconds().div(86400.0)
    max_gap = float(gaps.dropna().max()) if gaps.notna().any() else 0.0
    if max_gap > 10.0:
        raise RuntimeError(
            f"{symbol}: quebra temporal de {max_gap:.1f} dias excede 10."
        )

sobreposicao = sorted(set(frames_u56_raw).intersection(frames_random))
if sobreposicao:
    raise RuntimeError(
        "Ativos aleatorios ja existem no U56: " + ",".join(sobreposicao)
    )

frames_u76_raw = {
    **frames_u56_raw,
    **frames_random,
}
if len(frames_u76_raw) != 76:
    raise RuntimeError(f"U76 deveria ter 76 ativos; obtidos {len(frames_u76_raw)}.")

config_u56, _ = build_variant_configs(frames_u56_raw, CONFIG)
_, reference_calendar, reference_calendar_source = preparar_painel_rotacao(
    frames_u56_raw,
    config_u56,
)

config_u76, _ = build_variant_configs(frames_u76_raw, CONFIG)
(
    frames_u76,
    common_dates,
    calendar_source_asset,
    symbols_u76,
    folds,
    all_decision_dates,
    decision_to_fold,
    decision_metadata,
) = _construir_contexto_execucao(
    frames_u76_raw,
    config_u76,
    calendar_override=reference_calendar,
    calendar_source_label=(
        f"U56_FIXED:{reference_calendar_source}"
    ),
)
symbols_u56 = sorted(frames_u56_raw)
candidate_margins = tuple(
    float(value)
    for value in config_u76.rotation_switch_margin_candidates
)

print(
    f"[universe] U56={len(symbols_u56)} U76={len(symbols_u76)} "
    f"calendar={calendar_source_asset} common_dates={len(common_dates)} "
    f"folds={len(folds)}",
    flush=True,
)

benchmark_frames_u56 = {
    symbol: frames_u76[symbol]
    for symbol in symbols_u56
}
shared_benchmark = _benchmark_pesos_iguais(
    benchmark_frames_u56,
    symbols_u56,
    all_decision_dates[1:],
    float(config_u76.initial_capital),
    config_u76,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)
SHARED_BENCHMARK_NAME = (
    "Fixed U56 equal-weight buy-and-hold on the original reference calendar"
)
print(
    f"[benchmark] fixed=U56 ending={float(shared_benchmark.iloc[-1]):,.2f}",
    flush=True,
)


# %% 4 - Propriedades de mercado dos objetos

def _market_object_properties(symbol: str) -> dict[str, object]:
    frame = frames_u76[symbol]
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    volume = pd.to_numeric(frame["volume"], errors="coerce").reindex(
        close.index
    )
    returns = close.pct_change(fill_method=None).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    elapsed_years = (
        (close.index[-1] - close.index[0]).days / 365.25
        if len(close) > 1
        else 0.0
    )
    cagr = (
        float((close.iloc[-1] / close.iloc[0]) ** (1.0 / elapsed_years) - 1.0)
        if elapsed_years > 0.0 and close.iloc[0] > 0.0
        else None
    )
    ann_vol = (
        float(returns.std(ddof=1) * math.sqrt(252.0))
        if len(returns) > 1
        else None
    )
    running_peak = close.cummax()
    maxdd = float((close / running_peak - 1.0).min())
    dollar_volume = (close * volume).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    spy = pd.to_numeric(
        frames_u76["SPY"]["close"],
        errors="coerce",
    ).pct_change(fill_method=None)
    aligned = pd.concat(
        [returns.rename("asset"), spy.rename("spy")],
        axis=1,
        join="inner",
    ).dropna()
    spy_corr = (
        float(aligned["asset"].corr(aligned["spy"]))
        if len(aligned) > 2
        else None
    )
    spy_var = float(aligned["spy"].var(ddof=1)) if len(aligned) > 2 else 0.0
    beta = (
        float(aligned["asset"].cov(aligned["spy"]) / spy_var)
        if spy_var > 0.0
        else None
    )

    metadata = RANDOM_METADATA.get(symbol, {})
    return {
        "asset": symbol,
        "cohort": "random_20" if symbol in RANDOM_METADATA else "original_56",
        "is_random_extension": bool(symbol in RANDOM_METADATA),
        "name": metadata.get("name"),
        "exchange": metadata.get("exchange"),
        "structural_override": bool(symbol in {"CLMT", "DOC"}),
        "rows": int(len(close)),
        "first_date": str(close.index.min().date()),
        "last_date": str(close.index.max().date()),
        "raw_total_return": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "raw_price_cagr": cagr,
        "annualized_volatility": ann_vol,
        "maximum_drawdown": maxdd,
        "median_daily_dollar_volume": (
            float(dollar_volume.median())
            if not dollar_volume.empty
            else None
        ),
        "spy_return_correlation": spy_corr,
        "spy_beta": beta,
    }


object_rows = {
    symbol: _market_object_properties(symbol)
    for symbol in symbols_u76
}


# %% 5 - Treino uma vez por fold e caches reutilizaveis
fold_artifacts = {}
training_started = time.perf_counter()

for fold_position, fold in enumerate(folds, start=1):
    fold_id = int(fold["fold_id"])
    train_dates = common_dates[: int(fold["train_end_index"])]
    calibration_dates = common_dates[
        int(fold["calibration_start_index"]):
        int(fold["calibration_end_index"])
    ]
    final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
    decision_dates = pd.DatetimeIndex(fold["decision_dates"])

    print(
        f"[train] fold={fold_id} position={fold_position}/{len(folds)} "
        f"assets={len(symbols_u76)} calibration_models",
        flush=True,
    )
    calibration_models = _ajustar_modelos_lightgbm(
        frames_u76,
        symbols_u76,
        train_dates,
        config_u76,
        phase=f"u76_object_fold_{fold_id}_calibration",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    calibration_cache, _ = _precalcular_utilidades_modelo(
        calibration_models,
        frames_u76,
        symbols_u76,
        calibration_dates,
        config_u76,
    )

    print(
        f"[train] fold={fold_id} assets={len(symbols_u76)} final_models",
        flush=True,
    )
    final_models = _ajustar_modelos_lightgbm(
        frames_u76,
        symbols_u76,
        final_fit_dates,
        config_u76,
        phase=f"u76_object_fold_{fold_id}_final",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    decision_cache, _ = _precalcular_utilidades_modelo(
        final_models,
        frames_u76,
        symbols_u76,
        decision_dates,
        config_u76,
    )

    fold_artifacts[fold_id] = {
        "fold": fold,
        "calibration_dates": calibration_dates,
        "decision_dates": decision_dates,
        "calibration_models": calibration_models,
        "calibration_cache": calibration_cache,
        "final_models": final_models,
        "decision_cache": decision_cache,
    }


# %% 6 - Helpers para replay de subconjuntos sem retreinar os demais ativos

full_position = {
    symbol: index + 1
    for index, symbol in enumerate(symbols_u76)
}


def _slice_cache(cache, subset_symbols):
    indices = [0] + [
        full_position[symbol]
        for symbol in subset_symbols
    ]
    return {
        timestamp: np.asarray(values, dtype=np.float64)[indices].copy()
        for timestamp, values in cache.items()
    }


def _run_subset(label, subset_symbols, *, keep_result=False):
    subset_symbols = sorted(subset_symbols)
    subset_frames = {
        symbol: frames_u76[symbol]
        for symbol in subset_symbols
    }
    subset_config = config_u76.copiar_modelo(
        update={"assets": tuple(subset_symbols)}
    )
    policies = {}
    margins = []

    for fold_id in sorted(fold_artifacts):
        artifact = fold_artifacts[fold_id]
        calibration_models = {
            symbol: artifact["calibration_models"][symbol]
            for symbol in subset_symbols
            if symbol in artifact["calibration_models"]
        }
        final_models = {
            symbol: artifact["final_models"][symbol]
            for symbol in subset_symbols
            if symbol in artifact["final_models"]
        }
        calibration_cache = _slice_cache(
            artifact["calibration_cache"],
            subset_symbols,
        )
        decision_cache = _slice_cache(
            artifact["decision_cache"],
            subset_symbols,
        )

        candidate_scores = []
        for candidate in candidate_margins:
            calibration_policy = _politica_utilidade(
                calibration_models,
                subset_frames,
                subset_symbols,
                subset_config,
                float(candidate),
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                calibration_policy,
                subset_frames,
                subset_symbols,
                artifact["calibration_dates"],
                subset_config,
            )
            candidate_scores.append((float(candidate), float(score)))

        selection = _selecionar_switch_margin_fold(
            subset_config,
            fold_id,
            candidate_scores,
        )
        selected_margin = float(
            selection["selected_candidate_margin"]
        )
        effective_margin = max(
            float(subset_config.rotation_switch_margin),
            selected_margin,
        )
        policies[fold_id] = _politica_utilidade(
            final_models,
            subset_frames,
            subset_symbols,
            subset_config,
            effective_margin,
            fold_id=fold_id,
            calibrated_switch_margin=selected_margin,
            utility_cache=decision_cache,
        )
        margins.append(
            {
                "fold_id": int(fold_id),
                "selected_margin": selected_margin,
                "effective_margin": effective_margin,
                "calibration_score": float(
                    selection["selected_calibration_score"]
                ),
                "candidate_scores": [
                    {
                        "margin": float(margin),
                        "score": float(score),
                    }
                    for margin, score in candidate_scores
                ],
            }
        )

    scheduled = _politica_agendada(policies, decision_to_fold)
    result = _simular_exato(
        "object_universe_control",
        scheduled,
        subset_frames,
        subset_symbols,
        all_decision_dates,
        subset_config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"Control Object Universe - {label}",
        method_line=(
            "- Control LightGBM models are trained once per fold for U76. "
            "Object-level leave-one-out removes one asset from calibration "
            "and OOS competition without retraining independent models. "
            "All scenarios share the fixed original-U56 market calendar and "
            "the same fixed U56 equal-weight benchmark."
        ),
        benchmark_override=shared_benchmark,
        benchmark_override_name=SHARED_BENCHMARK_NAME,
    )
    metrics = summarize_metrics(
        result,
        folds,
        float(subset_config.initial_capital),
    )
    print(
        f"[replay] {label} assets={len(subset_symbols)} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%}",
        flush=True,
    )
    return {
        "label": label,
        "symbols": subset_symbols,
        "metrics": metrics,
        "margins": margins,
        "result": result if keep_result else None,
    }


# %% 7 - U56, U76 e propriedades comportamentais do U76
baseline_u56 = _run_subset(
    "U56_ORIGINAL",
    symbols_u56,
)
full_u76 = _run_subset(
    "U76_FULL",
    symbols_u76,
    keep_result=True,
)
full_result = full_u76["result"]
full_capital = float(full_u76["metrics"]["ending_capital"])
u56_capital = float(baseline_u56["metrics"]["ending_capital"])

selected_counts = (
    full_result.predictions["selected_asset"]
    .fillna("CASH")
    .astype(str)
    .value_counts()
)
selected_total = max(1, len(full_result.predictions))
trade_counts = (
    full_result.trades["asset"].astype(str).value_counts()
    if "asset" in full_result.trades.columns
    else pd.Series(dtype=int)
)

score_stats = {
    symbol: {
        "score_values": [],
        "ranks": [],
        "top1": 0,
        "top3": 0,
        "valid": 0,
    }
    for symbol in symbols_u76
}
for fold_id, artifact in fold_artifacts.items():
    for timestamp in artifact["decision_dates"][:-1]:
        utilities = artifact["decision_cache"].get(pd.Timestamp(timestamp))
        if utilities is None:
            continue
        scores = np.asarray(utilities[1:], dtype=float)
        finite = [
            index
            for index, value in enumerate(scores)
            if np.isfinite(value)
        ]
        if not finite:
            continue
        ranked = sorted(
            finite,
            key=lambda index: (
                -float(scores[index]),
                symbols_u76[index],
            ),
        )
        ranks = {
            position: rank
            for rank, position in enumerate(ranked, start=1)
        }
        for position in finite:
            symbol = symbols_u76[position]
            item = score_stats[symbol]
            item["score_values"].append(float(scores[position]))
            item["ranks"].append(int(ranks[position]))
            item["valid"] += 1
            if ranks[position] == 1:
                item["top1"] += 1
            if ranks[position] <= 3:
                item["top3"] += 1

for symbol in symbols_u76:
    stats = score_stats[symbol]
    values = np.asarray(stats["score_values"], dtype=float)
    ranks = np.asarray(stats["ranks"], dtype=float)
    valid = int(stats["valid"])
    object_rows[symbol].update(
        {
            "u76_selected_sessions": int(selected_counts.get(symbol, 0)),
            "u76_selected_share": float(
                selected_counts.get(symbol, 0) / selected_total
            ),
            "u76_trade_rows": int(trade_counts.get(symbol, 0)),
            "model_valid_score_sessions": valid,
            "model_score_mean": (
                float(values.mean()) if valid else None
            ),
            "model_score_std": (
                float(values.std()) if valid else None
            ),
            "model_positive_score_share": (
                float(np.mean(values > 0.0)) if valid else None
            ),
            "model_top1_share": (
                float(stats["top1"] / valid) if valid else None
            ),
            "model_top3_share": (
                float(stats["top3"] / valid) if valid else None
            ),
            "model_mean_rank": (
                float(ranks.mean()) if valid else None
            ),
            "model_mean_rank_percentile": (
                float(
                    1.0
                    - (ranks.mean() - 1.0)
                    / max(1.0, len(symbols_u76) - 1.0)
                )
                if valid
                else None
            ),
        }
    )


# %% 8 - Leave-one-out exato dos 76 objetos
full_margin_by_fold = {
    int(row["fold_id"]): row
    for row in full_u76["margins"]
}
loo_rows = []

for position, asset in enumerate(symbols_u76, start=1):
    print(
        f"[loo] {position}/{len(symbols_u76)} removing={asset}",
        flush=True,
    )
    subset = [
        symbol
        for symbol in symbols_u76
        if symbol != asset
    ]
    scenario = _run_subset(
        f"U76_MINUS_{asset}",
        subset,
    )
    loo_capital = float(scenario["metrics"]["ending_capital"])
    loo_margin_by_fold = {
        int(row["fold_id"]): row
        for row in scenario["margins"]
    }

    margin_flip_count = 0
    calibration_delta_sum = 0.0
    fold_margin_rows = []
    for fold_id in sorted(full_margin_by_fold):
        full_margin = full_margin_by_fold[fold_id]
        loo_margin = loo_margin_by_fold[fold_id]
        changed = (
            abs(
                float(full_margin["selected_margin"])
                - float(loo_margin["selected_margin"])
            )
            > 1e-12
        )
        margin_flip_count += int(changed)
        calibration_delta = (
            float(full_margin["calibration_score"])
            - float(loo_margin["calibration_score"])
        )
        calibration_delta_sum += calibration_delta
        fold_margin_rows.append(
            {
                "fold_id": int(fold_id),
                "full_margin": float(full_margin["selected_margin"]),
                "loo_margin": float(loo_margin["selected_margin"]),
                "margin_changed": bool(changed),
                "full_calibration_score": float(
                    full_margin["calibration_score"]
                ),
                "loo_calibration_score": float(
                    loo_margin["calibration_score"]
                ),
                "calibration_score_delta": float(calibration_delta),
            }
        )

    marginal_pct = (
        full_capital / loo_capital - 1.0
        if loo_capital > 0.0
        else None
    )
    row = {
        "asset": asset,
        "full_u76_capital": full_capital,
        "loo_ending_capital": loo_capital,
        "marginal_profit_abs": full_capital - loo_capital,
        "marginal_profit_pct": marginal_pct,
        "loo_sharpe": float(scenario["metrics"]["sharpe"]),
        "sharpe_contribution": (
            float(full_u76["metrics"]["sharpe"])
            - float(scenario["metrics"]["sharpe"])
        ),
        "loo_maximum_drawdown": float(
            scenario["metrics"]["maximum_drawdown"]
        ),
        "maxdd_contribution": (
            float(full_u76["metrics"]["maximum_drawdown"])
            - float(scenario["metrics"]["maximum_drawdown"])
        ),
        "margin_flip_count": int(margin_flip_count),
        "mean_calibration_score_contribution": float(
            calibration_delta_sum / max(1, len(fold_margin_rows))
        ),
        "fold_margin_effects": fold_margin_rows,
    }
    loo_rows.append(row)
    object_rows[asset].update(
        {
            key: value
            for key, value in row.items()
            if key != "asset" and key != "fold_margin_effects"
        }
    )


# %% 9 - Tabela de objetos e relacoes propriedade -> lucro
objects = pd.DataFrame(
    [object_rows[symbol] for symbol in symbols_u76]
)
loo_table = pd.DataFrame(
    [
        {
            key: value
            for key, value in row.items()
            if key != "fold_margin_effects"
        }
        for row in loo_rows
    ]
)

correlation_properties = (
    "raw_price_cagr",
    "annualized_volatility",
    "maximum_drawdown",
    "median_daily_dollar_volume",
    "spy_return_correlation",
    "spy_beta",
    "u76_selected_share",
    "u76_trade_rows",
    "model_score_mean",
    "model_score_std",
    "model_positive_score_share",
    "model_top1_share",
    "model_top3_share",
    "model_mean_rank_percentile",
    "margin_flip_count",
    "mean_calibration_score_contribution",
)


def _correlation_rows(frame, scope):
    rows = []
    target = pd.to_numeric(
        frame["marginal_profit_pct"],
        errors="coerce",
    )
    for property_name in correlation_properties:
        values = pd.to_numeric(
            frame[property_name],
            errors="coerce",
        )
        pair = pd.concat(
            [values.rename("x"), target.rename("y")],
            axis=1,
        ).dropna()
        if len(pair) < 5 or pair["x"].nunique() < 2:
            continue
        rows.append(
            {
                "scope": scope,
                "property": property_name,
                "n": int(len(pair)),
                "pearson": float(pair["x"].corr(pair["y"])),
                "spearman": float(
                    pair["x"].rank(method="average").corr(
                        pair["y"].rank(method="average")
                    )
                ),
            }
        )
    return rows


correlations = pd.DataFrame(
    _correlation_rows(objects, "all_76")
    + _correlation_rows(
        objects.loc[objects["is_random_extension"].astype(bool)],
        "random_20",
    )
)
if not correlations.empty:
    correlations["max_abs_correlation"] = correlations[
        ["pearson", "spearman"]
    ].abs().max(axis=1)
    correlations = correlations.sort_values(
        ["scope", "max_abs_correlation"],
        ascending=[True, False],
    ).reset_index(drop=True)

objects = objects.sort_values(
    "marginal_profit_pct",
    ascending=False,
    na_position="last",
).reset_index(drop=True)

random_objects = objects.loc[
    objects["is_random_extension"].astype(bool)
].copy()

print(
    "[group-effect] "
    f"U56={u56_capital:,.2f} "
    f"U76={full_capital:,.2f} "
    f"delta={full_capital - u56_capital:+,.2f} "
    f"ratio={full_capital / u56_capital - 1.0:+.4%}",
    flush=True,
)
print("[objects] top positive marginal contribution", flush=True)
print(
    objects[
        [
            "asset",
            "cohort",
            "marginal_profit_pct",
            "margin_flip_count",
            "u76_selected_share",
            "model_top3_share",
        ]
    ].head(15).to_string(index=False),
    flush=True,
)
print("[objects] strongest negative marginal contribution", flush=True)
print(
    objects[
        [
            "asset",
            "cohort",
            "marginal_profit_pct",
            "margin_flip_count",
            "u76_selected_share",
            "model_top3_share",
        ]
    ].tail(15).to_string(index=False),
    flush=True,
)


# %% 10 - Exportacao
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
for antigo in DIRETORIO_RESULTADOS.glob("*.csv"):
    antigo.unlink()
for antigo in DIRETORIO_RESULTADOS.glob("*.json"):
    antigo.unlink()
graficos = DIRETORIO_RESULTADOS / "graficos"
if graficos.exists():
    for antigo in graficos.glob("*.png"):
        antigo.unlink()

objects.to_csv(
    DIRETORIO_RESULTADOS / "asset_objects_76.csv",
    index=False,
)
random_objects.to_csv(
    DIRETORIO_RESULTADOS / "random_20_objects.csv",
    index=False,
)
loo_table.to_csv(
    DIRETORIO_RESULTADOS / "asset_leave_one_out_76.csv",
    index=False,
)
correlations.to_csv(
    DIRETORIO_RESULTADOS / "object_property_profit_correlations.csv",
    index=False,
)
full_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "u76_full_predictions.csv",
    index=False,
)
full_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "u76_full_trades.csv",
    index=False,
)

payload = {
    "research_version": RESEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "base_snapshot_sha256": manifesto_base.get("snapshot_sha256"),
    "extension_snapshot_sha256": manifesto_extensao.get("snapshot_sha256"),
    "question": (
        "How does expanding the original 56-ticker diagnostic universe with "
        "20 reproducibly random assets change profit, and which object "
        "properties are associated with positive or negative marginal "
        "contribution?"
    ),
    "protocol": {
        "original_universe_size": 56,
        "random_extension_size": 20,
        "expanded_universe_size": 76,
        "random_selection_seed": RANDOM_SELECTION_SEED,
        "random_selection_catalog_date": RANDOM_SELECTION_CATALOG_DATE,
        "random_selection_rule": RANDOM_SELECTION_RULE,
        "random_assets": list(RANDOM_ASSETS),
        "base_snapshot_unchanged": True,
        "extension_snapshot_separate": True,
        "control_only": True,
        "lightgbm_parameters_unchanged": True,
        "fold_method_unchanged": True,
        "switch_margin_candidates": list(candidate_margins),
        "models_trained_once_per_fold": True,
        "loo_retrains_other_asset_models": False,
        "loo_recalibrates_rotation_policy": True,
        "loo_uses_same_u76_folds": True,
        "calendar_is_fixed_to_original_u56": True,
        "reference_calendar_source": reference_calendar_source,
        "benchmark_is_fixed_across_all_replays": True,
        "benchmark_name": SHARED_BENCHMARK_NAME,
        "marginal_contributions_are_not_additive": True,
        "structural_overrides_in_original_56": ["CLMT", "DOC"],
    },
    "universe_group_effect": {
        "u56_metrics": baseline_u56["metrics"],
        "u76_metrics": full_u76["metrics"],
        "capital_delta": full_capital - u56_capital,
        "capital_ratio": (
            full_capital / u56_capital - 1.0
            if u56_capital > 0.0
            else None
        ),
    },
    "u76_full_margins": full_u76["margins"],
    "random_asset_metadata": list(RANDOM_ASSET_OBJECTS),
    "random_asset_diagnostics": diagnosticos_random,
    "random_asset_audit": auditoria_random,
    "u56_diagnostics": diagnosticos_u56,
    "u56_audit": auditoria_u56,
    "asset_objects": objects.to_dict(orient="records"),
    "leave_one_out_fold_effects": {
        row["asset"]: row["fold_margin_effects"]
        for row in loo_rows
    },
    "property_profit_correlations": correlations.to_dict(
        orient="records"
    ),
    "runtime_seconds": float(time.perf_counter() - training_started),
    "interpretation_rule": (
        "Positive marginal_profit_pct means removing the object lowers final "
        "capital, so the object contributes positively around the full U76 "
        "state. Negative values mean the full universe would have earned more "
        "without that object. Leave-one-out effects are local and path-"
        "dependent and must not be summed as if they were independent."
    ),
}

with (
    DIRETORIO_RESULTADOS / "object_universe_76.json"
).open("w", encoding="utf-8") as arquivo:
    json.dump(
        payload,
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

PACOTE_ANALISE = criar_pacote_analise(DIRETORIO_RESULTADOS)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)
sinal_sonoro_conclusao()
print(
    "[done] object universe U76 concluido "
    f"seconds={time.perf_counter() - training_started:.3f}",
    flush=True,
)
