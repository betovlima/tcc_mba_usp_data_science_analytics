"""Segundo lote aleatorio: 20 novos objetos contra o universo vencedor U56.

Objetivo:
- sortear outro lote reprodutivel de 20 ativos, sem reutilizar o lote anterior;
- medir cada ativo por insercao individual U56 + candidato;
- medir o efeito conjunto U56 + 20;
- correlacionar propriedades de mercado e de score de cada candidato com o
  comportamento do U56 vencedor;
- identificar quais candidatos aumentam ou diminuem capital sem usar retorno
  passado para escolher previamente os 20 nomes.

O calendario, folds, benchmark e modelos dos 56 ativos-base permanecem fixos.
"""

# %% 0 - Imports e configuracao
from pathlib import Path
import json
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
SNAPSHOT_BATCH2 = SnapshotPaths.from_root(
    RAIZ_PROJETO / "dados" / "pesquisa_expansao_76_b2"
)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"

SCRIPT_RESEARCH_VERSION = "1.14.1-dev.1"
EXECUTION_SCHEMA = "random-batch2-u56-correlation-v1"

RANDOM_SELECTION_SEED = 2026100402
RANDOM_SELECTION_CATALOG_DATE = "2026-10-04"
RANDOM_SELECTION_RULE = (
    "Alpaca active/tradable/marginable US equities; exchanges NYSE/NASDAQ/"
    "AMEX/ARCA/BATS; simple ticker; excludes original 56 and random batch 1; "
    "deterministic xorshift32 seed=2026100402; requires daily SIP RAW history "
    "covering 2016-01 through 2026-09 with >=2600 rows and max gap <=10 days; "
    "structural identity/ticker transitions are rejected before freezing."
)

BATCH1_ASSETS = (
    "FAF", "IJR", "GAB", "ELS", "AEIS", "VWOB", "BDJ", "DGX", "ESP", "BWZ",
    "PSF", "DBA", "HEEM", "NPKI", "MHK", "BLKB", "ARCO", "AGM", "NWFL", "SKOR",
)

RANDOM_ASSET_OBJECTS = (
    {"symbol": "VIOV", "name": "Vanguard S&P Small-Cap 600 Value ETF", "exchange": "ARCA"},
    {"symbol": "MBSD", "name": "Northern Trust Disciplined Duration MBS ETF", "exchange": "ARCA"},
    {"symbol": "MVIS", "name": "MicroVision, Inc.", "exchange": "NASDAQ"},
    {"symbol": "EWD", "name": "iShares MSCI Sweden ETF", "exchange": "ARCA"},
    {"symbol": "OPHC", "name": "OptimumBank Holdings, Inc.", "exchange": "AMEX"},
    {"symbol": "CASY", "name": "Casey's General Stores, Inc.", "exchange": "NASDAQ"},
    {"symbol": "COLB", "name": "Columbia Banking System, Inc.", "exchange": "NASDAQ"},
    {"symbol": "EES", "name": "WisdomTree U.S. SmallCap Fund", "exchange": "ARCA"},
    {"symbol": "GNK", "name": "Genco Shipping & Trading Ltd", "exchange": "NYSE"},
    {"symbol": "VUZI", "name": "Vuzix Corporation", "exchange": "NASDAQ"},
    {"symbol": "FOXF", "name": "Fox Factory Holding Corp.", "exchange": "NASDAQ"},
    {"symbol": "AMS", "name": "American Shared Hospital Services", "exchange": "AMEX"},
    {"symbol": "IQLT", "name": "iShares MSCI Intl Quality Factor ETF", "exchange": "ARCA"},
    {"symbol": "ISCF", "name": "iShares International Small Cap Equity Factor ETF", "exchange": "ARCA"},
    {"symbol": "FUTY", "name": "Fidelity MSCI Utilities Index ETF", "exchange": "ARCA"},
    {"symbol": "KB", "name": "KB Financial Group Inc", "exchange": "NYSE"},
    {"symbol": "PRN", "name": "Invesco Dorsey Wright Industrials Momentum ETF", "exchange": "NASDAQ"},
    {"symbol": "CE", "name": "Celanese Corporation", "exchange": "NYSE"},
    {"symbol": "XTNT", "name": "Xtant Medical Holdings, Inc.", "exchange": "AMEX"},
    {"symbol": "UEC", "name": "Uranium Energy Corp.", "exchange": "AMEX"},
)
RANDOM_ASSETS = tuple(item["symbol"] for item in RANDOM_ASSET_OBJECTS)
RANDOM_METADATA = {
    item["symbol"]: dict(item)
    for item in RANDOM_ASSET_OBJECTS
}

REJECTED_RANDOM_CANDIDATES = (
    {
        "symbol": "ONTO",
        "reason": "structural_ticker_identity_transition",
        "detail": "NANO -> ONTO name/ticker transition observed in Corporate Actions",
    },
)


# %% 1 - Guards e snapshot base
if EXECUTION_SCHEMA != EXPECTED_EXECUTION_SCHEMA:
    raise RuntimeError(
        "Script e modulo de pesquisa incompatíveis antes do replay: "
        f"script={EXECUTION_SCHEMA!r} modulo={EXPECTED_EXECUTION_SCHEMA!r}. "
        "Atualize a branch e reinicie o kernel do Spyder."
    )
if SCRIPT_RESEARCH_VERSION != RESEARCH_VERSION:
    raise RuntimeError(
        "Versao do runner e modulo incompatíveis antes do replay: "
        f"runner={SCRIPT_RESEARCH_VERSION!r} modulo={RESEARCH_VERSION!r}."
    )
if len(RANDOM_ASSETS) != 20 or len(set(RANDOM_ASSETS)) != 20:
    raise RuntimeError("O segundo lote precisa conter exatamente 20 ativos unicos.")
if set(RANDOM_ASSETS).intersection(BATCH1_ASSETS):
    raise RuntimeError("O segundo lote nao pode reutilizar ativos do primeiro lote.")

print("=" * 78, flush=True)
print("TCC - Random Batch 2 vs Winning U56", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print(f"versao_runner={SCRIPT_RESEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print(f"random_seed={RANDOM_SELECTION_SEED}", flush=True)
print("random_assets=" + ",".join(RANDOM_ASSETS), flush=True)
print("=" * 78, flush=True)

manifesto_base = validate_snapshot(SNAPSHOT_BASE)


# %% 2 - Snapshot congelado do segundo lote de 20

def _batch2_manifest_is_usable() -> bool:
    if not SNAPSHOT_BATCH2.manifest.exists():
        return False
    try:
        payload = validate_snapshot(SNAPSHOT_BATCH2)
    except Exception:
        return False
    return (
        tuple(payload.get("assets") or ()) == RANDOM_ASSETS
        and str(payload.get("parent_snapshot_sha256") or "")
        == str(manifesto_base.get("snapshot_sha256") or "")
        and str((payload.get("bars") or {}).get("bar_snapshot_as_of_end") or "")
        == str(BAR_SNAPSHOT_AS_OF_END)
    )


if not _batch2_manifest_is_usable():
    print(
        "[batch2] snapshot ausente/incompativel; baixando os 20 novos ativos",
        flush=True,
    )
    SNAPSHOT_BATCH2.clear_generated()
    credenciais = load_alpaca_credentials(RAIZ_PROJETO)
    arquivos_barras = download_raw_bars(
        credenciais,
        SNAPSHOT_BATCH2,
        assets=RANDOM_ASSETS,
        replace=True,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
    )
    arquivos_eventos = download_corporate_actions(
        credenciais,
        SNAPSHOT_BATCH2,
        assets=RANDOM_ASSETS,
        replace=True,
        query_end=ANALYSIS_END_DATE,
    )
    build_snapshot_manifest(
        SNAPSHOT_BATCH2,
        arquivos_barras,
        arquivos_eventos,
        credentials=credenciais,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
        assets=RANDOM_ASSETS,
        snapshot_name="tcc-random-extension-batch2-20-v1",
        parent_snapshot_sha256=str(
            manifesto_base.get("snapshot_sha256") or ""
        ),
    )

manifesto_batch2 = validate_snapshot(SNAPSHOT_BATCH2)


# %% 3 - U56 vencedor + segundo lote
frames_u56_raw, exclusoes_u56, diagnosticos_u56, auditoria_u56 = (
    prepare_model_frames(
        SNAPSHOT_BASE,
        assets=CONFIG.assets,
        comparar_snapshot_referencia=False,
        allow_structural_assets=frozenset({"CLMT", "DOC"}),
    )
)
if len(frames_u56_raw) != 56:
    raise RuntimeError(f"U56 deveria ter 56 ativos; obtidos {len(frames_u56_raw)}.")

frames_batch2, exclusoes_batch2, diagnosticos_batch2, auditoria_batch2 = (
    prepare_model_frames(
        SNAPSHOT_BATCH2,
        assets=RANDOM_ASSETS,
        comparar_snapshot_referencia=False,
    )
)
if exclusoes_batch2:
    raise RuntimeError(
        "Ativo do lote 2 apresentou problema estrutural: "
        + ",".join(str(row.get("symbol")) for row in exclusoes_batch2)
    )
if len(frames_batch2) != 20:
    raise RuntimeError(
        f"Lote 2 deveria manter 20 ativos; obtidos {len(frames_batch2)}."
    )

for symbol, frame in frames_batch2.items():
    dates = pd.DatetimeIndex(frame.index)
    if len(frame) < 2600:
        raise RuntimeError(f"{symbol}: historico insuficiente ({len(frame)} < 2600).")
    gaps = dates.to_series().diff().dt.total_seconds().div(86400.0)
    max_gap = float(gaps.dropna().max()) if gaps.notna().any() else 0.0
    if max_gap > 10.0:
        raise RuntimeError(f"{symbol}: quebra temporal de {max_gap:.1f} dias.")

frames_u76_raw = {**frames_u56_raw, **frames_batch2}
if len(frames_u76_raw) != 76:
    raise RuntimeError(f"U76 batch2 deveria ter 76 ativos; obtidos {len(frames_u76_raw)}.")

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
    calendar_source_label=f"U56_FIXED:{reference_calendar_source}",
)

symbols_u56 = sorted(frames_u56_raw)
candidate_margins = tuple(
    float(value)
    for value in config_u76.rotation_switch_margin_candidates
)
full_position = {
    symbol: index + 1
    for index, symbol in enumerate(symbols_u76)
}

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
    f"[universe] U56=56 U76_B2=76 calendar={calendar_source_asset} "
    f"common_dates={len(common_dates)} folds={len(folds)}",
    flush=True,
)
print(
    f"[benchmark] fixed=U56 ending={float(shared_benchmark.iloc[-1]):,.2f}",
    flush=True,
)


# %% 4 - Treino dos 76 modelos uma vez por fold
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
        f"[train] fold={fold_id} {fold_position}/{len(folds)} "
        f"calibration_models={len(symbols_u76)}",
        flush=True,
    )
    calibration_models = _ajustar_modelos_lightgbm(
        frames_u76,
        symbols_u76,
        train_dates,
        config_u76,
        phase=f"batch2_fold_{fold_id}_calibration",
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
    calibration_missing = sorted(
        set(symbols_u76).difference(calibration_models)
    )
    if calibration_missing:
        print(
            f"[train-diagnostic] fold={fold_id} "
            "calibration_missing=" + ",".join(calibration_missing),
            flush=True,
        )

    print(
        f"[train] fold={fold_id} final_models={len(symbols_u76)}",
        flush=True,
    )
    final_models = _ajustar_modelos_lightgbm(
        frames_u76,
        symbols_u76,
        final_fit_dates,
        config_u76,
        phase=f"batch2_fold_{fold_id}_final",
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
    final_missing = sorted(
        set(symbols_u76).difference(final_models)
    )
    if final_missing:
        print(
            f"[train-diagnostic] fold={fold_id} "
            "final_missing=" + ",".join(final_missing),
            flush=True,
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


# %% 5 - Replay de subconjuntos

def _slice_cache(cache, subset_symbols):
    indices = [0] + [full_position[symbol] for symbol in subset_symbols]
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
        for margin in candidate_margins:
            calibration_policy = _politica_utilidade(
                calibration_models,
                subset_frames,
                subset_symbols,
                subset_config,
                float(margin),
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                calibration_policy,
                subset_frames,
                subset_symbols,
                artifact["calibration_dates"],
                subset_config,
            )
            candidate_scores.append((float(margin), float(score)))

        selection = _selecionar_switch_margin_fold(
            subset_config,
            fold_id,
            candidate_scores,
        )
        selected_margin = float(selection["selected_candidate_margin"])
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
            }
        )

    scheduled = _politica_agendada(policies, decision_to_fold)
    result = _simular_exato(
        "random_batch2_u56",
        scheduled,
        subset_frames,
        subset_symbols,
        all_decision_dates,
        subset_config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"Control Random Batch 2 - {label}",
        method_line=(
            "- Models are trained once per fold for the frozen U76 batch-2 "
            "panel. Each insertion replay changes only the available object "
            "set and recalibrates the frozen switch-margin candidates. "
            "Calendar and benchmark remain fixed to the winning U56."
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


# %% 6 - Baseline U56 e lote completo
baseline_u56 = _run_subset(
    "U56_WINNER",
    symbols_u56,
    keep_result=True,
)
baseline_result = baseline_u56["result"]
baseline_capital = float(baseline_u56["metrics"]["ending_capital"])

full_u76 = _run_subset(
    "U76_BATCH2_FULL",
    symbols_u76,
    keep_result=True,
)
full_u76_result = full_u76["result"]
full_u76_capital = float(full_u76["metrics"]["ending_capital"])

baseline_margin_by_fold = {
    int(row["fold_id"]): float(row["selected_margin"])
    for row in baseline_u56["margins"]
}


# %% 7 - Correlacoes de cada candidato com o U56 vencedor
close_returns = {}
for symbol in symbols_u76:
    close = pd.to_numeric(
        frames_u76[symbol]["close"],
        errors="coerce",
    )
    close_returns[symbol] = close.pct_change(
        fill_method=None
    ).replace([np.inf, -np.inf], np.nan)

u56_return_frame = pd.DataFrame(
    {symbol: close_returns[symbol] for symbol in symbols_u56}
)
u56_equal_weight_return = u56_return_frame.mean(
    axis=1,
    skipna=True,
)

baseline_predictions = baseline_result.predictions.reset_index().copy()
baseline_predictions["strategy_return"] = pd.to_numeric(
    baseline_predictions["strategy_equity"],
    errors="coerce",
).pct_change(fill_method=None)
baseline_strategy_returns = (
    baseline_predictions.set_index("timestamp")["strategy_return"]
)
baseline_decision_score = pd.Series(
    pd.to_numeric(
        baseline_predictions["decision_score"],
        errors="coerce",
    ).to_numpy(),
    index=pd.to_datetime(
        baseline_predictions["decision_date"],
        utc=True,
    ),
)


def _corr_pair(left, right):
    pair = pd.concat(
        [
            pd.Series(left, dtype=float).rename("x"),
            pd.Series(right, dtype=float).rename("y"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 5 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {"n": int(len(pair)), "pearson": None, "spearman": None}
    return {
        "n": int(len(pair)),
        "pearson": float(pair["x"].corr(pair["y"])),
        "spearman": float(
            pair["x"].rank(method="average").corr(
                pair["y"].rank(method="average")
            )
        ),
    }


def _candidate_score_profile(candidate):
    candidate_index = full_position[candidate]
    original_indices = [
        full_position[symbol]
        for symbol in symbols_u56
    ]
    rows = []

    for artifact in fold_artifacts.values():
        for timestamp in artifact["decision_dates"][:-1]:
            utilities = artifact["decision_cache"].get(
                pd.Timestamp(timestamp)
            )
            if utilities is None:
                continue
            utilities = np.asarray(utilities, dtype=float)
            candidate_score = float(utilities[candidate_index])
            original_scores = utilities[original_indices]
            finite = original_scores[np.isfinite(original_scores)]
            if not np.isfinite(candidate_score) or len(finite) == 0:
                continue
            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamp),
                    "candidate_score": candidate_score,
                    "u56_score_mean": float(np.mean(finite)),
                    "u56_score_best": float(np.max(finite)),
                    "candidate_beats_u56_best": bool(
                        candidate_score > float(np.max(finite))
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "model_score_corr_u56_mean": None,
            "model_score_corr_u56_best": None,
            "model_score_corr_u56_decision_score": None,
            "candidate_beats_u56_best_share": None,
            "candidate_score_mean": None,
            "candidate_score_std": None,
            "candidate_positive_score_share": None,
            "model_score_sessions": 0,
        }

    frame = frame.set_index("timestamp").sort_index()
    corr_mean = _corr_pair(
        frame["candidate_score"],
        frame["u56_score_mean"],
    )
    corr_best = _corr_pair(
        frame["candidate_score"],
        frame["u56_score_best"],
    )
    corr_decision = _corr_pair(
        frame["candidate_score"],
        baseline_decision_score,
    )
    scores = frame["candidate_score"]
    return {
        "model_score_corr_u56_mean": corr_mean["pearson"],
        "model_score_spearman_u56_mean": corr_mean["spearman"],
        "model_score_corr_u56_best": corr_best["pearson"],
        "model_score_spearman_u56_best": corr_best["spearman"],
        "model_score_corr_u56_decision_score": corr_decision["pearson"],
        "candidate_beats_u56_best_share": float(
            frame["candidate_beats_u56_best"].mean()
        ),
        "candidate_score_mean": float(scores.mean()),
        "candidate_score_std": float(scores.std(ddof=0)),
        "candidate_positive_score_share": float((scores > 0.0).mean()),
        "model_score_sessions": int(len(frame)),
    }


def _candidate_return_profile(candidate):
    candidate_returns = close_returns[candidate]
    pairwise = []
    for symbol in symbols_u56:
        corr = _corr_pair(
            candidate_returns,
            close_returns[symbol],
        )["pearson"]
        if corr is not None:
            pairwise.append((symbol, float(corr)))

    corr_equal = _corr_pair(
        candidate_returns,
        u56_equal_weight_return,
    )
    corr_strategy = _corr_pair(
        candidate_returns,
        baseline_strategy_returns,
    )

    pairwise_values = np.asarray(
        [value for _, value in pairwise],
        dtype=float,
    )
    most_correlated = (
        max(pairwise, key=lambda item: item[1])
        if pairwise
        else (None, None)
    )
    least_correlated = (
        min(pairwise, key=lambda item: item[1])
        if pairwise
        else (None, None)
    )
    return {
        "return_corr_u56_equal_weight": corr_equal["pearson"],
        "return_spearman_u56_equal_weight": corr_equal["spearman"],
        "return_corr_u56_strategy": corr_strategy["pearson"],
        "mean_return_corr_u56_assets": (
            float(pairwise_values.mean())
            if len(pairwise_values)
            else None
        ),
        "median_return_corr_u56_assets": (
            float(np.median(pairwise_values))
            if len(pairwise_values)
            else None
        ),
        "mean_abs_return_corr_u56_assets": (
            float(np.abs(pairwise_values).mean())
            if len(pairwise_values)
            else None
        ),
        "most_correlated_u56_asset": most_correlated[0],
        "most_correlated_u56_asset_corr": most_correlated[1],
        "least_correlated_u56_asset": least_correlated[0],
        "least_correlated_u56_asset_corr": least_correlated[1],
        "pairwise_u56_correlation_count": int(len(pairwise)),
    }


# %% 8 - Insercao individual U56 + candidato
candidate_rows = []
candidate_fold_effects = {}

baseline_selected = (
    baseline_result.predictions["selected_asset"]
    .fillna("CASH")
    .astype(str)
)

for position, candidate in enumerate(RANDOM_ASSETS, start=1):
    print(
        f"[candidate] {position}/20 U56_PLUS_{candidate}",
        flush=True,
    )
    scenario = _run_subset(
        f"U56_PLUS_{candidate}",
        [*symbols_u56, candidate],
        keep_result=True,
    )
    result = scenario["result"]
    metrics = scenario["metrics"]
    ending = float(metrics["ending_capital"])

    candidate_selected = (
        result.predictions["selected_asset"]
        .fillna("CASH")
        .astype(str)
    )
    aligned = pd.concat(
        [
            baseline_selected.rename("baseline"),
            candidate_selected.rename("candidate"),
        ],
        axis=1,
        join="inner",
    )
    changed = aligned["baseline"] != aligned["candidate"]

    scenario_margin_by_fold = {
        int(row["fold_id"]): float(row["selected_margin"])
        for row in scenario["margins"]
    }
    fold_effects = []
    margin_flip_count = 0
    for fold_id in sorted(baseline_margin_by_fold):
        before = baseline_margin_by_fold[fold_id]
        after = scenario_margin_by_fold[fold_id]
        changed_margin = abs(before - after) > 1e-12
        margin_flip_count += int(changed_margin)
        fold_effects.append(
            {
                "fold_id": int(fold_id),
                "u56_margin": float(before),
                "u56_plus_candidate_margin": float(after),
                "margin_changed": bool(changed_margin),
            }
        )
    candidate_fold_effects[candidate] = fold_effects

    trades = result.trades
    candidate_sells = trades.loc[
        (trades.get("asset", pd.Series(dtype=str)).astype(str) == candidate)
        & trades.get("action", pd.Series(dtype=str)).isin(
            ["SELL", "FINAL_SELL"]
        )
    ].copy()
    realized_pnl = (
        float(pd.to_numeric(
            candidate_sells.get("realized_pnl", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0).sum())
        if not candidate_sells.empty
        else 0.0
    )
    position_returns = pd.to_numeric(
        candidate_sells.get("position_return", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()

    market_profile = _candidate_return_profile(candidate)
    score_profile = _candidate_score_profile(candidate)
    metadata = RANDOM_METADATA[candidate]

    row = {
        "asset": candidate,
        "name": metadata["name"],
        "exchange": metadata["exchange"],
        "u56_ending_capital": baseline_capital,
        "u56_plus_candidate_ending_capital": ending,
        "insertion_capital_delta": ending - baseline_capital,
        "insertion_capital_pct": (
            ending / baseline_capital - 1.0
            if baseline_capital > 0.0
            else None
        ),
        "sharpe_delta": (
            float(metrics["sharpe"])
            - float(baseline_u56["metrics"]["sharpe"])
        ),
        "maxdd_delta": (
            float(metrics["maximum_drawdown"])
            - float(baseline_u56["metrics"]["maximum_drawdown"])
        ),
        "worst_fold_delta": (
            float(metrics["worst_fold_return"])
            - float(baseline_u56["metrics"]["worst_fold_return"])
        ),
        "changed_selected_asset_sessions": int(changed.sum()),
        "changed_selected_asset_share": float(changed.mean()),
        "first_path_divergence": (
            str(aligned.index[changed][0])
            if bool(changed.any())
            else None
        ),
        "candidate_selected_sessions": int(
            (candidate_selected == candidate).sum()
        ),
        "candidate_sell_count": int(len(candidate_sells)),
        "candidate_realized_pnl_sum": realized_pnl,
        "candidate_mean_position_return": (
            float(position_returns.mean())
            if len(position_returns)
            else None
        ),
        "candidate_win_rate": (
            float((position_returns > 0.0).mean())
            if len(position_returns)
            else None
        ),
        "margin_flip_count": int(margin_flip_count),
        **market_profile,
        **score_profile,
    }
    candidate_rows.append(row)

candidates = pd.DataFrame(candidate_rows).sort_values(
    "insertion_capital_pct",
    ascending=False,
).reset_index(drop=True)


# %% 9 - Relacao propriedade -> contribuicao
correlation_properties = (
    "return_corr_u56_equal_weight",
    "return_corr_u56_strategy",
    "mean_return_corr_u56_assets",
    "mean_abs_return_corr_u56_assets",
    "model_score_corr_u56_mean",
    "model_score_corr_u56_best",
    "model_score_corr_u56_decision_score",
    "candidate_beats_u56_best_share",
    "candidate_score_mean",
    "candidate_score_std",
    "candidate_positive_score_share",
    "changed_selected_asset_share",
    "candidate_selected_sessions",
    "candidate_sell_count",
    "candidate_realized_pnl_sum",
    "margin_flip_count",
)

correlation_rows = []
target = pd.to_numeric(
    candidates["insertion_capital_pct"],
    errors="coerce",
)
for property_name in correlation_properties:
    values = pd.to_numeric(
        candidates[property_name],
        errors="coerce",
    )
    pair = pd.concat(
        [values.rename("x"), target.rename("y")],
        axis=1,
    ).dropna()
    if len(pair) < 5 or pair["x"].nunique() < 2:
        continue
    correlation_rows.append(
        {
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

property_correlations = pd.DataFrame(correlation_rows)
if not property_correlations.empty:
    property_correlations["max_abs_correlation"] = property_correlations[
        ["pearson", "spearman"]
    ].abs().max(axis=1)
    property_correlations = property_correlations.sort_values(
        "max_abs_correlation",
        ascending=False,
    ).reset_index(drop=True)

positive_count = int((candidates["insertion_capital_pct"] > 0.0).sum())
negative_count = int((candidates["insertion_capital_pct"] < 0.0).sum())
zero_count = int(len(candidates) - positive_count - negative_count)

print(
    "[batch2-group] "
    f"U56={baseline_capital:,.2f} "
    f"U76_B2={full_u76_capital:,.2f} "
    f"delta={full_u76_capital - baseline_capital:+,.2f} "
    f"ratio={full_u76_capital / baseline_capital - 1.0:+.4%}",
    flush=True,
)
print(
    f"[batch2-candidates] positive={positive_count} "
    f"negative={negative_count} zero={zero_count}",
    flush=True,
)
print("[batch2-candidates] ranking", flush=True)
print(
    candidates[
        [
            "asset",
            "insertion_capital_pct",
            "changed_selected_asset_sessions",
            "margin_flip_count",
            "return_corr_u56_equal_weight",
            "model_score_corr_u56_mean",
            "candidate_beats_u56_best_share",
        ]
    ].to_string(index=False),
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

candidates.to_csv(
    DIRETORIO_RESULTADOS / "random_batch2_candidates.csv",
    index=False,
)
property_correlations.to_csv(
    DIRETORIO_RESULTADOS / "random_batch2_property_correlations.csv",
    index=False,
)
baseline_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "u56_winner_predictions.csv",
    index=False,
)
baseline_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "u56_winner_trades.csv",
    index=False,
)
full_u76_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "u76_batch2_predictions.csv",
    index=False,
)
full_u76_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "u76_batch2_trades.csv",
    index=False,
)

payload = {
    "research_version": RESEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "base_snapshot_sha256": manifesto_base.get("snapshot_sha256"),
    "batch2_snapshot_sha256": manifesto_batch2.get("snapshot_sha256"),
    "question": (
        "Do a second 20-asset random replication, measure each object by "
        "one-at-a-time insertion into the winning U56, and determine which "
        "return/score relationships with U56 are associated with positive or "
        "negative contribution."
    ),
    "protocol": {
        "winner_universe_size": 56,
        "batch2_size": 20,
        "expanded_universe_size": 76,
        "random_selection_seed": RANDOM_SELECTION_SEED,
        "random_selection_catalog_date": RANDOM_SELECTION_CATALOG_DATE,
        "random_selection_rule": RANDOM_SELECTION_RULE,
        "batch1_assets_excluded": list(BATCH1_ASSETS),
        "batch2_assets": list(RANDOM_ASSETS),
        "rejected_random_candidates": list(REJECTED_RANDOM_CANDIDATES),
        "selection_uses_backtest_performance": False,
        "base_calendar_fixed_to_u56": True,
        "benchmark_fixed_to_u56": True,
        "lightgbm_parameters_unchanged": True,
        "fold_method_unchanged": True,
        "switch_margin_candidates": list(candidate_margins),
        "models_trained_once_per_fold": True,
        "individual_test": "U56 plus exactly one batch-2 candidate",
        "group_test": "U56 plus all 20 batch-2 candidates",
        "interpretation": (
            "Individual insertion effects are the primary candidate-level "
            "measure. The 20-object group effect is secondary because object "
            "interactions can be non-additive."
        ),
    },
    "u56_winner": {
        "metrics": baseline_u56["metrics"],
        "margins": baseline_u56["margins"],
        "calendar_source": reference_calendar_source,
    },
    "u76_batch2_group": {
        "metrics": full_u76["metrics"],
        "margins": full_u76["margins"],
        "capital_delta": full_u76_capital - baseline_capital,
        "capital_pct": (
            full_u76_capital / baseline_capital - 1.0
            if baseline_capital > 0.0
            else None
        ),
    },
    "candidate_counts": {
        "positive": positive_count,
        "negative": negative_count,
        "zero": zero_count,
    },
    "candidate_rows": candidates.to_dict(orient="records"),
    "candidate_fold_margin_effects": candidate_fold_effects,
    "property_profit_correlations": property_correlations.to_dict(
        orient="records"
    ),
    "batch2_diagnostics": diagnosticos_batch2,
    "batch2_audit": auditoria_batch2,
    "u56_diagnostics": diagnosticos_u56,
    "u56_audit": auditoria_u56,
    "runtime_seconds": float(time.perf_counter() - training_started),
    "interpretation_rule": (
        "A candidate improves U56 when its one-at-a-time insertion capital "
        "effect is positive. Correlations with U56 returns and model scores "
        "are descriptive/exploratory and must not be treated as causal or as "
        "a final asset-selection rule without another untouched replication."
    ),
}

with (
    DIRETORIO_RESULTADOS / "random_batch2_u56_correlation.json"
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
    "[done] random batch2 U56 correlation concluido "
    f"seconds={time.perf_counter() - training_started:.3f}",
    flush=True,
)
