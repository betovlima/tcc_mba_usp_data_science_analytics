"""Terceiro lote aleatorio para validacao da assinatura de objeto.

Objetivo:
- manter o U56 cientifico como controle intacto;
- manter apenas COLB, AMS e FOXF do lote 2 no universo enriquecido U59;
- adicionar 20 objetos aleatorios totalmente novos, formando U79;
- congelar antes dos replays a assinatura selective-specialist-v0.1;
- testar cada candidato contra U56 para validacao comparavel ao lote 2;
- testar cada candidato contra U59 para medir valor incremental no universo
  enriquecido solicitado pelo usuario;
- medir o efeito conjunto U59 + 20 sem reutilizar neutros ou negativos.

O calendario, folds e benchmark continuam fixos no U56 original.
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
SNAPSHOT_BATCH3 = SnapshotPaths.from_root(
    RAIZ_PROJETO / "dados" / "pesquisa_expansao_76_b3"
)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"

SCRIPT_RESEARCH_VERSION = "1.15.0-dev.1"
EXECUTION_SCHEMA = "signature-validation-batch3-u59-v1"

RANDOM_SELECTION_SEED = 2026100503
RANDOM_SELECTION_CATALOG_DATE = "2026-10-05"
RANDOM_SELECTION_RULE = (
    "Alpaca active/tradable/marginable US equities; exchanges NYSE/NASDAQ/"
    "AMEX/ARCA/BATS; simple ticker; excludes original 56 and random batches "
    "1/2; deterministic xorshift32 seed=2026100503; requires daily SIP RAW "
    "history covering 2016-01 through 2026-09 with >=2600 rows and max gap "
    "<=10 days; structural identity/ticker transitions are rejected before "
    "freezing."
)

BATCH1_ASSETS = (
    "FAF", "IJR", "GAB", "ELS", "AEIS", "VWOB", "BDJ", "DGX", "ESP", "BWZ",
    "PSF", "DBA", "HEEM", "NPKI", "MHK", "BLKB", "ARCO", "AGM", "NWFL", "SKOR",
)
BATCH2_ASSETS = (
    "VIOV", "MBSD", "MVIS", "EWD", "OPHC", "CASY", "COLB", "EES", "GNK", "VUZI",
    "FOXF", "AMS", "IQLT", "ISCF", "FUTY", "KB", "PRN", "CE", "XTNT", "UEC",
)
ENRICHED_POSITIVES = ("COLB", "AMS", "FOXF")

SIGNATURE_NAME = "selective-specialist-v0.1"
SIGNATURE_THRESHOLDS = {
    "beats_best_share_min_exclusive": 0.0,
    "beats_best_share_max_inclusive": 0.05,
    "score_std_max_inclusive": 0.15,
    "score_mean_max_inclusive": 0.11,
    "abs_score_corr_u56_mean_max_inclusive": 0.25,
}

RANDOM_ASSET_OBJECTS = (
    {"symbol": "MG", "name": "Mistras Group Inc.", "exchange": "NYSE"},
    {"symbol": "VSTM", "name": "Verastem, Inc.", "exchange": "NASDAQ"},
    {"symbol": "HEWJ", "name": "iShares Currency Hedged MSCI Japan ETF", "exchange": "ARCA"},
    {"symbol": "GBAB", "name": "Guggenheim Taxable Municipal Bond & Investment Grade Debt Trust", "exchange": "NYSE"},
    {"symbol": "CRESY", "name": "Cresud S.A.C.I.F. y A.", "exchange": "NASDAQ"},
    {"symbol": "BGT", "name": "BlackRock Floating Rate Income Trust", "exchange": "NYSE"},
    {"symbol": "DBJP", "name": "Xtrackers MSCI Japan Hedged Equity ETF", "exchange": "ARCA"},
    {"symbol": "UBND", "name": "VictoryShares Core Plus Bond ETF", "exchange": "NASDAQ"},
    {"symbol": "PPLT", "name": "abrdn Physical Platinum Shares ETF", "exchange": "ARCA"},
    {"symbol": "RXL", "name": "ProShares Ultra Health Care", "exchange": "ARCA"},
    {"symbol": "JPIN", "name": "JPMorgan Diversified Return International Equity ETF", "exchange": "ARCA"},
    {"symbol": "REXR", "name": "Rexford Industrial Realty, Inc.", "exchange": "NYSE"},
    {"symbol": "QVAL", "name": "Alpha Architect U.S. Quantitative Value ETF", "exchange": "NASDAQ"},
    {"symbol": "REM", "name": "iShares Mortgage Real Estate ETF", "exchange": "BATS"},
    {"symbol": "CEVA", "name": "CEVA, Inc.", "exchange": "NASDAQ"},
    {"symbol": "TRC", "name": "Tejon Ranch Co.", "exchange": "NYSE"},
    {"symbol": "SCHA", "name": "Schwab U.S. Small-Cap ETF", "exchange": "ARCA"},
    {"symbol": "CALM", "name": "Cal-Maine Foods, Inc.", "exchange": "NASDAQ"},
    {"symbol": "DRN", "name": "Direxion Daily Real Estate Bull 3X ETF", "exchange": "ARCA"},
    {"symbol": "EVH", "name": "Evolent Health, Inc.", "exchange": "NYSE"},
)
RANDOM_ASSETS = tuple(item["symbol"] for item in RANDOM_ASSET_OBJECTS)
RANDOM_METADATA = {
    item["symbol"]: dict(item)
    for item in RANDOM_ASSET_OBJECTS
}

REJECTED_RANDOM_CANDIDATES = (
    {"symbol": "FLG", "reason": "structural_ticker_identity_transition", "detail": "NYCB -> FLG"},
    {"symbol": "LBTYA", "reason": "structural_identity_change", "detail": "CUSIP transition under same ticker"},
    {"symbol": "DCOY", "reason": "structural_ticker_identity_transition", "detail": "SLRX -> DCOY"},
    {"symbol": "BLOX", "reason": "history_discontinuity", "detail": "large multi-year gap"},
    {"symbol": "VISN", "reason": "structural_ticker_identity_transition", "detail": "COMM -> VISN"},
    {"symbol": "COR", "reason": "structural_ticker_identity_transition", "detail": "ABC -> COR"},
    {"symbol": "ECON", "reason": "insufficient_history", "detail": "fewer than 2600 daily rows"},
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
    raise RuntimeError("O terceiro lote precisa conter exatamente 20 ativos unicos.")
if set(RANDOM_ASSETS).intersection(BATCH1_ASSETS):
    raise RuntimeError("O terceiro lote nao pode reutilizar ativos do primeiro lote.")
if set(RANDOM_ASSETS).intersection(BATCH2_ASSETS):
    raise RuntimeError("O terceiro lote nao pode reutilizar ativos do segundo lote.")

print("=" * 78, flush=True)
print("TCC - Batch 3 Signature Validation: U56 / U59 / U79", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print(f"versao_runner={SCRIPT_RESEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print(f"random_seed={RANDOM_SELECTION_SEED}", flush=True)
print("random_assets=" + ",".join(RANDOM_ASSETS), flush=True)
print("=" * 78, flush=True)

manifesto_base = validate_snapshot(SNAPSHOT_BASE)


# %% 2 - Snapshot congelado do terceiro lote de 20

manifesto_batch2 = validate_snapshot(SNAPSHOT_BATCH2)
if not set(ENRICHED_POSITIVES).issubset(
    set(manifesto_batch2.get("assets") or [])
):
    raise RuntimeError(
        "Snapshot do lote 2 nao contem COLB, AMS e FOXF. "
        "Restaure dados/pesquisa_expansao_76_b2 antes de executar."
    )


def _batch3_manifest_is_usable() -> bool:
    if not SNAPSHOT_BATCH3.manifest.exists():
        return False
    try:
        payload = validate_snapshot(SNAPSHOT_BATCH3)
    except Exception:
        return False
    return (
        tuple(payload.get("assets") or ()) == RANDOM_ASSETS
        and str(payload.get("parent_snapshot_sha256") or "")
        == str(manifesto_base.get("snapshot_sha256") or "")
        and str((payload.get("bars") or {}).get("bar_snapshot_as_of_end") or "")
        == str(BAR_SNAPSHOT_AS_OF_END)
    )


if not _batch3_manifest_is_usable():
    print(
        "[batch3] snapshot ausente/incompativel; baixando os 20 novos ativos",
        flush=True,
    )
    SNAPSHOT_BATCH3.clear_generated()
    credenciais = load_alpaca_credentials(RAIZ_PROJETO)
    arquivos_barras = download_raw_bars(
        credenciais,
        SNAPSHOT_BATCH3,
        assets=RANDOM_ASSETS,
        replace=True,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
    )
    arquivos_eventos = download_corporate_actions(
        credenciais,
        SNAPSHOT_BATCH3,
        assets=RANDOM_ASSETS,
        replace=True,
        query_end=ANALYSIS_END_DATE,
    )
    build_snapshot_manifest(
        SNAPSHOT_BATCH3,
        arquivos_barras,
        arquivos_eventos,
        credentials=credenciais,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
        assets=RANDOM_ASSETS,
        snapshot_name="tcc-random-extension-batch3-20-v1",
        parent_snapshot_sha256=str(
            manifesto_base.get("snapshot_sha256") or ""
        ),
    )

manifesto_batch3 = validate_snapshot(SNAPSHOT_BATCH3)


# %% 3 - U56 + positivos confirmados do lote 2 + terceiro lote
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

frames_positive, exclusoes_positive, diagnosticos_positive, auditoria_positive = (
    prepare_model_frames(
        SNAPSHOT_BATCH2,
        assets=ENRICHED_POSITIVES,
        comparar_snapshot_referencia=False,
    )
)
if exclusoes_positive or len(frames_positive) != len(ENRICHED_POSITIVES):
    raise RuntimeError(
        "COLB, AMS e FOXF precisam estar integralmente disponiveis no snapshot "
        "do lote 2."
    )

frames_batch3, exclusoes_batch3, diagnosticos_batch3, auditoria_batch3 = (
    prepare_model_frames(
        SNAPSHOT_BATCH3,
        assets=RANDOM_ASSETS,
        comparar_snapshot_referencia=False,
    )
)
if exclusoes_batch3:
    raise RuntimeError(
        "Ativo do lote 3 apresentou problema estrutural: "
        + ",".join(str(row.get("symbol")) for row in exclusoes_batch3)
    )
if len(frames_batch3) != 20:
    raise RuntimeError(
        f"Lote 3 deveria manter 20 ativos; obtidos {len(frames_batch3)}."
    )

for symbol, frame in frames_batch3.items():
    dates = pd.DatetimeIndex(frame.index)
    if len(frame) < 2600:
        raise RuntimeError(f"{symbol}: historico insuficiente ({len(frame)} < 2600).")
    gaps = dates.to_series().diff().dt.total_seconds().div(86400.0)
    max_gap = float(gaps.dropna().max()) if gaps.notna().any() else 0.0
    if max_gap > 10.0:
        raise RuntimeError(f"{symbol}: quebra temporal de {max_gap:.1f} dias.")

frames_u79_raw = {
    **frames_u56_raw,
    **frames_positive,
    **frames_batch3,
}
if len(frames_u79_raw) != 79:
    raise RuntimeError(f"U79 deveria ter 79 ativos; obtidos {len(frames_u79_raw)}.")

config_u56, _ = build_variant_configs(frames_u56_raw, CONFIG)
_, reference_calendar, reference_calendar_source = preparar_painel_rotacao(
    frames_u56_raw,
    config_u56,
)

config_u79, _ = build_variant_configs(frames_u79_raw, CONFIG)
(
    frames_u79,
    common_dates,
    calendar_source_asset,
    symbols_u79,
    folds,
    all_decision_dates,
    decision_to_fold,
    decision_metadata,
) = _construir_contexto_execucao(
    frames_u79_raw,
    config_u79,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_calendar_source}",
)

symbols_u56 = sorted(frames_u56_raw)
symbols_u59 = sorted([*symbols_u56, *ENRICHED_POSITIVES])
candidate_margins = tuple(
    float(value)
    for value in config_u79.rotation_switch_margin_candidates
)
full_position = {
    symbol: index + 1
    for index, symbol in enumerate(symbols_u79)
}

benchmark_frames_u56 = {
    symbol: frames_u79[symbol]
    for symbol in symbols_u56
}
shared_benchmark = _benchmark_pesos_iguais(
    benchmark_frames_u56,
    symbols_u56,
    all_decision_dates[1:],
    float(config_u79.initial_capital),
    config_u79,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)
SHARED_BENCHMARK_NAME = (
    "Fixed U56 equal-weight buy-and-hold on the original reference calendar"
)

print(
    f"[universe] U56=56 U59=59 U79_B3=79 calendar={calendar_source_asset} "
    f"common_dates={len(common_dates)} folds={len(folds)}",
    flush=True,
)
print(
    "[u59] retained_positive_objects=" + ",".join(ENRICHED_POSITIVES),
    flush=True,
)
print(
    f"[benchmark] fixed=U56 ending={float(shared_benchmark.iloc[-1]):,.2f}",
    flush=True,
)


# %% 4 - Treino dos 79 modelos uma vez por fold
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
        f"calibration_models={len(symbols_u79)}",
        flush=True,
    )
    calibration_models = _ajustar_modelos_lightgbm(
        frames_u79,
        symbols_u79,
        train_dates,
        config_u79,
        phase=f"batch3_fold_{fold_id}_calibration",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    calibration_cache, _ = _precalcular_utilidades_modelo(
        calibration_models,
        frames_u79,
        symbols_u79,
        calibration_dates,
        config_u79,
    )
    calibration_missing = sorted(
        set(symbols_u79).difference(calibration_models)
    )
    if calibration_missing:
        print(
            f"[train-diagnostic] fold={fold_id} "
            "calibration_missing=" + ",".join(calibration_missing),
            flush=True,
        )

    print(
        f"[train] fold={fold_id} final_models={len(symbols_u79)}",
        flush=True,
    )
    final_models = _ajustar_modelos_lightgbm(
        frames_u79,
        symbols_u79,
        final_fit_dates,
        config_u79,
        phase=f"batch3_fold_{fold_id}_final",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    decision_cache, _ = _precalcular_utilidades_modelo(
        final_models,
        frames_u79,
        symbols_u79,
        decision_dates,
        config_u79,
    )
    final_missing = sorted(
        set(symbols_u79).difference(final_models)
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
        symbol: frames_u79[symbol]
        for symbol in subset_symbols
    }
    subset_config = config_u79.copiar_modelo(
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
        model_label=f"Control Batch 3 Signature Validation - {label}",
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


# %% 6 - Baselines U56, U59 e grupo U79
baseline_u56 = _run_subset(
    "U56_WINNER",
    symbols_u56,
    keep_result=True,
)
baseline_result = baseline_u56["result"]
baseline_capital = float(baseline_u56["metrics"]["ending_capital"])

baseline_u59 = _run_subset(
    "U59_ENRICHED",
    symbols_u59,
    keep_result=True,
)
baseline_u59_result = baseline_u59["result"]
baseline_u59_capital = float(baseline_u59["metrics"]["ending_capital"])

full_u79 = _run_subset(
    "U79_BATCH3_FULL",
    symbols_u79,
    keep_result=True,
)
full_u79_result = full_u79["result"]
full_u79_capital = float(full_u79["metrics"]["ending_capital"])

baseline_margin_by_fold = {
    int(row["fold_id"]): float(row["selected_margin"])
    for row in baseline_u56["margins"]
}
baseline_u59_margin_by_fold = {
    int(row["fold_id"]): float(row["selected_margin"])
    for row in baseline_u59["margins"]
}


# %% 7 - Correlacoes de cada candidato com o U56 vencedor
close_returns = {}
for symbol in symbols_u79:
    close = pd.to_numeric(
        frames_u79[symbol]["close"],
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


# %% 8 - Assinatura congelada antes dos replays individuais

def _classificar_assinatura(score_profile):
    beats = score_profile.get("candidate_beats_u56_best_share")
    score_std = score_profile.get("candidate_score_std")
    score_mean = score_profile.get("candidate_score_mean")
    corr_mean = score_profile.get("model_score_corr_u56_mean")

    if beats is None or score_std is None or score_mean is None or corr_mean is None:
        return {
            "signature_name": SIGNATURE_NAME,
            "signature_predicted_positive": False,
            "signature_class": "insufficient_score_data",
        }

    if float(beats) <= 0.0:
        signature_class = "dormant"
        predicted = False
    elif (
        float(beats)
        <= SIGNATURE_THRESHOLDS["beats_best_share_max_inclusive"]
        and float(score_std)
        <= SIGNATURE_THRESHOLDS["score_std_max_inclusive"]
        and float(score_mean)
        <= SIGNATURE_THRESHOLDS["score_mean_max_inclusive"]
        and abs(float(corr_mean))
        <= SIGNATURE_THRESHOLDS["abs_score_corr_u56_mean_max_inclusive"]
    ):
        signature_class = "selective_specialist"
        predicted = True
    elif (
        float(beats)
        > SIGNATURE_THRESHOLDS["beats_best_share_max_inclusive"]
        or float(score_std)
        > SIGNATURE_THRESHOLDS["score_std_max_inclusive"]
    ):
        signature_class = "invasive_or_unstable"
        predicted = False
    else:
        signature_class = "uncertain"
        predicted = False

    return {
        "signature_name": SIGNATURE_NAME,
        "signature_predicted_positive": bool(predicted),
        "signature_class": signature_class,
    }


signature_profiles = {}
for candidate in RANDOM_ASSETS:
    score_profile = _candidate_score_profile(candidate)
    signature_profiles[candidate] = {
        **score_profile,
        **_classificar_assinatura(score_profile),
    }

signature_pre_replay = pd.DataFrame(
    [
        {
            "asset": candidate,
            **signature_profiles[candidate],
        }
        for candidate in RANDOM_ASSETS
    ]
)

print("[signature-pre-replay] predictions frozen before candidate capital replays", flush=True)
print(
    signature_pre_replay[
        [
            "asset",
            "signature_class",
            "signature_predicted_positive",
            "candidate_beats_u56_best_share",
            "candidate_score_std",
            "candidate_score_mean",
            "model_score_corr_u56_mean",
        ]
    ].to_string(index=False),
    flush=True,
)


# %% 9 - Insercao individual contra U56 e contra U59

candidate_rows = []
candidate_fold_effects = {}

baseline_selected = (
    baseline_result.predictions["selected_asset"]
    .fillna("CASH")
    .astype(str)
)

baseline_u59_selected = (
    baseline_u59_result.predictions["selected_asset"]
    .fillna("CASH")
    .astype(str)
)

for position, candidate in enumerate(RANDOM_ASSETS, start=1):
    print(
        f"[candidate] {position}/20 U56_PLUS_{candidate} / U59_PLUS_{candidate}",
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

    enriched_scenario = _run_subset(
        f"U59_PLUS_{candidate}",
        [*symbols_u59, candidate],
        keep_result=True,
    )
    enriched_result = enriched_scenario["result"]
    enriched_metrics = enriched_scenario["metrics"]
    enriched_ending = float(enriched_metrics["ending_capital"])

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

    enriched_selected = (
        enriched_result.predictions["selected_asset"]
        .fillna("CASH")
        .astype(str)
    )
    enriched_aligned = pd.concat(
        [
            baseline_u59_selected.rename("baseline"),
            enriched_selected.rename("candidate"),
        ],
        axis=1,
        join="inner",
    )
    enriched_changed = (
        enriched_aligned["baseline"] != enriched_aligned["candidate"]
    )

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
    score_profile = signature_profiles[candidate]
    metadata = RANDOM_METADATA[candidate]
    signature_actual_positive = bool(
        ending > baseline_capital
    )

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
        "u59_ending_capital": baseline_u59_capital,
        "u59_plus_candidate_ending_capital": enriched_ending,
        "u59_insertion_capital_delta": (
            enriched_ending - baseline_u59_capital
        ),
        "u59_insertion_capital_pct": (
            enriched_ending / baseline_u59_capital - 1.0
            if baseline_u59_capital > 0.0
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
        "u59_changed_selected_asset_sessions": int(
            enriched_changed.sum()
        ),
        "u59_changed_selected_asset_share": float(
            enriched_changed.mean()
        ),
        "u59_candidate_selected_sessions": int(
            (enriched_selected == candidate).sum()
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
        "signature_actual_positive_u56": signature_actual_positive,
        "signature_prediction_correct_u56": bool(
            score_profile["signature_predicted_positive"]
            == signature_actual_positive
        ),
        **market_profile,
        **score_profile,
    }
    candidate_rows.append(row)

candidates = pd.DataFrame(candidate_rows).sort_values(
    "insertion_capital_pct",
    ascending=False,
).reset_index(drop=True)


# %% 10 - Relacao propriedade -> contribuicao
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
    "[batch3-group] "
    f"U56={baseline_capital:,.2f} "
    f"U59={baseline_u59_capital:,.2f} "
    f"U79_B3={full_u79_capital:,.2f} "
    f"U79_vs_U59={full_u79_capital / baseline_u59_capital - 1.0:+.4%}",
    flush=True,
)
print(
    f"[batch3-candidates-vs-u56] positive={positive_count} "
    f"negative={negative_count} zero={zero_count}",
    flush=True,
)
print("[batch3-candidates] ranking", flush=True)
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



signature_accuracy = float(
    candidates["signature_prediction_correct_u56"].mean()
)
signature_predicted_positive_count = int(
    candidates["signature_predicted_positive"].sum()
)
signature_true_positive_count = int(
    candidates["signature_actual_positive_u56"].sum()
)
print(
    f"[signature-validation] name={SIGNATURE_NAME} "
    f"accuracy={signature_accuracy:.2%} "
    f"predicted_positive={signature_predicted_positive_count} "
    f"actual_positive={signature_true_positive_count}",
    flush=True,
)


# %% 11 - Exportacao
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
for antigo in DIRETORIO_RESULTADOS.glob("*.csv"):
    antigo.unlink()
for antigo in DIRETORIO_RESULTADOS.glob("*.json"):
    antigo.unlink()
graficos = DIRETORIO_RESULTADOS / "graficos"
if graficos.exists():
    for antigo in graficos.glob("*.png"):
        antigo.unlink()

signature_pre_replay.to_csv(
    DIRETORIO_RESULTADOS / "signature_batch3_pre_replay.csv",
    index=False,
)
candidates.to_csv(
    DIRETORIO_RESULTADOS / "signature_batch3_candidates.csv",
    index=False,
)
property_correlations.to_csv(
    DIRETORIO_RESULTADOS / "signature_batch3_property_correlations.csv",
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
baseline_u59_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "u59_enriched_predictions.csv",
    index=False,
)
baseline_u59_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "u59_enriched_trades.csv",
    index=False,
)
full_u79_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "u79_batch3_predictions.csv",
    index=False,
)
full_u79_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "u79_batch3_trades.csv",
    index=False,
)

payload = {
    "research_version": RESEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "base_snapshot_sha256": manifesto_base.get("snapshot_sha256"),
    "batch3_snapshot_sha256": manifesto_batch3.get("snapshot_sha256"),
    "question": (
        "Validate the predeclared selective-specialist signature on a third untouched "
        "20-asset random batch, while measuring both U56-control insertion and "
        "incremental insertion into U59 enriched by COLB, AMS and FOXF."
    ),
    "protocol": {
        "winner_universe_size": 56,
        "batch3_size": 20,
        "enriched_base_size": 59,
        "expanded_universe_size": 79,
        "random_selection_seed": RANDOM_SELECTION_SEED,
        "random_selection_catalog_date": RANDOM_SELECTION_CATALOG_DATE,
        "random_selection_rule": RANDOM_SELECTION_RULE,
        "batch1_assets_excluded": list(BATCH1_ASSETS),
        "batch2_assets_excluded_from_random_draw": list(BATCH2_ASSETS),
        "retained_positive_batch2_assets": list(ENRICHED_POSITIVES),
        "batch3_assets": list(RANDOM_ASSETS),
        "signature_name": SIGNATURE_NAME,
        "signature_thresholds": dict(SIGNATURE_THRESHOLDS),
        "rejected_random_candidates": list(REJECTED_RANDOM_CANDIDATES),
        "selection_uses_backtest_performance": False,
        "base_calendar_fixed_to_u56": True,
        "benchmark_fixed_to_u56": True,
        "lightgbm_parameters_unchanged": True,
        "fold_method_unchanged": True,
        "switch_margin_candidates": list(candidate_margins),
        "models_trained_once_per_fold": True,
        "individual_test_control": "U56 plus exactly one batch-3 candidate",
        "individual_test_enriched": "U59 plus exactly one batch-3 candidate",
        "group_test": "U59 plus all 20 batch-3 candidates",
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
    "u59_enriched": {
        "retained_assets": list(ENRICHED_POSITIVES),
        "metrics": baseline_u59["metrics"],
        "margins": baseline_u59["margins"],
        "capital_delta_vs_u56": baseline_u59_capital - baseline_capital,
        "capital_pct_vs_u56": (
            baseline_u59_capital / baseline_capital - 1.0
            if baseline_capital > 0.0
            else None
        ),
    },
    "signature_validation": {
        "name": SIGNATURE_NAME,
        "thresholds": dict(SIGNATURE_THRESHOLDS),
        "accuracy_u56": signature_accuracy,
        "predicted_positive_count": signature_predicted_positive_count,
        "actual_positive_count": signature_true_positive_count,
        "pre_replay_rows": signature_pre_replay.to_dict(orient="records"),
    },
    "u79_batch3_group": {
        "metrics": full_u79["metrics"],
        "margins": full_u79["margins"],
        "capital_delta_vs_u59": full_u79_capital - baseline_u59_capital,
        "capital_pct_vs_u59": (
            full_u79_capital / baseline_u59_capital - 1.0
            if baseline_u59_capital > 0.0
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
    "batch3_diagnostics": diagnosticos_batch3,
    "batch3_audit": auditoria_batch3,
    "u56_diagnostics": diagnosticos_u56,
    "u56_audit": auditoria_u56,
    "runtime_seconds": float(time.perf_counter() - training_started),
    "interpretation_rule": (
        "The selective-specialist signature was frozen before candidate capital "
        "replays. Its first confirmatory target is the sign of U56+candidate "
        "capital contribution. U59+candidate is a separate enriched-universe "
        "incremental test. Market-return correlation remains descriptive and "
        "is not itself part of the signature."
    ),
}

with (
    DIRETORIO_RESULTADOS / "signature_batch3_validation.json"
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
    "[done] batch3 signature validation concluido "
    f"seconds={time.perf_counter() - training_started:.3f}",
    flush=True,
)
