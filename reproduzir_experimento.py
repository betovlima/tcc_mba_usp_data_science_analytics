"""REPRODUCAO DE PARIDADE TCC x MCT U67.

Objetivo
--------
Executar no TCC exatamente o comportamento observado na Strategy #13 do MCT:

- carregar o snapshot U67 congelado e versionado no Git;
- validar os mesmos 67 ativos solicitados pelo MCT;
- usar barras 1Day, SIP, RAW e Corporate Actions congelados;
- normalizar splits;
- aplicar a mesma politica estrutural que deixou 65 ativos efetivos no job MCT;
- usar o mesmo LightGBM Control, folds, purge, custos e regras de rotacao;
- comparar o resultado com o job MCT auditado.

Execucao no Spyder
------------------
Abra reproduzir_experimento.py, reinicie o kernel e execute com F5.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import hashlib
import json
import math
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from engine.configuracao import (
    CONFIG,
    U67_U67_EXPECTED_EFFECTIVE_COUNT,
    U67_U67_EXPECTED_EXCLUSIONS,
    U67_U67_EXPECTED_REQUESTED_COUNT,
    U67_REQUESTED_ASSETS,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    selecionar_switch_margin,
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
from reproducao.artefatos import (
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.graficos import gerar_analises_backtest
from reproducao.preparacao import load_raw_bar_file, prepare_model_frames


# %% 0 - Contrato de paridade com o job MCT auditado
ROOT = Path(__file__).resolve().parent
DATA = SnapshotPaths.u67(ROOT)
OUT = ROOT / "output" / "reproducao"

REPRODUCTION_VERSION = "1.22.0-dev.5"
EXECUTION_SCHEMA = "u67-mct-operational-parity-v1"


# O job MCT 20261007T095423-60e489c0 foi auditado com dados ate 2026-10-06.
# Mantemos a mesma data para que a primeira execucao de paridade seja
# comparavel operacao por operacao.
MCT_JOB_ID = "20261007T095423-60e489c0"
MCT_ANALYSIS_END_DATE = "2026-10-06"
MCT_SAME_CUTOFF_DATE = "2026-09-17"
MCT_ENDING_CAPITAL = 76_927_051.38897176
MCT_CAPITAL_AT_2026_09_17 = 78_782_538.31270888
MCT_SNAPSHOT_SHA256 = (
    "3f1159fced345e6ed2888256a135e36a9337398c992ffc2e2b83fe16929d5d81"
)
MCT_MARKET_DATA_SIGNATURE = (
    "a88a2c658e493ca46ba2573a830d69a3ada07dba3f13feefd87bf963bc1591ce"
)

PARITY_ABS_TOL = 0.01
MODEL_DATA_COLUMNS = ("open", "high", "low", "close", "volume")


def _canonical_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Replica a canonicalizacao usada pelo MCT antes do SHA-256."""
    canonical = frame[list(MODEL_DATA_COLUMNS)].copy()
    canonical.index = pd.to_datetime(
        canonical.index,
        utc=True,
        errors="coerce",
    )
    canonical = canonical.loc[~canonical.index.isna()]
    canonical = canonical[
        ~canonical.index.duplicated(keep="last")
    ].sort_index()
    try:
        canonical.index = canonical.index.as_unit("ns")
    except AttributeError:
        canonical.index = pd.DatetimeIndex(
            canonical.index.to_numpy(dtype="datetime64[ns]"),
            tz="UTC",
        )
    for column in MODEL_DATA_COLUMNS:
        canonical[column] = pd.to_numeric(
            canonical[column],
            errors="coerce",
        ).astype(np.float64, copy=False)
    return canonical


def _history_frame_sha256(frame: pd.DataFrame) -> str:
    """Replica o hash OHLCV usado no manifest de reproducibilidade do MCT."""
    canonical = _canonical_history_frame(frame)
    row_hashes = pd.util.hash_pandas_object(
        canonical,
        index=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


if len(U67_REQUESTED_ASSETS) != EXPECTED_REQUESTED_COUNT:
    raise RuntimeError(
        "Contrato U67 invalido: "
        f"esperado={EXPECTED_REQUESTED_COUNT} "
        f"observado={len(U67_REQUESTED_ASSETS)}"
    )
if len(set(U67_REQUESTED_ASSETS)) != EXPECTED_REQUESTED_COUNT:
    raise RuntimeError("Contrato U67 contem ativos duplicados.")

started = time.perf_counter()

print("=" * 78, flush=True)
print("TCC MBA USP - PARIDADE COM MCT STRATEGY #13", flush=True)
print(
    f"version={REPRODUCTION_VERSION} schema={EXECUTION_SCHEMA}",
    flush=True,
)
print(f"mct_job={MCT_JOB_ID}", flush=True)
print(f"analysis_end={MCT_ANALYSIS_END_DATE}", flush=True)
print("data_source=FROZEN_GIT_SNAPSHOT", flush=True)
print("feed=SIP adjustment=RAW timeframe=1Day", flush=True)
print("database=NO", flush=True)
print("csv_float_precision=round_trip", flush=True)
print("requested_assets=67 expected_effective_assets=65", flush=True)
print("expected_runtime_exclusions=CLMT,DOC", flush=True)
print("=" * 78, flush=True)


# %% 1 - Snapshot U67 congelado e versionado
snapshot_manifest = validate_snapshot(DATA)

manifest_assets = tuple(snapshot_manifest.get("assets") or ())
if manifest_assets != tuple(U67_REQUESTED_ASSETS):
    raise RuntimeError(
        "Snapshot U67 possui universo diferente do contrato oficial. "
        f"esperado={len(U67_REQUESTED_ASSETS)} "
        f"observado={len(manifest_assets)}"
    )

snapshot_bar_end = str(
    (snapshot_manifest.get("bars") or {}).get("bar_snapshot_as_of_end") or ""
)
snapshot_action_end = str(
    (snapshot_manifest.get("corporate_actions") or {}).get("query_end") or ""
)
if snapshot_bar_end != MCT_ANALYSIS_END_DATE:
    raise RuntimeError(
        "Snapshot U67 possui cutoff de barras inesperado: "
        f"esperado={MCT_ANALYSIS_END_DATE} observado={snapshot_bar_end}"
    )
if snapshot_action_end != MCT_ANALYSIS_END_DATE:
    raise RuntimeError(
        "Snapshot U67 possui cutoff de Corporate Actions inesperado: "
        f"esperado={MCT_ANALYSIS_END_DATE} observado={snapshot_action_end}"
    )

print(
    "[snapshot] mode=frozen_git "
    f"requested_assets={len(U67_REQUESTED_ASSETS)} "
    f"sha256={snapshot_manifest.get('snapshot_sha256')}",
    flush=True,
)


# %% 2 - Mesmo processamento estrutural observado no MCT
frames_raw, exclusions, diagnostics, data_audit = prepare_model_frames(
    DATA,
    assets=U67_REQUESTED_ASSETS,
    comparar_snapshot_referencia=False,
    # O MCT usa os floats recebidos da Alpaca diretamente em memoria.
    # Como o TCC persiste CSV antes do treino, usamos o parser round_trip
    # para recuperar exatamente o float64 serializado com %.17g.
    csv_float_precision="round_trip",
)

excluded_symbols = frozenset(
    str(item.get("symbol") or "").strip().upper()
    for item in exclusions
)
if excluded_symbols != EXPECTED_EXCLUSIONS:
    raise RuntimeError(
        "O universo efetivo nao corresponde ao job MCT auditado. "
        f"esperado={sorted(EXPECTED_EXCLUSIONS)} "
        f"observado={sorted(excluded_symbols)} "
        f"exclusoes={json.dumps(exclusions, ensure_ascii=False, default=str)}"
    )
if len(frames_raw) != EXPECTED_EFFECTIVE_COUNT:
    raise RuntimeError(
        "O job de paridade precisa entregar exatamente 65 ativos ao modelo. "
        f"observado={len(frames_raw)}"
    )

effective_assets_requested_order = tuple(
    symbol
    for symbol in U67_REQUESTED_ASSETS
    if symbol in frames_raw
)
print(
    "[universe] requested=67 effective=65 excluded="
    + ",".join(sorted(excluded_symbols)),
    flush=True,
)
print(
    "[universe] effective_assets="
    + ",".join(effective_assets_requested_order),
    flush=True,
)

# Hashes calculados com exatamente a mesma canonicalizacao OHLCV do MCT.
# Eles permitem comparar os 65 ativos sem depender do hash fisico do CSV.
market_data_hash_rows = []
for symbol in effective_assets_requested_order:
    raw_frame = load_raw_bar_file(
        DATA.raw_bars / f"{symbol}.csv",
        float_precision="round_trip",
    )
    normalized_frame = frames_raw[symbol]
    raw_canonical = _canonical_history_frame(raw_frame)
    normalized_canonical = _canonical_history_frame(normalized_frame)
    row = {
        "asset": symbol,
        "raw_sha256": _history_frame_sha256(raw_frame),
        "normalized_sha256": _history_frame_sha256(normalized_frame),
        "raw_rows": int(len(raw_canonical)),
        "normalized_rows": int(len(normalized_canonical)),
        "raw_first_timestamp": (
            pd.Timestamp(raw_canonical.index.min()).isoformat()
            if len(raw_canonical)
            else None
        ),
        "raw_last_timestamp": (
            pd.Timestamp(raw_canonical.index.max()).isoformat()
            if len(raw_canonical)
            else None
        ),
        "normalized_first_timestamp": (
            pd.Timestamp(normalized_canonical.index.min()).isoformat()
            if len(normalized_canonical)
            else None
        ),
        "normalized_last_timestamp": (
            pd.Timestamp(normalized_canonical.index.max()).isoformat()
            if len(normalized_canonical)
            else None
        ),
    }
    market_data_hash_rows.append(row)
    print(
        f"[hash] {symbol} "
        f"raw={row['raw_sha256']} "
        f"normalized={row['normalized_sha256']}",
        flush=True,
    )


# %% 3 - Mesmo calendario U56 elegivel e mesma configuracao Control
u56_eligible = tuple(
    symbol
    for symbol in tuple(CONFIG.assets)
    if symbol in frames_raw
)
if len(u56_eligible) != 54:
    raise RuntimeError(
        "O calendario de paridade deveria usar 54 ativos elegiveis do U56 "
        f"(U56 sem CLMT/DOC); observado={len(u56_eligible)}"
    )

config_u56 = build_control_config(
    {symbol: frames_raw[symbol] for symbol in u56_eligible},
    CONFIG,
)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    {symbol: frames_raw[symbol] for symbol in u56_eligible},
    config_u56,
)

config_u67 = build_control_config(
    {symbol: frames_raw[symbol] for symbol in effective_assets_requested_order},
    CONFIG,
)
config_u67 = config_u67.copiar_modelo(
    update={
        "analysis_end_date": MCT_ANALYSIS_END_DATE,
    }
)

(
    frames,
    common_dates,
    calendar_source,
    symbols,
    folds,
    all_decision_dates,
    decision_to_fold,
    decision_metadata,
) = _construir_contexto_execucao(
    {
        symbol: frames_raw[symbol]
        for symbol in effective_assets_requested_order
    },
    config_u67,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_source}",
)

if len(symbols) != EXPECTED_EFFECTIVE_COUNT:
    raise RuntimeError(
        "O contexto modelavel nao manteve os 65 ativos efetivos. "
        f"observado={len(symbols)}"
    )

print(
    f"[calendar] reference_source={reference_source} "
    f"calendar_source={calendar_source} "
    f"common_dates={len(common_dates)} "
    f"folds={len(folds)} "
    f"decision_sessions={len(all_decision_dates)-1}",
    flush=True,
)


# %% 4 - Mesmo benchmark do MCT: U56 elegivel equal-weight
benchmark = _benchmark_pesos_iguais(
    {
        symbol: frames[symbol]
        for symbol in u56_eligible
        if symbol in frames
    },
    [
        symbol
        for symbol in u56_eligible
        if symbol in frames
    ],
    all_decision_dates[1:],
    float(config_u67.initial_capital),
    config_u67,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)

BENCHMARK_NAME = "Fixed eligible U56 equal-weight buy-and-hold"


# %% 5 - Treino, calibracao e politicas identicos ao MCT
candidate_margins = tuple(
    float(value)
    for value in config_u67.rotation_switch_margin_candidates
)
fold_policies = {}
fold_margins = []
fold_calibration_candidates = []

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
        f"assets={len(symbols)} calibration",
        flush=True,
    )

    calibration_models = _ajustar_modelos_lightgbm(
        frames,
        symbols,
        train_dates,
        config_u67,
        phase=f"mct_parity_fold_{fold_id}_calibration",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    calibration_cache, _ = _precalcular_utilidades_modelo(
        calibration_models,
        frames,
        symbols,
        calibration_dates,
        config_u67,
    )

    candidate_scores = []
    for margin in candidate_margins:
        policy = _politica_utilidade(
            calibration_models,
            frames,
            symbols,
            config_u67,
            float(margin),
            utility_cache=calibration_cache,
        )
        score = _crescimento_politica_simples(
            policy,
            frames,
            symbols,
            calibration_dates,
            config_u67,
        )
        candidate_scores.append((float(margin), float(score)))

    selection = selecionar_switch_margin(candidate_scores)
    selected_margin = float(selection["selected_candidate_margin"])
    effective_margin = max(
        float(config_u67.rotation_switch_margin),
        selected_margin,
    )

    for candidate_margin, candidate_score in candidate_scores:
        fold_calibration_candidates.append(
            {
                "fold_id": fold_id,
                "candidate_margin": float(candidate_margin),
                "calibration_score": float(candidate_score),
                "selected": bool(
                    abs(float(candidate_margin) - selected_margin) <= 1e-12
                ),
                "selected_margin": selected_margin,
                "effective_margin": effective_margin,
            }
        )

    print(
        f"[calibration] fold={fold_id} "
        f"selected_margin={selected_margin:.6f} "
        f"effective_margin={effective_margin:.6f} "
        f"candidates={candidate_scores}",
        flush=True,
    )

    final_models = _ajustar_modelos_lightgbm(
        frames,
        symbols,
        final_fit_dates,
        config_u67,
        phase=f"mct_parity_fold_{fold_id}_final",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    decision_cache, _ = _precalcular_utilidades_modelo(
        final_models,
        frames,
        symbols,
        decision_dates,
        config_u67,
    )

    fold_policies[fold_id] = _politica_utilidade(
        final_models,
        frames,
        symbols,
        config_u67,
        effective_margin,
        fold_id=fold_id,
        calibrated_switch_margin=selected_margin,
        utility_cache=decision_cache,
    )
    fold_margins.append(
        {
            "fold_id": fold_id,
            "selected_margin": selected_margin,
            "effective_margin": effective_margin,
            "calibration_score": float(
                selection["selected_calibration_score"]
            ),
        }
    )


# %% 6 - Replay financeiro
scheduled_policy = _politica_agendada(
    fold_policies,
    decision_to_fold,
)

result = _simular_exato(
    "tcc_u67_mct_operational_parity",
    scheduled_policy,
    frames,
    symbols,
    all_decision_dates,
    config_u67,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    decision_metadata=decision_metadata,
    model_label="TCC U67 v1.21.0 Control - MCT parity",
    method_line=(
        "- Same U67 scientific engine; fresh Alpaca RAW/SIP; "
        "same effective MCT universe; same Control settings."
    ),
    benchmark_override=benchmark,
    benchmark_override_name=BENCHMARK_NAME,
)

metrics = summarize_metrics(
    result,
    folds,
    float(config_u67.initial_capital),
)
ending_capital = float(metrics["ending_capital"])


# %% 7 - Comparacao direta com o job MCT auditado
predictions = result.predictions.reset_index().copy()
predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)

same_cutoff = pd.Timestamp(MCT_SAME_CUTOFF_DATE, tz="UTC")
same_cutoff_rows = predictions.loc[
    predictions["timestamp"].dt.normalize() == same_cutoff
]
if same_cutoff_rows.empty:
    capital_same_cutoff = None
else:
    capital_same_cutoff = float(
        same_cutoff_rows.iloc[-1]["strategy_equity"]
    )

final_delta = ending_capital - MCT_ENDING_CAPITAL
final_ratio = ending_capital / MCT_ENDING_CAPITAL - 1.0
same_cutoff_delta = (
    capital_same_cutoff - MCT_CAPITAL_AT_2026_09_17
    if capital_same_cutoff is not None
    else None
)

final_exact = math.isclose(
    ending_capital,
    MCT_ENDING_CAPITAL,
    rel_tol=0.0,
    abs_tol=PARITY_ABS_TOL,
)
same_cutoff_exact = bool(
    capital_same_cutoff is not None
    and math.isclose(
        capital_same_cutoff,
        MCT_CAPITAL_AT_2026_09_17,
        rel_tol=0.0,
        abs_tol=PARITY_ABS_TOL,
    )
)

print("=" * 78, flush=True)
print("[PARITY RESULT]", flush=True)
print(
    f"TCC fresh capital 2026-10-06 = US$ {ending_capital:,.8f}",
    flush=True,
)
print(
    f"MCT audited capital 2026-10-06 = US$ {MCT_ENDING_CAPITAL:,.8f}",
    flush=True,
)
print(
    f"delta_final = US$ {final_delta:+,.8f} "
    f"ratio={final_ratio:+.12%} exact_to_cent={final_exact}",
    flush=True,
)
if capital_same_cutoff is not None:
    print(
        f"TCC fresh capital 2026-09-17 = US$ {capital_same_cutoff:,.8f}",
        flush=True,
    )
    print(
        "MCT audited capital 2026-09-17 = "
        f"US$ {MCT_CAPITAL_AT_2026_09_17:,.8f}",
        flush=True,
    )
    print(
        f"delta_same_cutoff = US$ {same_cutoff_delta:+,.8f} "
        f"exact_to_cent={same_cutoff_exact}",
        flush=True,
    )
print(
    f"cagr={float(metrics['cagr']):.6%} "
    f"sharpe={float(metrics['sharpe']):.8f} "
    f"maxdd={float(metrics['maximum_drawdown']):.6%} "
    f"worst_fold={float(metrics['worst_fold_return']):.6%}",
    flush=True,
)
print("=" * 78, flush=True)


# %% 8 - Artefatos de auditoria
# Limpa completamente a pasta para impedir que pacotes antigos entrem no ZIP.
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

runtime_environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "lightgbm": _package_version("lightgbm"),
    "scikit_learn": _package_version("scikit-learn"),
    "threadpoolctl": _package_version("threadpoolctl"),
    "threadpool_runtime": threadpool_info(),
    "deterministic_execution": bool(config_u67.deterministic_execution),
    "numeric_thread_limit": int(config_u67.numeric_thread_limit),
    "lightgbm_n_jobs": (
        (config_u67.research_model_settings.get("lightgbm") or {}).get(
            "n_jobs"
        )
    ),
}

pd.DataFrame(
    {"asset": U67_REQUESTED_ASSETS}
).to_csv(
    OUT / "u67_requested_assets.csv",
    index=False,
)
pd.DataFrame(
    {"asset": symbols}
).to_csv(
    OUT / "u67_effective_assets.csv",
    index=False,
)
pd.DataFrame(exclusions).to_csv(
    OUT / "u67_runtime_exclusions.csv",
    index=False,
)
pd.DataFrame(fold_margins).to_csv(
    OUT / "u67_fold_margins.csv",
    index=False,
)
pd.DataFrame(fold_calibration_candidates).to_csv(
    OUT / "u67_fold_calibration_candidates.csv",
    index=False,
)
pd.DataFrame(market_data_hash_rows).to_csv(
    OUT / "u67_market_data_hashes.csv",
    index=False,
)
with (
    OUT / "u67_runtime_environment.json"
).open("w", encoding="utf-8") as handle:
    json.dump(
        runtime_environment,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
predictions.to_csv(
    OUT / "u67_predictions.csv",
    index=False,
)
result.trades.to_csv(
    OUT / "u67_trades.csv",
    index=False,
)

payload = {
    "reproduction_version": REPRODUCTION_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "status": "completed",
    "purpose": "independent_tcc_reproduction_of_mct_strategy_13",
    "mct_reference": {
        "job_id": MCT_JOB_ID,
        "analysis_end_date": MCT_ANALYSIS_END_DATE,
        "same_cutoff_date": MCT_SAME_CUTOFF_DATE,
        "ending_capital": MCT_ENDING_CAPITAL,
        "capital_at_same_cutoff": MCT_CAPITAL_AT_2026_09_17,
        "snapshot_sha256": MCT_SNAPSHOT_SHA256,
        "market_data_signature": MCT_MARKET_DATA_SIGNATURE,
    },
    "data": {
        "source": "alpaca",
        "feed": "sip",
        "timeframe": "1Day",
        "adjustment": "raw",
        "full_refresh": False,
        "snapshot_mode": "frozen_git",
        "database_used": False,
        "snapshot": snapshot_manifest,
        "data_audit": data_audit,
        "diagnostics": diagnostics,
        "market_data_hashes": market_data_hash_rows,
    },
    "universe": {
        "requested_count": len(U67_REQUESTED_ASSETS),
        "requested_assets": list(U67_REQUESTED_ASSETS),
        "effective_count": len(symbols),
        "effective_assets": list(symbols),
        "excluded_assets": exclusions,
    },
    "calendar": {
        "reference_source": reference_source,
        "calendar_source": calendar_source,
        "common_dates": len(common_dates),
        "decision_sessions": len(all_decision_dates) - 1,
    },
    "fold_margins": fold_margins,
    "fold_calibration_candidates": fold_calibration_candidates,
    "runtime_environment": runtime_environment,
    "metrics": metrics,
    "parity": {
        "tcc_ending_capital": ending_capital,
        "mct_ending_capital": MCT_ENDING_CAPITAL,
        "ending_capital_delta": final_delta,
        "ending_capital_ratio": final_ratio,
        "ending_capital_exact_to_cent": final_exact,
        "tcc_capital_at_2026_09_17": capital_same_cutoff,
        "mct_capital_at_2026_09_17": MCT_CAPITAL_AT_2026_09_17,
        "same_cutoff_delta": same_cutoff_delta,
        "same_cutoff_exact_to_cent": same_cutoff_exact,
    },
    "runtime_seconds": float(time.perf_counter() - started),
}

with (
    OUT / "reproducao_u67_mct_parity.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

gerar_analises_backtest(
    OUT,
    manifest=snapshot_manifest,
    result=result,
)

package = criar_pacote_analise(
    OUT,
    comparison_file="reproducao_u67_mct_parity.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_reproducao_u67_mct_parity.zip",
)

print(
    f"[done] package={package}",
    flush=True,
)
sinal_sonoro_conclusao()
