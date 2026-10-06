"""VALIDACAO FINANCEIRA PROSPECTIVA ONE-SHOT DA ASSINATURA v1.18.

IMPORTANTE
----------
A coorte e as metricas foram congeladas ANTES deste runner revelar capital.

Este arquivo:
- NAO procura novos ativos;
- NAO baixa dados da Alpaca;
- NAO altera a formula S;
- NAO substitui candidatos;
- treina os modelos uma unica vez por fold para U59 + 32 candidatos;
- reproduz primeiro o U59 e aborta se o baseline divergir;
- depois revela os 32 efeitos individuais contra U59;
- calcula os endpoints estatisticos pre-registrados;
- executa um unico replay secundario do grupo dos 8 ativos high-S.

Resultado primario pre-registrado:
    Spearman(S, capital_pct_vs_u59) entre os 24 candidatos ativos,
    com alternativa unilateral rho > 0 por permutacao deterministica.

Criterio:
    suporte prospectivo se rho > 0 e p_perm < 0.05.

Nenhum resultado desta execucao pode ser usado para reajustar a v1.18 e
continuar chamando a mesma analise de validacao prospectiva.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from engine.configuracao import CONFIG
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
    RESEARCH_VERSION,
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_variant_configs, summarize_metrics
from reproducao.preparacao import prepare_model_frames


# %% 0 - Protocolo congelado
ROOT = Path(__file__).resolve().parent

BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_expansao_76_b2"
)
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)

COHORT_FILE = (
    ROOT
    / "dados"
    / "assinatura_matematica"
    / "prospective_validation_cohort_v118.csv"
)
FREEZE_FILE = (
    ROOT
    / "dados"
    / "assinatura_matematica"
    / "prospective_validation_freeze_v118.json"
)
PLAN_FILE = (
    ROOT
    / "dados"
    / "assinatura_matematica"
    / "prospective_validation_plan_v118.json"
)
SEARCH_JSON = (
    ROOT / "output" / "busca_ativos" / "asset_search.json"
)

OUT = ROOT / "output" / "validacao_prospectiva_financeira_v118"
FIG = OUT / "graficos"

SCRIPT_VERSION = "1.18.2-dev.2"
EXECUTION_SCHEMA = "prospective-signature-financial-validation-v1"
EXPECTED_SHARED_MODULE_VERSION = "1.17.0-dev.1"
EXPECTED_SOURCE_COHORT_SHA256 = (
    "a1fe00ea2a7c7691d55396366be0375e64294ab43d97d4949a2433d106443978"
)
EXPECTED_COHORT_SEMANTIC_SHA256 = (
    "37fd32c3fc8c2a424f9a1f26a1fc0764ee8f6361eaec2ac7a13cd60a6439d2b6"
)
EXPECTED_COHORT_SIZE = 32
EXPECTED_ACTIVE_SIZE = 24
EXPECTED_DORMANT_SIZE = 8
EXPECTED_BASELINE = 30_080_091.008142874
BASELINE_REL_TOL = 1e-9

U59_ADDITIONS = ("COLB", "AMS", "FOXF")

PERMUTATIONS = 20_000
BOOTSTRAP_REPS = 5_000
RANDOM_SEED = 20261005


# %% 1 - Helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def semantic_cohort_sha256(path: Path) -> str:
    """Hash semantico independente de CRLF/LF e serializacao trivial de float.

    O hash bruto do CSV produzido no Windows e preservado no freeze original,
    mas Git pode normalizar quebras de linha e pandas/Git podem reserializar
    floats sem mudar a coorte. Este hash protege o conteudo cientificamente
    relevante: ordem, ativos, estratos, score e identidades do protocolo.
    """
    frame = pd.read_csv(path)

    columns = [
        "prospective_validation_rank",
        "asset",
        "validation_stratum",
        "activation",
        "dormant",
        "specialist_score",
        "validation_score",
        "active_percentile",
        "beats_best_share",
        "abs_corr_best",
        "score_std",
        "score_sessions",
        "research_version",
        "execution_schema",
        "financial_reference",
        "capital_observed_at_freeze",
        "selection_salt",
        "source_search_sha256",
        "frozen_signature_sha256",
    ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(
            "Coorte prospectiva incompleta para hash semantico: "
            + ",".join(missing)
        )

    boolean_columns = {
        "activation",
        "dormant",
        "capital_observed_at_freeze",
    }
    integer_columns = {
        "prospective_validation_rank",
        "score_sessions",
    }
    float_columns = {
        "specialist_score",
        "validation_score",
        "active_percentile",
        "beats_best_share",
        "abs_corr_best",
        "score_std",
    }

    rows = []
    for _, row in frame[columns].iterrows():
        normalized = {}
        for column in columns:
            value = row[column]
            if pd.isna(value):
                normalized[column] = None
            elif column in boolean_columns:
                if isinstance(value, (bool, np.bool_)):
                    normalized[column] = bool(value)
                else:
                    normalized[column] = (
                        str(value).strip().lower()
                        in {"true", "1", "yes"}
                    )
            elif column in integer_columns:
                normalized[column] = int(value)
            elif column in float_columns:
                normalized[column] = round(float(value), 12)
            else:
                normalized[column] = str(value)
        rows.append(normalized)

    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _safe_ap(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if int(y.sum()) == 0:
        return None
    return float(average_precision_score(y, score))


def _percentiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"p025": None, "median": None, "p975": None}
    p = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "p025": float(p[0]),
        "median": float(p[1]),
        "p975": float(p[2]),
    }


def _permutation_spearman_one_sided(x, y, reps, rng):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(spearmanr(x, y).statistic)
    count = 0
    for _ in range(int(reps)):
        perm = rng.permutation(y)
        statistic = float(spearmanr(x, perm).statistic)
        if statistic >= observed - 1e-15:
            count += 1
    pvalue = (count + 1.0) / (float(reps) + 1.0)
    return observed, pvalue


def _permutation_auc_one_sided(score, label, reps, rng):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    observed = _safe_auc(label, score)
    if observed is None:
        return None, None
    count = 0
    for _ in range(int(reps)):
        perm = rng.permutation(label)
        statistic = _safe_auc(perm, score)
        if statistic is not None and statistic >= observed - 1e-15:
            count += 1
    pvalue = (count + 1.0) / (float(reps) + 1.0)
    return float(observed), float(pvalue)


def _bootstrap_active(frame, reps, rng):
    rho_values = []
    auc_values = []
    n = len(frame)
    for _ in range(int(reps)):
        sample = frame.iloc[
            rng.integers(0, n, n)
        ]
        rho = spearmanr(
            sample["specialist_score"],
            sample["capital_pct_vs_u59"],
        ).statistic
        if np.isfinite(rho):
            rho_values.append(float(rho))
        auc = _safe_auc(
            sample["positive"].astype(int),
            sample["specialist_score"],
        )
        if auc is not None:
            auc_values.append(float(auc))
    return rho_values, auc_values


# %% 2 - Guards antes de qualquer treino/replay
if RESEARCH_VERSION != EXPECTED_SHARED_MODULE_VERSION:
    raise RuntimeError(
        "Modulo compartilhado inesperado. "
        f"esperado={EXPECTED_SHARED_MODULE_VERSION!r} "
        f"observado={RESEARCH_VERSION!r}."
    )

for path in (
    COHORT_FILE,
    FREEZE_FILE,
    PLAN_FILE,
    SEARCH_JSON,
):
    if not path.exists():
        raise RuntimeError(
            "Artefato congelado ausente: " + str(path)
        )

cohort_worktree_sha = sha256_file(COHORT_FILE)
cohort_semantic_sha = semantic_cohort_sha256(COHORT_FILE)

if cohort_semantic_sha != EXPECTED_COHORT_SEMANTIC_SHA256:
    raise RuntimeError(
        "A coorte prospectiva mudou semanticamente depois do congelamento. "
        f"esperado={EXPECTED_COHORT_SEMANTIC_SHA256} "
        f"observado={cohort_semantic_sha}"
    )

if cohort_worktree_sha != EXPECTED_SOURCE_COHORT_SHA256:
    print(
        "[freeze-guard] hash bruto do arquivo difere do pacote original, "
        "mas o hash semantico e identico. Isso e esperado quando Git/Windows "
        "normaliza CRLF/LF ou a representacao textual de floats. "
        f"source_raw={EXPECTED_SOURCE_COHORT_SHA256} "
        f"worktree_raw={cohort_worktree_sha}",
        flush=True,
    )

freeze = json.loads(
    FREEZE_FILE.read_text(encoding="utf-8")
)
plan = json.loads(
    PLAN_FILE.read_text(encoding="utf-8")
)
search_meta = json.loads(
    SEARCH_JSON.read_text(encoding="utf-8")
)

if freeze.get("status") != "frozen_before_financial_replay":
    raise RuntimeError("Freeze prospectivo invalido.")
if freeze.get("financial_reference") != "U59_WINNER":
    raise RuntimeError("Referencia financeira do freeze nao e U59.")
if plan.get("status") != "preregistered_before_outcome_reveal":
    raise RuntimeError("Plano estatistico nao esta pre-registrado.")
if plan.get("frozen_cohort_sha256") != EXPECTED_SOURCE_COHORT_SHA256:
    raise RuntimeError("Plano estatistico aponta para outra coorte.")

cohort = pd.read_csv(COHORT_FILE)
cohort["asset"] = (
    cohort["asset"].astype(str).str.strip().str.upper()
)

if len(cohort) != EXPECTED_COHORT_SIZE:
    raise RuntimeError(
        f"Coorte deveria ter 32 ativos; observados={len(cohort)}."
    )
if cohort["asset"].duplicated().any():
    raise RuntimeError("Duplicata na coorte prospectiva.")

active_count = int(
    cohort["activation"].astype(bool).sum()
)
dormant_count = int(
    cohort["dormant"].astype(bool).sum()
)
if active_count != EXPECTED_ACTIVE_SIZE:
    raise RuntimeError(
        f"Ativos deveria ser 24; observado={active_count}."
    )
if dormant_count != EXPECTED_DORMANT_SIZE:
    raise RuntimeError(
        f"Dormant deveria ser 8; observado={dormant_count}."
    )

expected_assets = list(
    freeze["selection"]["assets"]
)
if cohort["asset"].tolist() != expected_assets:
    raise RuntimeError(
        "A ordem/conteudo da coorte nao coincide com o freeze."
    )

print("=" * 78, flush=True)
print(
    "TCC - VALIDACAO PROSPECTIVA ONE-SHOT v1.18",
    flush=True,
)
print(
    f"runner={SCRIPT_VERSION} schema={EXECUTION_SCHEMA}",
    flush=True,
)
print(
    f"source_cohort_sha256={EXPECTED_SOURCE_COHORT_SHA256}",
    flush=True,
)
print(
    f"semantic_cohort_sha256={cohort_semantic_sha}",
    flush=True,
)
print(
    f"worktree_cohort_sha256={cohort_worktree_sha}",
    flush=True,
)
print(
    f"cohort={len(cohort)} active={active_count} "
    f"dormant={dormant_count}",
    flush=True,
)
print(
    "[primary] Spearman(S, capital_pct_vs_u59), "
    "one-sided permutation, alpha=0.05",
    flush=True,
)
print("=" * 78, flush=True)


# %% 3 - Snapshots e U59
started = time.perf_counter()

manifest_base = validate_snapshot(BASE)
manifest_b2 = validate_snapshot(B2)
manifest_smart = validate_snapshot(SMART)

expected_smart_hash = (
    (search_meta.get("stage1") or {})
    .get("snapshot_sha256")
)
observed_smart_hash = manifest_smart.get("snapshot_sha256")
if (
    expected_smart_hash
    and str(expected_smart_hash) != str(observed_smart_hash)
):
    raise RuntimeError(
        "Snapshot SMART mudou desde a busca congelada. "
        f"esperado={expected_smart_hash} "
        f"observado={observed_smart_hash}"
    )

frames_u56, exclusions_u56, _, _ = prepare_model_frames(
    BASE,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=False,
    allow_structural_assets=frozenset({"CLMT", "DOC"}),
)
if len(frames_u56) != 56:
    raise RuntimeError(
        f"U56 deveria conter 56 ativos; obtidos={len(frames_u56)}."
    )

frames_b2, exclusions_b2, _, _ = prepare_model_frames(
    B2,
    assets=U59_ADDITIONS,
    comparar_snapshot_referencia=False,
)
if exclusions_b2 or len(frames_b2) != 3:
    raise RuntimeError(
        "COLB, AMS e FOXF precisam estar integralmente disponiveis."
    )

frames_u59 = {
    **frames_u56,
    **frames_b2,
}
symbols_u59 = sorted(frames_u59)
if len(symbols_u59) != 59:
    raise RuntimeError(
        f"U59 deveria ter 59 ativos; obtidos={len(symbols_u59)}."
    )

validation_symbols = cohort["asset"].tolist()

frames_validation, validation_exclusions, _, _ = prepare_model_frames(
    SMART,
    assets=tuple(validation_symbols),
    comparar_snapshot_referencia=False,
)
if validation_exclusions:
    raise RuntimeError(
        "Ativo congelado apresentou exclusao estrutural. "
        "Nao substitua o candidato. Exclusoes="
        + json.dumps(
            validation_exclusions,
            ensure_ascii=False,
            default=str,
        )
    )
missing_validation = sorted(
    set(validation_symbols).difference(frames_validation)
)
if missing_validation:
    raise RuntimeError(
        "Ativo congelado nao ficou modelavel. "
        "Nao substitua. Ausentes="
        + ",".join(missing_validation)
    )


# %% 4 - Contexto compartilhado, calendario/benchmark fixos no U56
frames_all_raw = {
    **frames_u59,
    **frames_validation,
}

config_u56, _ = build_variant_configs(
    frames_u56,
    CONFIG,
)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    frames_u56,
    config_u56,
)

config_all, _ = build_variant_configs(
    frames_all_raw,
    CONFIG,
)
(
    frames_all,
    common_dates,
    calendar_source,
    symbols_all,
    folds,
    all_decision_dates,
    decision_to_fold,
    decision_metadata,
) = _construir_contexto_execucao(
    frames_all_raw,
    config_all,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_source}",
)

candidate_margins = tuple(
    float(value)
    for value in config_all.rotation_switch_margin_candidates
)
full_position = {
    symbol: index + 1
    for index, symbol in enumerate(symbols_all)
}

benchmark_frames_u56 = {
    symbol: frames_all[symbol]
    for symbol in sorted(frames_u56)
}
shared_benchmark = _benchmark_pesos_iguais(
    benchmark_frames_u56,
    sorted(frames_u56),
    all_decision_dates[1:],
    float(config_all.initial_capital),
    config_all,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)
SHARED_BENCHMARK_NAME = (
    "Fixed U56 equal-weight buy-and-hold on the original reference calendar"
)

print(
    f"[context] train_universe={len(symbols_all)} "
    f"u59=59 prospective={len(validation_symbols)} "
    f"folds={len(folds)} calendar={calendar_source}",
    flush=True,
)


# %% 5 - Treino unico por fold
fold_artifacts = {}

for fold_position, fold in enumerate(folds, start=1):
    fold_id = int(fold["fold_id"])
    train_dates = common_dates[
        : int(fold["train_end_index"])
    ]
    calibration_dates = common_dates[
        int(fold["calibration_start_index"]):
        int(fold["calibration_end_index"])
    ]
    final_fit_dates = common_dates[
        : int(fold["final_fit_end_index"])
    ]
    decision_dates = pd.DatetimeIndex(
        fold["decision_dates"]
    )

    print(
        f"[train] fold={fold_id} "
        f"{fold_position}/{len(folds)} "
        f"models={len(symbols_all)} calibration",
        flush=True,
    )
    calibration_models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        train_dates,
        config_all,
        phase=(
            f"prospective_v118_fold_"
            f"{fold_id}_calibration"
        ),
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    calibration_cache, _ = _precalcular_utilidades_modelo(
        calibration_models,
        frames_all,
        symbols_all,
        calibration_dates,
        config_all,
    )

    print(
        f"[train] fold={fold_id} "
        f"models={len(symbols_all)} final",
        flush=True,
    )
    final_models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        final_fit_dates,
        config_all,
        phase=(
            f"prospective_v118_fold_"
            f"{fold_id}_final"
        ),
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )
    decision_cache, _ = _precalcular_utilidades_modelo(
        final_models,
        frames_all,
        symbols_all,
        decision_dates,
        config_all,
    )

    fold_artifacts[fold_id] = {
        "calibration_dates": calibration_dates,
        "decision_dates": decision_dates,
        "calibration_models": calibration_models,
        "calibration_cache": calibration_cache,
        "final_models": final_models,
        "decision_cache": decision_cache,
    }


# %% 6 - Replay por subset, sem novo treino
def _slice_cache(cache, subset_symbols):
    indices = [0] + [
        full_position[symbol]
        for symbol in subset_symbols
    ]
    return {
        timestamp: np.asarray(
            values,
            dtype=np.float64,
        )[indices].copy()
        for timestamp, values in cache.items()
    }


def _run_subset(label, subset_symbols, *, keep_result=False):
    subset_symbols = sorted(set(subset_symbols))
    subset_frames = {
        symbol: frames_all[symbol]
        for symbol in subset_symbols
    }
    subset_config = config_all.copiar_modelo(
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
            policy = _politica_utilidade(
                calibration_models,
                subset_frames,
                subset_symbols,
                subset_config,
                float(margin),
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                policy,
                subset_frames,
                subset_symbols,
                artifact["calibration_dates"],
                subset_config,
            )
            candidate_scores.append(
                (float(margin), float(score))
            )

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
                "fold_id": fold_id,
                "selected_margin": selected_margin,
                "effective_margin": effective_margin,
                "calibration_score": float(
                    selection[
                        "selected_calibration_score"
                    ]
                ),
            }
        )

    scheduled = _politica_agendada(
        policies,
        decision_to_fold,
    )

    result = _simular_exato(
        "prospective_signature_validation_v118",
        scheduled,
        subset_frames,
        subset_symbols,
        all_decision_dates,
        subset_config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"Prospective v1.18 - {label}",
        method_line=(
            "- One-shot prospective validation. "
            "The candidate cohort and statistical endpoints "
            "were frozen before financial outcomes were revealed."
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
        f"[financial] {label} "
        f"assets={len(subset_symbols)} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"cagr={metrics['cagr']:.4%} "
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


# %% 7 - Guard financeiro: reproduzir U59 ANTES de abrir candidatos
baseline = _run_subset(
    "U59_WINNER",
    symbols_u59,
    keep_result=True,
)
baseline_capital = float(
    baseline["metrics"]["ending_capital"]
)
baseline_rel_error = (
    baseline_capital / EXPECTED_BASELINE - 1.0
)

print(
    "[baseline-guard] "
    f"expected={EXPECTED_BASELINE:,.8f} "
    f"observed={baseline_capital:,.8f} "
    f"relative_error={baseline_rel_error:+.12e}",
    flush=True,
)

if not math.isclose(
    baseline_capital,
    EXPECTED_BASELINE,
    rel_tol=BASELINE_REL_TOL,
    abs_tol=0.01,
):
    raise RuntimeError(
        "U59 nao reproduziu o checkpoint congelado. "
        "ABORTANDO antes de revelar os 32 candidatos."
    )


# %% 8 - ABERTURA DO GABARITO: 32 replays individuais
outcome_rows = []

for position, frozen_row in cohort.iterrows():
    symbol = str(frozen_row["asset"])
    print(
        f"[reveal] {position + 1}/{len(cohort)} "
        f"U59_PLUS_{symbol}",
        flush=True,
    )

    scenario = _run_subset(
        f"U59_PLUS_{symbol}",
        [*symbols_u59, symbol],
        keep_result=False,
    )
    ending = float(
        scenario["metrics"]["ending_capital"]
    )
    delta = ending - baseline_capital
    pct = ending / baseline_capital - 1.0

    outcome_rows.append(
        {
            **frozen_row.to_dict(),
            "ending_capital": ending,
            "capital_delta_vs_u59": delta,
            "capital_pct_vs_u59": pct,
            "positive": bool(pct > 1e-12),
            "exact_zero": bool(
                math.isclose(
                    pct,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            "cagr": scenario["metrics"]["cagr"],
            "sharpe": scenario["metrics"]["sharpe"],
            "maximum_drawdown": (
                scenario["metrics"]["maximum_drawdown"]
            ),
            "worst_fold_return": (
                scenario["metrics"]["worst_fold_return"]
            ),
            "margins": json.dumps(
                scenario["margins"],
                ensure_ascii=False,
                default=str,
            ),
        }
    )

outcomes = pd.DataFrame(outcome_rows)


# %% 9 - Endpoint primario e secundarios pre-registrados
rng = np.random.default_rng(RANDOM_SEED)

active = outcomes[
    outcomes["activation"].astype(bool)
].copy()
dormant = outcomes[
    outcomes["dormant"].astype(bool)
].copy()

rho, rho_perm_p = _permutation_spearman_one_sided(
    active["specialist_score"],
    active["capital_pct_vs_u59"],
    PERMUTATIONS,
    rng,
)

auc, auc_perm_p = _permutation_auc_one_sided(
    active["specialist_score"],
    active["positive"].astype(int),
    PERMUTATIONS,
    rng,
)
ap = _safe_ap(
    active["positive"].astype(int),
    active["specialist_score"],
)

boot_rho, boot_auc = _bootstrap_active(
    active,
    BOOTSTRAP_REPS,
    rng,
)

PRIMARY_SUPPORTED = bool(
    rho > 0.0
    and rho_perm_p < 0.05
)

stratum_rows = []
for stratum in (
    "active_high_S",
    "active_mid_S",
    "active_low_S",
    "dormant_A0",
):
    part = outcomes[
        outcomes["validation_stratum"] == stratum
    ]
    stratum_rows.append(
        {
            "validation_stratum": stratum,
            "n": int(len(part)),
            "positive_count": int(
                part["positive"].sum()
            ),
            "positive_rate": float(
                part["positive"].mean()
            ),
            "exact_zero_count": int(
                part["exact_zero"].sum()
            ),
            "exact_zero_rate": float(
                part["exact_zero"].mean()
            ),
            "median_capital_pct_vs_u59": float(
                part["capital_pct_vs_u59"].median()
            ),
            "mean_capital_pct_vs_u59": float(
                part["capital_pct_vs_u59"].mean()
            ),
            "min_capital_pct_vs_u59": float(
                part["capital_pct_vs_u59"].min()
            ),
            "max_capital_pct_vs_u59": float(
                part["capital_pct_vs_u59"].max()
            ),
        }
    )

stratum_summary = pd.DataFrame(stratum_rows)

high = outcomes[
    outcomes["validation_stratum"] == "active_high_S"
]
low = outcomes[
    outcomes["validation_stratum"] == "active_low_S"
]

high_pos = int(high["positive"].sum())
low_pos = int(low["positive"].sum())

fisher_table = [
    [high_pos, len(high) - high_pos],
    [low_pos, len(low) - low_pos],
]
fisher = fisher_exact(
    fisher_table,
    alternative="greater",
)


# %% 10 - Teste economico secundario do grupo high-S
high_symbols = high["asset"].astype(str).tolist()

high_group = _run_subset(
    "U59_PLUS_PROSPECTIVE_HIGH_S_8",
    [*symbols_u59, *high_symbols],
    keep_result=True,
)
high_group_capital = float(
    high_group["metrics"]["ending_capital"]
)
high_group_pct = (
    high_group_capital / baseline_capital - 1.0
)


# %% 11 - Exportacao tabular
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

outcomes.to_csv(
    OUT / "prospective_individual_outcomes.csv",
    index=False,
)
stratum_summary.to_csv(
    OUT / "prospective_stratum_summary.csv",
    index=False,
)

pd.DataFrame(
    [
        {
            "endpoint": (
                "primary_spearman_active"
            ),
            "n": int(len(active)),
            "statistic": rho,
            "p_one_sided_permutation": rho_perm_p,
            "alpha": 0.05,
            "supported": PRIMARY_SUPPORTED,
        },
        {
            "endpoint": "secondary_auc_active",
            "n": int(len(active)),
            "statistic": auc,
            "p_one_sided_permutation": auc_perm_p,
            "alpha": 0.05,
            "supported": (
                bool(
                    auc is not None
                    and auc > 0.5
                    and auc_perm_p is not None
                    and auc_perm_p < 0.05
                )
            ),
        },
    ]
).to_csv(
    OUT / "prospective_statistical_tests.csv",
    index=False,
)

cohort.to_csv(
    OUT / "frozen_cohort_copy.csv",
    index=False,
)

baseline["result"].predictions.reset_index().to_csv(
    OUT / "u59_predictions.csv",
    index=False,
)
baseline["result"].trades.to_csv(
    OUT / "u59_trades.csv",
    index=False,
)
high_group["result"].predictions.reset_index().to_csv(
    OUT / "u59_plus_high_s8_predictions.csv",
    index=False,
)
high_group["result"].trades.to_csv(
    OUT / "u59_plus_high_s8_trades.csv",
    index=False,
)


# %% 12 - Graficos da validacao
fig, ax = plt.subplots(figsize=(11.0, 6.2))
ax.scatter(
    active["specialist_score"],
    100.0 * active["capital_pct_vs_u59"],
    s=55,
)
for _, row in active.iterrows():
    ax.annotate(
        str(row["asset"]),
        (
            row["specialist_score"],
            100.0 * row["capital_pct_vs_u59"],
        ),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7,
    )
ax.axhline(0.0, linewidth=1.0)
ax.set_xlabel("Score especialista congelado S")
ax.set_ylabel("Efeito individual vs U59 (%)")
ax.set_title(
    "Validacao prospectiva: score S vs contribuicao financeira"
)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIG / "prospective_S_vs_capital_effect.png",
    dpi=180,
)
fig.savefig(
    FIG / "prospective_S_vs_capital_effect.svg",
)
plt.close(fig)

plot_summary = stratum_summary.set_index(
    "validation_stratum"
)
fig, ax = plt.subplots(figsize=(10.0, 5.8))
ax.bar(
    plot_summary.index,
    100.0 * plot_summary["positive_rate"],
)
ax.set_ylim(0.0, 100.0)
ax.set_ylabel("Ativos com delta de capital positivo (%)")
ax.set_xlabel("Estrato congelado")
ax.set_title(
    "Taxa prospectiva de contribuicao positiva por estrato"
)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIG / "prospective_positive_rate_by_stratum.png",
    dpi=180,
)
fig.savefig(
    FIG / "prospective_positive_rate_by_stratum.svg",
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.0, 5.8))
ax.bar(
    plot_summary.index,
    100.0 * plot_summary["median_capital_pct_vs_u59"],
)
ax.axhline(0.0, linewidth=1.0)
ax.set_ylabel("Mediana do efeito vs U59 (%)")
ax.set_xlabel("Estrato congelado")
ax.set_title(
    "Efeito financeiro prospectivo por estrato"
)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIG / "prospective_median_effect_by_stratum.png",
    dpi=180,
)
fig.savefig(
    FIG / "prospective_median_effect_by_stratum.svg",
)
plt.close(fig)


# %% 13 - Relatorio machine-readable e decisao
payload = {
    "research_version": SCRIPT_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "status": "completed_one_shot_prospective_validation",
    "frozen_before_outcome": True,
    "cohort_sha256": EXPECTED_SOURCE_COHORT_SHA256,
    "cohort_worktree_sha256": cohort_worktree_sha,
    "cohort_semantic_sha256": cohort_semantic_sha,
    "financial_reference": "U59_WINNER",
    "baseline": {
        "expected_ending_capital": EXPECTED_BASELINE,
        "observed_metrics": baseline["metrics"],
        "relative_error": baseline_rel_error,
        "reproduced": True,
    },
    "primary_endpoint": {
        "population": "24 active candidates",
        "predictor": "specialist_score",
        "outcome": "capital_pct_vs_u59",
        "statistic": "spearman_rho",
        "rho": rho,
        "permutation_reps": PERMUTATIONS,
        "p_one_sided": rho_perm_p,
        "alpha": 0.05,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_95": _percentiles(
            boot_rho
        ),
        "supported": PRIMARY_SUPPORTED,
    },
    "secondary": {
        "auc": auc,
        "auc_permutation_p_one_sided": auc_perm_p,
        "average_precision": ap,
        "auc_bootstrap_95": _percentiles(
            boot_auc
        ),
        "high_vs_low_positive_rate_fisher": {
            "table": fisher_table,
            "odds_ratio": _finite(
                fisher.statistic
            ),
            "p_one_sided": _finite(
                fisher.pvalue
            ),
        },
        "strata": stratum_rows,
        "dormant_control": {
            "n": int(len(dormant)),
            "positive_count": int(
                dormant["positive"].sum()
            ),
            "positive_rate": float(
                dormant["positive"].mean()
            ),
            "exact_zero_count": int(
                dormant["exact_zero"].sum()
            ),
            "exact_zero_rate": float(
                dormant["exact_zero"].mean()
            ),
            "median_capital_pct_vs_u59": float(
                dormant[
                    "capital_pct_vs_u59"
                ].median()
            ),
        },
        "high_S_group": {
            "assets": high_symbols,
            "metrics": high_group["metrics"],
            "margins": high_group["margins"],
            "capital_delta_vs_u59": (
                high_group_capital
                - baseline_capital
            ),
            "capital_pct_vs_u59": (
                high_group_pct
            ),
            "interpretation": (
                "Secondary economic interaction test. "
                "Not part of the primary signature inference."
            ),
        },
    },
    "decision": {
        "prospective_ranking_generalization_supported": (
            PRIMARY_SUPPORTED
        ),
        "rule": (
            "Support only if primary Spearman rho > 0 "
            "and one-sided permutation p < 0.05."
        ),
        "post_outcome_retuning_allowed": False,
        "second_selection_attempt_allowed": False,
    },
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "prospective_financial_validation.json"
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

package = criar_pacote_analise(
    OUT,
    comparison_file="prospective_financial_validation.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name=(
        "pacote_validacao_prospectiva_financeira_v118.zip"
    ),
)

print("=" * 78, flush=True)
print("[PROSPECTIVE RESULT]", flush=True)
print(
    f"primary_rho={rho:.6f} "
    f"p_one_sided={rho_perm_p:.6g} "
    f"supported={PRIMARY_SUPPORTED}",
    flush=True,
)
print(
    f"secondary_auc={auc} "
    f"auc_p={auc_perm_p} "
    f"average_precision={ap}",
    flush=True,
)
print("[strata]", flush=True)
print(
    stratum_summary.to_string(index=False),
    flush=True,
)
print(
    "[high-vs-low] "
    f"fisher_odds={_finite(fisher.statistic)} "
    f"p_one_sided={_finite(fisher.pvalue)}",
    flush=True,
)
print(
    "[high-S-group] "
    f"capital={high_group_capital:,.2f} "
    f"pct_vs_u59={high_group_pct:+.4%}",
    flush=True,
)
print(f"[package] pronto={package}", flush=True)
print(
    "[final-rule] Nao reajustar a assinatura v1.18 "
    "com base neste resultado.",
    flush=True,
)
sinal_sonoro_conclusao()
