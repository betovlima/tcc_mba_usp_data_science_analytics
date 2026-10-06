"""TESTE FINANCEIRO ONE-SHOT DO FILTRO DE ADERENCIA v1.19.

Nova hipotese posterior a v1.18.

A lista TRST, DBB, MCD, CSB, WMK, WFC, RJF e FIBK foi congelada antes de
qualquer resultado financeiro v1.19.

Endpoint primario pre-registrado:
    ending_capital(U59 + adherence8) > ending_capital(U59)

Este runner:
- NAO procura novos ativos;
- NAO baixa dados;
- NAO troca candidatos;
- NAO altera R;
- reproduz U59 antes de abrir o gabarito;
- revela os oito efeitos individuais;
- executa uma unica vez o grupo congelado adherence8;
- preserva resultados negativos sem segunda tentativa.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

DATA = ROOT / "dados" / "assinatura_matematica"
COHORT_FILE = DATA / "capital_adherence_cohort_v119.csv"
FREEZE_FILE = DATA / "capital_adherence_freeze_v119.json"
MODEL_FILE = DATA / "capital_adherence_filter_v1_19_frozen.json"
PLAN_FILE = DATA / "capital_adherence_validation_plan_v119.json"
SEARCH_JSON = ROOT / "output" / "busca_ativos" / "asset_search.json"

OUT = ROOT / "output" / "validacao_aderencia_capital_v119"
FIG = OUT / "graficos"

SCRIPT_VERSION = "1.19.1-dev.1"
EXECUTION_SCHEMA = "capital-adherence-financial-validation-v1"
EXPECTED_SHARED_MODULE_VERSION = "1.17.0-dev.1"
EXPECTED_BASELINE = 30_080_091.008142874
STRETCH_BENCHMARK = 58_557_157.67496595
BASELINE_REL_TOL = 1e-9

EXPECTED_ASSETS = [
    "TRST", "DBB", "MCD", "CSB",
    "WMK", "WFC", "RJF", "FIBK",
]
EXPECTED_RISK = {
    "TRST": 0.028465346534653466,
    "DBB": 0.03094059405940594,
    "MCD": 0.04455445544554455,
    "CSB": 0.06559405940594059,
    "WMK": 0.06683168316831684,
    "WFC": 0.06806930693069307,
    "RJF": 0.07054455445544554,
    "FIBK": 0.07425742574257425,
}
EXPECTED_SELECTION_RAW_SHA = (
    "dcb6b8746719c24a59b6b9a2c056366446372d9c38811a0885b1abab877ef3b2"
)
EXPECTED_FREEZE_PACKAGE_SHA = (
    "8919f98a4613a67c55b595441234bc9b74cf0fb1c8ae49498ea7454f230ab38c"
)
U59_ADDITIONS = ("COLB", "AMS", "FOXF")


# %% 1 - Guards de protocolo
if RESEARCH_VERSION != EXPECTED_SHARED_MODULE_VERSION:
    raise RuntimeError(
        "Modulo compartilhado inesperado. "
        f"esperado={EXPECTED_SHARED_MODULE_VERSION!r} "
        f"observado={RESEARCH_VERSION!r}"
    )

for path in (
    COHORT_FILE,
    FREEZE_FILE,
    MODEL_FILE,
    PLAN_FILE,
    SEARCH_JSON,
):
    if not path.exists():
        raise RuntimeError(
            "Artefato congelado ausente: " + str(path)
        )

freeze = json.loads(
    FREEZE_FILE.read_text(encoding="utf-8")
)
model = json.loads(
    MODEL_FILE.read_text(encoding="utf-8")
)
plan = json.loads(
    PLAN_FILE.read_text(encoding="utf-8")
)
search_meta = json.loads(
    SEARCH_JSON.read_text(encoding="utf-8")
)

if freeze.get("status") != "frozen_before_new_candidate_capital":
    raise RuntimeError("Freeze v1.19 invalido.")
if freeze.get("financial_reference") != "U59_WINNER":
    raise RuntimeError("Freeze v1.19 nao usa U59.")
if (
    model.get("status")
    != "new_hypothesis_frozen_before_new_candidate_capital"
):
    raise RuntimeError("Modelo v1.19 nao esta congelado.")
if (
    plan.get("status")
    != "preregistered_before_v119_outcome_reveal"
):
    raise RuntimeError("Plano financeiro v1.19 nao esta pre-registrado.")
if plan.get("source_freeze_package_sha256") != EXPECTED_FREEZE_PACKAGE_SHA:
    raise RuntimeError("Plano aponta para outro pacote de congelamento.")
if plan.get("frozen_selection_raw_sha256") != EXPECTED_SELECTION_RAW_SHA:
    raise RuntimeError("Plano aponta para outra selecao.")

cohort = pd.read_csv(COHORT_FILE)
cohort["asset"] = (
    cohort["asset"].astype(str).str.strip().str.upper()
)

if cohort["asset"].tolist() != EXPECTED_ASSETS:
    raise RuntimeError(
        "Lista/order dos oito candidatos mudou depois do congelamento."
    )
if len(cohort) != 8 or cohort["asset"].duplicated().any():
    raise RuntimeError("Coorte v1.19 deve conter oito ativos unicos.")

observed_flags = (
    cohort["capital_observed_at_freeze"]
    .astype(str)
    .str.lower()
    .isin({"true", "1", "yes"})
)
if bool(observed_flags.any()):
    raise RuntimeError(
        "Freeze invalido: capital_observed_at_freeze deveria ser False."
    )

for _, row in cohort.iterrows():
    symbol = str(row["asset"])
    observed = float(row["capital_adherence_risk"])
    expected = float(EXPECTED_RISK[symbol])
    if not math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Risco congelado mudou para {symbol}: "
            f"esperado={expected} observado={observed}"
        )

if list(freeze["selection"]["assets"]) != EXPECTED_ASSETS:
    raise RuntimeError("Freeze JSON aponta para outra lista.")
if list(plan["selected_assets"]) != EXPECTED_ASSETS:
    raise RuntimeError("Plano pre-registrado aponta para outra lista.")

print("=" * 78, flush=True)
print("TCC - TESTE FINANCEIRO ADERENCIA v1.19", flush=True)
print(
    f"runner={SCRIPT_VERSION} schema={EXECUTION_SCHEMA}",
    flush=True,
)
print(
    "[frozen] " + ",".join(EXPECTED_ASSETS),
    flush=True,
)
print(
    "[primary] ending_capital(U59+adherence8) > ending_capital(U59)",
    flush=True,
)
print("=" * 78, flush=True)


# %% 2 - Snapshots e U59
started = time.perf_counter()

validate_snapshot(BASE)
validate_snapshot(B2)
manifest_smart = validate_snapshot(SMART)

expected_smart_hash = (
    (search_meta.get("stage1") or {}).get("snapshot_sha256")
)
observed_smart_hash = manifest_smart.get("snapshot_sha256")
if (
    expected_smart_hash
    and str(expected_smart_hash) != str(observed_smart_hash)
):
    raise RuntimeError(
        "Snapshot SMART mudou desde a busca congelada. "
        f"esperado={expected_smart_hash} observado={observed_smart_hash}"
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

frames_selected, selected_exclusions, _, _ = prepare_model_frames(
    SMART,
    assets=tuple(EXPECTED_ASSETS),
    comparar_snapshot_referencia=False,
)
if selected_exclusions:
    raise RuntimeError(
        "Candidato congelado apresentou exclusao estrutural. "
        "Nao substitua. Exclusoes="
        + json.dumps(
            selected_exclusions,
            ensure_ascii=False,
            default=str,
        )
    )
missing_selected = sorted(
    set(EXPECTED_ASSETS).difference(frames_selected)
)
if missing_selected:
    raise RuntimeError(
        "Candidato congelado nao ficou modelavel. "
        "Nao substitua. Ausentes="
        + ",".join(missing_selected)
    )


# %% 3 - Contexto compartilhado
frames_all_raw = {
    **frames_u59,
    **frames_selected,
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
    f"u59=59 adherence8=8 folds={len(folds)} "
    f"calendar={calendar_source}",
    flush=True,
)


# %% 4 - Treino unico por fold
fold_artifacts = {}

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
        f"models={len(symbols_all)} calibration",
        flush=True,
    )
    calibration_models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        train_dates,
        config_all,
        phase=f"adherence_v119_fold_{fold_id}_calibration",
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
        f"[train] fold={fold_id} models={len(symbols_all)} final",
        flush=True,
    )
    final_models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        final_fit_dates,
        config_all,
        phase=f"adherence_v119_fold_{fold_id}_final",
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


# %% 5 - Replay reutilizando os mesmos modelos
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
                    selection["selected_calibration_score"]
                ),
            }
        )

    scheduled = _politica_agendada(
        policies,
        decision_to_fold,
    )

    result = _simular_exato(
        "capital_adherence_validation_v119",
        scheduled,
        subset_frames,
        subset_symbols,
        all_decision_dates,
        subset_config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"Capital adherence v1.19 - {label}",
        method_line=(
            "- New post-v1.18 hypothesis. The eight candidates and the "
            "economic endpoint were frozen before v1.19 capital outcomes."
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
        f"[financial] {label} assets={len(subset_symbols)} "
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


# %% 6 - Guard U59 antes do gabarito
baseline = _run_subset(
    "U59_WINNER",
    symbols_u59,
    keep_result=True,
)
baseline_capital = float(baseline["metrics"]["ending_capital"])
baseline_rel_error = baseline_capital / EXPECTED_BASELINE - 1.0

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
        "ABORTANDO antes de revelar adherence8."
    )


# %% 7 - Oito efeitos individuais
rows = []

for position, frozen_row in cohort.iterrows():
    symbol = str(frozen_row["asset"])
    print(
        f"[reveal] {position + 1}/8 U59_PLUS_{symbol}",
        flush=True,
    )

    scenario = _run_subset(
        f"U59_PLUS_{symbol}",
        [*symbols_u59, symbol],
        keep_result=False,
    )
    ending = float(scenario["metrics"]["ending_capital"])
    pct = ending / baseline_capital - 1.0

    rows.append(
        {
            **frozen_row.to_dict(),
            "ending_capital": ending,
            "capital_delta_vs_u59": ending - baseline_capital,
            "capital_pct_vs_u59": pct,
            "positive": bool(pct > 1e-12),
            "harm10": bool(pct <= -0.10),
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

outcomes = pd.DataFrame(rows)


# %% 8 - Endpoint economico primario: grupo adherence8
group = _run_subset(
    "U59_PLUS_ADHERENCE8",
    [*symbols_u59, *EXPECTED_ASSETS],
    keep_result=True,
)
group_capital = float(group["metrics"]["ending_capital"])
group_delta = group_capital - baseline_capital
group_pct = group_capital / baseline_capital - 1.0

PRIMARY_SUPPORTED = bool(group_capital > baseline_capital)
STRETCH_REACHED = bool(group_capital > STRETCH_BENCHMARK)

harm10_count = int(outcomes["harm10"].sum())
positive_count = int(outcomes["positive"].sum())
zero_count = int(outcomes["exact_zero"].sum())
median_effect = float(outcomes["capital_pct_vs_u59"].median())
mean_effect = float(outcomes["capital_pct_vs_u59"].mean())


# %% 9 - Exportacao
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

outcomes.to_csv(
    OUT / "capital_adherence_individual_outcomes_v119.csv",
    index=False,
)
cohort.to_csv(
    OUT / "frozen_adherence8_copy.csv",
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
group["result"].predictions.reset_index().to_csv(
    OUT / "u59_plus_adherence8_predictions.csv",
    index=False,
)
group["result"].trades.to_csv(
    OUT / "u59_plus_adherence8_trades.csv",
    index=False,
)

summary_table = pd.DataFrame(
    [
        {
            "scenario": "U59_WINNER",
            "ending_capital": baseline_capital,
            "delta_vs_u59": 0.0,
            "pct_vs_u59": 0.0,
        },
        {
            "scenario": "U59_PLUS_ADHERENCE8",
            "ending_capital": group_capital,
            "delta_vs_u59": group_delta,
            "pct_vs_u59": group_pct,
        },
        {
            "scenario": "EXPLORATORY_U59_PLUS_POSITIVE8",
            "ending_capital": STRETCH_BENCHMARK,
            "delta_vs_u59": STRETCH_BENCHMARK - baseline_capital,
            "pct_vs_u59": STRETCH_BENCHMARK / baseline_capital - 1.0,
        },
    ]
)
summary_table.to_csv(
    OUT / "capital_adherence_scenario_summary_v119.csv",
    index=False,
)


# %% 10 - Graficos
plot_individual = outcomes.sort_values(
    "capital_pct_vs_u59",
    ascending=False,
)

fig, ax = plt.subplots(figsize=(10.5, 5.8))
ax.bar(
    plot_individual["asset"],
    100.0 * plot_individual["capital_pct_vs_u59"],
)
ax.axhline(0.0, linewidth=1.0)
ax.axhline(-10.0, linewidth=1.0, linestyle="--")
ax.set_xlabel("Ativo congelado")
ax.set_ylabel("Efeito individual vs U59 (%)")
ax.set_title(
    "Filtro de aderencia v1.19: efeitos individuais"
)
fig.tight_layout()
fig.savefig(
    FIG / "individual_effects_v119.png",
    dpi=180,
)
fig.savefig(
    FIG / "individual_effects_v119.svg",
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.bar(
    ["U59", "U59 + adherence8", "U59 + positive8*"],
    [
        baseline_capital / 1_000_000.0,
        group_capital / 1_000_000.0,
        STRETCH_BENCHMARK / 1_000_000.0,
    ],
)
ax.set_ylabel("Capital final (US$ milhoes)")
ax.set_title(
    "Teste economico do filtro de aderencia v1.19"
)
fig.tight_layout()
fig.savefig(
    FIG / "scenario_capital_v119.png",
    dpi=180,
)
fig.savefig(
    FIG / "scenario_capital_v119.svg",
)
plt.close(fig)


# %% 11 - Relatorio machine-readable
payload = {
    "research_version": SCRIPT_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "status": "completed_one_shot_capital_adherence_test",
    "new_hypothesis_after_v118": True,
    "frozen_before_v119_outcome": True,
    "financial_reference": "U59_WINNER",
    "selection": {
        "assets": EXPECTED_ASSETS,
        "source_freeze_package_sha256": EXPECTED_FREEZE_PACKAGE_SHA,
        "source_selection_raw_sha256": EXPECTED_SELECTION_RAW_SHA,
    },
    "baseline": {
        "expected_ending_capital": EXPECTED_BASELINE,
        "observed_metrics": baseline["metrics"],
        "relative_error": baseline_rel_error,
        "reproduced": True,
    },
    "primary_endpoint": {
        "name": "joint_capital_improvement",
        "rule": "U59+adherence8 ending capital > U59 ending capital",
        "group_metrics": group["metrics"],
        "capital_delta_vs_u59": group_delta,
        "capital_pct_vs_u59": group_pct,
        "supported": PRIMARY_SUPPORTED,
    },
    "secondary_diagnostics": {
        "individual_count": int(len(outcomes)),
        "positive_count": positive_count,
        "harm10_count": harm10_count,
        "exact_zero_count": zero_count,
        "median_capital_pct_vs_u59": median_effect,
        "mean_capital_pct_vs_u59": mean_effect,
        "stretch_benchmark": STRETCH_BENCHMARK,
        "stretch_reached": STRETCH_REACHED,
    },
    "decision": {
        "capital_adherence_hypothesis_supported": PRIMARY_SUPPORTED,
        "post_outcome_retuning_allowed": False,
        "second_selection_attempt_allowed": False,
        "interpretation_rule": (
            "v1.19 is a new hypothesis. Its result does not modify the "
            "completed prospective validation conclusions of v1.18."
        ),
    },
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "capital_adherence_financial_validation_v119.json"
).open("w", encoding="utf-8") as handle:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

package = criar_pacote_analise(
    OUT,
    comparison_file="capital_adherence_financial_validation_v119.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_validacao_aderencia_capital_v119.zip",
)

print("=" * 78, flush=True)
print("[CAPITAL ADHERENCE RESULT]", flush=True)
print(
    f"U59={baseline_capital:,.2f} "
    f"U59+adherence8={group_capital:,.2f} "
    f"delta={group_delta:+,.2f} "
    f"pct={group_pct:+.4%}",
    flush=True,
)
print(
    f"primary_supported={PRIMARY_SUPPORTED} "
    f"stretch_reached={STRETCH_REACHED}",
    flush=True,
)
print(
    f"individual_positive={positive_count}/8 "
    f"harm10={harm10_count}/8 "
    f"zero={zero_count}/8 "
    f"median={median_effect:+.4%} "
    f"mean={mean_effect:+.4%}",
    flush=True,
)
print("[individual-outcomes]", flush=True)
print(
    outcomes[
        [
            "asset",
            "capital_adherence_risk",
            "capital_pct_vs_u59",
            "positive",
            "harm10",
        ]
    ].to_string(index=False),
    flush=True,
)
print(f"[package] pronto={package}", flush=True)
print(
    "[final-rule] NAO selecionar outro adherence8 depois de ver este resultado.",
    flush=True,
)
sinal_sonoro_conclusao()
