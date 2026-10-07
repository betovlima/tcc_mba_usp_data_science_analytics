"""AUDITORIA CAUSAL U67 x MCT: atribuicao CLMT/DOC.

Objetivo
--------
Responder UMA pergunta:

    Por que o TCC U67 congelado termina em ~US$ 58,56 milhoes enquanto a
    Strategy #13 do MCT chegou a ~US$ 76,93 milhoes?

O export real do MCT mostrou 65 ativos elegiveis. Em relacao ao U67 oficial,
somente CLMT e DOC ficaram fora. Esta pesquisa isola exatamente essas duas
diferencas usando os MESMOS snapshots congelados do TCC.

Variantes executadas
--------------------
1. U67 completo, 67 ativos: guarda do checkpoint oficial.
2. U67 sem DOC.
3. U67 sem CLMT.
4. U67 sem CLMT e DOC: mesmo universo de 65 ativos observado no MCT.

Nada mais muda: mesmos dados, calendario U56, LightGBM, folds, purge, custos,
slippage e calibracao de switch margin.

Esta pesquisa NAO seleciona ativos, NAO usa CUSIP como criterio, NAO procura
um universo melhor e NAO altera a main.

Execucao no Spyder
------------------
Abra buscar_ativos.py, reinicie o kernel e execute com F5.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

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
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_variant_configs, summarize_metrics
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuracao congelada
ROOT = Path(__file__).resolve().parent
BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(ROOT / "dados" / "pesquisa_expansao_76_b2")
SMART = SnapshotPaths.from_root(ROOT / "dados" / "pesquisa_smart_candidates")
OUT = ROOT / "output" / "auditoria_u67_mct"

RESEARCH_VERSION = "1.22.1-dev.1"
EXECUTION_SCHEMA = "u67-mct-clmt-doc-attribution-v1"
SOURCE_MAIN_COMMIT = "4b5f16030afa8b850790747bb0e3e3063d233e79"

U59_ADDITIONS = ("COLB", "AMS", "FOXF")
U67_ADDITIONS = (
    "THO", "WDAY", "EXR", "XEL",
    "SBFG", "PAYX", "MUX", "SXC",
)
EXPECTED_U67_COUNT = 67
EXPECTED_U67_CAPITAL = 58_557_157.67496595

started = time.perf_counter()

print("=" * 78, flush=True)
print("TCC - AUDITORIA CAUSAL U67 x MCT", flush=True)
print(f"version={RESEARCH_VERSION} schema={EXECUTION_SCHEMA}", flush=True)
print("question=why_mct_65_assets_differs_from_tcc_u67", flush=True)
print("selection_experiment=False", flush=True)
print("cusip_selection=False", flush=True)
print("changed_factor=CLMT_DOC_membership_only", flush=True)
print("=" * 78, flush=True)


# %% 1 - Carregar exatamente o U67 oficial
manifest_u56 = validate_snapshot(BASE)
manifest_b2 = validate_snapshot(B2)
manifest_smart = validate_snapshot(SMART)

frames_u56, exclusions_u56, diagnostics_u56, audit_u56 = prepare_model_frames(
    BASE,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=False,
    # O U67 oficial inclui CLMT e DOC. Nesta auditoria eles precisam entrar
    # primeiro, e somente depois sao removidos nas variantes controladas.
    allow_structural_assets=frozenset({"CLMT", "DOC"}),
)
if len(frames_u56) != 56:
    raise RuntimeError(
        "U56 oficial nao foi carregado integralmente. "
        f"observado={len(frames_u56)} exclusoes={exclusions_u56}"
    )

frames_b2, exclusions_b2, diagnostics_b2, audit_b2 = prepare_model_frames(
    B2,
    assets=U59_ADDITIONS,
    comparar_snapshot_referencia=False,
)
if exclusions_b2 or len(frames_b2) != len(U59_ADDITIONS):
    raise RuntimeError(f"U59 additions invalidas: {exclusions_b2}")

frames_smart, exclusions_smart, diagnostics_smart, audit_smart = (
    prepare_model_frames(
        SMART,
        assets=U67_ADDITIONS,
        comparar_snapshot_referencia=False,
    )
)
if exclusions_smart or len(frames_smart) != len(U67_ADDITIONS):
    raise RuntimeError(f"U67 additions invalidas: {exclusions_smart}")

frames_u67_raw = {**frames_u56, **frames_b2, **frames_smart}
if len(frames_u67_raw) != EXPECTED_U67_COUNT:
    raise RuntimeError(
        f"U67 deveria conter 67 ativos; observado={len(frames_u67_raw)}"
    )
if "CLMT" not in frames_u67_raw or "DOC" not in frames_u67_raw:
    raise RuntimeError("CLMT e DOC precisam existir no U67 de origem.")


# %% 2 - Calendario e benchmark fixos, identicos ao checkpoint oficial
config_u56, _ = build_variant_configs(frames_u56, CONFIG)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    frames_u56,
    config_u56,
)

config_full, _ = build_variant_configs(frames_u67_raw, CONFIG)
(
    frames_full_for_benchmark,
    _,
    _,
    _,
    _,
    all_decision_dates_full,
    _,
    _,
) = _construir_contexto_execucao(
    frames_u67_raw,
    config_full,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_source}",
)

benchmark_frames_u56 = {
    symbol: frames_full_for_benchmark[symbol]
    for symbol in sorted(frames_u56)
}
shared_benchmark = _benchmark_pesos_iguais(
    benchmark_frames_u56,
    sorted(frames_u56),
    all_decision_dates_full[1:],
    float(config_full.initial_capital),
    config_full,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)
BENCHMARK_NAME = (
    "Fixed U56 equal-weight buy-and-hold on the original reference calendar"
)


# %% 3 - Executor identico para cada universo
def run_universe(
    label: str,
    raw_frames: dict[str, pd.DataFrame],
):
    config, _ = build_variant_configs(raw_frames, CONFIG)
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
        raw_frames,
        config,
        calendar_override=reference_calendar,
        calendar_source_label=f"U56_FIXED:{reference_source}",
    )

    candidate_margins = tuple(
        float(value)
        for value in config.rotation_switch_margin_candidates
    )
    fold_policies = {}
    fold_margins = []

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
            f"[{label}] fold={fold_id} {fold_position}/{len(folds)} "
            f"assets={len(symbols)} calibration",
            flush=True,
        )

        calibration_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config,
            phase=f"{label}_fold_{fold_id}_calibration",
            technical_log_callback=lambda message: print(
                f"[technical] {label} {message}",
                flush=True,
            ),
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_models,
            frames,
            symbols,
            calibration_dates,
            config,
        )

        candidate_scores = []
        for margin in candidate_margins:
            policy = _politica_utilidade(
                calibration_models,
                frames,
                symbols,
                config,
                float(margin),
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                policy,
                frames,
                symbols,
                calibration_dates,
                config,
            )
            candidate_scores.append((float(margin), float(score)))

        selection = _selecionar_switch_margin_fold(
            config,
            fold_id,
            candidate_scores,
        )
        selected_margin = float(selection["selected_candidate_margin"])
        effective_margin = max(
            float(config.rotation_switch_margin),
            selected_margin,
        )

        final_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"{label}_fold_{fold_id}_final",
            technical_log_callback=lambda message: print(
                f"[technical] {label} {message}",
                flush=True,
            ),
        )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_models,
            frames,
            symbols,
            decision_dates,
            config,
        )
        fold_policies[fold_id] = _politica_utilidade(
            final_models,
            frames,
            symbols,
            config,
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

    scheduled_policy = _politica_agendada(
        fold_policies,
        decision_to_fold,
    )
    result = _simular_exato(
        label,
        scheduled_policy,
        frames,
        symbols,
        all_decision_dates,
        config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"U67 causal attribution - {label}",
        method_line=(
            "- Same frozen U67 engine/data; only CLMT/DOC membership changes."
        ),
        benchmark_override=shared_benchmark,
        benchmark_override_name=BENCHMARK_NAME,
    )
    metrics = summarize_metrics(
        result,
        folds,
        float(config.initial_capital),
    )
    meta = {
        "label": label,
        "asset_count": len(symbols),
        "assets": list(symbols),
        "calendar_source": calendar_source,
        "decision_sessions": len(all_decision_dates),
        "fold_margins": fold_margins,
    }
    print(
        f"[{label}] capital=US$ {float(metrics['ending_capital']):,.2f} "
        f"sharpe={float(metrics['sharpe']):.6f} "
        f"maxdd={float(metrics['maximum_drawdown']):.4%}",
        flush=True,
    )
    return result, metrics, meta


# %% 4 - Quatro variantes, uma unica pergunta
variants = {
    "u67_full": dict(frames_u67_raw),
    "u67_without_doc": {
        symbol: frame
        for symbol, frame in frames_u67_raw.items()
        if symbol != "DOC"
    },
    "u67_without_clmt": {
        symbol: frame
        for symbol, frame in frames_u67_raw.items()
        if symbol != "CLMT"
    },
    "u67_without_clmt_doc": {
        symbol: frame
        for symbol, frame in frames_u67_raw.items()
        if symbol not in {"CLMT", "DOC"}
    },
}

results = {}
for label, frames in variants.items():
    results[label] = run_universe(label, frames)

full_result, full_metrics, full_meta = results["u67_full"]
full_capital = float(full_metrics["ending_capital"])
if not math.isclose(
    full_capital,
    EXPECTED_U67_CAPITAL,
    rel_tol=1e-9,
    abs_tol=0.01,
):
    raise RuntimeError(
        "A guarda U67 falhou. Nao interpretar a atribuicao. "
        f"observado={full_capital:,.8f} "
        f"esperado={EXPECTED_U67_CAPITAL:,.8f}"
    )
print("[guard] U67 oficial reproduzido exatamente", flush=True)


# %% 5 - Atribuicao de capital
capitals = {
    label: float(metrics["ending_capital"])
    for label, (_, metrics, _) in results.items()
}
without_doc = capitals["u67_without_doc"]
without_clmt = capitals["u67_without_clmt"]
without_both = capitals["u67_without_clmt_doc"]

attribution = {
    "u67_full": full_capital,
    "u67_without_doc": without_doc,
    "u67_without_clmt": without_clmt,
    "u67_without_clmt_doc": without_both,
    "delta_remove_doc": without_doc - full_capital,
    "ratio_remove_doc": without_doc / full_capital - 1.0,
    "delta_remove_clmt": without_clmt - full_capital,
    "ratio_remove_clmt": without_clmt / full_capital - 1.0,
    "delta_remove_both": without_both - full_capital,
    "ratio_remove_both": without_both / full_capital - 1.0,
    "interaction_clmt_doc": (
        without_both - without_clmt - without_doc + full_capital
    ),
}


def first_state_divergence(
    base_predictions: pd.DataFrame,
    other_predictions: pd.DataFrame,
):
    left = base_predictions.reset_index().rename(
        columns={"selected_asset": "selected_asset_full"}
    )
    right = other_predictions.reset_index().rename(
        columns={"selected_asset": "selected_asset_variant"}
    )
    merged = left[["timestamp", "selected_asset_full"]].merge(
        right[["timestamp", "selected_asset_variant"]],
        on="timestamp",
        how="inner",
    )
    changed = merged.loc[
        merged["selected_asset_full"] != merged["selected_asset_variant"]
    ]
    if changed.empty:
        return None
    return changed.iloc[0].to_dict()


divergences = {
    label: first_state_divergence(
        full_result.predictions,
        result.predictions,
    )
    for label, (result, _, _) in results.items()
    if label != "u67_full"
}

print("=" * 78, flush=True)
print("[ATTRIBUTION]", flush=True)
for key, value in attribution.items():
    if key.startswith("ratio_"):
        print(f"{key}={float(value):+.6%}", flush=True)
    elif key.startswith("delta_") or key == "interaction_clmt_doc":
        print(f"{key}=US$ {float(value):+,.2f}", flush=True)
    else:
        print(f"{key}=US$ {float(value):,.2f}", flush=True)
print("[FIRST_DIVERGENCES]", json.dumps(divergences, default=str), flush=True)
print("=" * 78, flush=True)


# %% 6 - Artefatos
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.rglob("*"):
    if old.is_file():
        old.unlink()

comparison_rows = []
for label, (result, metrics, meta) in results.items():
    comparison_rows.append(
        {
            "variant": label,
            "asset_count": meta["asset_count"],
            "ending_capital": metrics["ending_capital"],
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "maximum_drawdown": metrics["maximum_drawdown"],
            "worst_fold_return": metrics["worst_fold_return"],
            "buy_hold_ending_capital": metrics["buy_hold_ending_capital"],
        }
    )
    result.predictions.reset_index().to_csv(
        OUT / f"{label}_predictions.csv",
        index=False,
    )
    result.trades.to_csv(
        OUT / f"{label}_trades.csv",
        index=False,
    )

pd.DataFrame(comparison_rows).to_csv(
    OUT / "comparison_variants.csv",
    index=False,
)

payload = {
    "research_version": RESEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "source_main_commit": SOURCE_MAIN_COMMIT,
    "question": (
        "Quanto da diferenca entre o U67 do TCC e o MCT e causado "
        "exclusivamente pela ausencia de CLMT e/ou DOC?"
    ),
    "changed_factor_only": "CLMT_DOC_universe_membership",
    "snapshots": {
        "u56": manifest_u56,
        "b2": manifest_b2,
        "smart": manifest_smart,
    },
    "variants": {
        label: {
            "metadata": meta,
            "metrics": metrics,
        }
        for label, (_, metrics, meta) in results.items()
    },
    "attribution": attribution,
    "first_state_divergences_vs_full": divergences,
    "runtime_seconds": float(time.perf_counter() - started),
}
(OUT / "comparison_u67_mct_attribution.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

package = criar_pacote_analise(
    OUT,
    comparison_file="comparison_u67_mct_attribution.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_analise.zip",
)
print(f"[done] package={package}", flush=True)
sinal_sonoro_conclusao()
