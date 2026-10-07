"""PESQUISA DE SELECAO CAUSAL DO UNIVERSO U67.

Pergunta principal
------------------
Com o mesmo capital inicial, uma estrategia de rotacao LightGBM aplicada a um
grupo de ativos selecionado sem olhar o OOS supera o buy-and-hold dos mesmos
ativos?

Protocolo
---------
1. carrega exatamente o U67 oficial e os mesmos snapshots congelados;
2. NAO usa CUSIP, capital OOS, retorno OOS ou excecao por ticker na selecao;
3. usa somente o primeiro bloco de calibracao, anterior a qualquer teste OOS;
4. treina os modelos apenas no bloco de treino anterior;
5. mantem o ativo se a correlacao de ranking entre score previsto e utilidade
   futura realizada na calibracao for estritamente positiva;
6. congela o universo selecionado antes de observar qualquer resultado OOS;
7. executa Control U67 completo e Control no universo selecionado;
8. compara cada rotacao com buy-and-hold equal-weight do MESMO universo.

A selecao e estatica durante todo o OOS. O segundo purge do fold separa os
labels da calibracao do inicio do teste.

Execucao no Spyder
------------------
Abra buscar_ativos.py, reinicie o kernel e execute com F5.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

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
    ROTATION_FEATURES,
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


# %% 0 - Configuracao preregistrada da pesquisa
ROOT = Path(__file__).resolve().parent
BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_expansao_76_b2"
)
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)
OUT = ROOT / "output" / "selecao_universo"

RESEARCH_VERSION = "1.22.0-dev.1"
EXECUTION_SCHEMA = "u67-causal-universe-selection-v1"
SOURCE_CHECKPOINT = "U67 Control v1.21.0"
SOURCE_MAIN_COMMIT = "4b5f16030afa8b850790747bb0e3e3063d233e79"

U59_ADDITIONS = ("COLB", "AMS", "FOXF")
U67_ADDITIONS = (
    "THO", "WDAY", "EXR", "XEL",
    "SBFG", "PAYX", "MUX", "SXC",
)
EXPECTED_U67_COUNT = 67
EXPECTED_U67_CAPITAL = 58_557_157.67496595

# Criterio fixado ANTES do OOS. Nao ajustar depois de ver o capital.
MIN_CALIBRATION_ROWS = 63
MIN_VALIDATION_RANK_CORRELATION = 0.0

started = time.perf_counter()

print("=" * 78, flush=True)
print("TCC - SELECAO CAUSAL DO UNIVERSO U67", flush=True)
print(
    f"version={RESEARCH_VERSION} schema={EXECUTION_SCHEMA}",
    flush=True,
)
print("selection_uses_oos_capital=False", flush=True)
print("selection_uses_oos_returns=False", flush=True)
print("selection_uses_cusip=False", flush=True)
print("selection_symbol_exceptions=False", flush=True)
print("selection_window=fold_1_calibration_only", flush=True)
print(
    "selection_rule=rows>=63 AND validation_rank_correlation>0",
    flush=True,
)
print("=" * 78, flush=True)


# %% 1 - Carregamento exato do U67 oficial
manifest_u56 = validate_snapshot(BASE)
manifest_b2 = validate_snapshot(B2)
manifest_smart = validate_snapshot(SMART)

frames_u56, exclusions_u56, diagnostics_u56, audit_u56 = (
    prepare_model_frames(
        BASE,
        assets=CONFIG.assets,
        comparar_snapshot_referencia=False,
        # O checkpoint oficial U67 inclui estes ativos. O processo de selecao
        # abaixo decide por evidencia de calibracao, nunca por CUSIP/ticker.
        allow_structural_assets=frozenset({"CLMT", "DOC"}),
    )
)
if len(frames_u56) != 56:
    raise RuntimeError(
        "O U56 oficial precisa ser carregado integralmente. "
        f"observado={len(frames_u56)} exclusoes={exclusions_u56}"
    )

frames_b2, exclusions_b2, diagnostics_b2, audit_b2 = prepare_model_frames(
    B2,
    assets=U59_ADDITIONS,
    comparar_snapshot_referencia=False,
)
if exclusions_b2 or len(frames_b2) != len(U59_ADDITIONS):
    raise RuntimeError(
        "As adicoes U59 precisam estar integralmente disponiveis. "
        f"exclusoes={exclusions_b2}"
    )

frames_smart, exclusions_smart, diagnostics_smart, audit_smart = (
    prepare_model_frames(
        SMART,
        assets=U67_ADDITIONS,
        comparar_snapshot_referencia=False,
    )
)
if exclusions_smart or len(frames_smart) != len(U67_ADDITIONS):
    raise RuntimeError(
        "As adicoes U67 precisam estar integralmente disponiveis. "
        f"exclusoes={exclusions_smart}"
    )

frames_u67_raw = {
    **frames_u56,
    **frames_b2,
    **frames_smart,
}
if len(frames_u67_raw) != EXPECTED_U67_COUNT:
    raise RuntimeError(
        "O universo de origem precisa conter exatamente 67 ativos. "
        f"observado={len(frames_u67_raw)}"
    )


# %% 2 - Calendario fixo U56 e contexto temporal do U67
config_u56, _ = build_variant_configs(frames_u56, CONFIG)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    frames_u56,
    config_u56,
)

config_u67, _ = build_variant_configs(frames_u67_raw, CONFIG)
(
    frames_u67,
    common_dates_u67,
    calendar_source_u67,
    symbols_u67,
    folds_u67,
    all_decision_dates_u67,
    decision_to_fold_u67,
    decision_metadata_u67,
) = _construir_contexto_execucao(
    frames_u67_raw,
    config_u67,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_source}",
)

if len(symbols_u67) != EXPECTED_U67_COUNT:
    raise RuntimeError(
        "O contexto modelavel precisa conter 67 ativos. "
        f"observado={len(symbols_u67)}"
    )
if not folds_u67:
    raise RuntimeError("Nenhum fold walk-forward foi construido.")


# %% 3 - Selecao estatica usando SOMENTE calibracao anterior ao primeiro OOS
first_fold = folds_u67[0]
selection_train_dates = common_dates_u67[
    : int(first_fold["train_end_index"])
]
selection_calibration_dates = common_dates_u67[
    int(first_fold["calibration_start_index"]):
    int(first_fold["calibration_end_index"])
]

print(
    "[selection] fitting pre-OOS models "
    f"train_sessions={len(selection_train_dates)} "
    f"calibration_sessions={len(selection_calibration_dates)}",
    flush=True,
)

selection_models = _ajustar_modelos_lightgbm(
    frames_u67,
    symbols_u67,
    selection_train_dates,
    config_u67,
    phase="universe_selection_pre_oos",
    technical_log_callback=lambda message: print(
        f"[technical] {message}",
        flush=True,
    ),
)


def _rank_correlation(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    pair = pd.concat(
        [
            pd.Series(actual, dtype=float).rename("actual"),
            pd.Series(predicted, dtype=float).rename("predicted"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return None
    actual_rank = pair["actual"].rank(method="average")
    predicted_rank = pair["predicted"].rank(method="average")
    if actual_rank.nunique() < 2 or predicted_rank.nunique() < 2:
        return None
    value = actual_rank.corr(predicted_rank)
    return float(value) if pd.notna(value) else None


selection_rows: list[dict[str, object]] = []
selected_symbols: list[str] = []

for position, symbol in enumerate(symbols_u67, start=1):
    model = selection_models.get(symbol)
    frame = frames_u67[symbol]
    sample = frame.reindex(selection_calibration_dates).dropna(
        subset=["forward_risk_adjusted_utility", *ROTATION_FEATURES]
    )

    rank_corr: float | None = None
    mae: float | None = None
    predicted_mean: float | None = None
    actual_mean: float | None = None

    if model is not None and not sample.empty:
        predicted = pd.Series(
            np.asarray(
                model.predict(sample[ROTATION_FEATURES]),
                dtype=float,
            ),
            index=sample.index,
            dtype=float,
        )
        actual = pd.Series(
            sample["forward_risk_adjusted_utility"].to_numpy(dtype=float),
            index=sample.index,
            dtype=float,
        )
        rank_corr = _rank_correlation(actual, predicted)
        mae = float(np.mean(np.abs(predicted.to_numpy() - actual.to_numpy())))
        predicted_mean = float(predicted.mean())
        actual_mean = float(actual.mean())

    keep = bool(
        model is not None
        and len(sample) >= MIN_CALIBRATION_ROWS
        and rank_corr is not None
        and rank_corr > MIN_VALIDATION_RANK_CORRELATION
    )
    if keep:
        selected_symbols.append(symbol)

    row = {
        "asset": symbol,
        "calibration_rows": int(len(sample)),
        "validation_rank_correlation": rank_corr,
        "validation_mae": mae,
        "predicted_utility_mean": predicted_mean,
        "realized_utility_mean": actual_mean,
        "selected": keep,
        "selection_reason": (
            "positive_pre_oos_validation_rank_correlation"
            if keep
            else "failed_pre_oos_validation_rule"
        ),
    }
    selection_rows.append(row)
    print(
        f"[selection] {position}/{len(symbols_u67)} {symbol} "
        f"rows={len(sample)} rank_corr={rank_corr} selected={keep}",
        flush=True,
    )

selected_symbols = sorted(selected_symbols)
removed_symbols = sorted(set(symbols_u67) - set(selected_symbols))

if len(selected_symbols) < 2:
    raise RuntimeError(
        "A regra preregistrada selecionou menos de dois ativos. "
        "Nao ajuste o limiar com base no OOS; encerre a tentativa."
    )

print(
    f"[selection] frozen selected={len(selected_symbols)}/67 "
    "assets=" + ",".join(selected_symbols),
    flush=True,
)
print(
    f"[selection] removed={len(removed_symbols)} "
    "assets=" + ",".join(removed_symbols),
    flush=True,
)


# %% 4 - Executor Control comum aos dois universos
def _run_control_universe(
    label: str,
    raw_frames: dict[str, pd.DataFrame],
) -> tuple[object, dict[str, object], dict[str, object]]:
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

    benchmark_dates = pd.DatetimeIndex(all_decision_dates[1:])
    incomplete_benchmark_assets: list[str] = []
    for symbol in symbols:
        window = frames[symbol].reindex(benchmark_dates)
        first_open = (
            float(window.iloc[0]["open"])
            if not window.empty
            else float("nan")
        )
        closes = pd.to_numeric(window["close"], errors="coerce")
        if (
            not np.isfinite(first_open)
            or first_open <= 0.0
            or closes.isna().any()
            or bool((closes <= 0.0).any())
        ):
            incomplete_benchmark_assets.append(symbol)
    if incomplete_benchmark_assets:
        raise RuntimeError(
            "O buy-and-hold precisa usar exatamente o mesmo universo da "
            "rotacao. Ativos sem cobertura completa: "
            + ",".join(incomplete_benchmark_assets)
        )

    benchmark = _benchmark_pesos_iguais(
        {symbol: frames[symbol] for symbol in symbols},
        symbols,
        benchmark_dates,
        float(config.initial_capital),
        config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
    )
    benchmark_name = (
        f"Equal-weight buy-and-hold of the same {len(symbols)}-asset universe"
    )

    candidate_margins = tuple(
        float(value)
        for value in config.rotation_switch_margin_candidates
    )
    fold_policies = {}
    fold_margins: list[dict[str, object]] = []

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
            f"calibration models={len(symbols)}",
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

        candidate_scores: list[tuple[float, float]] = []
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

        margin_selection = _selecionar_switch_margin_fold(
            config,
            fold_id,
            candidate_scores,
        )
        selected_margin = float(
            margin_selection["selected_candidate_margin"]
        )
        effective_margin = max(
            float(config.rotation_switch_margin),
            selected_margin,
        )

        print(
            f"[{label}] fold={fold_id} "
            f"selected_margin={selected_margin:.6f} "
            f"effective_margin={effective_margin:.6f}",
            flush=True,
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
                    margin_selection["selected_calibration_score"]
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
        model_label=f"U67 Control - {label}",
        method_line=(
            "- LightGBM Control; universo congelado antes do OOS; "
            "calendario de referencia U56 fixo."
        ),
        benchmark_override=benchmark,
        benchmark_override_name=benchmark_name,
    )
    metrics = summarize_metrics(
        result,
        folds,
        float(config.initial_capital),
    )
    metadata = {
        "label": label,
        "assets": list(symbols),
        "asset_count": len(symbols),
        "calendar_source": calendar_source,
        "common_dates": len(common_dates),
        "decision_sessions": len(all_decision_dates),
        "fold_margins": fold_margins,
    }
    print(
        f"[{label}] capital={float(metrics['ending_capital']):,.2f} "
        f"buy_hold={float(metrics['buy_hold_ending_capital']):,.2f} "
        f"sharpe={float(metrics['sharpe']):.4f} "
        f"maxdd={float(metrics['maximum_drawdown']):.4%}",
        flush=True,
    )
    return result, metrics, metadata


# %% 5 - Controle U67 completo; guarda de reproducao
full_result, full_metrics, full_meta = _run_control_universe(
    "u67_full_control",
    frames_u67_raw,
)

full_capital = float(full_metrics["ending_capital"])
if not math.isclose(
    full_capital,
    EXPECTED_U67_CAPITAL,
    rel_tol=1e-9,
    abs_tol=0.01,
):
    raise RuntimeError(
        "O Control U67 completo nao reproduziu o checkpoint antes da "
        "comparacao de universo. "
        f"observado={full_capital:,.8f} "
        f"esperado={EXPECTED_U67_CAPITAL:,.8f}"
    )

print(
    "[guard] full U67 checkpoint reproduced exactly",
    flush=True,
)


# %% 6 - Control no universo selecionado, sem mudar a regra
selected_raw = {
    symbol: frames_u67_raw[symbol]
    for symbol in selected_symbols
}
selected_result, selected_metrics, selected_meta = _run_control_universe(
    "u67_selected_control",
    selected_raw,
)


# %% 7 - Decomposicao da pergunta de pesquisa
full_buy_hold = float(full_metrics["buy_hold_ending_capital"])
selected_buy_hold = float(selected_metrics["buy_hold_ending_capital"])
selected_capital = float(selected_metrics["ending_capital"])

comparison = {
    "initial_capital": float(CONFIG.initial_capital),
    "full_universe_count": len(symbols_u67),
    "selected_universe_count": len(selected_symbols),
    "removed_universe_count": len(removed_symbols),
    "full_rotation_ending_capital": full_capital,
    "full_buy_hold_ending_capital": full_buy_hold,
    "selected_rotation_ending_capital": selected_capital,
    "selected_buy_hold_ending_capital": selected_buy_hold,
    "selection_effect_on_buy_hold_capital": selected_buy_hold - full_buy_hold,
    "selection_effect_on_buy_hold_ratio": (
        selected_buy_hold / full_buy_hold - 1.0
        if full_buy_hold > 0
        else None
    ),
    "rotation_effect_full_capital": full_capital - full_buy_hold,
    "rotation_effect_full_ratio": (
        full_capital / full_buy_hold - 1.0
        if full_buy_hold > 0
        else None
    ),
    "rotation_effect_selected_capital": selected_capital - selected_buy_hold,
    "rotation_effect_selected_ratio": (
        selected_capital / selected_buy_hold - 1.0
        if selected_buy_hold > 0
        else None
    ),
    "selected_rotation_vs_full_rotation_capital": (
        selected_capital - full_capital
    ),
    "selected_rotation_vs_full_rotation_ratio": (
        selected_capital / full_capital - 1.0
        if full_capital > 0
        else None
    ),
    "interaction_capital": (
        (selected_capital - selected_buy_hold)
        - (full_capital - full_buy_hold)
    ),
    "selected_rotation_beats_same_universe_buy_hold": bool(
        selected_capital > selected_buy_hold
    ),
}

print("=" * 78, flush=True)
print("[QUESTION] same capital, same selected assets", flush=True)
print(
    f"[FULL] rotation=US$ {full_capital:,.2f} "
    f"buy_hold=US$ {full_buy_hold:,.2f}",
    flush=True,
)
print(
    f"[SELECTED] assets={len(selected_symbols)} "
    f"rotation=US$ {selected_capital:,.2f} "
    f"buy_hold=US$ {selected_buy_hold:,.2f}",
    flush=True,
)
print(
    "[ANSWER] selected_rotation_minus_same_universe_buy_hold="
    f"US$ {selected_capital - selected_buy_hold:,.2f}",
    flush=True,
)
print(
    "[ANSWER] selected_rotation_beats_same_universe_buy_hold="
    f"{selected_capital > selected_buy_hold}",
    flush=True,
)
print("=" * 78, flush=True)


# %% 8 - Artefatos auditaveis
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.rglob("*"):
    if old.is_file():
        old.unlink()

selection_frame = pd.DataFrame(selection_rows).sort_values("asset")
selection_frame.to_csv(
    OUT / "universe_selection.csv",
    index=False,
)
pd.DataFrame({"asset": selected_symbols}).to_csv(
    OUT / "selected_assets.csv",
    index=False,
)

full_result.predictions.reset_index().to_csv(
    OUT / "u67_full_predictions.csv",
    index=False,
)
full_result.trades.to_csv(
    OUT / "u67_full_trades.csv",
    index=False,
)
selected_result.predictions.reset_index().to_csv(
    OUT / "selected_predictions.csv",
    index=False,
)
selected_result.trades.to_csv(
    OUT / "selected_trades.csv",
    index=False,
)

payload = {
    "research_version": RESEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "source_checkpoint": SOURCE_CHECKPOINT,
    "source_main_commit": SOURCE_MAIN_COMMIT,
    "status": "completed",
    "research_question": (
        "Com o mesmo capital inicial, a rotacao LightGBM sobre um grupo "
        "selecionado sem olhar o OOS supera o buy-and-hold dos mesmos ativos?"
    ),
    "selection_protocol": {
        "static_before_all_oos": True,
        "source_fold": int(first_fold["fold_id"]),
        "selection_window": "first_fold_calibration",
        "training_sessions": len(selection_train_dates),
        "calibration_sessions": len(selection_calibration_dates),
        "training_start": str(selection_train_dates.min()),
        "training_end": str(selection_train_dates.max()),
        "calibration_start": str(selection_calibration_dates.min()),
        "calibration_end": str(selection_calibration_dates.max()),
        "minimum_calibration_rows": MIN_CALIBRATION_ROWS,
        "minimum_validation_rank_correlation_exclusive": (
            MIN_VALIDATION_RANK_CORRELATION
        ),
        "uses_oos_capital": False,
        "uses_oos_returns": False,
        "uses_buy_hold_result": False,
        "uses_cusip": False,
        "uses_symbol_specific_exceptions": False,
        "uses_future_test_information": False,
        "rule": (
            "keep asset iff model exists, calibration rows >= 63, and "
            "Spearman-like rank correlation(predicted utility, realized "
            "forward utility) > 0 in the first pre-OOS calibration block"
        ),
    },
    "universe": {
        "full_assets": list(symbols_u67),
        "selected_assets": list(selected_symbols),
        "removed_assets": list(removed_symbols),
    },
    "snapshots": {
        "u56": manifest_u56,
        "b2": manifest_b2,
        "smart": manifest_smart,
    },
    "full_control": {
        "metadata": full_meta,
        "metrics": full_metrics,
    },
    "selected_control": {
        "metadata": selected_meta,
        "metrics": selected_metrics,
    },
    "comparison": comparison,
    "runtime_seconds": float(time.perf_counter() - started),
}

comparison_path = OUT / "comparison_universe_selection.json"
comparison_path.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)

package = criar_pacote_analise(
    OUT,
    comparison_file="comparison_universe_selection.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_analise.zip",
)

print(
    f"[done] output={OUT}",
    flush=True,
)
print(
    f"[done] package={package}",
    flush=True,
)
sinal_sonoro_conclusao()
