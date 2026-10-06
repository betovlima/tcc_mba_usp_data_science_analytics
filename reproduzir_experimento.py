"""REPRODUCAO DO BASELINE DE TRABALHO U67 (~US$ 58,56 MILHOES).

Este e o runner principal da etapa atual da pesquisa.

Objetivo
--------
Reproduzir o universo U67:
- U56 congelado em dados/pesquisa;
- + COLB, AMS e FOXF, congelados em dados/pesquisa_expansao_76_b2;
- + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC do snapshot SMART local.

Resultado de referencia:
    capital final = US$ 58.557.157,67496595

Nota metodologica:
os oito ativos adicionais foram identificados retrospectivamente a partir dos
resultados individuais. Portanto, U67 passa a ser o baseline de trabalho das
proximas analises, mas permanece rotulado como resultado exploratorio no TCC.

Regras
------
- nao usa banco de dados;
- nao baixa dados da Alpaca;
- nao executa busca de ativos;
- usa somente snapshots ja congelados;
- usa LightGBM Control, sem Soft Horizon Consensus;
- fixa o calendario de referencia no U56 original;
- aborta se o capital final nao reproduzir o checkpoint U67.

Execucao no Spyder:
- abra reproduzir_experimento.py;
- reinicie o kernel;
- execute com F5.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import shutil
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
from reproducao.graficos_rotacoes import gerar_graficos_rotacoes
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuracao congelada
ROOT = Path(__file__).resolve().parent

BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_expansao_76_b2"
)
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)

OUT = ROOT / "output" / "reproducao"

REPRODUCTION_VERSION = "1.21.0-dev.3"
EXECUTION_SCHEMA = "u67-positive8-reproduction-v1"

U59_ADDITIONS = ("COLB", "AMS", "FOXF")
POSITIVE8 = (
    "THO", "WDAY", "EXR", "XEL",
    "SBFG", "PAYX", "MUX", "SXC",
)
EXPECTED_U56_COUNT = 56
EXPECTED_U59_COUNT = 59
EXPECTED_U67_COUNT = 67

EXPECTED_ENDING_CAPITAL = 58_557_157.67496595
CAPITAL_REL_TOL = 1e-9
CAPITAL_ABS_TOL = 0.01


# %% 1 - Validacao dos snapshots congelados
started = time.perf_counter()

manifest_u56 = validate_snapshot(BASE)
manifest_b2 = validate_snapshot(B2)
if not SMART.root.exists():
    raise RuntimeError(
        "Snapshot local dados/pesquisa_smart_candidates ausente. "
        "O baseline U67 de US$ 58,56 milhoes depende dos oito ativos "
        "THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC."
    )
manifest_smart = validate_snapshot(SMART)

print("=" * 78, flush=True)
print("TCC MBA USP - REPRODUCAO U67 / BASELINE 58,56M", flush=True)
print(
    f"version={REPRODUCTION_VERSION} schema={EXECUTION_SCHEMA}",
    flush=True,
)
print("data_source=frozen_csv_snapshots", flush=True)
print("database=NO", flush=True)
print("alpaca_download=NO", flush=True)
print("variant=CONTROL", flush=True)
print(
    f"expected_capital={EXPECTED_ENDING_CAPITAL:,.8f}",
    flush=True,
)
print("=" * 78, flush=True)


# %% 2 - U56 congelado
frames_u56, exclusions_u56, diagnostics_u56, audit_u56 = (
    prepare_model_frames(
        BASE,
        assets=CONFIG.assets,
        comparar_snapshot_referencia=False,
        allow_structural_assets=frozenset({"CLMT", "DOC"}),
    )
)

if len(frames_u56) != EXPECTED_U56_COUNT:
    raise RuntimeError(
        "U56 nao foi reproduzido integralmente. "
        f"esperado={EXPECTED_U56_COUNT} observado={len(frames_u56)} "
        f"exclusoes={json.dumps(exclusions_u56, ensure_ascii=False, default=str)}"
    )


# %% 3 - Adicoes que transformam U56 em U59
frames_b2, exclusions_b2, diagnostics_b2, audit_b2 = (
    prepare_model_frames(
        B2,
        assets=U59_ADDITIONS,
        comparar_snapshot_referencia=False,
    )
)

if exclusions_b2 or len(frames_b2) != len(U59_ADDITIONS):
    raise RuntimeError(
        "COLB, AMS e FOXF precisam estar integralmente disponiveis. "
        f"exclusoes={json.dumps(exclusions_b2, ensure_ascii=False, default=str)}"
    )

frames_u59_raw = {
    **frames_u56,
    **frames_b2,
}
symbols_u59_requested = sorted(frames_u59_raw)

if len(symbols_u59_requested) != EXPECTED_U59_COUNT:
    raise RuntimeError(
        "U59 intermediario deveria conter 59 ativos. "
        f"observado={len(symbols_u59_requested)}"
    )

print(
    "[universe] U56=56 additions=COLB,AMS,FOXF U59=59",
    flush=True,
)
print(
    "[universe] assets=" + ",".join(symbols_u59_requested),
    flush=True,
)


# %% 4 - Oito impulsionadores exploratorios e montagem U67
frames_positive8, exclusions_positive8, diagnostics_positive8, audit_positive8 = (
    prepare_model_frames(
        SMART,
        assets=POSITIVE8,
        comparar_snapshot_referencia=False,
    )
)

if exclusions_positive8 or len(frames_positive8) != len(POSITIVE8):
    raise RuntimeError(
        "Os oito ativos do baseline U67 precisam estar integralmente "
        "disponiveis no snapshot SMART. "
        f"exclusoes={json.dumps(exclusions_positive8, ensure_ascii=False, default=str)}"
    )

frames_u67_raw = {
    **frames_u59_raw,
    **frames_positive8,
}
symbols_u67_requested = sorted(frames_u67_raw)

if len(symbols_u67_requested) != EXPECTED_U67_COUNT:
    raise RuntimeError(
        "U67 deveria conter 67 ativos. "
        f"observado={len(symbols_u67_requested)}"
    )

print(
    "[universe] U59=59 positive8="
    + ",".join(POSITIVE8)
    + " U67=67",
    flush=True,
)

# %% 5 - Calendario original U56 e contexto U67
config_u56, _ = build_variant_configs(
    frames_u56,
    CONFIG,
)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    frames_u56,
    config_u56,
)

config_u67, _ = build_variant_configs(
    frames_u67_raw,
    CONFIG,
)

(
    frames_u67,
    common_dates,
    calendar_source,
    symbols_u67,
    folds,
    all_decision_dates,
    decision_to_fold,
    decision_metadata,
) = _construir_contexto_execucao(
    frames_u67_raw,
    config_u67,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_source}",
)

if len(symbols_u67) != EXPECTED_U67_COUNT:
    raise RuntimeError(
        "Contexto modelavel U67 nao contem 67 ativos. "
        f"observado={len(symbols_u67)}"
    )

candidate_margins = tuple(
    float(value)
    for value in config_u67.rotation_switch_margin_candidates
)


# %% 5 - Benchmark fixo do universo original U56
benchmark_frames_u56 = {
    symbol: frames_u67[symbol]
    for symbol in sorted(frames_u56)
}

shared_benchmark = _benchmark_pesos_iguais(
    benchmark_frames_u56,
    sorted(frames_u56),
    all_decision_dates[1:],
    float(config_u67.initial_capital),
    config_u67,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)

BENCHMARK_NAME = (
    "Fixed U56 equal-weight buy-and-hold on the original reference calendar"
)


# %% 6 - Treino e calibracao walk-forward
fold_policies = {}
fold_margins = []
decision_diagnostics = {}

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
        f"[train] fold={fold_id} {fold_position}/{len(folds)} "
        f"models={len(symbols_u67)} calibration",
        flush=True,
    )

    calibration_models = _ajustar_modelos_lightgbm(
        frames_u67,
        symbols_u67,
        train_dates,
        config_u67,
        phase=f"reproduction_u67_fold_{fold_id}_calibration",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )

    calibration_cache, _ = _precalcular_utilidades_modelo(
        calibration_models,
        frames_u67,
        symbols_u67,
        calibration_dates,
        config_u67,
    )

    candidate_scores = []
    for margin in candidate_margins:
        policy = _politica_utilidade(
            calibration_models,
            frames_u67,
            symbols_u67,
            config_u67,
            float(margin),
            utility_cache=calibration_cache,
        )
        score = _crescimento_politica_simples(
            policy,
            frames_u67,
            symbols_u67,
            calibration_dates,
            config_u67,
        )
        candidate_scores.append(
            (float(margin), float(score))
        )

    selection = _selecionar_switch_margin_fold(
        config_u67,
        fold_id,
        candidate_scores,
    )

    selected_margin = float(
        selection["selected_candidate_margin"]
    )
    effective_margin = max(
        float(config_u67.rotation_switch_margin),
        selected_margin,
    )

    print(
        f"[calibration] fold={fold_id} "
        f"selected_margin={selected_margin:.6f} "
        f"effective_margin={effective_margin:.6f}",
        flush=True,
    )

    print(
        f"[train] fold={fold_id} models={len(symbols_u67)} final",
        flush=True,
    )

    final_models = _ajustar_modelos_lightgbm(
        frames_u67,
        symbols_u67,
        final_fit_dates,
        config_u67,
        phase=f"reproduction_u67_fold_{fold_id}_final",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )

    decision_cache, _ = _precalcular_utilidades_modelo(
        final_models,
        frames_u67,
        symbols_u67,
        decision_dates,
        config_u67,
    )

    fold_policies[fold_id] = _politica_utilidade(
        final_models,
        frames_u67,
        symbols_u67,
        config_u67,
        effective_margin,
        fold_id=fold_id,
        calibrated_switch_margin=selected_margin,
        utility_cache=decision_cache,
        decision_diagnostics=decision_diagnostics,
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


# %% 7 - Replay U67
scheduled_policy = _politica_agendada(
    fold_policies,
    decision_to_fold,
)

result = _simular_exato(
    "u67_positive8_reproduction",
    scheduled_policy,
    frames_u67,
    symbols_u67,
    all_decision_dates,
    config_u67,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    decision_metadata=decision_metadata,
    policy_decision_diagnostics=decision_diagnostics,
    model_label="U67 Positive8 - baseline de trabalho",
    method_line=(
        "- Baseline U67: U59 + THO + WDAY + EXR + XEL + SBFG + PAYX + MUX + SXC; "
        "LightGBM Control; calendario U56 fixo."
    ),
    benchmark_override=shared_benchmark,
    benchmark_override_name=BENCHMARK_NAME,
)

metrics = summarize_metrics(
    result,
    folds,
    float(config_u67.initial_capital),
)

ending_capital = float(metrics["ending_capital"])
relative_error = (
    ending_capital / EXPECTED_ENDING_CAPITAL - 1.0
)

print(
    "[result] "
    f"capital={ending_capital:,.8f} "
    f"expected={EXPECTED_ENDING_CAPITAL:,.8f} "
    f"relative_error={relative_error:+.12e}",
    flush=True,
)
print(
    "[result] "
    f"cagr={float(metrics['cagr']):.6%} "
    f"sharpe={float(metrics['sharpe']):.8f} "
    f"maxdd={float(metrics['maximum_drawdown']):.6%} "
    f"worst_fold={float(metrics['worst_fold_return']):.6%}",
    flush=True,
)

reproduced = math.isclose(
    ending_capital,
    EXPECTED_ENDING_CAPITAL,
    rel_tol=CAPITAL_REL_TOL,
    abs_tol=CAPITAL_ABS_TOL,
)

if not reproduced:
    raise RuntimeError(
        "A reproducao U67 divergiu do checkpoint de US$ 58.557.157,67. "
        f"observado={ending_capital:,.8f} "
        f"esperado={EXPECTED_ENDING_CAPITAL:,.8f}"
    )


# %% 8 - Artefatos finais de reproducao
# A pasta e recriada para impedir mistura de artefatos U59/U67 entre rodadas.
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    {"asset": symbols_u67}
).to_csv(
    OUT / "u67_assets.csv",
    index=False,
)

pd.DataFrame(fold_margins).to_csv(
    OUT / "u67_fold_margins.csv",
    index=False,
)

result.predictions.reset_index().to_csv(
    OUT / "u67_predictions.csv",
    index=False,
)

result.trades.to_csv(
    OUT / "u67_trades.csv",
    index=False,
)

# %% 9 - Visualizacoes exploratorias das rotacoes
rotation_graphs = gerar_graficos_rotacoes(
    OUT,
    result=result,
    universe_label="U67 · baseline US$ 58,56M",
    highlight_assets=POSITIVE8,
    show=True,
)

payload = {
    "reproduction_version": REPRODUCTION_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "status": "reproduced" if reproduced else "failed",
    "universe": {
        "base_u56_count": len(frames_u56),
        "u59_additions": list(U59_ADDITIONS),
        "positive8": list(POSITIVE8),
        "u67_count": len(symbols_u67),
        "assets": list(symbols_u67),
        "scientific_status": "exploratory_working_baseline",
    },
    "snapshots": {
        "u56": manifest_u56,
        "b2": manifest_b2,
        "smart": manifest_smart,
    },
    "calendar": {
        "source": calendar_source,
        "common_dates": len(common_dates),
        "decision_sessions": len(all_decision_dates),
    },
    "fold_margins": fold_margins,
    "expected_ending_capital": EXPECTED_ENDING_CAPITAL,
    "observed_ending_capital": ending_capital,
    "relative_error": relative_error,
    "metrics": metrics,
    "rotation_visualizations": {
        key: str(path.relative_to(ROOT))
        for key, path in rotation_graphs.items()
        if isinstance(path, Path)
    },
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "reproducao_u67.json"
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
    comparison_file="reproducao_u67.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_reproducao_u67_58m.zip",
)

print("=" * 78, flush=True)
print(
    "[done] U67 REPRODUZIDO COM SUCESSO",
    flush=True,
)
print(
    f"[done] capital_final=US$ {ending_capital:,.2f}",
    flush=True,
)
print(
    f"[done] package={package}",
    flush=True,
)
print(
    f"[done] graficos_rotacoes={OUT / 'graficos_rotacoes'}",
    flush=True,
)
print("=" * 78, flush=True)

sinal_sonoro_conclusao()
