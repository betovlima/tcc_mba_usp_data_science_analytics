"""AVALIACAO FINANCEIRA - runner independente para Spyder.

Este arquivo executa a estrategia e mede capital, CAGR, Sharpe, MaxDD e folds.

Baseline financeiro congelado:
    U59 = U56 + COLB + AMS + FOXF

Esse foi o conjunto que atingiu aproximadamente US$ 30,08 milhoes e passa a
ser o ponto de partida da avaliacao financeira.

Importante:
- este arquivo NAO procura ativos;
- a busca fica em buscar_ativos_spyder.py;
- por padrao, uma lista nova so e avaliada depois de ter sido congelada pelo
  arquivo de busca;
- avaliacao individual e opcional e fica desligada por padrao para nao
  confundir diagnostico financeiro com mecanismo de descoberta.

Execucao no Spyder:
- executar o arquivo inteiro com F5; ou
- executar as celulas "# %%" em ordem, de cima para baixo.
"""

from __future__ import annotations

from pathlib import Path
import json
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
from reproducao.graficos import gerar_graficos_pesquisa_financeira
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuracao da avaliacao financeira
ROOT = Path(__file__).resolve().parent
BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_expansao_76_b2"
)
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)

OUT = ROOT / "output" / "avaliacao_financeira"
SELECTION_FILE = SMART.root / "selected_candidates.csv"

SCRIPT_RESEARCH_VERSION = "1.17.2-dev.1"
EXPECTED_SHARED_MODULE_VERSION = "1.17.0-dev.1"
SOURCE_SEARCH_VERSION = "1.17.0-dev.1"
EXECUTION_SCHEMA = "financial-evaluation-u59-positive8-v2"

# U59: conjunto financeiro que produziu aproximadamente US$ 30,08 milhoes.
U59_ADDITIONS = ("COLB", "AMS", "FOXF")
HISTORICAL_U59_ENDING_CAPITAL = 30_080_091.008142874

# Checkpoints financeiros ja observados nesta linha de pesquisa.
# Servem apenas para visualizacao cumulativa e auditoria; nao entram no modelo.
HISTORICAL_RESEARCH_CHECKPOINTS = (
    {
        "label": "U56",
        "ending_capital": 10_082_425.910911141,
        "status": "historical_baseline",
    },
    {
        "label": "U59",
        "ending_capital": 30_080_091.008142874,
        "status": "validated_baseline",
    },
    {
        "label": "U59 + 20",
        "ending_capital": 2_017_935.5138941268,
        "status": "failed_frozen_validation",
    },
    {
        "label": "U59 + 8",
        "ending_capital": 58_557_157.67496595,
        "status": "exploratory_positive_subset",
    },
)

# Diagnostico individual da lista congelada 1.17.0. Mantido aqui para que os
# graficos da pesquisa continuem mostrando o que foi aprendido, mesmo depois
# que a pasta de output da rodada anterior for limpa.
HISTORICAL_INDIVIDUAL_EFFECTS = (
    {"asset": "SGA", "capital_pct_vs_u59": -0.1213898031782675},
    {"asset": "THO", "capital_pct_vs_u59": 0.2985966485004629},
    {"asset": "XNTK", "capital_pct_vs_u59": -0.0060642003161124},
    {"asset": "CIVB", "capital_pct_vs_u59": 0.0},
    {"asset": "WDAY", "capital_pct_vs_u59": 0.1611279104700289},
    {"asset": "EXR", "capital_pct_vs_u59": 0.1502297859145460},
    {"asset": "PAYX", "capital_pct_vs_u59": 0.0280505245194395},
    {"asset": "FMBH", "capital_pct_vs_u59": -0.2443273602314529},
    {"asset": "SBFG", "capital_pct_vs_u59": 0.0557892408537519},
    {"asset": "ALNY", "capital_pct_vs_u59": -0.2584354333761023},
    {"asset": "SXC", "capital_pct_vs_u59": 0.0177801914510296},
    {"asset": "ICCC", "capital_pct_vs_u59": -0.1689313194878604},
    {"asset": "XEL", "capital_pct_vs_u59": 0.1088855014744620},
    {"asset": "EBMT", "capital_pct_vs_u59": -0.2749713629367226},
    {"asset": "VLRS", "capital_pct_vs_u59": -0.6210466934933634},
    {"asset": "PDFS", "capital_pct_vs_u59": -0.1787288258097510},
    {"asset": "SITC", "capital_pct_vs_u59": -0.3939835532403795},
    {"asset": "FDX", "capital_pct_vs_u59": -0.1993113709999824},
    {"asset": "FNWB", "capital_pct_vs_u59": -0.2032560878178345},
    {"asset": "MUX", "capital_pct_vs_u59": 0.0210742632222769},
)

SEARCH_CANDIDATES_CSV = (
    ROOT / "output" / "busca_ativos" / "intelligent_candidates_ranked.csv"
)

# Oito positivos observados SOMENTE depois do congelamento e do replay
# individual da lista 1.17.0. Este grupo e exploratorio, nao confirmatorio.
DIAGNOSTIC_POSITIVE_ASSETS = (
    "THO", "WDAY", "EXR", "XEL",
    "SBFG", "PAYX", "MUX", "SXC",
)

# Controles de execucao para o Spyder.
EXECUTAR_BASELINE_U59 = True

# Precisa permanecer True para carregar e validar a lista congelada original.
AVALIAR_LISTA_CONGELADA = True

# O grupo completo dos 20 ja foi testado e falhou. Deixe False para nao
# repetir esse replay nesta rodada.
AVALIAR_GRUPO_CONGELADO = False

# Experimento atual: U59 + os oito positivos individuais, todos juntos.
# Resultado deve ser interpretado como exploratorio, pois os oito foram
# escolhidos depois de observar seus resultados financeiros individuais.
AVALIAR_GRUPO_POSITIVOS_DIAGNOSTICOS = True

# Diagnostico individual ja concluido na versao anterior. Nao repetir.
AVALIAR_CANDIDATOS_INDIVIDUALMENTE = False


# %% 1 - Guards e snapshots congelados
if RESEARCH_VERSION != EXPECTED_SHARED_MODULE_VERSION:
    raise RuntimeError(
        "Modulo compartilhado inesperado para esta campanha financeira: "
        f"esperado={EXPECTED_SHARED_MODULE_VERSION!r} "
        f"observado={RESEARCH_VERSION!r}. "
        "Atualize a branch e reinicie o kernel do Spyder."
    )

print("=" * 78, flush=True)
print("TCC - Avaliacao Financeira U59", flush=True)
print(f"financial_version={SCRIPT_RESEARCH_VERSION}", flush=True)
print(f"shared_module_version={RESEARCH_VERSION}", flush=True)
print(f"source_search_version={SOURCE_SEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print("baseline=U59=U56+COLB+AMS+FOXF", flush=True)
print(
    f"avaliar_lista_congelada={AVALIAR_LISTA_CONGELADA} "
    f"avaliar_grupo20={AVALIAR_GRUPO_CONGELADO} "
    f"avaliar_grupo8={AVALIAR_GRUPO_POSITIVOS_DIAGNOSTICOS} "
    f"avaliar_individuais={AVALIAR_CANDIDATOS_INDIVIDUALMENTE}",
    flush=True,
)
print("=" * 78, flush=True)

started = time.perf_counter()
manifest_base = validate_snapshot(BASE)
manifest_b2 = validate_snapshot(B2)


# %% 2 - Montagem do U59 vencedor
frames_u56, u56_exclusions, u56_diagnostics, u56_audit = prepare_model_frames(
    BASE,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=False,
    allow_structural_assets=frozenset({"CLMT", "DOC"}),
)
if len(frames_u56) != 56:
    raise RuntimeError(
        f"U56 deveria conter 56 ativos; obtidos {len(frames_u56)}."
    )

frames_b2_positive, b2_exclusions, b2_diagnostics, b2_audit = (
    prepare_model_frames(
        B2,
        assets=U59_ADDITIONS,
        comparar_snapshot_referencia=False,
    )
)
if b2_exclusions or len(frames_b2_positive) != len(U59_ADDITIONS):
    raise RuntimeError(
        "COLB, AMS e FOXF precisam estar integralmente disponiveis. "
        "Nao e permitido alterar silenciosamente o U59 financeiro."
    )

frames_u59 = {
    **frames_u56,
    **frames_b2_positive,
}
symbols_u59 = sorted(frames_u59)
if len(symbols_u59) != 59:
    raise RuntimeError(
        f"U59 deveria conter 59 ativos; obtidos {len(symbols_u59)}."
    )

print(
    "[baseline] U59 montado com adicionais="
    + ",".join(U59_ADDITIONS),
    flush=True,
)


# %% 3 - Leitura opcional da lista congelada pelo arquivo de busca
selected_symbols: list[str] = []
frames_selected: dict[str, pd.DataFrame] = {}
smart_manifest = None
selected_table = pd.DataFrame()

if AVALIAR_LISTA_CONGELADA:
    if not SELECTION_FILE.exists():
        print(
            "[selection] arquivo ainda nao existe: "
            f"{SELECTION_FILE}. Sera executado somente o U59.",
            flush=True,
        )
    else:
        selected_table = pd.read_csv(SELECTION_FILE)
        if "asset" not in selected_table.columns:
            raise RuntimeError(
                "selected_candidates.csv nao possui a coluna asset."
            )

        required_identity = {
            "research_version",
            "execution_schema",
            "search_reference",
            "smart_snapshot_sha256",
        }
        missing_identity = sorted(
            required_identity.difference(selected_table.columns)
        )
        if missing_identity:
            raise RuntimeError(
                "Lista congelada pertence a protocolo antigo ou incompleto. "
                "Rode buscar_ativos_spyder.py novamente. Colunas ausentes: "
                + ",".join(missing_identity)
            )
        if not (
            selected_table["research_version"].astype(str)
            == SOURCE_SEARCH_VERSION
        ).all():
            raise RuntimeError(
                "Lista congelada foi produzida por outra versao da pesquisa."
            )
        if not (
            selected_table["execution_schema"].astype(str)
            == "intelligent-asset-search-u59-v1"
        ).all():
            raise RuntimeError(
                "Lista congelada nao veio do runner de busca U59 atual."
            )
        if not (
            selected_table["search_reference"].astype(str)
            == "U59_WINNER"
        ).all():
            raise RuntimeError(
                "Lista congelada nao foi selecionada contra o U59 vencedor."
            )

        if "signature_predicted_positive" in selected_table.columns:
            flags = (
                selected_table["signature_predicted_positive"]
                .astype(str)
                .str.lower()
                .isin({"true", "1", "yes"})
            )
            if not bool(flags.all()):
                raise RuntimeError(
                    "A lista congelada contem candidato que nao passou "
                    "a assinatura. Rode novamente buscar_ativos_spyder.py."
                )

        selected_symbols = (
            selected_table["asset"]
            .astype(str)
            .str.strip()
            .str.upper()
            .drop_duplicates()
            .tolist()
        )

        if not selected_symbols:
            print(
                "[selection] busca concluiu sem candidatos aprovados. "
                "Sera executado somente o U59.",
                flush=True,
            )
        else:
            smart_manifest = validate_snapshot(SMART)
            observed_snapshot = str(
                smart_manifest.get("snapshot_sha256") or ""
            )
            frozen_hashes = set(
                selected_table["smart_snapshot_sha256"]
                .astype(str)
                .str.strip()
            )
            if frozen_hashes != {observed_snapshot}:
                raise RuntimeError(
                    "A lista congelada nao corresponde ao snapshot SMART atual. "
                    "Rode buscar_ativos_spyder.py novamente antes da avaliacao."
                )

            smart_assets = set(smart_manifest.get("assets") or [])
            missing = sorted(set(selected_symbols).difference(smart_assets))
            if missing:
                raise RuntimeError(
                    "Candidato congelado ausente do snapshot SMART: "
                    + ",".join(missing)
                )

            (
                frames_selected,
                selected_exclusions,
                selected_diagnostics,
                selected_audit,
            ) = prepare_model_frames(
                SMART,
                assets=tuple(selected_symbols),
                comparar_snapshot_referencia=False,
            )
            if selected_exclusions:
                raise RuntimeError(
                    "Candidato congelado apresentou exclusao estrutural: "
                    + ",".join(
                        str(row.get("symbol"))
                        for row in selected_exclusions
                    )
                )
            if len(frames_selected) != len(selected_symbols):
                raise RuntimeError(
                    "Nem todos os candidatos congelados ficaram modelaveis."
                )

            print(
                "[selection] candidatos congelados para avaliacao="
                + ",".join(selected_symbols),
                flush=True,
            )


# %% 4 - Contexto de treino compartilhado e calendario fixo no U56
frames_all_raw = {
    **frames_u59,
    **frames_selected,
}

config_u56, _ = build_variant_configs(frames_u56, CONFIG)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    frames_u56,
    config_u56,
)

config_all, _ = build_variant_configs(frames_all_raw, CONFIG)
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
    f"u59=59 selected={len(selected_symbols)} "
    f"calendar={calendar_source} folds={len(folds)}",
    flush=True,
)


# %% 5 - Treino LightGBM uma vez por fold
fold_artifacts: dict[int, dict] = {}

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
        f"calibration_models={len(symbols_all)}",
        flush=True,
    )
    calibration_models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        train_dates,
        config_all,
        phase=f"financial_u59_fold_{fold_id}_calibration",
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
        f"[train] fold={fold_id} final_models={len(symbols_all)}",
        flush=True,
    )
    final_models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        final_fit_dates,
        config_all,
        phase=f"financial_u59_fold_{fold_id}_final",
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
        "fold": fold,
        "calibration_dates": calibration_dates,
        "decision_dates": decision_dates,
        "calibration_models": calibration_models,
        "calibration_cache": calibration_cache,
        "final_models": final_models,
        "decision_cache": decision_cache,
    }


# %% 6 - Funcoes de replay financeiro
def _slice_cache(cache, subset_symbols):
    indices = [0] + [full_position[symbol] for symbol in subset_symbols]
    return {
        timestamp: np.asarray(values, dtype=np.float64)[indices].copy()
        for timestamp, values in cache.items()
    }


def _run_subset(label: str, subset_symbols, *, keep_result: bool = False):
    subset_symbols = sorted(subset_symbols)
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
        "financial_evaluation_u59",
        scheduled,
        subset_frames,
        subset_symbols,
        all_decision_dates,
        subset_config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"Financial Evaluation - {label}",
        method_line=(
            "- Financial runner separated from asset discovery. "
            "U59 is the frozen baseline. Candidate lists are evaluated only "
            "after being frozen by buscar_ativos_spyder.py. Calendar and "
            "benchmark remain fixed to U56."
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


# %% 7 - Execucao financeira principal
baseline_u59 = None
baseline_u59_capital = None

if EXECUTAR_BASELINE_U59:
    baseline_u59 = _run_subset(
        "U59_WINNER",
        symbols_u59,
        keep_result=True,
    )
    baseline_u59_capital = float(
        baseline_u59["metrics"]["ending_capital"]
    )
    reference_delta = (
        baseline_u59_capital - HISTORICAL_U59_ENDING_CAPITAL
    )
    reference_pct = (
        baseline_u59_capital / HISTORICAL_U59_ENDING_CAPITAL - 1.0
    )
    print(
        "[u59-reference] "
        f"historical={HISTORICAL_U59_ENDING_CAPITAL:,.2f} "
        f"current={baseline_u59_capital:,.2f} "
        f"delta={reference_delta:+,.2f} "
        f"pct={reference_pct:+.6%}",
        flush=True,
    )

group_result = None
if selected_symbols and AVALIAR_GRUPO_CONGELADO:
    group_result = _run_subset(
        "U59_PLUS_FROZEN_SELECTION",
        [*symbols_u59, *selected_symbols],
        keep_result=True,
    )

diagnostic_positive_group_result = None
if AVALIAR_GRUPO_POSITIVOS_DIAGNOSTICOS:
    missing_positive = sorted(
        set(DIAGNOSTIC_POSITIVE_ASSETS).difference(selected_symbols)
    )
    if missing_positive:
        raise RuntimeError(
            "O grupo exploratorio de oito positivos nao e subconjunto da "
            "lista congelada original. Ausentes: "
            + ",".join(missing_positive)
        )
    diagnostic_positive_group_result = _run_subset(
        "U59_PLUS_DIAGNOSTIC_POSITIVE8",
        [*symbols_u59, *DIAGNOSTIC_POSITIVE_ASSETS],
        keep_result=True,
    )

individual_rows = []
if (
    selected_symbols
    and AVALIAR_CANDIDATOS_INDIVIDUALMENTE
):
    if baseline_u59_capital is None:
        raise RuntimeError(
            "Para avaliar candidatos individualmente, execute o baseline U59."
        )

    for position_index, symbol in enumerate(
        selected_symbols,
        start=1,
    ):
        print(
            f"[individual] {position_index}/{len(selected_symbols)} "
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
        individual_rows.append(
            {
                "asset": symbol,
                "ending_capital": ending,
                "capital_delta_vs_u59": (
                    ending - baseline_u59_capital
                ),
                "capital_pct_vs_u59": (
                    ending / baseline_u59_capital - 1.0
                ),
                "cagr": scenario["metrics"]["cagr"],
                "sharpe": scenario["metrics"]["sharpe"],
                "maximum_drawdown": (
                    scenario["metrics"]["maximum_drawdown"]
                ),
                "worst_fold_return": (
                    scenario["metrics"]["worst_fold_return"]
                ),
            }
        )


# %% 8 - Resumo e comparacao
summary_rows = []

if baseline_u59 is not None:
    summary_rows.append(
        {
            "scenario": "U59_WINNER",
            "asset_count": 59,
            **baseline_u59["metrics"],
            "capital_delta_vs_u59": 0.0,
            "capital_pct_vs_u59": 0.0,
        }
    )

if group_result is not None:
    group_capital = float(
        group_result["metrics"]["ending_capital"]
    )
    if baseline_u59_capital is None:
        delta = None
        pct = None
    else:
        delta = group_capital - baseline_u59_capital
        pct = group_capital / baseline_u59_capital - 1.0

    summary_rows.append(
        {
            "scenario": "U59_PLUS_FROZEN_SELECTION",
            "asset_count": 59 + len(selected_symbols),
            **group_result["metrics"],
            "capital_delta_vs_u59": delta,
            "capital_pct_vs_u59": pct,
        }
    )

if diagnostic_positive_group_result is not None:
    positive8_capital = float(
        diagnostic_positive_group_result["metrics"]["ending_capital"]
    )
    if baseline_u59_capital is None:
        positive8_delta = None
        positive8_pct = None
    else:
        positive8_delta = positive8_capital - baseline_u59_capital
        positive8_pct = positive8_capital / baseline_u59_capital - 1.0

    summary_rows.append(
        {
            "scenario": "U59_PLUS_DIAGNOSTIC_POSITIVE8",
            "asset_count": 59 + len(DIAGNOSTIC_POSITIVE_ASSETS),
            **diagnostic_positive_group_result["metrics"],
            "capital_delta_vs_u59": positive8_delta,
            "capital_pct_vs_u59": positive8_pct,
        }
    )

summary = pd.DataFrame(summary_rows)

if not summary.empty:
    print("[financial-summary]", flush=True)
    columns = [
        column
        for column in (
            "scenario",
            "asset_count",
            "ending_capital",
            "cagr",
            "sharpe",
            "maximum_drawdown",
            "worst_fold_return",
            "capital_delta_vs_u59",
            "capital_pct_vs_u59",
        )
        if column in summary.columns
    ]
    print(
        summary[columns].to_string(index=False),
        flush=True,
    )


# %% 9 - Exportacao dos dados financeiros
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.rglob("*"):
    if old.is_file():
        old.unlink()

summary.to_csv(
    OUT / "financial_summary.csv",
    index=False,
)

if not selected_table.empty:
    selected_table.to_csv(
        OUT / "frozen_selected_candidates.csv",
        index=False,
    )

if individual_rows:
    pd.DataFrame(individual_rows).to_csv(
        OUT / "financial_individual_candidates.csv",
        index=False,
    )

if baseline_u59 is not None:
    baseline_u59["result"].predictions.reset_index().to_csv(
        OUT / "u59_predictions.csv",
        index=False,
    )
    baseline_u59["result"].trades.to_csv(
        OUT / "u59_trades.csv",
        index=False,
    )

if group_result is not None:
    group_result["result"].predictions.reset_index().to_csv(
        OUT / "u59_plus_selection_predictions.csv",
        index=False,
    )
    group_result["result"].trades.to_csv(
        OUT / "u59_plus_selection_trades.csv",
        index=False,
    )

if diagnostic_positive_group_result is not None:
    diagnostic_positive_group_result["result"].predictions.reset_index().to_csv(
        OUT / "u59_plus_positive8_predictions.csv",
        index=False,
    )
    diagnostic_positive_group_result["result"].trades.to_csv(
        OUT / "u59_plus_positive8_trades.csv",
        index=False,
    )

# %% 10 - Graficos cumulativos da pesquisa
scenario_results_for_graphs = {}
if baseline_u59 is not None:
    scenario_results_for_graphs["U59_WINNER"] = baseline_u59["result"]
if group_result is not None:
    scenario_results_for_graphs["U59_PLUS_FROZEN_SELECTION"] = group_result["result"]
if diagnostic_positive_group_result is not None:
    scenario_results_for_graphs[
        "U59_PLUS_DIAGNOSTIC_POSITIVE8"
    ] = diagnostic_positive_group_result["result"]

added_assets_for_graphs = {}
if group_result is not None:
    added_assets_for_graphs["U59_PLUS_FROZEN_SELECTION"] = selected_symbols
if diagnostic_positive_group_result is not None:
    added_assets_for_graphs[
        "U59_PLUS_DIAGNOSTIC_POSITIVE8"
    ] = DIAGNOSTIC_POSITIVE_ASSETS

research_graphs = gerar_graficos_pesquisa_financeira(
    OUT,
    summary=summary,
    scenario_results=scenario_results_for_graphs,
    historical_checkpoints=HISTORICAL_RESEARCH_CHECKPOINTS,
    individual_effects=HISTORICAL_INDIVIDUAL_EFFECTS,
    added_assets_by_scenario=added_assets_for_graphs,
    search_candidates_path=SEARCH_CANDIDATES_CSV,
)


# %% 11 - Metadados e pacote final
payload = {
    "research_version": SCRIPT_RESEARCH_VERSION,
    "shared_module_version": RESEARCH_VERSION,
    "source_search_version": SOURCE_SEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "question": (
        "Measure financial performance separately from asset discovery, "
        "keeping U59 (U56 + COLB + AMS + FOXF) as the frozen baseline."
    ),
    "protocol": {
        "financial_baseline": "U59",
        "u59_additions": list(U59_ADDITIONS),
        "u59_asset_count": 59,
        "historical_u59_ending_capital": HISTORICAL_U59_ENDING_CAPITAL,
        "base_calendar_fixed_to_u56": True,
        "benchmark_fixed_to_u56": True,
        "search_runner": "buscar_ativos_spyder.py",
        "financial_runner": "avaliar_resultado_financeiro_spyder.py",
        "selection_file": str(SELECTION_FILE),
        "selected_assets": selected_symbols,
        "evaluate_frozen_group": bool(
            selected_symbols and AVALIAR_GRUPO_CONGELADO
        ),
        "evaluate_diagnostic_positive8": bool(
            AVALIAR_GRUPO_POSITIVOS_DIAGNOSTICOS
        ),
        "diagnostic_positive_assets": list(DIAGNOSTIC_POSITIVE_ASSETS),
        "diagnostic_positive8_is_confirmatory": False,
        "evaluate_individual_candidates": bool(
            AVALIAR_CANDIDATOS_INDIVIDUALMENTE
        ),
        "individual_replays_are_not_used_for_discovery": True,
        "research_graphs_generated": True,
        "research_graphs_directory": str(OUT / "graficos_pesquisa"),
    },
    "research_graph_files": {
        key: str(value)
        for key, value in research_graphs.items()
    },
    "snapshots": {
        "base_snapshot_sha256": manifest_base.get("snapshot_sha256"),
        "batch2_snapshot_sha256": manifest_b2.get("snapshot_sha256"),
        "smart_snapshot_sha256": (
            smart_manifest.get("snapshot_sha256")
            if smart_manifest
            else None
        ),
    },
    "u59": (
        {
            "metrics": baseline_u59["metrics"],
            "margins": baseline_u59["margins"],
        }
        if baseline_u59 is not None
        else None
    ),
    "u59_plus_frozen_selection": (
        {
            "selected_assets": selected_symbols,
            "metrics": group_result["metrics"],
            "margins": group_result["margins"],
            "capital_delta_vs_u59": (
                float(group_result["metrics"]["ending_capital"])
                - baseline_u59_capital
                if baseline_u59_capital is not None
                else None
            ),
            "capital_pct_vs_u59": (
                float(group_result["metrics"]["ending_capital"])
                / baseline_u59_capital
                - 1.0
                if baseline_u59_capital
                else None
            ),
        }
        if group_result is not None
        else None
    ),
    "u59_plus_diagnostic_positive8": (
        {
            "assets": list(DIAGNOSTIC_POSITIVE_ASSETS),
            "metrics": diagnostic_positive_group_result["metrics"],
            "margins": diagnostic_positive_group_result["margins"],
            "capital_delta_vs_u59": (
                float(
                    diagnostic_positive_group_result["metrics"]["ending_capital"]
                ) - baseline_u59_capital
                if baseline_u59_capital is not None
                else None
            ),
            "capital_pct_vs_u59": (
                float(
                    diagnostic_positive_group_result["metrics"]["ending_capital"]
                ) / baseline_u59_capital - 1.0
                if baseline_u59_capital
                else None
            ),
            "interpretation": (
                "Exploratory only: these eight assets were chosen after "
                "observing their individual financial effects."
            ),
        }
        if diagnostic_positive_group_result is not None
        else None
    ),
    "individual_candidates": individual_rows,
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "financial_evaluation.json"
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
    comparison_file="financial_evaluation.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_avaliacao_financeira_positivos8_graficos.zip",
)
print(f"[package] pronto={package}", flush=True)
sinal_sonoro_conclusao()
print(
    "[done] financial evaluation completed "
    f"seconds={time.perf_counter() - started:.3f}",
    flush=True,
)
