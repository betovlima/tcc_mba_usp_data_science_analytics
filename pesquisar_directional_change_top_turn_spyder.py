"""Pesquisa Directional Change Top-Turn + LightGBM no TCC.

Execute no Spyder por celulas (# %%). Esta pesquisa:
- usa somente o snapshot congelado versionado em dados/pesquisa;
- nao altera o Control oficial;
- compara Control vs Directional Change Top-Turn + LightGBM;
- gera ZIP compacto para analise;
- emite aviso sonoro quando todo o processamento termina.
"""

# %% 0 - Imports e configuracao
from pathlib import Path
import json
import time

import pandas as pd

from engine.configuracao import CONFIG
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.rotacao import preparar_painel_rotacao
from pesquisas.directional_change_lightgbm import calcular_peak_exit
from pesquisas.directional_change_top_turn import (
    RESEARCH_VERSION,
    comparar_control_top_turn,
    executar_directional_change_top_turn,
)
from pesquisas.utilitarios import criar_pacote_analise, sinal_sonoro_conclusao
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import (
    build_folds,
    build_variant_configs,
    run_variant,
    summarize_metrics,
)
from reproducao.preparacao import prepare_model_frames

RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHOS = SnapshotPaths.research(RAIZ_PROJETO)
DIRETORIO_RESULTADOS = (
    RAIZ_PROJETO
    / "output"
    / "directional_change_top_turn"
    / RESEARCH_VERSION.replace(".", "_").replace("-", "_")
)

print("=" * 78, flush=True)
print("TCC - Directional Change Top-Turn + LightGBM", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print("comparacao=CONTROL vs DIRECTIONAL_CHANGE_TOP_TURN", flush=True)
print("=" * 78, flush=True)


# %% 1 - Validacao do snapshot congelado
manifesto = validate_snapshot(CAMINHOS)
print(
    "[snapshot] validado "
    f"sha256={manifesto.get('snapshot_sha256')}",
    flush=True,
)


# %% 2 - Preparacao dos dados
inicio_preparacao = time.perf_counter()
frames, exclusoes, diagnosticos_dados, auditoria_dados = prepare_model_frames(
    CAMINHOS,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=True,
)
ativos_elegiveis = tuple(frames)
print(
    f"[stage] preparation eligible={len(frames)} "
    f"seconds={time.perf_counter() - inicio_preparacao:.3f}",
    flush=True,
)


# %% 3 - Control e folds identicos ao baseline
config_control, _ = build_variant_configs(frames, CONFIG)
datas_comuns, folds = build_folds(frames, config_control)

print(
    f"[folds] common_dates={len(datas_comuns)} folds={len(folds)} "
    f"first={datas_comuns.min().date()} last={datas_comuns.max().date()}",
    flush=True,
)


# %% 4 - CONTROL
inicio_control = time.perf_counter()
control_result, control_metrics = run_variant(
    "CONTROL",
    frames,
    config_control,
    folds,
)
print(
    f"[stage] CONTROL seconds={time.perf_counter() - inicio_control:.3f}",
    flush=True,
)


# %% 5 - DIRECTIONAL CHANGE TOP-TURN + LIGHTGBM
inicio_top_turn = time.perf_counter()
top_turn_result = executar_directional_change_top_turn(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=lambda p, stage, completed: print(
        f"[top-turn] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
)
top_turn_metrics = summarize_metrics(
    top_turn_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in top_turn_result.metrics.items():
    if str(chave).startswith("directional_change_"):
        top_turn_metrics[str(chave)] = valor

print(
    "[stage] DIRECTIONAL_CHANGE_TOP_TURN "
    f"capital={top_turn_metrics['ending_capital']:,.2f} "
    f"sharpe={top_turn_metrics['sharpe']:.4f} "
    f"maxdd={top_turn_metrics['maximum_drawdown']:.4%} "
    f"triggers={top_turn_metrics.get('directional_change_exit_triggers')} "
    f"seconds={time.perf_counter() - inicio_top_turn:.3f}",
    flush=True,
)


# %% 6 - Peak Exit: mesma definicao para Control e challenger
frames_alinhados, _, _ = preparar_painel_rotacao(
    frames,
    config_control,
)
control_peak, control_peak_trades = calcular_peak_exit(
    control_result.trades,
    frames_alinhados,
)
top_turn_peak, top_turn_peak_trades = calcular_peak_exit(
    top_turn_result.trades,
    frames_alinhados,
)

print(
    "[peak] CONTROL "
    f"distance={control_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={control_peak.get('median_peak_capture_pct')}",
    flush=True,
)
print(
    "[peak] TOP_TURN "
    f"distance={top_turn_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={top_turn_peak.get('median_peak_capture_pct')}",
    flush=True,
)


# %% 7 - Comparacao final
comparacao = comparar_control_top_turn(
    control_metrics,
    top_turn_metrics,
    control_peak,
    top_turn_peak,
)

print(
    "[comparison] "
    f"CONTROL={comparacao['control_ending_capital']:,.2f} "
    f"TOP_TURN={comparacao['top_turn_ending_capital']:,.2f} "
    f"delta={comparacao['top_turn_minus_control_capital']:,.2f} "
    f"ratio={comparacao['top_turn_vs_control_ratio']:+.4%}",
    flush=True,
)
print(
    "[comparison] "
    "median_peak_distance_improvement_pp="
    f"{comparacao.get('median_exit_distance_improvement_pct_points')} "
    "top_turn_exit_triggers="
    f"{comparacao.get('top_turn_exit_triggers')}",
    flush=True,
)


# %% 8 - Exportacao dos artefatos
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

control_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "control_predictions.csv",
    index=False,
)
control_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "control_trades.csv",
    index=False,
)
top_turn_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "top_turn_predictions.csv",
    index=False,
)
top_turn_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "top_turn_trades.csv",
    index=False,
)
control_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_control.csv",
    index=False,
)
top_turn_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_top_turn.csv",
    index=False,
)

calibration_rows = top_turn_result.metrics.get(
    "directional_change_calibration",
    [],
)
pd.DataFrame(calibration_rows).to_csv(
    DIRETORIO_RESULTADOS / "top_turn_calibration.csv",
    index=False,
)

comparison_table = pd.DataFrame(
    [
        {
            "metric": "ending_capital",
            "control": control_metrics["ending_capital"],
            "top_turn": top_turn_metrics["ending_capital"],
        },
        {
            "metric": "cagr",
            "control": control_metrics["cagr"],
            "top_turn": top_turn_metrics["cagr"],
        },
        {
            "metric": "sharpe",
            "control": control_metrics["sharpe"],
            "top_turn": top_turn_metrics["sharpe"],
        },
        {
            "metric": "maximum_drawdown",
            "control": control_metrics["maximum_drawdown"],
            "top_turn": top_turn_metrics["maximum_drawdown"],
        },
        {
            "metric": "worst_fold_return",
            "control": control_metrics["worst_fold_return"],
            "top_turn": top_turn_metrics["worst_fold_return"],
        },
        {
            "metric": "median_exit_distance_from_peak_pct",
            "control": control_peak["median_exit_distance_from_peak_pct"],
            "top_turn": top_turn_peak[
                "median_exit_distance_from_peak_pct"
            ],
        },
        {
            "metric": "median_peak_capture_pct",
            "control": control_peak["median_peak_capture_pct"],
            "top_turn": top_turn_peak["median_peak_capture_pct"],
        },
        {
            "metric": "median_days_from_peak_to_exit",
            "control": control_peak["median_days_from_peak_to_exit"],
            "top_turn": top_turn_peak["median_days_from_peak_to_exit"],
        },
    ]
)
comparison_table.to_csv(
    DIRETORIO_RESULTADOS / "comparison_top_turn.csv",
    index=False,
)

with (DIRETORIO_RESULTADOS / "comparison_top_turn.json").open(
    "w",
    encoding="utf-8",
) as arquivo:
    json.dump(
        {
            "comparison": comparacao,
            "control_metrics": control_metrics,
            "top_turn_metrics": top_turn_metrics,
            "control_peak": control_peak,
            "top_turn_peak": top_turn_peak,
            "structural_exclusions": exclusoes,
            "snapshot_sha256": manifesto.get("snapshot_sha256"),
        },
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print(f"[output] diretorio={DIRETORIO_RESULTADOS}", flush=True)


# %% 9 - PACOTE ZIP PARA ANALISE
# Este e o unico arquivo que voce precisa enviar para analise.
PACOTE_ANALISE = criar_pacote_analise(
    DIRETORIO_RESULTADOS,
    versao=RESEARCH_VERSION,
)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)


# %% 10 - SINAL SONORO DE CONCLUSAO
# Dois tons no Windows. Em outros sistemas tenta o bell do terminal.
sinal_sonoro_conclusao()
print("[done] pesquisa e pacote de analise concluidos", flush=True)
