"""Pesquisa Directional Change + LightGBM no TCC.

Execute no Spyder por celulas (# %%). Esta pesquisa:
- usa somente o snapshot congelado versionado em dados/pesquisa;
- nao altera o Control oficial;
- compara Control vs Top-Turn vs BOCPD;
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
from pesquisas.directional_change_lightgbm import (
    RESEARCH_VERSION,
    calcular_peak_exit,
    comparar_control_directional_change,
    criar_pacote_analise,
    executar_bocpd_overlay,
    executar_directional_change_lightgbm,
    sinal_sonoro_conclusao,
)
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
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"

print("=" * 78, flush=True)
print("TCC - Directional Change + LightGBM", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print("comparacao=CONTROL vs TOP_TURN vs BOCPD", flush=True)
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


# %% 5 - DIRECTIONAL CHANGE + LIGHTGBM
inicio_directional_change = time.perf_counter()
directional_change_result = executar_directional_change_lightgbm(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=lambda p, stage, completed: print(
        f"[top-turn] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
    run_ablation=False,
)
directional_change_metrics = summarize_metrics(
    directional_change_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in directional_change_result.metrics.items():
    if str(chave).startswith("directional_change_"):
        directional_change_metrics[str(chave)] = valor

print(
    "[stage] DIRECTIONAL_CHANGE "
    f"capital={directional_change_metrics['ending_capital']:,.2f} "
    f"sharpe={directional_change_metrics['sharpe']:.4f} "
    f"maxdd={directional_change_metrics['maximum_drawdown']:.4%} "
    f"triggers={directional_change_metrics.get('directional_change_exit_triggers')} "
    f"seconds={time.perf_counter() - inicio_directional_change:.3f}",
    flush=True,
)


# %% 6 - BOCPD
inicio_bocpd = time.perf_counter()
bocpd_result = executar_bocpd_overlay(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=lambda p, stage, completed: print(
        f"[bocpd] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
)
bocpd_metrics = summarize_metrics(
    bocpd_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in bocpd_result.metrics.items():
    if str(chave).startswith("bocpd_"):
        bocpd_metrics[str(chave)] = valor

print(
    "[stage] BOCPD "
    f"capital={bocpd_metrics['ending_capital']:,.2f} "
    f"sharpe={bocpd_metrics['sharpe']:.4f} "
    f"maxdd={bocpd_metrics['maximum_drawdown']:.4%} "
    f"triggers={bocpd_metrics.get('bocpd_exit_triggers')} "
    f"seconds={time.perf_counter() - inicio_bocpd:.3f}",
    flush=True,
)


# %% 7 - Peak Exit: mesma definicao para os tres cenarios
frames_alinhados, _, _ = preparar_painel_rotacao(
    frames,
    config_control,
)
control_peak, control_peak_trades = calcular_peak_exit(
    control_result.trades,
    frames_alinhados,
)
directional_change_peak, directional_change_peak_trades = calcular_peak_exit(
    directional_change_result.trades,
    frames_alinhados,
)
bocpd_peak, bocpd_peak_trades = calcular_peak_exit(
    bocpd_result.trades,
    frames_alinhados,
)

print(
    "[peak] CONTROL "
    f"distance={control_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={control_peak.get('median_peak_capture_pct')}",
    flush=True,
)
print(
    "[peak] DIRECTIONAL_CHANGE "
    f"distance={directional_change_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={directional_change_peak.get('median_peak_capture_pct')}",
    flush=True,
)
print(
    "[peak] BOCPD "
    f"distance={bocpd_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={bocpd_peak.get('median_peak_capture_pct')}",
    flush=True,
)


# %% 8 - Comparacao final
comparacao = comparar_control_directional_change(
    control_metrics,
    directional_change_metrics,
    control_peak,
    directional_change_peak,
)

print(
    "[comparison] "
    f"CONTROL={comparacao['control_ending_capital']:,.2f} "
    f"DIRECTIONAL_CHANGE={comparacao['directional_change_ending_capital']:,.2f} "
    f"delta={comparacao['directional_change_minus_control_capital']:,.2f} "
    f"ratio={comparacao['directional_change_vs_control_ratio']:+.4%}",
    flush=True,
)
print(
    "[comparison] "
    "median_peak_distance_improvement_pp="
    f"{comparacao.get('median_exit_distance_improvement_pct_points')} "
    "directional_change_exit_triggers="
    f"{comparacao.get('directional_change_exit_triggers')}",
    flush=True,
)

print(
    "[comparison-bocpd] "
    f"CONTROL={float(control_metrics['ending_capital']):,.2f} "
    f"TOP_TURN={float(directional_change_metrics['ending_capital']):,.2f} "
    f"BOCPD={float(bocpd_metrics['ending_capital']):,.2f} "
    f"BOCPD_vs_CONTROL="
    f"{float(bocpd_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%} "
    f"BOCPD_vs_TOP_TURN="
    f"{float(bocpd_metrics['ending_capital']) / float(directional_change_metrics['ending_capital']) - 1.0:+.4%}",
    flush=True,
)

# %% 9 - Exportacao dos artefatos
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

control_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "control_predictions.csv",
    index=False,
)
control_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "control_trades.csv",
    index=False,
)
directional_change_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "directional_change_predictions.csv",
    index=False,
)
directional_change_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "directional_change_trades.csv",
    index=False,
)
bocpd_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "bocpd_predictions.csv",
    index=False,
)
bocpd_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "bocpd_trades.csv",
    index=False,
)
control_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_control.csv",
    index=False,
)
directional_change_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_directional_change.csv",
    index=False,
)
bocpd_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_bocpd.csv",
    index=False,
)

calibration_rows = directional_change_result.metrics.get(
    "directional_change_calibration",
    [],
)
pd.DataFrame(calibration_rows).to_csv(
    DIRETORIO_RESULTADOS / "directional_change_calibration.csv",
    index=False,
)

bocpd_calibration_rows = bocpd_result.metrics.get(
    "bocpd_calibration",
    [],
)
pd.DataFrame(bocpd_calibration_rows).to_csv(
    DIRETORIO_RESULTADOS / "bocpd_calibration.csv",
    index=False,
)

comparison_table = pd.DataFrame(
    [
        {
            "metric": "ending_capital",
            "control": control_metrics["ending_capital"],
            "directional_change": directional_change_metrics["ending_capital"],
            "bocpd": bocpd_metrics["ending_capital"],
        },
        {
            "metric": "cagr",
            "control": control_metrics["cagr"],
            "directional_change": directional_change_metrics["cagr"],
            "bocpd": bocpd_metrics["cagr"],
        },
        {
            "metric": "sharpe",
            "control": control_metrics["sharpe"],
            "directional_change": directional_change_metrics["sharpe"],
            "bocpd": bocpd_metrics["sharpe"],
        },
        {
            "metric": "maximum_drawdown",
            "control": control_metrics["maximum_drawdown"],
            "directional_change": directional_change_metrics["maximum_drawdown"],
            "bocpd": bocpd_metrics["maximum_drawdown"],
        },
        {
            "metric": "worst_fold_return",
            "control": control_metrics["worst_fold_return"],
            "directional_change": directional_change_metrics["worst_fold_return"],
            "bocpd": bocpd_metrics["worst_fold_return"],
        },
        {
            "metric": "median_exit_distance_from_peak_pct",
            "control": control_peak["median_exit_distance_from_peak_pct"],
            "directional_change": directional_change_peak[
                "median_exit_distance_from_peak_pct"
            ],
            "bocpd": bocpd_peak[
                "median_exit_distance_from_peak_pct"
            ],
        },
        {
            "metric": "median_peak_capture_pct",
            "control": control_peak["median_peak_capture_pct"],
            "directional_change": directional_change_peak["median_peak_capture_pct"],
            "bocpd": bocpd_peak["median_peak_capture_pct"],
        },
        {
            "metric": "median_days_from_peak_to_exit",
            "control": control_peak["median_days_from_peak_to_exit"],
            "directional_change": directional_change_peak["median_days_from_peak_to_exit"],
            "bocpd": bocpd_peak["median_days_from_peak_to_exit"],
        },
    ]
)
comparison_table.to_csv(
    DIRETORIO_RESULTADOS / "comparison_directional_change.csv",
    index=False,
)

with (DIRETORIO_RESULTADOS / "comparison_directional_change.json").open(
    "w",
    encoding="utf-8",
) as arquivo:
    json.dump(
        {
            "comparison": comparacao,
            "control_metrics": control_metrics,
            "directional_change_metrics": directional_change_metrics,
            "bocpd_metrics": bocpd_metrics,
            "control_peak": control_peak,
            "directional_change_peak": directional_change_peak,
            "bocpd_peak": bocpd_peak,
            "structural_exclusions": exclusoes,
            "snapshot_sha256": manifesto.get("snapshot_sha256"),
        },
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print(f"[output] diretorio={DIRETORIO_RESULTADOS}", flush=True)


# %% 10 - PACOTE ZIP PARA ANALISE
# Este e o unico arquivo que voce precisa enviar para analise.
PACOTE_ANALISE = criar_pacote_analise(DIRETORIO_RESULTADOS)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)


# %% 11 - SINAL SONORO DE CONCLUSAO
# Dois tons no Windows. Em outros sistemas tenta o bell do terminal.
sinal_sonoro_conclusao()
print("[done] pesquisa e pacote de analise concluidos", flush=True)
