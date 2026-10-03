"""Pesquisa Directional Change + LightGBM no TCC.

Execute no Spyder por celulas (# %%). Esta pesquisa:
- usa somente o snapshot congelado versionado em dados/pesquisa;
- nao altera o Control oficial;
- compara Control vs Top-Turn vs BOCPD vs HSMM vs Hazard/Survival;
- gera ZIP compacto para analise;
- emite aviso sonoro quando todo o processamento termina.
"""

# %% 0 - Imports e configuracao
from pathlib import Path
import json
import time

try:
    from IPython import get_ipython

    _ipython = get_ipython()
    if _ipython is not None:
        _ipython.run_line_magic("matplotlib", "inline")
except Exception:
    _ipython = None

import matplotlib.pyplot as plt
import pandas as pd

from engine.configuracao import CONFIG
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.rotacao import preparar_painel_rotacao
from pesquisas.directional_change_lightgbm import (
    RESEARCH_VERSION,
    calcular_metricas_peak_gatilhos,
    calcular_peak_exit,
    comparar_control_directional_change,
    criar_pacote_analise,
    executar_bocpd_overlay,
    executar_directional_change_lightgbm,
    executar_hazard_survival_overlay,
    executar_hsmm_overlay,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import (
    build_folds,
    build_variant_configs,
    run_variant,
    summarize_metrics,
)
from reproducao.preparacao import (
    KNOWN_STRUCTURAL_EXCLUSIONS,
    prepare_model_frames,
)

RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHOS = SnapshotPaths.research(RAIZ_PROJETO)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"
DIRETORIO_GRAFICOS = DIRETORIO_RESULTADOS / "graficos"


def _trigger_rows(predictions: pd.DataFrame, trigger_column: str) -> pd.DataFrame:
    frame = predictions.copy()
    if trigger_column not in frame.columns:
        return pd.DataFrame()
    mask = frame[trigger_column].fillna(False).astype(bool)
    if not bool(mask.any()):
        return pd.DataFrame()
    output = frame.loc[mask].copy()
    output = output.reset_index()
    if "timestamp" not in output.columns:
        first = output.columns[0]
        output = output.rename(columns={first: "timestamp"})
    return output


def _salvar_e_publicar_grafico(fig, destino: Path) -> None:
    """Salva PNG e publica a figura no console/aba Plots do Spyder."""
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        try:
            plt.show(block=False)
        except Exception:
            pass
    plt.close(fig)


def _gerar_graficos_comparacao(
    *,
    control_result,
    top_turn_result,
    bocpd_result,
    hsmm_result,
    hazard_result,
    control_peak,
    top_turn_peak,
    bocpd_peak,
    hsmm_peak,
    hazard_peak,
    top_turn_peak_trades,
    bocpd_peak_trades,
    hsmm_peak_trades,
    hazard_peak_trades,
    frames_alinhados,
) -> list[Path]:
    DIRETORIO_GRAFICOS.mkdir(parents=True, exist_ok=True)
    for antigo in DIRETORIO_GRAFICOS.glob("*.png"):
        antigo.unlink()

    gerados: list[Path] = []

    # 1. Curvas de capital OOS.
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, result in (
        ("Control", control_result),
        ("Top-Turn", top_turn_result),
        ("BOCPD", bocpd_result),
        ("HSMM", hsmm_result),
        ("Hazard/Survival", hazard_result),
    ):
        curve = pd.to_numeric(
            result.predictions["strategy_equity"],
            errors="coerce",
        )
        ax.plot(curve.index, curve.values, label=label, linewidth=1.6)
    ax.set_title("Curvas de capital OOS")
    ax.set_xlabel("Data")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25)
    ax.legend()
    destino = DIRETORIO_GRAFICOS / "capital_comparison.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    # 2. Capital relativo ao Control.
    control_curve = pd.to_numeric(
        control_result.predictions["strategy_equity"],
        errors="coerce",
    )
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for label, result in (
        ("Top-Turn", top_turn_result),
        ("BOCPD", bocpd_result),
        ("HSMM", hsmm_result),
        ("Hazard/Survival", hazard_result),
    ):
        curve = pd.to_numeric(
            result.predictions["strategy_equity"],
            errors="coerce",
        ).reindex(control_curve.index)
        relative = (curve / control_curve - 1.0) * 100.0
        ax.plot(
            relative.index,
            relative.values,
            label=label,
            linewidth=1.5,
        )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Capital relativo ao Control")
    ax.set_xlabel("Data")
    ax.set_ylabel("Diferença percentual")
    ax.grid(True, alpha=0.25)
    ax.legend()
    destino = DIRETORIO_GRAFICOS / "relative_equity_vs_control.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    # 3. Linha do tempo dos gatilhos.
    trigger_specs = (
        (
            "Top-Turn",
            top_turn_result.predictions,
            "directional_change_exit_triggered",
        ),
        ("BOCPD", bocpd_result.predictions, "bocpd_exit_triggered"),
        ("HSMM", hsmm_result.predictions, "hsmm_exit_triggered"),
        (
            "Hazard/Survival",
            hazard_result.predictions,
            "hazard_exit_triggered",
        ),
    )
    fig, ax = plt.subplots(figsize=(15, 5.5))
    lanes = {
        "Top-Turn": 3,
        "BOCPD": 2,
        "HSMM": 1,
        "Hazard/Survival": 0,
    }
    marker_by_label = {
        "Top-Turn": "o",
        "BOCPD": "s",
        "HSMM": "^",
        "Hazard/Survival": "D",
    }
    for label, predictions, trigger_column in trigger_specs:
        rows = _trigger_rows(predictions, trigger_column)
        if rows.empty:
            continue
        y = [lanes[label]] * len(rows)
        ax.scatter(
            pd.to_datetime(rows["timestamp"], utc=True),
            y,
            label=label,
            marker=marker_by_label[label],
            s=55,
        )
        for _, row in rows.iterrows():
            asset = str(
                row.get("current_asset")
                or row.get("previous_asset")
                or row.get("selected_asset")
                or ""
            )
            if asset and asset != "nan":
                ax.annotate(
                    asset,
                    (
                        pd.Timestamp(row["timestamp"]),
                        lanes[label],
                    ),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=8,
                    rotation=45,
                )
    ax.set_yticks(
        [0, 1, 2, 3],
        labels=["Hazard/Survival", "HSMM", "BOCPD", "Top-Turn"],
    )
    ax.set_title("Linha do tempo dos gatilhos de saída OOS")
    ax.set_xlabel("Data")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper left")
    destino = DIRETORIO_GRAFICOS / "trigger_timeline.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    # 3. Distância mediana do topo.
    labels = [
        "Control",
        "Top-Turn",
        "BOCPD",
        "HSMM",
        "Hazard/Survival",
    ]
    peaks = [
        control_peak,
        top_turn_peak,
        bocpd_peak,
        hsmm_peak,
        hazard_peak,
    ]
    distance_values = [
        peak.get("median_exit_distance_from_peak_pct")
        for peak in peaks
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, distance_values)
    ax.set_title("Distância mediana da saída ao topo")
    ax.set_ylabel("Percentual")
    ax.grid(True, axis="y", alpha=0.25)
    destino = DIRETORIO_GRAFICOS / "peak_distance_comparison.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    # 4. Captura mediana do movimento até o topo.
    capture_values = [
        peak.get("median_peak_capture_pct")
        for peak in peaks
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, capture_values)
    ax.set_title("Captura mediana do movimento até o topo")
    ax.set_ylabel("Percentual")
    ax.grid(True, axis="y", alpha=0.25)
    destino = DIRETORIO_GRAFICOS / "peak_capture_comparison.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    # 6. Peak Exit somente nas saídas provocadas pelos overlays.
    trigger_peak = {
        "Top-Turn": calcular_metricas_peak_gatilhos(
            top_turn_peak_trades,
            top_turn_result.predictions,
            "directional_change_exit_triggered",
        ),
        "BOCPD": calcular_metricas_peak_gatilhos(
            bocpd_peak_trades,
            bocpd_result.predictions,
            "bocpd_exit_triggered",
        ),
        "HSMM": calcular_metricas_peak_gatilhos(
            hsmm_peak_trades,
            hsmm_result.predictions,
            "hsmm_exit_triggered",
        ),
        "Hazard/Survival": calcular_metricas_peak_gatilhos(
            hazard_peak_trades,
            hazard_result.predictions,
            "hazard_exit_triggered",
        ),
    }
    trigger_labels = list(trigger_peak)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        trigger_labels,
        [
            trigger_peak[label][
                "median_exit_distance_from_peak_pct"
            ]
            for label in trigger_labels
        ],
    )
    ax.set_title("Distância mediana do topo somente nos gatilhos")
    ax.set_ylabel("Percentual")
    ax.grid(True, axis="y", alpha=0.25)
    destino = DIRETORIO_GRAFICOS / "trigger_peak_distance.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        trigger_labels,
        [
            trigger_peak[label]["median_peak_capture_pct"]
            for label in trigger_labels
        ],
    )
    ax.set_title("Captura mediana do topo somente nos gatilhos")
    ax.set_ylabel("Percentual")
    ax.grid(True, axis="y", alpha=0.25)
    destino = DIRETORIO_GRAFICOS / "trigger_peak_capture.png"
    _salvar_e_publicar_grafico(fig, destino)
    gerados.append(destino)

    # 7. Preço por ativo com os gatilhos das quatro técnicas.
    triggers_by_method: dict[str, pd.DataFrame] = {}
    for label, predictions, trigger_column in trigger_specs:
        triggers_by_method[label] = _trigger_rows(
            predictions,
            trigger_column,
        )

    assets: set[str] = set()
    for rows in triggers_by_method.values():
        if rows.empty:
            continue
        for _, row in rows.iterrows():
            asset = str(
                row.get("current_asset")
                or row.get("previous_asset")
                or row.get("selected_asset")
                or ""
            )
            if asset and asset != "nan" and asset != "CASH":
                assets.add(asset)

    markers = {
        "Top-Turn": "o",
        "BOCPD": "s",
        "HSMM": "^",
        "Hazard/Survival": "D",
    }
    for asset in sorted(assets):
        frame = frames_alinhados.get(asset)
        if frame is None or frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(close.index, close.values, linewidth=1.25, label=f"{asset} close")

        for label, rows in triggers_by_method.items():
            if rows.empty:
                continue
            subset_rows: list[tuple[pd.Timestamp, float]] = []
            for _, row in rows.iterrows():
                row_asset = str(
                    row.get("current_asset")
                    or row.get("previous_asset")
                    or row.get("selected_asset")
                    or ""
                )
                if row_asset != asset:
                    continue
                timestamp = pd.Timestamp(row["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                if timestamp not in close.index:
                    nearest_position = close.index.get_indexer(
                        [timestamp],
                        method="nearest",
                    )[0]
                    if nearest_position < 0:
                        continue
                    timestamp = pd.Timestamp(close.index[nearest_position])
                value = close.loc[timestamp]
                if pd.notna(value):
                    subset_rows.append((timestamp, float(value)))

            if subset_rows:
                ax.scatter(
                    [item[0] for item in subset_rows],
                    [item[1] for item in subset_rows],
                    marker=markers[label],
                    s=65,
                    label=label,
                )

        ax.set_title(f"{asset}: preço e gatilhos de reversão")
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço de fechamento")
        ax.grid(True, alpha=0.25)
        ax.legend()
        destino = DIRETORIO_GRAFICOS / f"triggers_{asset}.png"
        _salvar_e_publicar_grafico(fig, destino)
        gerados.append(destino)

    return gerados


print("=" * 78, flush=True)
print("TCC - Directional Change + LightGBM", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print(
    "comparacao=CONTROL vs TOP_TURN vs BOCPD vs HSMM vs HAZARD_SURVIVAL",
    flush=True,
)
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
known_still_present = sorted(
    set(KNOWN_STRUCTURAL_EXCLUSIONS).intersection(frames)
)
if known_still_present:
    raise RuntimeError(
        "Exclusao estrutural nao aplicada para: "
        + ", ".join(known_still_present)
        + ". Reinicie o kernel do Spyder e execute novamente."
    )

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


# %% 7 - HSMM
inicio_hsmm = time.perf_counter()
hsmm_result = executar_hsmm_overlay(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=lambda p, stage, completed: print(
        f"[hsmm] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
)
hsmm_metrics = summarize_metrics(
    hsmm_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in hsmm_result.metrics.items():
    if str(chave).startswith("hsmm_"):
        hsmm_metrics[str(chave)] = valor

print(
    "[stage] HSMM "
    f"capital={hsmm_metrics['ending_capital']:,.2f} "
    f"sharpe={hsmm_metrics['sharpe']:.4f} "
    f"maxdd={hsmm_metrics['maximum_drawdown']:.4%} "
    f"triggers={hsmm_metrics.get('hsmm_exit_triggers')} "
    f"seconds={time.perf_counter() - inicio_hsmm:.3f}",
    flush=True,
)


# %% 8 - HAZARD / SURVIVAL
inicio_hazard = time.perf_counter()
hazard_result = executar_hazard_survival_overlay(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=lambda p, stage, completed: print(
        f"[hazard] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
)
hazard_metrics = summarize_metrics(
    hazard_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in hazard_result.metrics.items():
    if str(chave).startswith("hazard_"):
        hazard_metrics[str(chave)] = valor

print(
    "[stage] HAZARD_SURVIVAL "
    f"capital={hazard_metrics['ending_capital']:,.2f} "
    f"sharpe={hazard_metrics['sharpe']:.4f} "
    f"maxdd={hazard_metrics['maximum_drawdown']:.4%} "
    f"triggers={hazard_metrics.get('hazard_exit_triggers')} "
    f"seconds={time.perf_counter() - inicio_hazard:.3f}",
    flush=True,
)


# %% 9 - Peak Exit: mesma definicao para os cinco cenarios
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
hsmm_peak, hsmm_peak_trades = calcular_peak_exit(
    hsmm_result.trades,
    frames_alinhados,
)
hazard_peak, hazard_peak_trades = calcular_peak_exit(
    hazard_result.trades,
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
print(
    "[peak] HSMM "
    f"distance={hsmm_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={hsmm_peak.get('median_peak_capture_pct')}",
    flush=True,
)
print(
    "[peak] HAZARD_SURVIVAL "
    f"distance={hazard_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={hazard_peak.get('median_peak_capture_pct')}",
    flush=True,
)


# %% 10 - Comparacao final
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
    "[comparison-reversal] "
    f"CONTROL={float(control_metrics['ending_capital']):,.2f} "
    f"TOP_TURN={float(directional_change_metrics['ending_capital']):,.2f} "
    f"BOCPD={float(bocpd_metrics['ending_capital']):,.2f} "
    f"HSMM={float(hsmm_metrics['ending_capital']):,.2f} "
    f"HAZARD={float(hazard_metrics['ending_capital']):,.2f} "
    f"BOCPD_vs_CONTROL="
    f"{float(bocpd_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%} "
    f"HSMM_vs_CONTROL="
    f"{float(hsmm_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%} "
    f"HSMM_vs_TOP_TURN="
    f"{float(hsmm_metrics['ending_capital']) / float(directional_change_metrics['ending_capital']) - 1.0:+.4%} "
    f"HAZARD_vs_CONTROL="
    f"{float(hazard_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%} "
    f"HAZARD_vs_TOP_TURN="
    f"{float(hazard_metrics['ending_capital']) / float(directional_change_metrics['ending_capital']) - 1.0:+.4%}",
    flush=True,
)

# %% 11 - Exportacao dos artefatos
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
hsmm_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "hsmm_predictions.csv",
    index=False,
)
hsmm_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "hsmm_trades.csv",
    index=False,
)
hazard_result.predictions.reset_index().to_csv(
    DIRETORIO_RESULTADOS / "hazard_predictions.csv",
    index=False,
)
hazard_result.trades.to_csv(
    DIRETORIO_RESULTADOS / "hazard_trades.csv",
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
hsmm_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_hsmm.csv",
    index=False,
)
hazard_peak_trades.to_csv(
    DIRETORIO_RESULTADOS / "peak_exit_hazard.csv",
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

hsmm_calibration_rows = hsmm_result.metrics.get(
    "hsmm_calibration",
    [],
)
pd.DataFrame(hsmm_calibration_rows).to_csv(
    DIRETORIO_RESULTADOS / "hsmm_calibration.csv",
    index=False,
)

hazard_calibration_rows = hazard_result.metrics.get(
    "hazard_calibration",
    [],
)
pd.DataFrame(hazard_calibration_rows).to_csv(
    DIRETORIO_RESULTADOS / "hazard_calibration.csv",
    index=False,
)

comparison_table = pd.DataFrame(
    [
        {
            "metric": "ending_capital",
            "control": control_metrics["ending_capital"],
            "directional_change": directional_change_metrics["ending_capital"],
            "bocpd": bocpd_metrics["ending_capital"],
            "hsmm": hsmm_metrics["ending_capital"],
            "hazard_survival": hazard_metrics["ending_capital"],
        },
        {
            "metric": "cagr",
            "control": control_metrics["cagr"],
            "directional_change": directional_change_metrics["cagr"],
            "bocpd": bocpd_metrics["cagr"],
            "hsmm": hsmm_metrics["cagr"],
            "hazard_survival": hazard_metrics["cagr"],
        },
        {
            "metric": "sharpe",
            "control": control_metrics["sharpe"],
            "directional_change": directional_change_metrics["sharpe"],
            "bocpd": bocpd_metrics["sharpe"],
            "hsmm": hsmm_metrics["sharpe"],
            "hazard_survival": hazard_metrics["sharpe"],
        },
        {
            "metric": "maximum_drawdown",
            "control": control_metrics["maximum_drawdown"],
            "directional_change": directional_change_metrics["maximum_drawdown"],
            "bocpd": bocpd_metrics["maximum_drawdown"],
            "hsmm": hsmm_metrics["maximum_drawdown"],
            "hazard_survival": hazard_metrics["maximum_drawdown"],
        },
        {
            "metric": "worst_fold_return",
            "control": control_metrics["worst_fold_return"],
            "directional_change": directional_change_metrics["worst_fold_return"],
            "bocpd": bocpd_metrics["worst_fold_return"],
            "hsmm": hsmm_metrics["worst_fold_return"],
            "hazard_survival": hazard_metrics["worst_fold_return"],
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
            "hsmm": hsmm_peak[
                "median_exit_distance_from_peak_pct"
            ],
            "hazard_survival": hazard_peak[
                "median_exit_distance_from_peak_pct"
            ],
        },
        {
            "metric": "median_peak_capture_pct",
            "control": control_peak["median_peak_capture_pct"],
            "directional_change": directional_change_peak["median_peak_capture_pct"],
            "bocpd": bocpd_peak["median_peak_capture_pct"],
            "hsmm": hsmm_peak["median_peak_capture_pct"],
            "hazard_survival": hazard_peak[
                "median_peak_capture_pct"
            ],
        },
        {
            "metric": "median_days_from_peak_to_exit",
            "control": control_peak["median_days_from_peak_to_exit"],
            "directional_change": directional_change_peak["median_days_from_peak_to_exit"],
            "bocpd": bocpd_peak["median_days_from_peak_to_exit"],
            "hsmm": hsmm_peak["median_days_from_peak_to_exit"],
            "hazard_survival": hazard_peak[
                "median_days_from_peak_to_exit"
            ],
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
            "hsmm_metrics": hsmm_metrics,
            "hazard_metrics": hazard_metrics,
            "control_peak": control_peak,
            "directional_change_peak": directional_change_peak,
            "bocpd_peak": bocpd_peak,
            "hsmm_peak": hsmm_peak,
            "hazard_peak": hazard_peak,
            "structural_exclusions": exclusoes,
            "snapshot_sha256": manifesto.get("snapshot_sha256"),
        },
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

graficos_gerados = _gerar_graficos_comparacao(
    control_result=control_result,
    top_turn_result=directional_change_result,
    bocpd_result=bocpd_result,
    hsmm_result=hsmm_result,
    hazard_result=hazard_result,
    control_peak=control_peak,
    top_turn_peak=directional_change_peak,
    bocpd_peak=bocpd_peak,
    hsmm_peak=hsmm_peak,
    hazard_peak=hazard_peak,
    top_turn_peak_trades=directional_change_peak_trades,
    bocpd_peak_trades=bocpd_peak_trades,
    hsmm_peak_trades=hsmm_peak_trades,
    hazard_peak_trades=hazard_peak_trades,
    frames_alinhados=frames_alinhados,
)
print(
    f"[graphs] gerados={len(graficos_gerados)} "
    f"diretorio={DIRETORIO_GRAFICOS}",
    flush=True,
)

print(f"[output] diretorio={DIRETORIO_RESULTADOS}", flush=True)


# %% 12 - PACOTE ZIP PARA ANALISE
# Este e o unico arquivo que voce precisa enviar para analise.
PACOTE_ANALISE = criar_pacote_analise(DIRETORIO_RESULTADOS)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)


# %% 13 - SINAL SONORO DE CONCLUSAO
# Dois tons no Windows. Em outros sistemas tenta o bell do terminal.
sinal_sonoro_conclusao()
print("[done] pesquisa e pacote de analise concluidos", flush=True)
