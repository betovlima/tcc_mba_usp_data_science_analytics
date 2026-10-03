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
import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.rotacao import preparar_painel_rotacao
from pesquisas.directional_change_lightgbm import (
    RESEARCH_VERSION,
    calcular_bottom_entry,
    calcular_peak_exit,
    comparar_control_directional_change,
    criar_pacote_analise,
    executar_bottom_turn_lightgbm,
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


def _month_key(index: pd.Index) -> pd.PeriodIndex:
    return pd.to_datetime(index, utc=True).to_period("M")


def _story_strategy_monthly(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame["period"] = _month_key(frame.index)
    frame["selected_asset"] = (
        frame["selected_asset"].fillna("CASH").astype(str)
    )
    frame["strategy_equity"] = pd.to_numeric(
        frame["strategy_equity"],
        errors="coerce",
    )

    rows: list[dict[str, object]] = []
    for period, month in frame.groupby("period", sort=True):
        equity = month["strategy_equity"].dropna()
        monthly_return = None
        if len(equity) >= 2 and float(equity.iloc[0]) > 0:
            monthly_return = float(
                equity.iloc[-1] / equity.iloc[0] - 1.0
            )

        counts = month["selected_asset"].value_counts()
        dominant = str(counts.index[0]) if not counts.empty else "CASH"
        exposure = (
            float(counts.iloc[0] / len(month))
            if len(month) and not counts.empty
            else 0.0
        )
        rotations = int(month["trade_action"].notna().sum())

        rows.append(
            {
                "period": period,
                "year": int(period.year),
                "month": int(period.month),
                "dominant_asset": dominant,
                "dominant_exposure": exposure,
                "monthly_return": monthly_return,
                "rotations": rotations,
            }
        )
    return pd.DataFrame(rows)


def _asset_monthly_returns(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset, frame in sorted(frames.items()):
        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if close.empty:
            continue
        local = pd.DataFrame({"close": close})
        local["period"] = _month_key(local.index)
        for period, month in local.groupby("period", sort=True):
            values = month["close"].dropna()
            if len(values) < 2 or float(values.iloc[0]) <= 0:
                continue
            rows.append(
                {
                    "asset": str(asset),
                    "period": period,
                    "year": int(period.year),
                    "month": int(period.month),
                    "monthly_return": float(
                        values.iloc[-1] / values.iloc[0] - 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _calendar_axes(
    data: pd.DataFrame,
    value_column: str,
) -> tuple[list[int], np.ndarray]:
    years = sorted(
        int(value)
        for value in data["year"].dropna().unique().tolist()
    )
    matrix = np.full((len(years), 12), np.nan, dtype=float)
    year_to_row = {year: index for index, year in enumerate(years)}
    for _, row in data.iterrows():
        year = int(row["year"])
        month = int(row["month"])
        value = row.get(value_column)
        if pd.notna(value):
            matrix[year_to_row[year], month - 1] = float(value)
    return years, matrix


def _plot_calendar_story(
    data: pd.DataFrame,
    *,
    title: str,
    text_builder,
    destino: Path,
) -> None:
    if data.empty:
        return

    years, values = _calendar_axes(data, "monthly_return")
    finite = values[np.isfinite(values)]
    limit = (
        float(np.nanpercentile(np.abs(finite), 90))
        if finite.size
        else 0.10
    )
    limit = max(limit, 0.05)

    fig, ax = plt.subplots(
        figsize=(16, max(5.5, 0.65 * len(years) + 2.5))
    )
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
    )
    labels = [
        "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
        "Jul", "Ago", "Set", "Out", "Nov", "Dez",
    ]
    ax.set_xticks(range(12), labels)
    ax.set_yticks(range(len(years)), [str(year) for year in years])
    ax.set_xlabel("Mês")
    ax.set_ylabel("Ano")
    ax.set_title(title)

    lookup = {
        (int(row["year"]), int(row["month"])): row
        for _, row in data.iterrows()
    }
    for row_index, year in enumerate(years):
        for month_index in range(12):
            row = lookup.get((year, month_index + 1))
            if row is None:
                continue
            text = text_builder(row)
            ax.text(
                month_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
            )

    fig.colorbar(image, ax=ax, label="Retorno mensal")
    _salvar_e_publicar_grafico(fig, destino)


def _trigger_counts_monthly(
    trigger_specs: tuple[tuple[str, pd.DataFrame, str], ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for short_label, predictions, trigger_column in trigger_specs:
        rows = _trigger_rows(predictions, trigger_column)
        if rows.empty:
            continue
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
        for _, row in rows.iterrows():
            asset = str(
                row.get("current_asset")
                or row.get("previous_asset")
                or row.get("selected_asset")
                or ""
            )
            if not asset or asset in {"nan", "CASH"}:
                continue
            period = pd.Timestamp(row["timestamp"]).to_period("M")
            records.append(
                {
                    "asset": asset,
                    "period": period,
                    "year": int(period.year),
                    "month": int(period.month),
                    "method": short_label,
                }
            )
    if not records:
        return pd.DataFrame(
            columns=["asset", "period", "year", "month", "method", "count"]
        )
    frame = pd.DataFrame(records)
    return (
        frame.groupby(
            ["asset", "period", "year", "month", "method"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )


def _gerar_graficos_comparacao(
    *,
    control_result,
    bottom_turn_result,
    top_turn_result,
    top_bottom_result,
    frames_alinhados,
) -> list[Path]:
    DIRETORIO_GRAFICOS.mkdir(parents=True, exist_ok=True)
    for antigo in DIRETORIO_GRAFICOS.glob("*.png"):
        antigo.unlink()

    gerados: list[Path] = []
    strategies = (
        ("Control", control_result),
        ("Bottom-Turn", bottom_turn_result),
        ("Top-Turn", top_turn_result),
        ("Top+Bottom", top_bottom_result),
    )

    for label, result in strategies:
        monthly = _story_strategy_monthly(result.predictions)
        destino = DIRETORIO_GRAFICOS / (
            "story_" + label.lower().replace("+", "_").replace("-", "_")
            + "_monthly.png"
        )

        def strategy_text(row):
            value = row.get("monthly_return")
            value_text = (
                f"{float(value):+.1%}"
                if pd.notna(value)
                else "n/a"
            )
            return (
                f"{row.get('dominant_asset', 'CASH')}\n"
                f"{value_text}\n"
                f"{float(row.get('dominant_exposure', 0.0)):.0%} exp"
            )

        _plot_calendar_story(
            monthly,
            title=f"{label}: ativo dominante e retorno por mês/ano",
            text_builder=strategy_text,
            destino=destino,
        )
        if destino.exists():
            gerados.append(destino)

    asset_monthly = _asset_monthly_returns(frames_alinhados)
    oos_start = pd.Timestamp(
        control_result.predictions.index.min()
    ).to_period("M")
    oos_end = pd.Timestamp(
        control_result.predictions.index.max()
    ).to_period("M")
    asset_monthly = asset_monthly.loc[
        (asset_monthly["period"] >= oos_start)
        & (asset_monthly["period"] <= oos_end)
    ].copy()

    trigger_specs = (
        (
            "TT↓",
            top_turn_result.predictions,
            "directional_change_exit_triggered",
        ),
        (
            "BT↑",
            bottom_turn_result.predictions,
            "bottom_turn_entry_triggered",
        ),
    )
    trigger_counts = _trigger_counts_monthly(trigger_specs)

    if not asset_monthly.empty:
        leaders = (
            asset_monthly.sort_values(
                ["period", "monthly_return"],
                ascending=[True, False],
            )
            .groupby("period", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        destino = DIRETORIO_GRAFICOS / "market_monthly_leaders.png"

        def leader_text(row):
            return f"{row['asset']}\n{float(row['monthly_return']):+.1%}"

        _plot_calendar_story(
            leaders,
            title="Melhor ativo de cada mês",
            text_builder=leader_text,
            destino=destino,
        )
        if destino.exists():
            gerados.append(destino)

        laggards = (
            asset_monthly.sort_values(
                ["period", "monthly_return"],
                ascending=[True, True],
            )
            .groupby("period", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        destino = DIRETORIO_GRAFICOS / "market_monthly_laggards.png"

        def laggard_text(row):
            return f"{row['asset']}\n{float(row['monthly_return']):+.1%}"

        _plot_calendar_story(
            laggards,
            title="Pior ativo de cada mês",
            text_builder=laggard_text,
            destino=destino,
        )
        if destino.exists():
            gerados.append(destino)

    trigger_assets = (
        sorted(trigger_counts["asset"].unique().tolist())
        if not trigger_counts.empty
        else []
    )
    for asset in trigger_assets:
        calendar = asset_monthly.loc[
            asset_monthly["asset"] == asset
        ].copy()
        if calendar.empty:
            continue

        counts_asset = trigger_counts.loc[
            trigger_counts["asset"] == asset
        ].copy()
        count_lookup: dict[tuple[int, int], list[str]] = {}
        for _, row in counts_asset.iterrows():
            key = (int(row["year"]), int(row["month"]))
            count = int(row["count"])
            tag = str(row["method"])
            count_lookup.setdefault(key, []).append(
                f"{tag}{count}" if count > 1 else tag
            )

        destino = DIRETORIO_GRAFICOS / f"asset_story_{asset}.png"

        def asset_text(row):
            key = (int(row["year"]), int(row["month"]))
            signals = " ".join(count_lookup.get(key, []))
            base = f"{float(row['monthly_return']):+.1%}"
            return f"{base}\n{signals}" if signals else base

        _plot_calendar_story(
            calendar,
            title=(
                f"{asset}: retorno mensal e ciclo de reversão "
                "(TT↓=saída Top-Turn, BT↑=entrada Bottom-Turn)"
            ),
            text_builder=asset_text,
            destino=destino,
        )
        if destino.exists():
            gerados.append(destino)

    return gerados


print("=" * 78, flush=True)
print("TCC - Top-Turn / Bottom-Turn Cycle Research", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print(
    "comparacao=CONTROL vs BOTTOM_TURN vs TOP_TURN vs TOP_BOTTOM",
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


# %% 6 - BOTTOM-TURN: melhora apenas CASH -> ativo
inicio_bottom = time.perf_counter()
bottom_turn_result = executar_bottom_turn_lightgbm(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    include_top_turn_exit=False,
    progress_callback=lambda p, stage, completed: print(
        f"[bottom-turn] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
)
bottom_turn_metrics = summarize_metrics(
    bottom_turn_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in bottom_turn_result.metrics.items():
    if str(chave).startswith("bottom_turn_"):
        bottom_turn_metrics[str(chave)] = valor

print(
    "[stage] BOTTOM_TURN "
    f"capital={bottom_turn_metrics['ending_capital']:,.2f} "
    f"sharpe={bottom_turn_metrics['sharpe']:.4f} "
    f"maxdd={bottom_turn_metrics['maximum_drawdown']:.4%} "
    f"entries={bottom_turn_metrics.get('bottom_turn_entry_triggers')} "
    f"blocked={bottom_turn_metrics.get('bottom_turn_entry_blocks')} "
    f"seconds={time.perf_counter() - inicio_bottom:.3f}",
    flush=True,
)


# %% 7 - TOP-TURN + BOTTOM-TURN: ciclo completo
inicio_top_bottom = time.perf_counter()
top_bottom_result = executar_bottom_turn_lightgbm(
    frames,
    config_control,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    include_top_turn_exit=True,
    progress_callback=lambda p, stage, completed: print(
        f"[top-bottom] progress={p:.1f}% completed={completed} stage={stage}",
        flush=True,
    ),
)
top_bottom_metrics = summarize_metrics(
    top_bottom_result,
    folds,
    float(config_control.initial_capital),
)
for chave, valor in top_bottom_result.metrics.items():
    if (
        str(chave).startswith("bottom_turn_")
        or str(chave).startswith("combined_top_turn_")
    ):
        top_bottom_metrics[str(chave)] = valor

print(
    "[stage] TOP_BOTTOM "
    f"capital={top_bottom_metrics['ending_capital']:,.2f} "
    f"sharpe={top_bottom_metrics['sharpe']:.4f} "
    f"maxdd={top_bottom_metrics['maximum_drawdown']:.4%} "
    f"bottom_entries={top_bottom_metrics.get('bottom_turn_entry_triggers')} "
    f"top_exits={top_bottom_metrics.get('combined_top_turn_exit_triggers')} "
    f"seconds={time.perf_counter() - inicio_top_bottom:.3f}",
    flush=True,
)


# %% 8 - reservado: BOCPD/HSMM/Hazard ficam congelados no historico
print(
    "[research-focus] BOCPD, HSMM e Hazard permanecem documentados, "
    "mas nao sao executados nesta campanha Bottom-Turn.",
    flush=True,
)


# %% 9 - Diagnosticos de topo e fundo
frames_alinhados, _, _ = preparar_painel_rotacao(
    frames,
    config_control,
)
oos_start = pd.Timestamp(control_result.predictions.index.min())

control_peak, control_peak_trades = calcular_peak_exit(
    control_result.trades,
    frames_alinhados,
)
top_turn_peak, top_turn_peak_trades = calcular_peak_exit(
    directional_change_result.trades,
    frames_alinhados,
)
bottom_turn_peak, bottom_turn_peak_trades = calcular_peak_exit(
    bottom_turn_result.trades,
    frames_alinhados,
)
top_bottom_peak, top_bottom_peak_trades = calcular_peak_exit(
    top_bottom_result.trades,
    frames_alinhados,
)

control_bottom, control_bottom_entries = calcular_bottom_entry(
    control_result.trades,
    frames_alinhados,
    oos_start=oos_start,
)
bottom_turn_bottom, bottom_turn_entries = calcular_bottom_entry(
    bottom_turn_result.trades,
    frames_alinhados,
    oos_start=oos_start,
)
top_turn_bottom, top_turn_entries = calcular_bottom_entry(
    directional_change_result.trades,
    frames_alinhados,
    oos_start=oos_start,
)
top_bottom_bottom, top_bottom_entries = calcular_bottom_entry(
    top_bottom_result.trades,
    frames_alinhados,
    oos_start=oos_start,
)

print(
    "[top-diagnostic] CONTROL "
    f"distance={control_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={control_peak.get('median_peak_capture_pct')}",
    flush=True,
)
print(
    "[top-diagnostic] TOP_TURN "
    f"distance={top_turn_peak.get('median_exit_distance_from_peak_pct')} "
    f"capture={top_turn_peak.get('median_peak_capture_pct')}",
    flush=True,
)
print(
    "[bottom-diagnostic] CONTROL "
    f"distance={control_bottom.get('median_entry_distance_from_bottom_pct')} "
    f"capture={control_bottom.get('median_bottom_capture_pct')} "
    f"days={control_bottom.get('median_days_from_bottom_to_entry')}",
    flush=True,
)
print(
    "[bottom-diagnostic] BOTTOM_TURN "
    f"distance={bottom_turn_bottom.get('median_entry_distance_from_bottom_pct')} "
    f"capture={bottom_turn_bottom.get('median_bottom_capture_pct')} "
    f"days={bottom_turn_bottom.get('median_days_from_bottom_to_entry')}",
    flush=True,
)
print(
    "[bottom-diagnostic] TOP_BOTTOM "
    f"distance={top_bottom_bottom.get('median_entry_distance_from_bottom_pct')} "
    f"capture={top_bottom_bottom.get('median_bottom_capture_pct')} "
    f"days={top_bottom_bottom.get('median_days_from_bottom_to_entry')}",
    flush=True,
)


# %% 10 - Comparacao final do ciclo
comparacao = comparar_control_directional_change(
    control_metrics,
    directional_change_metrics,
    control_peak,
    top_turn_peak,
)

print(
    "[comparison-top-turn] "
    f"CONTROL={float(control_metrics['ending_capital']):,.2f} "
    f"TOP_TURN={float(directional_change_metrics['ending_capital']):,.2f} "
    f"ratio="
    f"{float(directional_change_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%}",
    flush=True,
)
print(
    "[comparison-cycle] "
    f"CONTROL={float(control_metrics['ending_capital']):,.2f} "
    f"BOTTOM_TURN={float(bottom_turn_metrics['ending_capital']):,.2f} "
    f"TOP_TURN={float(directional_change_metrics['ending_capital']):,.2f} "
    f"TOP_BOTTOM={float(top_bottom_metrics['ending_capital']):,.2f} "
    f"BOTTOM_vs_CONTROL="
    f"{float(bottom_turn_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%} "
    f"TOP_vs_CONTROL="
    f"{float(directional_change_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%} "
    f"TOP_BOTTOM_vs_CONTROL="
    f"{float(top_bottom_metrics['ending_capital']) / float(control_metrics['ending_capital']) - 1.0:+.4%}",
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
