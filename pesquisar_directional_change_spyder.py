"""Sensibilidade de universo do Top-Turn no TCC.

Execute no Spyder por celulas (# %%). Esta campanha:
- usa somente o snapshot congelado versionado em dados/pesquisa;
- mantem modelos, folds e parametros congelados;
- compara Control e Top-Turn nos universos U54, U55 e U56;
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
    EXPECTED_EXECUTION_SCHEMA,
    RESEARCH_VERSION,
    calcular_peak_exit,
    criar_pacote_analise,
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
    prepare_model_frames,
)

RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHOS = SnapshotPaths.research(RAIZ_PROJETO)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"
DIRETORIO_GRAFICOS = DIRETORIO_RESULTADOS / "graficos"
EXECUTION_SCHEMA = "universe-sensitivity-54-55-56-v1"


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
    # Primeiro normaliza em UTC; depois remove o timezone deliberadamente,
    # pois Period[M] representa apenas ano/mes e nao carrega fuso horario.
    normalized = pd.to_datetime(index, utc=True).tz_localize(None)
    return normalized.to_period("M")


def _month_period(value) -> pd.Period:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.tz_localize(None).to_period("M")


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
            period = _month_period(row["timestamp"])
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
    top_turn_result,
    top_cooldown5_result,
    top_bottom_v2_result,
    frames_alinhados,
) -> list[Path]:
    DIRETORIO_GRAFICOS.mkdir(parents=True, exist_ok=True)
    for antigo in DIRETORIO_GRAFICOS.glob("*.png"):
        antigo.unlink()

    gerados: list[Path] = []
    strategies = (
        ("Control", control_result),
        ("Top-Turn", top_turn_result),
        ("Top+Cooldown5", top_cooldown5_result),
        ("Top+Bottom-v2", top_bottom_v2_result),
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
    oos_start = _month_period(
        control_result.predictions.index.min()
    )
    oos_end = _month_period(
        control_result.predictions.index.max()
    )
    asset_monthly = asset_monthly.loc[
        (asset_monthly["period"] >= oos_start)
        & (asset_monthly["period"] <= oos_end)
    ].copy()

    trigger_specs = (
        (
            "TT↓",
            top_bottom_v2_result.predictions,
            "directional_change_exit_triggered",
        ),
        (
            "BT↑",
            top_bottom_v2_result.predictions,
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
                "(TT↓=saída Top-Turn, BT↑=entrada Bottom-Turn v2)"
            ),
            text_builder=asset_text,
            destino=destino,
        )
        if destino.exists():
            gerados.append(destino)

    return gerados


if EXECUTION_SCHEMA != EXPECTED_EXECUTION_SCHEMA:
    raise RuntimeError(
        "Script e modulo de pesquisa incompatíveis antes do replay: "
        f"script={EXECUTION_SCHEMA!r} "
        f"modulo={EXPECTED_EXECUTION_SCHEMA!r}. "
        "Atualize a branch e reinicie o kernel do Spyder."
    )


print("=" * 78, flush=True)
print("TCC - Top-Turn Universe Sensitivity Research", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print(f"script_path={Path(__file__).resolve()}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print(
    "comparacao=U54 vs U55_CLMT vs U56_RAW | CONTROL vs TOP_TURN",
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


# %% 2 - Campanha de sensibilidade do universo
UNIVERSE_SCENARIOS = (
    {
        "slug": "u54_current",
        "label": "U54 atual",
        "allow_structural_assets": frozenset(),
        "scientific_role": "baseline_current",
        "reference_comparison": True,
    },
    {
        "slug": "u55_clmt",
        "label": "U55 com CLMT restaurado",
        "allow_structural_assets": frozenset({"CLMT"}),
        "scientific_role": "historical_reference_candidate",
        "reference_comparison": True,
    },
    {
        "slug": "u56_raw",
        "label": "U56 original por ticker",
        "allow_structural_assets": frozenset({"CLMT", "DOC"}),
        "scientific_role": "diagnostic_only",
        "reference_comparison": False,
    },
)


def _structural_overrides(diagnostics):
    return [
        {
            "symbol": row.get("symbol"),
            "structural_issue": row.get("structural_issue"),
        }
        for row in diagnostics
        if bool(row.get("structural_override"))
    ]


def _executar_cenario_universo(spec):
    slug = str(spec["slug"])
    label = str(spec["label"])
    allowed = frozenset(spec["allow_structural_assets"])

    print(
        f"[universe] start slug={slug} label={label} "
        f"allow_structural={','.join(sorted(allowed)) or 'NONE'}",
        flush=True,
    )
    inicio = time.perf_counter()

    frames_local, exclusoes_local, diagnosticos_local, auditoria_local = (
        prepare_model_frames(
            CAMINHOS,
            assets=CONFIG.assets,
            comparar_snapshot_referencia=bool(
                spec["reference_comparison"]
            ),
            allow_structural_assets=allowed,
        )
    )
    overrides_local = _structural_overrides(diagnosticos_local)

    expected_count = {
        "u54_current": 54,
        "u55_clmt": 55,
        "u56_raw": 56,
    }[slug]
    if len(frames_local) != expected_count:
        raise RuntimeError(
            f"{slug}: esperados {expected_count} ativos, "
            f"obtidos {len(frames_local)}."
        )

    config_local, _ = build_variant_configs(frames_local, CONFIG)
    datas_comuns_local, folds_local = build_folds(
        frames_local,
        config_local,
    )

    control_result_local, control_metrics_local = run_variant(
        f"CONTROL_{slug.upper()}",
        frames_local,
        config_local,
        folds_local,
    )

    top_result_local = executar_directional_change_lightgbm(
        frames_local,
        config_local,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        progress_callback=lambda p, stage, completed: print(
            f"[top-turn:{slug}] progress={p:.1f}% "
            f"completed={completed} stage={stage}",
            flush=True,
        ),
        run_ablation=False,
    )
    top_metrics_local = summarize_metrics(
        top_result_local,
        folds_local,
        float(config_local.initial_capital),
    )
    for chave, valor in top_result_local.metrics.items():
        if str(chave).startswith("directional_change_"):
            top_metrics_local[str(chave)] = valor

    frames_alinhados_local, _, _ = preparar_painel_rotacao(
        frames_local,
        config_local,
    )
    control_peak_local, control_peak_trades_local = calcular_peak_exit(
        control_result_local.trades,
        frames_alinhados_local,
    )
    top_peak_local, top_peak_trades_local = calcular_peak_exit(
        top_result_local.trades,
        frames_alinhados_local,
    )

    control_capital = float(control_metrics_local["ending_capital"])
    top_capital = float(top_metrics_local["ending_capital"])
    top_vs_control = top_capital / control_capital - 1.0

    summary = {
        "slug": slug,
        "label": label,
        "scientific_role": spec["scientific_role"],
        "eligible_assets": len(frames_local),
        "assets": list(frames_local),
        "allowed_structural_assets": sorted(allowed),
        "structural_exclusions": exclusoes_local,
        "structural_overrides": overrides_local,
        "data_audit": auditoria_local,
        "common_dates": len(datas_comuns_local),
        "fold_count": len(folds_local),
        "first_common_date": str(datas_comuns_local.min().date()),
        "last_common_date": str(datas_comuns_local.max().date()),
        "control_metrics": control_metrics_local,
        "top_turn_metrics": top_metrics_local,
        "control_peak": control_peak_local,
        "top_turn_peak": top_peak_local,
        "top_turn_vs_control": top_vs_control,
    }

    print(
        f"[universe-result] {slug} assets={len(frames_local)} "
        f"CONTROL={control_capital:,.2f} "
        f"TOP_TURN={top_capital:,.2f} "
        f"TOP_vs_CONTROL={top_vs_control:+.4%} "
        f"seconds={time.perf_counter() - inicio:.3f}",
        flush=True,
    )

    return {
        "summary": summary,
        "frames": frames_local,
        "control_result": control_result_local,
        "top_result": top_result_local,
        "control_peak_trades": control_peak_trades_local,
        "top_peak_trades": top_peak_trades_local,
    }


resultados_universo = {}
for scenario in UNIVERSE_SCENARIOS:
    resultado = _executar_cenario_universo(scenario)
    resultados_universo[str(scenario["slug"])] = resultado


# %% 3 - Comparacao 54 vs 55 vs 56
summaries = {
    slug: item["summary"]
    for slug, item in resultados_universo.items()
}

u54 = summaries["u54_current"]
u55 = summaries["u55_clmt"]
u56 = summaries["u56_raw"]

print(
    "[comparison-universe] "
    f"U54_CONTROL={float(u54['control_metrics']['ending_capital']):,.2f} "
    f"U54_TOP={float(u54['top_turn_metrics']['ending_capital']):,.2f} "
    f"U55_CONTROL={float(u55['control_metrics']['ending_capital']):,.2f} "
    f"U55_TOP={float(u55['top_turn_metrics']['ending_capital']):,.2f} "
    f"U56_CONTROL={float(u56['control_metrics']['ending_capital']):,.2f} "
    f"U56_TOP={float(u56['top_turn_metrics']['ending_capital']):,.2f}",
    flush=True,
)

print(
    "[universe-deltas] "
    f"CLMT_effect_control="
    f"{float(u55['control_metrics']['ending_capital']) / float(u54['control_metrics']['ending_capital']) - 1.0:+.4%} "
    f"CLMT_effect_top="
    f"{float(u55['top_turn_metrics']['ending_capital']) / float(u54['top_turn_metrics']['ending_capital']) - 1.0:+.4%} "
    f"DOC_increment_control="
    f"{float(u56['control_metrics']['ending_capital']) / float(u55['control_metrics']['ending_capital']) - 1.0:+.4%} "
    f"DOC_increment_top="
    f"{float(u56['top_turn_metrics']['ending_capital']) / float(u55['top_turn_metrics']['ending_capital']) - 1.0:+.4%}",
    flush=True,
)


# %% 4 - Exportacao
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
for antigo in DIRETORIO_RESULTADOS.glob("*.csv"):
    antigo.unlink()
for antigo in DIRETORIO_RESULTADOS.glob("*.json"):
    antigo.unlink()

comparison_rows = []
for slug, item in resultados_universo.items():
    summary = item["summary"]
    control_metrics_local = summary["control_metrics"]
    top_metrics_local = summary["top_turn_metrics"]

    comparison_rows.append(
        {
            "universe": slug,
            "scientific_role": summary["scientific_role"],
            "eligible_assets": summary["eligible_assets"],
            "control_ending_capital": control_metrics_local[
                "ending_capital"
            ],
            "top_turn_ending_capital": top_metrics_local[
                "ending_capital"
            ],
            "top_turn_vs_control": summary["top_turn_vs_control"],
            "control_cagr": control_metrics_local["cagr"],
            "top_turn_cagr": top_metrics_local["cagr"],
            "control_sharpe": control_metrics_local["sharpe"],
            "top_turn_sharpe": top_metrics_local["sharpe"],
            "control_maxdd": control_metrics_local[
                "maximum_drawdown"
            ],
            "top_turn_maxdd": top_metrics_local[
                "maximum_drawdown"
            ],
            "control_worst_fold": control_metrics_local[
                "worst_fold_return"
            ],
            "top_turn_worst_fold": top_metrics_local[
                "worst_fold_return"
            ],
        }
    )

    item["control_result"].predictions.reset_index().to_csv(
        DIRETORIO_RESULTADOS / f"{slug}_control_predictions.csv",
        index=False,
    )
    item["control_result"].trades.to_csv(
        DIRETORIO_RESULTADOS / f"{slug}_control_trades.csv",
        index=False,
    )
    item["top_result"].predictions.reset_index().to_csv(
        DIRETORIO_RESULTADOS / f"{slug}_top_turn_predictions.csv",
        index=False,
    )
    item["top_result"].trades.to_csv(
        DIRETORIO_RESULTADOS / f"{slug}_top_turn_trades.csv",
        index=False,
    )
    item["control_peak_trades"].to_csv(
        DIRETORIO_RESULTADOS / f"{slug}_control_peak_exit.csv",
        index=False,
    )
    item["top_peak_trades"].to_csv(
        DIRETORIO_RESULTADOS / f"{slug}_top_turn_peak_exit.csv",
        index=False,
    )

comparison_table = pd.DataFrame(comparison_rows)
comparison_table.to_csv(
    DIRETORIO_RESULTADOS / "comparison_universe_sensitivity.csv",
    index=False,
)

with (
    DIRETORIO_RESULTADOS / "comparison_universe_sensitivity.json"
).open("w", encoding="utf-8") as arquivo:
    json.dump(
        {
            "research_version": RESEARCH_VERSION,
            "execution_schema": EXECUTION_SCHEMA,
            "snapshot_sha256": manifesto.get("snapshot_sha256"),
            "question": (
                "How do Control and Top-Turn change across the current "
                "54-asset universe, the historical 55-asset universe with "
                "CLMT restored, and the raw 56-ticker diagnostic universe?"
            ),
            "protocol": {
                "model_parameters_unchanged": True,
                "top_turn_parameters_unchanged": True,
                "snapshot_unchanged": True,
                "fold_method_unchanged": True,
                "u54_current": {
                    "allow_structural_assets": [],
                    "role": "current baseline",
                },
                "u55_clmt": {
                    "allow_structural_assets": ["CLMT"],
                    "role": (
                        "historical reference candidate; CLMT continuity "
                        "override only"
                    ),
                },
                "u56_raw": {
                    "allow_structural_assets": ["CLMT", "DOC"],
                    "role": (
                        "diagnostic only; DOC crosses a documented merger/"
                        "ticker-identity transition"
                    ),
                },
            },
            "reference_snapshot": {
                "reference_eligible_assets": 55,
                "reference_includes_clmt": True,
                "reference_excludes_doc": True,
            },
            "scenarios": summaries,
        },
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print(f"[output] diretorio={DIRETORIO_RESULTADOS}", flush=True)


# %% 5 - PACOTE ZIP PARA ANALISE
PACOTE_ANALISE = criar_pacote_analise(DIRETORIO_RESULTADOS)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)


# %% 6 - SINAL SONORO DE CONCLUSAO
sinal_sonoro_conclusao()
print("[done] sensibilidade do universo concluida", flush=True)
