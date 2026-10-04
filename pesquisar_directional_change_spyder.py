"""Contribuicao marginal de ativos para a inteligencia de rotacao do TCC.

Execute no Spyder por celulas (# %%). Esta campanha:
- usa somente o snapshot congelado versionado em dados/pesquisa;
- preserva modelos, folds, parametros e candidatos de switch margin;
- reproduz U54/U55/U56 e usa CLMT em um desenho fatorial 2x2;
- separa efeito de disponibilidade do ativo, efeito indireto de calibracao e
  interacao entre ambos;
- usa DOC como controle negativo de sensibilidade;
- gera ZIP compacto para analise e emite aviso sonoro ao final.
"""

# %% 0 - Imports e configuracao
from copy import deepcopy
from pathlib import Path
import json
import math
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
EXECUTION_SCHEMA = "rotation-contribution-factorial-v1"


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
print("TCC - Rotation Contribution Factorial Research", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print(f"script_path={Path(__file__).resolve()}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print(
    "comparacao=U54/U55/U56 + CLMT factorial 2x2 | CONTROL + TOP_TURN",
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


# %% 2 - Reproducao natural U54/U55/U56
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
        "scientific_role": "diagnostic_negative_control",
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


def _executar_par(
    *,
    slug,
    label,
    frames_local,
    config_local,
    folds_local,
    metadata,
):
    print(
        f"[scenario] start slug={slug} label={label}",
        flush=True,
    )
    inicio = time.perf_counter()

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
        **dict(metadata),
        "eligible_assets": len(frames_local),
        "assets": list(frames_local),
        "control_metrics": control_metrics_local,
        "top_turn_metrics": top_metrics_local,
        "control_peak": control_peak_local,
        "top_turn_peak": top_peak_local,
        "top_turn_vs_control": top_vs_control,
    }

    print(
        f"[scenario-result] {slug} assets={len(frames_local)} "
        f"CONTROL={control_capital:,.2f} "
        f"TOP_TURN={top_capital:,.2f} "
        f"TOP_vs_CONTROL={top_vs_control:+.4%} "
        f"seconds={time.perf_counter() - inicio:.3f}",
        flush=True,
    )

    return {
        "summary": summary,
        "frames": frames_local,
        "config": config_local,
        "folds": folds_local,
        "control_result": control_result_local,
        "top_result": top_result_local,
        "control_peak_trades": control_peak_trades_local,
        "top_peak_trades": top_peak_trades_local,
    }


def _executar_cenario_universo(spec):
    slug = str(spec["slug"])
    label = str(spec["label"])
    allowed = frozenset(spec["allow_structural_assets"])

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

    return _executar_par(
        slug=slug,
        label=label,
        frames_local=frames_local,
        config_local=config_local,
        folds_local=folds_local,
        metadata={
            "scientific_role": spec["scientific_role"],
            "allowed_structural_assets": sorted(allowed),
            "structural_exclusions": exclusoes_local,
            "structural_overrides": overrides_local,
            "data_audit": auditoria_local,
            "common_dates": len(datas_comuns_local),
            "fold_count": len(folds_local),
            "first_common_date": str(datas_comuns_local.min().date()),
            "last_common_date": str(datas_comuns_local.max().date()),
            "policy_margin_source": slug,
            "counterfactual": False,
        },
    )


resultados_universo = {}
for scenario in UNIVERSE_SCENARIOS:
    resultado = _executar_cenario_universo(scenario)
    resultados_universo[str(scenario["slug"])] = resultado


# %% 3 - Recuperacao das margens naturais por fold

def _margens_por_fold(result) -> dict[int, float]:
    predictions = result.predictions
    required = {"decision_fold_id", "calibrated_switch_margin"}
    missing = required.difference(predictions.columns)
    if missing:
        raise RuntimeError(
            "Predictions sem diagnostico de margem por fold: "
            + ", ".join(sorted(missing))
        )

    output: dict[int, float] = {}
    rows = predictions.dropna(
        subset=["decision_fold_id", "calibrated_switch_margin"]
    )
    for fold_id, group in rows.groupby("decision_fold_id"):
        values = pd.to_numeric(
            group["calibrated_switch_margin"],
            errors="coerce",
        ).dropna().unique()
        if len(values) != 1:
            raise RuntimeError(
                f"Fold {fold_id}: margem nao e unica: {values.tolist()}"
            )
        output[int(fold_id)] = float(values[0])
    if not output:
        raise RuntimeError("Nenhuma margem calibrada foi recuperada.")
    return output


margens_u54 = _margens_por_fold(
    resultados_universo["u54_current"]["control_result"]
)
margens_u55 = _margens_por_fold(
    resultados_universo["u55_clmt"]["control_result"]
)
margens_u56 = _margens_por_fold(
    resultados_universo["u56_raw"]["control_result"]
)

print(
    "[natural-margins] "
    f"U54={margens_u54} U55={margens_u55} U56={margens_u56}",
    flush=True,
)


# %% 4 - Contrafactuais fatoriais de CLMT

def _config_com_margens_forcadas(
    frames_local,
    margens_por_fold,
):
    config_local, _ = build_variant_configs(frames_local, CONFIG)
    settings = deepcopy(config_local.research_model_settings)
    settings["counterfactual_switch_margin_by_fold"] = {
        str(int(fold_id)): float(value)
        for fold_id, value in sorted(margens_por_fold.items())
    }
    return config_local.copiar_modelo(
        update={"research_model_settings": settings}
    )


def _executar_counterfactual(
    *,
    slug,
    label,
    asset_universe_slug,
    policy_margin_slug,
    forced_margins,
):
    base = resultados_universo[asset_universe_slug]
    frames_local = base["frames"]
    config_local = _config_com_margens_forcadas(
        frames_local,
        forced_margins,
    )
    datas_comuns_local, folds_local = build_folds(
        frames_local,
        config_local,
    )

    return _executar_par(
        slug=slug,
        label=label,
        frames_local=frames_local,
        config_local=config_local,
        folds_local=folds_local,
        metadata={
            "scientific_role": "factorial_counterfactual",
            "asset_universe_source": asset_universe_slug,
            "policy_margin_source": policy_margin_slug,
            "forced_switch_margin_by_fold": {
                str(k): float(v)
                for k, v in sorted(forced_margins.items())
            },
            "common_dates": len(datas_comuns_local),
            "fold_count": len(folds_local),
            "first_common_date": str(datas_comuns_local.min().date()),
            "last_common_date": str(datas_comuns_local.max().date()),
            "counterfactual": True,
        },
    )


resultados_counterfactual = {
    "u54_policy_u55": _executar_counterfactual(
        slug="u54_policy_u55",
        label="U54 ativos + politica calibrada no U55",
        asset_universe_slug="u54_current",
        policy_margin_slug="u55_clmt",
        forced_margins=margens_u55,
    ),
    "u55_policy_u54": _executar_counterfactual(
        slug="u55_policy_u54",
        label="U55 ativos + politica calibrada no U54",
        asset_universe_slug="u55_clmt",
        policy_margin_slug="u54_current",
        forced_margins=margens_u54,
    ),
}


# %% 5 - Decomposicao fatorial e controle negativo DOC
factorial_cells = {
    "asset0_policy0": resultados_universo["u54_current"],
    "asset0_policy1": resultados_counterfactual["u54_policy_u55"],
    "asset1_policy0": resultados_counterfactual["u55_policy_u54"],
    "asset1_policy1": resultados_universo["u55_clmt"],
}


def _capital_cell(cell, strategy):
    key = (
        "control_metrics"
        if strategy == "control"
        else "top_turn_metrics"
    )
    return float(factorial_cells[cell]["summary"][key]["ending_capital"])


def _decomposicao_fatorial(strategy):
    a00 = _capital_cell("asset0_policy0", strategy)
    a01 = _capital_cell("asset0_policy1", strategy)
    a10 = _capital_cell("asset1_policy0", strategy)
    a11 = _capital_cell("asset1_policy1", strategy)
    if min(a00, a01, a10, a11) <= 0:
        raise RuntimeError("Capital fatorial precisa ser positivo.")

    policy_log = math.log(a01 / a00)
    asset_log = math.log(a10 / a00)
    full_log = math.log(a11 / a00)
    interaction_log = full_log - policy_log - asset_log

    return {
        "strategy": strategy,
        "cell_asset0_policy0": a00,
        "cell_asset0_policy1": a01,
        "cell_asset1_policy0": a10,
        "cell_asset1_policy1": a11,
        "full_clmt_effect_pct": a11 / a00 - 1.0,
        "policy_only_effect_pct": a01 / a00 - 1.0,
        "investability_only_effect_pct": a10 / a00 - 1.0,
        "policy_effect_log": policy_log,
        "investability_effect_log": asset_log,
        "interaction_effect_log": interaction_log,
        "interaction_multiplier_pct": math.exp(interaction_log) - 1.0,
        "full_effect_log": full_log,
        "additive_interaction_capital": (
            a11 - a01 - a10 + a00
        ),
    }


def _path_difference(left_result, right_result):
    left = left_result.predictions[["selected_asset"]].copy()
    right = right_result.predictions[["selected_asset"]].copy()
    left.columns = ["left_asset"]
    right.columns = ["right_asset"]
    joined = left.join(right, how="inner")
    different = (
        joined["left_asset"].fillna("CASH").astype(str)
        != joined["right_asset"].fillna("CASH").astype(str)
    )
    first = (
        str(joined.index[different][0])
        if bool(different.any())
        else None
    )
    return {
        "comparable_sessions": int(len(joined)),
        "different_selected_asset_sessions": int(different.sum()),
        "different_selected_asset_rate": (
            float(different.mean()) if len(joined) else None
        ),
        "first_divergence": first,
    }


factorial_decomposition = {
    "control": _decomposicao_fatorial("control"),
    "top_turn": _decomposicao_fatorial("top_turn"),
}

policy_path_control = _path_difference(
    factorial_cells["asset0_policy0"]["control_result"],
    factorial_cells["asset0_policy1"]["control_result"],
)
policy_path_top = _path_difference(
    factorial_cells["asset0_policy0"]["top_result"],
    factorial_cells["asset0_policy1"]["top_result"],
)
doc_negative_control = {
    "control_capital_delta": (
        float(
            resultados_universo["u56_raw"]["summary"][
                "control_metrics"
            ]["ending_capital"]
        )
        - float(
            resultados_universo["u55_clmt"]["summary"][
                "control_metrics"
            ]["ending_capital"]
        )
    ),
    "top_turn_capital_delta": (
        float(
            resultados_universo["u56_raw"]["summary"][
                "top_turn_metrics"
            ]["ending_capital"]
        )
        - float(
            resultados_universo["u55_clmt"]["summary"][
                "top_turn_metrics"
            ]["ending_capital"]
        )
    ),
    "control_path": _path_difference(
        resultados_universo["u55_clmt"]["control_result"],
        resultados_universo["u56_raw"]["control_result"],
    ),
    "top_turn_path": _path_difference(
        resultados_universo["u55_clmt"]["top_result"],
        resultados_universo["u56_raw"]["top_result"],
    ),
    "margins_u55": {str(k): v for k, v in margens_u55.items()},
    "margins_u56": {str(k): v for k, v in margens_u56.items()},
}

print(
    "[factorial-control] "
    f"full={factorial_decomposition['control']['full_clmt_effect_pct']:+.4%} "
    f"policy_only={factorial_decomposition['control']['policy_only_effect_pct']:+.4%} "
    f"investability_only={factorial_decomposition['control']['investability_only_effect_pct']:+.4%} "
    f"interaction_multiplier={factorial_decomposition['control']['interaction_multiplier_pct']:+.4%}",
    flush=True,
)
print(
    "[factorial-top-turn] "
    f"full={factorial_decomposition['top_turn']['full_clmt_effect_pct']:+.4%} "
    f"policy_only={factorial_decomposition['top_turn']['policy_only_effect_pct']:+.4%} "
    f"investability_only={factorial_decomposition['top_turn']['investability_only_effect_pct']:+.4%} "
    f"interaction_multiplier={factorial_decomposition['top_turn']['interaction_multiplier_pct']:+.4%}",
    flush=True,
)
print(
    "[policy-path] "
    f"CONTROL_changed={policy_path_control['different_selected_asset_sessions']} "
    f"TOP_changed={policy_path_top['different_selected_asset_sessions']}",
    flush=True,
)


# %% 6 - Exportacao
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
for antigo in DIRETORIO_RESULTADOS.glob("*.csv"):
    antigo.unlink()
for antigo in DIRETORIO_RESULTADOS.glob("*.json"):
    antigo.unlink()
if DIRETORIO_GRAFICOS.exists():
    for antigo in DIRETORIO_GRAFICOS.glob("*.png"):
        antigo.unlink()

todos_resultados = {
    **resultados_universo,
    **resultados_counterfactual,
}

comparison_rows = []
for slug, item in todos_resultados.items():
    summary = item["summary"]
    control_metrics_local = summary["control_metrics"]
    top_metrics_local = summary["top_turn_metrics"]

    comparison_rows.append(
        {
            "scenario": slug,
            "scientific_role": summary.get("scientific_role"),
            "eligible_assets": summary["eligible_assets"],
            "policy_margin_source": summary.get("policy_margin_source"),
            "counterfactual": bool(summary.get("counterfactual", False)),
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
    DIRETORIO_RESULTADOS / "rotation_contribution_factorial.csv",
    index=False,
)

factorial_cell_rows = []
for cell, item in factorial_cells.items():
    summary = item["summary"]
    factorial_cell_rows.append(
        {
            "cell": cell,
            "scenario": summary["slug"],
            "clmt_investable": cell.startswith("asset1"),
            "u55_policy_margins": cell.endswith("policy1"),
            "control_ending_capital": summary[
                "control_metrics"
            ]["ending_capital"],
            "top_turn_ending_capital": summary[
                "top_turn_metrics"
            ]["ending_capital"],
        }
    )
pd.DataFrame(factorial_cell_rows).to_csv(
    DIRETORIO_RESULTADOS / "rotation_contribution_factorial_cells.csv",
    index=False,
)

with (
    DIRETORIO_RESULTADOS / "rotation_contribution_factorial.json"
).open("w", encoding="utf-8") as arquivo:
    json.dump(
        {
            "research_version": RESEARCH_VERSION,
            "execution_schema": EXECUTION_SCHEMA,
            "snapshot_sha256": manifesto.get("snapshot_sha256"),
            "question": (
                "Does CLMT improve the portfolio mainly by being an "
                "investable opportunity, by changing the calibrated rotation "
                "policy for the rest of the universe, or through interaction "
                "between both mechanisms?"
            ),
            "primary_causal_target": "Control",
            "secondary_robustness_target": "Top-Turn end-to-end",
            "protocol": {
                "snapshot_unchanged": True,
                "lightgbm_parameters_unchanged": True,
                "top_turn_parameters_unchanged": True,
                "fold_method_unchanged": True,
                "switch_margin_candidate_set_unchanged": True,
                "new_tuning_allowed": False,
                "factor_asset": {
                    "0": "U54; CLMT not investable",
                    "1": "U55; CLMT investable",
                },
                "factor_policy": {
                    "0": "natural U54 fold margins",
                    "1": "natural U55 fold margins",
                },
                "cells": {
                    "asset0_policy0": "U54 natural baseline",
                    "asset0_policy1": (
                        "information-only counterfactual: U54 investable "
                        "assets with U55 fold margins"
                    ),
                    "asset1_policy0": (
                        "investability-only counterfactual: U55 investable "
                        "assets with U54 fold margins"
                    ),
                    "asset1_policy1": "U55 natural full effect",
                },
                "doc_negative_control": (
                    "U56 keeps CLMT and adds raw DOC only as diagnostic; "
                    "DOC is not eligible for promotion to baseline."
                ),
            },
            "natural_margins": {
                "u54": {str(k): v for k, v in margens_u54.items()},
                "u55": {str(k): v for k, v in margens_u55.items()},
                "u56": {str(k): v for k, v in margens_u56.items()},
            },
            "factorial_decomposition": factorial_decomposition,
            "policy_only_path_change": {
                "control": policy_path_control,
                "top_turn": policy_path_top,
            },
            "doc_negative_control": doc_negative_control,
            "natural_universe_scenarios": {
                slug: item["summary"]
                for slug, item in resultados_universo.items()
            },
            "counterfactual_scenarios": {
                slug: item["summary"]
                for slug, item in resultados_counterfactual.items()
            },
            "interpretation_rule": (
                "A positive policy-only effect with CLMT non-investable is "
                "evidence of indirect rotation contribution. A positive "
                "investability-only effect is direct opportunity value. The "
                "interaction term quantifies the non-additive contribution. "
                "No candidate asset search is performed in this campaign."
            ),
        },
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print(f"[output] diretorio={DIRETORIO_RESULTADOS}", flush=True)


# %% 7 - PACOTE ZIP PARA ANALISE
PACOTE_ANALISE = criar_pacote_analise(DIRETORIO_RESULTADOS)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)


# %% 8 - SINAL SONORO DE CONCLUSAO
sinal_sonoro_conclusao()
print("[done] contribuicao marginal de rotacao concluida", flush=True)
