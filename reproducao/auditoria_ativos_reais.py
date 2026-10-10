"""Auditoria reproduzivel de decisoes usando ativos reais do snapshot U67.

O modulo seleciona casos reais por uma regra cronologica independente do
resultado financeiro: a primeira posicao concluida de cada ativo-alvo. Em cada
caso, compara a trajetoria de precos dos ativos de foco usando exatamente os
frames preparados que alimentam o replay oficial.

A auditoria e descritiva. Ela nao altera treinamento, ranking, politica,
custos, execucao ou capital do experimento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CASE_ASSETS = ("TSLA", "NVDA", "VNCE")
DEFAULT_FOCUS_ASSETS = ("TSLA", "NVDA", "VNCE", "SPY")


def _timestamp_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _iso(value: Any) -> str | None:
    timestamp = _timestamp_utc(value)
    return timestamp.isoformat() if timestamp is not None else None


def _matching_buy(
    trades: pd.DataFrame,
    *,
    asset: str,
    entry_timestamp: pd.Timestamp,
    exit_timestamp: pd.Timestamp,
) -> pd.Series | None:
    if trades.empty:
        return None

    actions = trades["action"].astype(str).str.upper()
    assets = trades["asset"].astype(str).str.upper()
    timestamps = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    candidates = trades.loc[
        (actions == "BUY")
        & (assets == asset)
        & (timestamps == entry_timestamp)
        & (timestamps < exit_timestamp)
    ]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def selecionar_casos_cronologicos(
    trades: pd.DataFrame,
    *,
    case_assets: tuple[str, ...] = DEFAULT_CASE_ASSETS,
) -> pd.DataFrame:
    """Seleciona a primeira posicao concluida de cada ativo-alvo.

    A regra usa somente a ordem cronologica das operacoes. Retorno, PnL, MFE,
    MAE e custo de oportunidade nao participam da selecao do exemplo.
    """

    required = {"timestamp", "action", "asset", "entry_timestamp"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(
            "Trades sem colunas obrigatorias para a auditoria: "
            + ", ".join(sorted(missing))
        )

    frame = trades.copy()
    frame["_timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["_entry_timestamp"] = pd.to_datetime(
        frame["entry_timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["_action"] = frame["action"].astype(str).str.upper()
    frame["_asset"] = frame["asset"].astype(str).str.upper()
    frame = frame.sort_values("_timestamp").reset_index(drop=True)

    cases: list[dict[str, Any]] = []
    for case_number, requested_asset in enumerate(case_assets, start=1):
        asset = str(requested_asset).strip().upper()
        sells = frame.loc[
            (frame["_action"] == "SELL")
            & (frame["_asset"] == asset)
            & frame["_timestamp"].notna()
            & frame["_entry_timestamp"].notna()
        ]
        if sells.empty:
            continue

        sell = sells.iloc[0]
        entry_timestamp = pd.Timestamp(sell["_entry_timestamp"])
        exit_timestamp = pd.Timestamp(sell["_timestamp"])
        buy = _matching_buy(
            frame,
            asset=asset,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
        )
        if buy is None:
            continue

        entry_decision = _timestamp_utc(buy.get("decision_timestamp"))
        exit_decision = _timestamp_utc(sell.get("decision_timestamp"))
        if entry_decision is None or exit_decision is None:
            continue

        cases.append(
            {
                "case_id": f"caso_{case_number:02d}_{asset.lower()}",
                "asset": asset,
                "selection_rule": (
                    "first_completed_position_chronologically"
                ),
                "entry_decision_timestamp": entry_decision,
                "entry_execution_timestamp": entry_timestamp,
                "exit_decision_timestamp": exit_decision,
                "exit_execution_timestamp": exit_timestamp,
                "entry_price": _number(sell.get("entry_price")),
                "exit_price": _number(sell.get("execution_price")),
                "holding_bars": _number(sell.get("holding_bars")),
                "position_return": _number(sell.get("position_return")),
                "realized_pnl": _number(sell.get("realized_pnl")),
                "walk_forward_fold": _number(sell.get("walk_forward_fold")),
                "entry_score": _number(buy.get("final_action_score")),
                "entry_top_1_asset": _text(buy.get("top_1_asset")),
                "entry_top_1_score": _number(buy.get("top_1_score")),
                "entry_top_2_asset": _text(buy.get("top_2_asset")),
                "entry_top_2_score": _number(buy.get("top_2_score")),
                "entry_top_3_asset": _text(buy.get("top_3_asset")),
                "entry_top_3_score": _number(buy.get("top_3_score")),
                "exit_current_score": _number(sell.get("current_score")),
                "exit_best_asset": _text(sell.get("best_asset")),
                "exit_best_score": _number(sell.get("best_score")),
                "exit_best_vs_current_gap": _number(
                    sell.get("best_vs_current_gap")
                ),
                "rotation_id": _text(sell.get("rotation_id")),
            }
        )

    return pd.DataFrame(cases)


def _common_dates_utc(common_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(
        pd.to_datetime(common_dates, utc=True, errors="coerce")
    )
    dates = dates[~dates.isna()]
    return dates.drop_duplicates().sort_values()


def _position_of_date(
    dates: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
) -> int:
    normalized = timestamp.normalize()
    normalized_dates = dates.normalize()
    locations = normalized_dates.get_indexer([normalized])
    position = int(locations[0])
    if position < 0:
        raise ValueError(
            f"Data {timestamp.isoformat()} nao encontrada no calendario comum."
        )
    return position


def construir_movimentos_casos(
    cases: pd.DataFrame,
    *,
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    focus_assets: tuple[str, ...] = DEFAULT_FOCUS_ASSETS,
    sessions_before: int = 5,
    sessions_after: int = 3,
) -> pd.DataFrame:
    """Constroi trajetorias reais alinhadas ao calendario do experimento."""

    if cases.empty:
        return pd.DataFrame()

    dates = _common_dates_utc(common_dates)
    focus = tuple(str(asset).strip().upper() for asset in focus_assets)
    missing_assets = [asset for asset in focus if asset not in frames]
    if missing_assets:
        raise ValueError(
            "Ativos de foco ausentes dos frames do experimento: "
            + ", ".join(missing_assets)
        )

    rows: list[dict[str, Any]] = []
    for case in cases.to_dict(orient="records"):
        decision = _timestamp_utc(case["entry_decision_timestamp"])
        entry = _timestamp_utc(case["entry_execution_timestamp"])
        exit_decision = _timestamp_utc(case["exit_decision_timestamp"])
        exit_execution = _timestamp_utc(case["exit_execution_timestamp"])
        if None in (decision, entry, exit_decision, exit_execution):
            continue

        start_pos = max(
            0,
            _position_of_date(dates, decision) - int(sessions_before),
        )
        end_pos = min(
            len(dates) - 1,
            _position_of_date(dates, exit_execution) + int(sessions_after),
        )
        window = dates[start_pos : end_pos + 1]

        for asset in focus:
            asset_frame = frames[asset].copy()
            asset_frame.index = pd.to_datetime(
                asset_frame.index,
                utc=True,
                errors="coerce",
            )
            asset_frame = asset_frame.sort_index()
            if decision not in asset_frame.index:
                raise ValueError(
                    f"{asset} sem barra na data de decisao "
                    f"{decision.isoformat()}."
                )
            base_close = _number(asset_frame.loc[decision].get("close"))
            if base_close is None or base_close <= 0:
                raise ValueError(
                    f"{asset} sem fechamento valido na decisao "
                    f"{decision.isoformat()}."
                )

            for timestamp in window:
                if timestamp not in asset_frame.index:
                    continue
                bar = asset_frame.loc[timestamp]
                close = _number(bar.get("close"))
                if close is None or close <= 0:
                    continue

                if timestamp < decision:
                    phase = "before_entry_decision"
                elif timestamp == decision:
                    phase = "entry_decision"
                elif timestamp == entry:
                    phase = "entry_execution"
                elif timestamp < exit_decision:
                    phase = "holding"
                elif timestamp == exit_decision:
                    phase = "exit_decision"
                elif timestamp == exit_execution:
                    phase = "exit_execution"
                else:
                    phase = "after_exit"

                rows.append(
                    {
                        "case_id": case["case_id"],
                        "selected_asset": case["asset"],
                        "asset": asset,
                        "timestamp": timestamp,
                        "phase": phase,
                        "open": _number(bar.get("open")),
                        "high": _number(bar.get("high")),
                        "low": _number(bar.get("low")),
                        "close": close,
                        "volume": _number(bar.get("volume")),
                        "decision_close": base_close,
                        "normalized_close": close / base_close * 100.0,
                        "return_from_entry_decision_close": (
                            close / base_close - 1.0
                        ),
                        "available_at_entry_decision": bool(
                            timestamp <= decision
                        ),
                        "is_entry_decision": bool(timestamp == decision),
                        "is_entry_execution": bool(timestamp == entry),
                        "is_exit_decision": bool(timestamp == exit_decision),
                        "is_exit_execution": bool(
                            timestamp == exit_execution
                        ),
                    }
                )

    return pd.DataFrame(rows)


def construir_comparacao_janelas(
    cases: pd.DataFrame,
    *,
    frames: dict[str, pd.DataFrame],
    focus_assets: tuple[str, ...] = DEFAULT_FOCUS_ASSETS,
) -> pd.DataFrame:
    """Compara o retorno bruto open-to-open na mesma janela de cada caso."""

    rows: list[dict[str, Any]] = []
    focus = tuple(str(asset).strip().upper() for asset in focus_assets)

    for case in cases.to_dict(orient="records"):
        entry = _timestamp_utc(case["entry_execution_timestamp"])
        exit_execution = _timestamp_utc(case["exit_execution_timestamp"])
        if entry is None or exit_execution is None:
            continue

        selected_return = _number(case.get("position_return"))
        for asset in focus:
            frame = frames.get(asset)
            if frame is None or frame.empty:
                continue
            local = frame.copy()
            local.index = pd.to_datetime(
                local.index,
                utc=True,
                errors="coerce",
            )
            local = local.sort_index()
            if entry not in local.index or exit_execution not in local.index:
                continue

            entry_open = _number(local.loc[entry].get("open"))
            exit_open = _number(local.loc[exit_execution].get("open"))
            if (
                entry_open is None
                or exit_open is None
                or entry_open <= 0
                or exit_open <= 0
            ):
                continue

            gross_return = exit_open / entry_open - 1.0
            rows.append(
                {
                    "case_id": case["case_id"],
                    "selected_asset": case["asset"],
                    "asset": asset,
                    "is_selected_asset": bool(asset == case["asset"]),
                    "entry_execution_timestamp": entry,
                    "exit_execution_timestamp": exit_execution,
                    "entry_open": entry_open,
                    "exit_open": exit_open,
                    "gross_open_to_open_return": gross_return,
                    "selected_position_return": selected_return,
                    "gross_return_minus_selected": (
                        gross_return - selected_return
                        if selected_return is not None
                        else None
                    ),
                }
            )

    return pd.DataFrame(rows)


def _plot_case(
    output_dir: Path,
    *,
    case: dict[str, Any],
    movements: pd.DataFrame,
    focus_assets: tuple[str, ...],
) -> list[str]:
    case_rows = movements.loc[movements["case_id"] == case["case_id"]]
    if case_rows.empty:
        return []

    plt.figure(figsize=(10, 6))
    for asset in focus_assets:
        asset_rows = case_rows.loc[case_rows["asset"] == asset].sort_values(
            "timestamp"
        )
        if asset_rows.empty:
            continue
        plt.plot(
            asset_rows["timestamp"],
            asset_rows["normalized_close"],
            label=asset,
            linewidth=1.6,
        )

    decision = _timestamp_utc(case["entry_decision_timestamp"])
    entry = _timestamp_utc(case["entry_execution_timestamp"])
    exit_execution = _timestamp_utc(case["exit_execution_timestamp"])
    if decision is not None:
        plt.axvline(decision, linestyle="--", linewidth=1.0)
    if entry is not None:
        plt.axvline(entry, linestyle="-.", linewidth=1.0)
    if exit_execution is not None:
        plt.axvline(exit_execution, linestyle=":", linewidth=1.2)

    asset = str(case["asset"])
    plt.title(
        f"Caso real {case['case_id']}: primeira posicao concluida em {asset}"
    )
    plt.xlabel("Sessao")
    plt.ylabel("Fechamento normalizado (decisao de entrada = 100)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    base_name = f"{case['case_id']}_movimentos_reais"
    files = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{base_name}.{suffix}"
        plt.savefig(path, dpi=180 if suffix == "png" else None)
        files.append(path.name)
    plt.close()
    return files


def _format_pct(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.2%}"


def _format_money(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"US$ {number:,.2f}"


def _write_markdown(
    path: Path,
    *,
    cases: pd.DataFrame,
    comparison: pd.DataFrame,
    focus_assets: tuple[str, ...],
) -> None:
    lines = [
        "# Auditoria de decisoes com ativos reais",
        "",
        (
            "Os casos abaixo usam somente movimentos presentes no snapshot "
            "congelado do experimento. Nao ha ativo, preco ou retorno "
            "hipotetico."
        ),
        "",
        (
            "A regra de escolha dos exemplos e cronologica: para TSLA, NVDA "
            "e VNCE foi usada a primeira posicao concluida registrada no "
            "replay oficial. O resultado financeiro da operacao nao participa "
            "da selecao do caso."
        ),
        "",
        (
            "SPY participa como comparador de mercado nas mesmas janelas, "
            "mesmo que nao tenha sido selecionado pela politica."
        ),
        "",
        "| Caso | Ativo | Decisao entrada | Execucao entrada | Execucao saida "
        "| Retorno | PnL realizado | Fold |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for row in cases.to_dict(orient="records"):
        lines.append(
            "| {case_id} | {asset} | {entry_decision} | {entry_execution} "
            "| {exit_execution} | {position_return} | {pnl} | {fold} |".format(
                case_id=row["case_id"],
                asset=row["asset"],
                entry_decision=_iso(row["entry_decision_timestamp"]) or "",
                entry_execution=_iso(row["entry_execution_timestamp"]) or "",
                exit_execution=_iso(row["exit_execution_timestamp"]) or "",
                position_return=_format_pct(row.get("position_return")),
                pnl=_format_money(row.get("realized_pnl")),
                fold=(
                    int(row["walk_forward_fold"])
                    if _number(row.get("walk_forward_fold")) is not None
                    else ""
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Ranking registrado nas decisoes",
            "",
            "| Caso | Ativo escolhido | Top 1 na entrada | Top 2 na entrada | "
            "Top 3 na entrada | Melhor ativo na saida | Gap melhor-atual |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in cases.to_dict(orient="records"):
        def ranked(asset_key: str, score_key: str) -> str:
            asset_name = _text(row.get(asset_key))
            score = _number(row.get(score_key))
            if asset_name is None:
                return ""
            return (
                f"{asset_name} ({score:.6f})"
                if score is not None
                else asset_name
            )

        gap = _number(row.get("exit_best_vs_current_gap"))
        lines.append(
            "| {case_id} | {asset} | {top1} | {top2} | {top3} | "
            "{best_exit} | {gap} |".format(
                case_id=row["case_id"],
                asset=row["asset"],
                top1=ranked("entry_top_1_asset", "entry_top_1_score"),
                top2=ranked("entry_top_2_asset", "entry_top_2_score"),
                top3=ranked("entry_top_3_asset", "entry_top_3_score"),
                best_exit=ranked("exit_best_asset", "exit_best_score"),
                gap="" if gap is None else f"{gap:.6f}",
            )
        )

    lines.extend(
        [
            "",
            "## Movimento dos mesmos ativos na janela executada",
            "",
            (
                "A tabela seguinte calcula, para TSLA, NVDA, VNCE e SPY, o "
                "retorno bruto entre a abertura da entrada e a abertura da "
                "saida do caso. E uma comparacao ex post da mesma janela, "
                "sem taxas e sem afirmar que os outros ativos poderiam ter "
                "sido escolhidos com conhecimento do desfecho."
            ),
            "",
            "| Caso | Ativo | Selecionado | Retorno bruto na mesma janela | "
            "Diferenca para o selecionado |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in comparison.to_dict(orient="records"):
        lines.append(
            "| {case_id} | {asset} | {selected} | {ret} | {delta} |".format(
                case_id=row["case_id"],
                asset=row["asset"],
                selected="sim" if bool(row["is_selected_asset"]) else "nao",
                ret=_format_pct(row.get("gross_open_to_open_return")),
                delta=_format_pct(row.get("gross_return_minus_selected")),
            )
        )

    lines.extend(
        [
            "",
            "Ativos comparados em cada janela: "
            + ", ".join(focus_assets)
            + ".",
            "",
            (
                "As linhas verticais dos graficos marcam decisao de entrada, "
                "execucao de entrada e execucao de saida. Os precos sao "
                "normalizados para 100 no fechamento da sessao da decisao de "
                "entrada. Valores posteriores a essa sessao sao exibidos "
                "apenas para descrever o que ocorreu depois; eles nao sao "
                "tratados como informacao disponivel na decisao."
            ),
            "",
            (
                "Esta auditoria nao mede contribuicao causal de um ativo ou "
                "feature e nao altera o resultado oficial. Ela apenas torna "
                "observaveis, com dados reais, algumas decisoes efetivamente "
                "registradas pelo replay."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def gerar_auditoria_ativos_reais(
    output_dir: Path,
    *,
    trades: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    case_assets: tuple[str, ...] = DEFAULT_CASE_ASSETS,
    focus_assets: tuple[str, ...] = DEFAULT_FOCUS_ASSETS,
    sessions_before: int = 5,
    sessions_after: int = 3,
) -> dict[str, Any]:
    """Gera tabelas, relatorio e graficos dos casos reais."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = selecionar_casos_cronologicos(
        trades,
        case_assets=case_assets,
    )
    if len(cases) != len(case_assets):
        found = set(cases.get("asset", pd.Series(dtype=str)).astype(str))
        missing = [asset for asset in case_assets if asset not in found]
        raise RuntimeError(
            "Nao foi possivel localizar um caso completo para: "
            + ", ".join(missing)
        )

    movements = construir_movimentos_casos(
        cases,
        frames=frames,
        common_dates=common_dates,
        focus_assets=focus_assets,
        sessions_before=sessions_before,
        sessions_after=sessions_after,
    )
    comparison = construir_comparacao_janelas(
        cases,
        frames=frames,
        focus_assets=focus_assets,
    )

    cases_csv = output_dir / "casos_ativos_reais.csv"
    movements_csv = output_dir / "movimentos_ativos_reais.csv"
    comparison_csv = output_dir / "comparacao_janela_ativos_reais.csv"
    report_md = output_dir / "auditoria_ativos_reais.md"
    metadata_json = output_dir / "auditoria_ativos_reais.json"

    cases_export = cases.copy()
    for column in (
        "entry_decision_timestamp",
        "entry_execution_timestamp",
        "exit_decision_timestamp",
        "exit_execution_timestamp",
    ):
        cases_export[column] = cases_export[column].map(_iso)
    cases_export.to_csv(cases_csv, index=False)

    movements_export = movements.copy()
    if not movements_export.empty:
        movements_export["timestamp"] = movements_export["timestamp"].map(_iso)
    movements_export.to_csv(movements_csv, index=False)

    comparison_export = comparison.copy()
    for column in (
        "entry_execution_timestamp",
        "exit_execution_timestamp",
    ):
        if column in comparison_export:
            comparison_export[column] = comparison_export[column].map(_iso)
    comparison_export.to_csv(comparison_csv, index=False)

    figure_files: list[str] = []
    for case in cases.to_dict(orient="records"):
        figure_files.extend(
            _plot_case(
                output_dir,
                case=case,
                movements=movements,
                focus_assets=focus_assets,
            )
        )

    _write_markdown(
        report_md,
        cases=cases,
        comparison=comparison,
        focus_assets=focus_assets,
    )

    metadata = {
        "schema_version": 1,
        "purpose": "descriptive_real_asset_decision_audit",
        "selection_rule": "first_completed_position_chronologically",
        "case_assets": list(case_assets),
        "focus_assets": list(focus_assets),
        "sessions_before_entry_decision": int(sessions_before),
        "sessions_after_exit_execution": int(sessions_after),
        "case_count": int(len(cases)),
        "movement_rows": int(len(movements)),
        "comparison_rows": int(len(comparison)),
        "uses_same_prepared_frames_as_official_replay": True,
        "outcome_used_to_select_cases": False,
        "hypothetical_asset_used": False,
        "files": {
            "cases": cases_csv.name,
            "movements": movements_csv.name,
            "same_window_comparison": comparison_csv.name,
            "report": report_md.name,
            "figures": figure_files,
        },
    }
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
