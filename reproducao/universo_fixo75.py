"""Executor experimental: somente 75 ativos elegiveis, unico universo Control.

Nao chama exploracao, selecao ou expansao de ativos. Utiliza as mesmas
funcoes matematicas de treino/rotacao do checkpoint historico, porem ajusta
os modelos de TODOS os 75 ativos na mesma etapa e calibra a politica em
conjunto. Isso e um experimento novo; nao e a reproducao v1.23.0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

import pandas as pd

from engine.configuracao import U67_EXPECTED_EXCLUSIONS
from reproducao.artefatos import criar_pacote_analise, sinal_sonoro_conclusao
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config
from reproducao.graficos import gerar_analises_backtest
from reproducao.preparacao import prepare_model_frames


def universo_75(manifest: dict[str, Any], base_assets: tuple[str, ...]) -> tuple[str, ...]:
    """Deriva uma vez a composicao historica vencedora, sem ativos excluidos."""
    from engine.configuracao import U67_REQUESTED_ASSETS

    candidates = tuple(manifest["selected_assets"])
    requested = (*U67_REQUESTED_ASSETS, *candidates)
    selected = tuple(asset for asset in requested if asset not in U67_EXPECTED_EXCLUSIONS)
    if len(requested) != 77 or len(set(requested)) != 77:
        raise RuntimeError("Contrato historico de identidade divergente.")
    if len(selected) != 75 or len(set(selected)) != 75:
        raise RuntimeError("O unico universo da pesquisa deve conter 75 ativos.")
    if base_assets and selected != base_assets:
        raise RuntimeError("Lista de ativos diferente do universo vencedor fixo.")
    return selected


def reproduzir_universo_fixo75(
    *,
    root: Path,
    output: Path,
    snapshot_root: Path,
    analysis_end_date: str,
    version: str = "1.23.7",
    execution_schema: str = "single-fixed-75-v1",
) -> dict[str, Any]:
    """Treina e simula todos os 75 ativos em conjunto; nao ha base 65+10."""
    import pesquisar_expansao_u67_spyder as motor
    from reproducao.experimento import summarize_metrics

    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    source = SnapshotPaths.from_root(Path(snapshot_root).resolve())
    snapshot = validate_snapshot(source)
    original_contract = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    selected = universo_75(original_contract, ())
    if tuple(snapshot.get("assets", ())) != selected:
        raise RuntimeError("O snapshot deve conter somente os 75 ativos elegiveis.")
    if snapshot["bars"]["bar_snapshot_as_of_end"] < analysis_end_date:
        raise RuntimeError("Data de analise posterior ao ultimo pregao coletado.")

    prepared, exclusions, data_diagnostics, data_audit = prepare_model_frames(
        source, assets=selected, csv_float_precision="round_trip",
    )
    if exclusions or set(prepared) != set(selected) or len(prepared) != 75:
        raise RuntimeError(
            "Universo incompleto ou exclusao estrutural inesperada; "
            "a pesquisa nao modifica silenciosamente os 75 ativos."
        )
    config = build_control_config(prepared, motor.CONFIG).copiar_modelo(
        update={"analysis_end_date": analysis_end_date}
    )
    (
        frames, common_dates, calendar_source, symbols, folds,
        decision_dates, decision_to_fold, decision_metadata,
    ) = motor._construir_contexto_execucao(prepared, config)
    if set(symbols) != set(selected) or len(symbols) != 75:
        raise RuntimeError("Painel de modelagem nao manteve os 75 ativos.")

    print(
        f"[fixed75] total=75, calibracao e treinamento conjuntos, "
        f"cutoff={analysis_end_date}, folds={len(folds)}",
        flush=True,
    )
    _, policies, calibration_rows = motor._train_baseline_states(
        frames, common_dates, symbols, folds, config
    )
    result = motor._simular_exato(
        "tcc_single_fixed_75",
        motor._politica_agendada(policies, decision_to_fold),
        frames, symbols, decision_dates, config,
        motor.calcular_taxas_referencia,
        motor.aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="TCC Control - unico universo fixo de 75 ativos",
        method_line=(
            "- Todos os 75 ativos elegiveis sao ajustados e avaliados "
            "na mesma etapa; sem selecao nem expansao."
        ),
    )
    metrics = summarize_metrics(result, folds, float(config.initial_capital))
    capital = float(metrics["ending_capital"])

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"asset": selected}).to_csv(
        output / "universe_75.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        output / "fold_calibration.csv", index=False
    )
    result.predictions.to_csv(
        output / "predictions.csv", index=True, float_format="%.17g"
    )
    if isinstance(getattr(result, "trades", None), pd.DataFrame):
        result.trades.to_csv(output / "trades.csv", index=False)

    payload = {
        "reproduction_version": version,
        "execution_schema": execution_schema,
        "status": "completed",
        "purpose": "pesquisa_universo_unico_75",
        "research_design": "fixed_75_trained_together_from_first_fold",
        "analysis_end_date": analysis_end_date,
        "checkpoint": {
            "observed_capital": capital,
            "expected_capital": None,
            "delta": None,
        },
        "data": {
            "source": "alpaca",
            "feed": "sip",
            "adjustment": "raw",
            "timeframe": "1Day",
            "database_used": False,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "snapshot_root": str(source.root),
            "data_audit": data_audit,
            "diagnostics": data_diagnostics,
        },
        "universe": {
            "requested_assets": list(selected),
            "requested_count": 75,
            "effective_assets": list(symbols),
            "effective_count": 75,
            "excluded_assets": [],
            "disallowed_assets": sorted(U67_EXPECTED_EXCLUSIONS),
        },
        "calendar": {
            "calendar_source": calendar_source,
            "decision_sessions": len(decision_dates) - 1,
        },
        "metrics": metrics,
        "baseline_reference": None,
        "runtime_seconds": time.perf_counter() - started,
        "comparison_warning": (
            "Treinamento conjunto de 75 ativos difere do executor legado "
            "65+10; nenhuma igualdade com o checkpoint historico e imposta."
        ),
    }
    (output / "reproducao_universo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    gerar_analises_backtest(output, manifest=snapshot, result=result)
    pacote = criar_pacote_analise(
        output,
        comparison_file="reproducao_universo.json",
        execution_schema=execution_schema,
        archive_name="pacote_universo_75.zip",
    )
    print(f"[fixed75] capital=US$ {capital:,.2f} pacote={pacote}", flush=True)
    sinal_sonoro_conclusao()
    return payload
