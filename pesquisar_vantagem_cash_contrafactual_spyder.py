"""Pesquisa v1.0.6: vantagem contrafactual CASH versus continuar Control.

Spyder F5. Universo imutavel de 75 ativos e mesmo checkpoint Control.
Aprende na calibracao o valor logaritmico de ficar em CASH por uma
decisao e retornar a regra Control nas decisoes seguintes (horizonte 10).
Uma regressao Ridge fixa estima vantagem contextual; a previsao OOS
usa apenas informacoes disponiveis no fechamento da decisao.
Gera ZIP automatico em exito ou erro, sem dados RAW nem credenciais.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import traceback

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG, construir_configuracao_controle
from engine.execucao import calcular_taxas_referencia, aplicar_deslizamento
from engine.modelo_lightgbm import _construir_contexto_execucao
from engine.rotacao import _politica_agendada, _simular_exato
from reproducao.experimento import summarize_metrics
from pesquisar_expansao_u67_spyder import _write_json
from pesquisar_optimal_switching_cash_spyder import (
    ROOT,
    carregar_universo_oficial,
    construir_calendario_control_75,
    verificar_protocolo_cronologico,
    ajustar_control_75,
)
from pesquisa.vantagem_cash_contrafactual import (
    FEATURE_COLUMNS,
    CounterfactualSettings,
    build_calibration_samples,
    fit_counterfactual_advantage,
    build_counterfactual_overlay,
)
from pesquisa.pacote_analise_switching import gerar_zip_resultados
from pesquisa.persistencia_resultados import salvar_csv, salvar_json, salvar_texto

SCRIPT_VERSION = "cash-counterfactual-continuation-v1.0.7"
EXPERIMENT_SCHEMA = "counterfactual-cash-continuation-v1"
SETTINGS = CounterfactualSettings(
    horizon=10,
    train_fraction=0.75,
    ridge_alpha=10.0,
    min_train_samples=45,
    min_validation_samples=12,
)
OUTPUT = ROOT / "output" / "counterfactual_cash_continuation"


def _run(output: Path) -> dict:
    # %% 1 - Carregamento integral e calendario historico oficial
    raw, contract, snapshot, universe = carregar_universo_oficial()
    config = construir_configuracao_controle(
        CONFIG, assets=tuple(sorted(raw)),
    ).copiar_modelo(update={"analysis_end_date": contract["analysis_end_date"]})
    calendar = construir_calendario_control_75(raw, config)
    (
        frames, common_dates, calendar_source, symbols, folds,
        decision_dates, decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao(
        raw, config, calendar_override=calendar,
        calendar_source_label="AAPL",
    )
    if len(symbols) != 75 or tuple(symbols) != tuple(universe["assets"]):
        raise RuntimeError("Universo efetivo nao contem exatamente os 75 ativos.")
    protocol = verificar_protocolo_cronologico(
        common_dates, folds, decision_dates,
    )
    for symbol in symbols:
        prices = frames[symbol].reindex(decision_dates[1:])[
            ["open", "high", "low", "close"]
        ].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise RuntimeError(
                f"{symbol}: preco nao executavel; sem excluir nenhum ativo."
            )
    print(
        f"[universe] official_assets={len(symbols)} "
        f"folds={len(folds)} oos_sessions={len(decision_dates)-1}",
        flush=True,
    )

    # %% 2 - Mesmo Control congelado; valida checkpoint ANTES de inovar
    policies, margin_calibration, calibration_policies = ajustar_control_75(
        frames, common_dates, symbols, folds, config,
    )
    control = _simular_exato(
        "tcc_control_frozen",
        _politica_agendada(policies, decision_to_fold),
        frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="Control oficial 75 ativos",
    )
    control_metrics = summarize_metrics(control, folds, float(config.initial_capital))
    expected = float(contract["expected_ending_capital"])
    if not math.isclose(
        control_metrics["ending_capital"], expected, abs_tol=0.01, rel_tol=0,
    ):
        raise RuntimeError(
            "Checkpoint Control de 75 ativos divergente: "
            f"observado={control_metrics['ending_capital']:.8f} "
            f"esperado={expected:.8f}"
        )
    salvar_csv(control.predictions, output / "control_predictions.csv", float_format="%.17g")
    salvar_csv(control.trades, output / "control_trades.csv", index=False)
    salvar_csv(pd.DataFrame(margin_calibration), 
        output / "control_75_fold_calibration.csv", index=False,
    )
    print(
        f"[control] checkpoint=validado capital=US$ {control_metrics['ending_capital']:,.2f}",
        flush=True,
    )

    # %% 3 - Rotulos contrafactuais calculados SO na calibracao
    models_by_fold = {}
    diagnostics_by_fold = []
    samples_parts = []
    validation_parts = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        start = int(fold["calibration_start_index"])
        end = int(fold["calibration_end_index"])
        calibration_dates = common_dates[start:end]
        if not pd.Timestamp(calibration_dates[-1]) < pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Calibracao posterior ao inicio da avaliacao OOS.")
        if not pd.Timestamp(common_dates[int(fold["train_end_index"]) - 1]) < pd.Timestamp(
            calibration_dates[0]
        ):
            raise RuntimeError("Control na calibracao nao precede a janela de treino.")

        # A politica de rotacao desta janela ja estava treinada antes de
        # calibration; trajeto real de Control nao e usado como rotulo OOS.
        samples = build_calibration_samples(
            calibration_policies[fold_id],
            frames, symbols, calibration_dates,
            config=config, settings=SETTINGS, fold_id=fold_id,
        )
        if samples.empty:
            raise RuntimeError(f"fold={fold_id}: nenhuma amostra contrafactual.")
        if pd.Timestamp(samples["label_last_date"].max()) >= pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Rotulo contrafactual invadiu periodo OOS.")
        fitted, validation = fit_counterfactual_advantage(
            samples, calibration_dates, settings=SETTINGS,
        )
        if fitted.fit_last_label_date >= pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Modelo treinou com labels do teste.")
        models_by_fold[fold_id] = fitted
        samples_parts.append(samples)
        validation_parts.append(validation)
        ridge = fitted.model.named_steps["ridge"]
        scaler = fitted.model.named_steps["standardscaler"]
        diagnostics_by_fold.append({
            "fold_id": fold_id,
            "train_end": pd.Timestamp(
                common_dates[int(fold["train_end_index"]) - 1]
            ).isoformat(),
            "calibration_first": pd.Timestamp(calibration_dates[0]).isoformat(),
            "calibration_last": pd.Timestamp(calibration_dates[-1]).isoformat(),
            "test_first": pd.Timestamp(fold["test_start"]).isoformat(),
            "train_rows_inner": fitted.training_rows,
            "validation_rows_inner": fitted.validation_rows,
            "calibration_rows_total": fitted.calibration_rows,
            "last_calibration_label_date": fitted.fit_last_label_date.isoformat(),
            "validation_first_date": fitted.validation_first_date.isoformat(),
            "validation_mae_log": fitted.validation_mae,
            "validation_direction_accuracy": fitted.validation_direction_accuracy,
            "ridge_alpha": SETTINGS.ridge_alpha,
            "ridge_intercept": float(ridge.intercept_),
            "ridge_coefficients": list(map(float, ridge.coef_)),
            "standard_scaler_mean": list(map(float, scaler.mean_)),
            "standard_scaler_scale": list(map(float, scaler.scale_)),
            "features": list(FEATURE_COLUMNS),
            "historical_test_used_for_fitting": False,
        })
        print(
            f"[counterfactual] fold={fold_id}/{len(folds)} "
            f"train={fitted.training_rows} validation={fitted.validation_rows} "
            f"MAE={fitted.validation_mae:.6f}", flush=True,
        )
    salvar_csv(pd.concat(samples_parts, ignore_index=True), 
        output / "calibration_counterfactual_labels.csv", index=False,
    )
    salvar_csv(pd.concat(validation_parts, ignore_index=True), 
        output / "inner_validation_predictions.csv", index=False,
    )
    salvar_csv(pd.DataFrame(diagnostics_by_fold), 
        output / "counterfactual_fold_models.csv", index=False,
    )
    salvar_json(output / "counterfactual_model_manifest.json", {
        "settings": SETTINGS.__dict__,
        "fold_models": diagnostics_by_fold,
        "features": list(FEATURE_COLUMNS),
    }, _write_json)

    # %% 4 - Decisoes OOS, unica nova politica versus Control
    diagnostics: dict[pd.Timestamp, dict] = {}
    counterfactual = build_counterfactual_overlay(
        _politica_agendada(policies, decision_to_fold),
        fits_by_fold=models_by_fold,
        decision_to_fold=decision_to_fold,
        frames=frames, symbols=symbols,
        diagnostics=diagnostics,
    )
    switching = _simular_exato(
        "tcc_cash_counterfactual_continuation",
        counterfactual,
        frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Pesquisa Control75 versus CASH contrafactual",
        method_line=(
            "- Reentrada pela politica Control; a vantagem de CASH "
            "e aprendida na calibracao sem futuro OOS."
        ),
    )
    switching_metrics = summarize_metrics(
        switching, folds, float(config.initial_capital),
    )
    salvar_csv(switching.predictions, 
        output / "counterfactual_predictions.csv", float_format="%.17g",
    )
    salvar_csv(switching.trades, 
        output / "counterfactual_trades.csv", index=False,
    )
    decision_rows = [
        {"decision_date": date.isoformat(), **row}
        for date, row in sorted(diagnostics.items())
    ]
    salvar_csv(pd.DataFrame(decision_rows), 
        output / "counterfactual_decisions.csv", index=False,
    )

    # %% 5 - Resultados e pacote de auditoria
    count_cash = sum(
        row.get("reason") == "predicted_cash_advantage" for row in diagnostics.values()
    )
    count_control = sum(
        row.get("reason") == "continue_control" for row in diagnostics.values()
    )
    summary = {
        "version": SCRIPT_VERSION,
        "schema": EXPERIMENT_SCHEMA,
        "status": "completed",
        "experiment": "cash_vs_continue_control",
        "research_only": True,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "universe_id": universe["universe_id"],
        "asset_count": len(symbols),
        "assets": symbols,
        "historical_cutoff": contract["analysis_end_date"],
        "calendar_source": calendar_source,
        "calendar_contract": protocol,
        "fold_count": len(folds),
        "oos_sessions": len(decision_dates) - 1,
        "initial_capital": float(config.initial_capital),
        "settings": SETTINGS.__dict__,
        "label_definition": (
            "log terminal wealth over horizon: forced CASH first decision "
            "then continue Control, minus continuing Control from the start; "
            "branch position and holding states evolve independently."
        ),
        "label_accounting": (
            "log close-to-next-close with proportional switching cost as "
            "defined in engine.rotacao._retorno_log_transicao_treinamento. "
            "Final OOS replay uses full broker fees/slippage."
        ),
        "control_metrics": control_metrics,
        "counterfactual_metrics": switching_metrics,
        "comparison": {
            "delta_ending_capital": (
                switching_metrics["ending_capital"] - control_metrics["ending_capital"]
            ),
            "delta_sharpe": switching_metrics["sharpe"] - control_metrics["sharpe"],
            "delta_maximum_drawdown": (
                switching_metrics["maximum_drawdown"] - control_metrics["maximum_drawdown"]
            ),
            "control_cash_days": int(control.metrics.get("cash_days", 0)),
            "counterfactual_cash_days": int(switching.metrics.get("cash_days", 0)),
            "control_total_fees": float(control.metrics.get("total_transaction_fees", 0)),
            "counterfactual_total_fees": float(
                switching.metrics.get("total_transaction_fees", 0)
            ),
            "predicted_cash_advantage_decisions": int(count_cash),
            "continue_control_decisions": int(count_control),
        },
        "fold_models": diagnostics_by_fold,
        "limitations": [
            "The 75-asset universe is fixed and historically selected.",
            "The control trading margin uses the same calibration window for selection; label policy may inherit selection bias.",
            "Counterfactual training labels include future outcomes only within prior calibration, not OOS.",
            "The cash alternative forces one first-day CASH and then resumes the Control rule; recursive OOS decisions may keep CASH for longer.",
            "Counterfactual labels use the training cost approximation; actual replay applies exact fees.",
            "Ridge alpha and horizon were fixed after seeing prior v1.0.4/v1.0.5 OOS failures. Historical window is not untouched for cross-version model selection.",
            "The value estimate is supervised finite-horizon approximation, not a solved full 76-state Markov Bellman problem.",
            "No model or result is promoted to the operational application.",
        ],
    }
    salvar_json(output / "resultado.json", summary, _write_json)
    report = (
        "# CASH versus continuar Control: evidencia experimental\n\n"
        f"Versao: {SCRIPT_VERSION}. Universo: {len(symbols)} ativos.\n"
        f"Folds: {len(folds)}; decisoes OOS: {len(decision_dates)-1}.\n\n"
        f"Control: US$ {control_metrics['ending_capital']:,.2f}; "
        f"Sharpe {control_metrics['sharpe']:.4f}; "
        f"MaxDD {control_metrics['maximum_drawdown']:.2%}.\n"
        f"Contrafactual: US$ {switching_metrics['ending_capital']:,.2f}; "
        f"Sharpe {switching_metrics['sharpe']}; "
        f"MaxDD {switching_metrics['maximum_drawdown']:.2%}.\n"
        f"CASH: Control {summary['comparison']['control_cash_days']} sessoes; "
        f"contrafactual {summary['comparison']['counterfactual_cash_days']}.\n"
        f"Decisoes com vantagem CASH prevista: {count_cash}.\n\n"
        "Os rotulos historicos foram criados apenas dentro de calibration; "
        "o fold OOS nao e usado para treino. Esta rodada nao constitui "
        "validacao externa independente, pois pesquisas anteriores ja "
        "exploraram o mesmo intervalo de teste.\n"
    )
    salvar_texto(output / "resumo_para_analise.md", report, encoding="utf-8")
    print(
        f"[resultado] Control=US$ {control_metrics['ending_capital']:,.2f} "
        f"Contrafactual=US$ {switching_metrics['ending_capital']:,.2f}",
        flush=True,
    )
    print(f"[resultado] cash_preferred={count_cash}", flush=True)
    return summary


def main() -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(output)
        package = gerar_zip_resultados(output, versao=SCRIPT_VERSION, status="completed")
        print(f"[pacote] ZIP para analise: {package}", flush=True)
    except Exception as exc:
        try:
            salvar_json(output / "erro_execucao.json", {
                "version": SCRIPT_VERSION,
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=30),
            }, _write_json)
            package = gerar_zip_resultados(
                output, versao=SCRIPT_VERSION, status="failed",
            )
            print(f"[pacote] erro registrado: {package}", flush=True)
        except Exception as archive_error:
            print(f"[pacote] falha ao compactar erro: {archive_error}", flush=True)
        raise
    return result


if __name__ == "__main__":
    main()
