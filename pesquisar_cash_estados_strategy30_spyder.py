"""Pesquisa CASH dependente do estado, usando Control Strategy #30 reproduzido.

Spyder F5. Refaz o treino Control a partir do snapshot RAW/SIP de 75 ativos,
verifica todas as 1576 decisoes com o backtest oficial ANTES de comparar.
Amostras para CASH e reentrada sao criadas so nas janelas de calibracao.
Nenhuma otimização de parametros usando resultados OOS.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import traceback
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pesquisa.control_operacional_strategy30 import (
    ROOT, carregar_contrato, validar_control_strategy30,
)
from pesquisa.paridade_strategy30 import (
    config_from_frozen_manifest, load_snapshot_strategy30,
    compare_control_trajectories,
)
from pesquisa.referencia_strategy30.modelo_lightgbm import (
    _ajustar_modelos_lightgbm, _construir_contexto_execucao,
    _selecionar_switch_margin_fold,
)
from pesquisa.referencia_strategy30.rotacao import (
    _politica_utilidade, _politica_agendada,
    _crescimento_politica_simples, _precalcular_utilidades_modelo,
    _simular_exato, _desempenho_folds,
)
from pesquisa.referencia_strategy30.execucao import (
    calcular_taxas_referencia, aplicar_deslizamento,
)
from pesquisa.cash_estados_strategy30 import (
    CounterfactualSettings, build_calibration_samples,
    fit_counterfactual_advantage, build_counterfactual_overlay,
)
from pesquisa.pacote_analise_switching import gerar_zip_resultados
from pesquisa.persistencia_resultados import (
    salvar_csv, salvar_json, salvar_texto,
)
from pesquisar_controle_strategy30_spyder import localizar_zip_original
from pesquisar_expansao_u67_spyder import _write_json

SCRIPT_VERSION = "cash-state-aware-strategy30-v1.0.0"
OUTPUT = ROOT / "output" / "cash_state_aware_strategy30"
SNAPSHOT = ROOT / "dados" / "strategy30_snapshot"

# Protocolo predeclarado. Nao ajustar a partir do patrimonio fora da amostra.
SETTINGS = CounterfactualSettings(
    horizon=10,
    train_fraction=0.75,
    ridge_alpha=10.0,
    min_train_samples=45,
    min_validation_samples=12,
)
EXPECTED_FOLDS = (504, 504, 568)


def _audit_cash_spells(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    states = predictions["selected_asset"].astype(str)
    dates = pd.DatetimeIndex(predictions.index)
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise RuntimeError("Previsoes sem DatetimeIndex.")
    spells = []
    first = None
    for i, (date, position) in enumerate(zip(dates, states)):
        if position == "CASH" and first is None:
            first = i
        if first is not None and (position != "CASH" or i == len(states) - 1):
            last = i if position == "CASH" else i - 1
            spells.append({
                "start": pd.Timestamp(dates[first]).isoformat(),
                "end": pd.Timestamp(dates[last]).isoformat(),
                "sessions": int(last - first + 1),
                "reentered": bool(position != "CASH"),
                "reentry_asset": position if position != "CASH" else None,
            })
            first = None
    transitions = [
        (str(states.iloc[i - 1]), str(states.iloc[i]))
        for i in range(1, len(states))
    ]
    summary = {
        "cash_sessions": int(states.eq("CASH").sum()),
        "cash_intervals": len(spells),
        "entered_cash_from_asset": int(sum(a != "CASH" and b == "CASH" for a, b in transitions)),
        "reentered_asset_from_cash": int(sum(a == "CASH" and b != "CASH" for a, b in transitions)),
        "maximum_cash_interval_sessions": max((x["sessions"] for x in spells), default=0),
        "cash_reentries_same_session": 0,
        "notes": "Uma decisao de alvo por sessao. Reentrada so e possivel na sessao seguinte.",
    }
    return summary, pd.DataFrame(spells, columns=[
        "start", "end", "sessions", "reentered", "reentry_asset",
    ])


def _run(output: Path) -> dict:
    # %% 1 - Carregar evidencias e CSVs independentes, sem banco
    contract = carregar_contrato()
    reference_zip = localizar_zip_original()
    evidence, reference, _ = validar_control_strategy30(
        reference_zip, contrato=contract,
    )
    with ZipFile(reference_zip) as archive:
        strategy = json.loads(archive.read("strategy_manifest.json"))
        experiment = json.loads(archive.read("experiment_manifest.json"))
    if experiment.get("market_data_signature_sha256") != contract["market_data_signature_sha256"]:
        raise RuntimeError("Assinatura do mercado nao coincide com a referencia.")
    root = Path(os.environ.get("STRATEGY30_SNAPSHOT_DIR", str(SNAPSHOT))).expanduser()
    local = load_snapshot_strategy30(
        root, contract=contract, experiment_manifest=experiment,
    )
    salvar_csv(local.records, output / "auditoria_snapshot_75.csv", index=False)
    config = config_from_frozen_manifest(strategy, contract)
    (
        frames, dates, calendar_source, symbols, folds, decision_dates,
        decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao(local.frames, config)
    if len(symbols) != 75 or tuple(symbols) != tuple(contract["assets"]):
        raise RuntimeError("Universo OOS diferente dos 75 ativos Control.")
    fold_sizes = tuple(
        int(f["test_end_index"] - f["test_start_index"]) for f in folds
    )
    if fold_sizes != EXPECTED_FOLDS or tuple(
        int(x) for x in contract["expected_fold_counts"].values()
    ) != EXPECTED_FOLDS:
        raise RuntimeError(f"Folds de Control divergentes: {fold_sizes}.")
    if len(decision_dates) != int(contract["expected_rows"]) + 1:
        raise RuntimeError("Quantidade de decisoes OOS diferente da referencia.")
    if (
        pd.Timestamp(folds[0]["test_start"]).strftime("%Y-%m-%d")
        != contract["first_session"]
        or pd.Timestamp(folds[-1]["test_end"]).strftime("%Y-%m-%d")
        != contract["last_session"]
    ):
        raise RuntimeError("Periodo de avaliacoes divergente.")
    print(
        f"[universe] 75 ativos calendario={calendar_source} "
        f"folds={fold_sizes} OOS=1576", flush=True,
    )

    # %% 2 - Treino Control por fold, calibracao independente e etiquetas CASH
    policies = {}
    fits = {}
    fold_rows = []
    label_parts = []
    validation_parts = []
    for fold in folds:
        fid = int(fold["fold_id"])
        training = dates[:int(fold["train_end_index"])]
        calibration = dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_training = dates[:int(fold["final_fit_end_index"])]
        assert pd.Timestamp(training[-1]) < pd.Timestamp(calibration[0])
        assert pd.Timestamp(calibration[-1]) < pd.Timestamp(fold["test_start"])
        print(f"[fit] fold={fid}/3 calibration models=75", flush=True)
        calibration_models = _ajustar_modelos_lightgbm(
            frames, symbols, training, config, phase=f"state_cash_fold_{fid}_calibration",
        )
        candidate_scores = []
        for margin in config.rotation_switch_margin_candidates:
            candidate_policy = _politica_utilidade(
                calibration_models, frames, symbols, config, float(margin),
            )
            score = _crescimento_politica_simples(
                candidate_policy, frames, symbols, calibration, config,
            )
            candidate_scores.append((float(margin), float(score)))
        margin = _selecionar_switch_margin_fold(config, fid, candidate_scores)
        if margin["selection_source"] != "calibration":
            raise RuntimeError("Override de margem nao permitido na pesquisa oficial.")
        effective_margin = max(
            float(config.rotation_switch_margin),
            float(margin["selected_candidate_margin"]),
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_models, frames, symbols, calibration, config,
        )
        historical_policy = _politica_utilidade(
            calibration_models, frames, symbols, config,
            effective_margin, utility_cache=calibration_cache,
        )
        samples = build_calibration_samples(
            historical_policy, frames, symbols, calibration,
            config=config, settings=SETTINGS, fold_id=fid,
        )
        if samples["label_last_date"].max() >= pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Uso de retorno futuro da janela OOS na calibracao CASH.")
        fit, validation = fit_counterfactual_advantage(
            samples, calibration, settings=SETTINGS,
        )
        cash_n = int(samples["cash_position"].gt(0).sum())
        print(
            f"[cash-fit] fold={fid} samples={len(samples)} "
            f"cash_samples={cash_n} unique_dates={samples['decision_date'].nunique()} "
            f"validation_mae={fit.validation_mae:.6f} "
            f"validation_accuracy={fit.validation_direction_accuracy:.2%}",
            flush=True,
        )
        print(f"[fit] fold={fid}/3 final models=75", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames, symbols, final_training, config,
            phase=f"state_cash_fold_{fid}_final",
        )
        oos_cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols,
            pd.DatetimeIndex(fold["decision_dates"]), config,
        )
        policies[fid] = _politica_utilidade(
            final_models, frames, symbols, config, effective_margin,
            utility_cache=oos_cache,
        )
        fits[fid] = fit
        y = validation["label_cash_minus_control_log_return"].to_numpy(dtype=float)
        fold_rows.append({
            "fold_id": fid,
            "calibration_start": pd.Timestamp(calibration[0]).isoformat(),
            "calibration_end": pd.Timestamp(calibration[-1]).isoformat(),
            "train_end": pd.Timestamp(training[-1]).isoformat(),
            "last_training_label_date": fit.fit_last_label_date.isoformat(),
            "first_inner_validation_date": fit.validation_first_date.isoformat(),
            "oos_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "oos_end": pd.Timestamp(fold["test_end"]).isoformat(),
            "calibrated_margin": float(margin["selected_candidate_margin"]),
            "effective_margin": float(effective_margin),
            "margin_selection_source": margin["selection_source"],
            "sample_rows": len(samples),
            "cash_state_rows": cash_n,
            "invested_state_rows": int(len(samples) - cash_n),
            "independent_calibration_dates": samples["decision_date"].nunique(),
            "training_rows_after_purge": fit.training_rows,
            "inner_validation_rows": fit.validation_rows,
            "inner_validation_distinct_dates": validation["decision_date"].nunique(),
            "inner_validation_mae": fit.validation_mae,
            "inner_validation_direction_accuracy": fit.validation_direction_accuracy,
            "always_control_direction_accuracy": float(np.mean(y <= 0)),
            "label_horizon": SETTINGS.horizon,
        })
        label_parts.append(samples)
        validation_parts.append(validation)

    salvar_csv(
        pd.concat(label_parts, ignore_index=True),
        output / "calibration_state_aware_labels.csv", index=False,
        float_format="%.17g",
    )
    salvar_csv(
        pd.concat(validation_parts, ignore_index=True),
        output / "inner_validation_predictions.csv", index=False,
        float_format="%.17g",
    )
    salvar_csv(pd.DataFrame(fold_rows), output / "fold_model_diagnostics.csv", index=False)

    # %% 3 - Reexecutar o Control com a politica reproduzida e verificar TUDO
    base = _politica_agendada(policies, decision_to_fold)
    control = _simular_exato(
        "strategy30_control_verification", base, frames, symbols,
        decision_dates, config, calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata, model_label="Control Strategy #30",
    )
    salvar_csv(
        control.predictions, output / "control_predictions.csv",
        index=True, index_label="timestamp", float_format="%.17g",
    )
    salvar_csv(control.trades, output / "control_trades.csv", index=False)
    parity, comparison = compare_control_trajectories(
        reference, control.predictions, contract=contract,
    )
    salvar_csv(comparison, output / "control_parity_by_session.csv", index=False)
    salvar_json(output / "control_parity.json", parity, _write_json)
    if not parity["fully_reproduced"] or not math.isclose(
        float(control.metrics["strategy_ending_capital"]),
        float(contract["ending_capital"]), abs_tol=0.01, rel_tol=0,
    ):
        raise RuntimeError(
            "Control Strategy #30 nao reproduzido integralmente. "
            f"Primeira divergencia: {parity['first_mismatch_date']}, "
            f"capital reproduzido: {parity['replay_final_capital']:.2f}. "
            "Experimento CASH bloqueado."
        )
    print(
        f"[control] 1576 sessoes identicas capital=US$ "
        f"{control.metrics['strategy_ending_capital']:,.2f}", flush=True,
    )

    # %% 4 - Politica CASH: uma unica decisao por dia, custos reais no replay
    diagnostics: dict[pd.Timestamp, dict] = {}
    switching_policy = build_counterfactual_overlay(
        base, fits_by_fold=fits, decision_to_fold=decision_to_fold,
        frames=frames, symbols=symbols, diagnostics=diagnostics,
    )
    switching = _simular_exato(
        "cash_state_aware_strategy30", switching_policy, frames, symbols,
        decision_dates, config, calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="Control Strategy #30 + state-aware CASH",
    )
    if len(switching.predictions) != 1576 or len(diagnostics) != 1576:
        raise RuntimeError("Replay CASH incompleto ou quantidade de decisoes alterada.")
    salvar_csv(
        switching.predictions, output / "cash_predictions.csv",
        index=True, index_label="timestamp", float_format="%.17g",
    )
    salvar_csv(switching.trades, output / "cash_trades.csv", index=False)
    salvar_csv(
        pd.DataFrame([
            {"decision_date": date.isoformat(), **data}
            for date, data in sorted(diagnostics.items())
        ]),
        output / "cash_decisions.csv", index=False, float_format="%.17g",
    )
    spells_summary, spells = _audit_cash_spells(switching.predictions)
    salvar_csv(spells, output / "cash_intervals.csv", index=False)
    control_metrics = control.metrics
    cash_metrics = switching.metrics
    key_metrics = (
        "strategy_ending_capital", "strategy_sharpe",
        "strategy_maximum_drawdown", "cash_days", "capital_rotations",
        "total_transaction_fees",
    )
    summary = {
        "version": SCRIPT_VERSION,
        "status": "completed",
        "experiment": "control30_cash_state_aware_counterfactual",
        "research_only": True,
        "universe_count": 75,
        "out_of_sample_sessions": 1576,
        "snapshot_manifest_file": str(root / "manifest.json"),
        "source_reference_sha256": evidence["source_zip_sha256"],
        "control_reproduced": True,
        "control_parity": parity,
        "calibration_settings": SETTINGS.__dict__,
        "exploration_scenarios": [
            "control_shadow", "cash_1_of_7", "cash_2_of_11",
            "cash_4_of_15", "cash_8_of_20", "cash_13_of_31",
        ],
        "fold_diagnostics": fold_rows,
        "control_metrics": {k: control_metrics.get(k) for k in key_metrics},
        "cash_metrics": {k: cash_metrics.get(k) for k in key_metrics},
        "comparison": {
            "delta_ending_capital": float(
                cash_metrics["strategy_ending_capital"] - control_metrics["strategy_ending_capital"]
            ),
            "delta_sharpe": float(
                cash_metrics["strategy_sharpe"] - control_metrics["strategy_sharpe"]
            ),
            "delta_max_drawdown": float(
                cash_metrics["strategy_maximum_drawdown"]
                - control_metrics["strategy_maximum_drawdown"]
            ),
            "relative_capital": float(
                cash_metrics["strategy_ending_capital"]
                / control_metrics["strategy_ending_capital"]
            ),
        },
        "cash_state_diagnostics": spells_summary,
        "control_folds": _desempenho_folds(
            control.predictions, folds, float(config.initial_capital),
        ),
        "cash_folds": _desempenho_folds(
            switching.predictions, folds, float(config.initial_capital),
        ),
        "methodological_limits": [
            "Exploracao CASH usa agenda deterministica criada em calibration, nao politica aprendida em trajetorias completas.",
            "Rotulos de horizonte 10 avaliam uma acao CASH seguida da politica Control; inferencia OOS pode escolher CASH repetidamente.",
            "Custos de rotulos historicos sao aproximados; replay OOS usa taxas financeiras do Control.",
            "Uma data historica pode contribuir com estados correlacionados; tamanho efetivo e menor que contagem de linhas.",
            "Os folds OOS ja foram inspecionados em versoes anteriores. Resultado exploratorio, nao novo teste virgem.",
            "Universo historico congelado nao demonstra validade prospectiva, nem retorno garantido.",
            "Nenhuma selecao de parametros foi realizada pelo resultado financeiro OOS nesta execucao.",
        ],
    }
    salvar_json(output / "resultado.json", summary, _write_json)
    salvar_texto(
        output / "resumo_para_analise.md",
        "# Pesquisa CASH dependente de estado, Strategy #30\n\n"
        f"Versao: {SCRIPT_VERSION}\n"
        "Control validado: 75 ativos, 1576 sessoes, US$ 80.685.454,79.\n"
        f"Control final: US$ {control_metrics['strategy_ending_capital']:,.2f}\n"
        f"Politica CASH final: US$ {cash_metrics['strategy_ending_capital']:,.2f}\n"
        f"Sharpe: Control {control_metrics['strategy_sharpe']:.4f}, "
        f"CASH {cash_metrics['strategy_sharpe']:.4f}\n"
        f"MaxDD: Control {control_metrics['strategy_maximum_drawdown']:.2%}, "
        f"CASH {cash_metrics['strategy_maximum_drawdown']:.2%}\n"
        f"CASH sessoes: {spells_summary['cash_sessions']}; "
        f"periodos: {spells_summary['cash_intervals']}; "
        f"reentradas: {spells_summary['reentered_asset_from_cash']}\n"
        "Validação interna foi cronologica com purga do horizonte. "
        "Dados dos 1576 dias OOS nao treinam o modelo CASH.\n"
        "Estas series historicas OOS nao sao teste externo inedito; "
        "estudo e exploratorio.\n",
    )
    print(
        f"[results] Control US$ {control_metrics['strategy_ending_capital']:,.2f}; "
        f"CASH US$ {cash_metrics['strategy_ending_capital']:,.2f}; "
        f"cash_days={spells_summary['cash_sessions']}", flush=True,
    )
    return summary


def main() -> dict:
    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(output)
        archive = gerar_zip_resultados(
            output, versao=SCRIPT_VERSION, status="completed",
        )
        print(f"[pacote] ZIP de pesquisa: {archive}", flush=True)
        return result
    except Exception as exc:
        try:
            salvar_json(output / "erro_execucao.json", {
                "status": "failed", "version": SCRIPT_VERSION,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=30),
            }, _write_json)
            archive = gerar_zip_resultados(
                output, versao=SCRIPT_VERSION, status="failed",
            )
            print(f"[pacote] diagnostico de falha: {archive}", flush=True)
        except Exception as archive_error:
            print(f"[pacote] falha para compactar: {archive_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
