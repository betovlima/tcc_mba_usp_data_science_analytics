"""TCC Spyder F5: contextual historical risk and CASH against frozen Control.

Scientific protocol v1.0.0: state-only, asset-aware and state+shape comparisons.
Full Control parity is mandatory before any protected financial simulation.
No downloads, no database, no new universe, no changes to ambiente operacional.
"""
# %% 0 - Contract and dependencies
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import math
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    _politica_utilidade, _politica_agendada, _crescimento_politica_simples,
    _precalcular_utilidades_modelo, _simular_exato,
)
from pesquisa.referencia_strategy30.execucao import (
    calcular_taxas_referencia, aplicar_deslizamento,
)
from pesquisa.risco_contextual_cash_u75 import (
    VERSION as CONTEXT_VERSION,
    AdvantageSettings, construct_market_context,
    ContextualCashRisk, contextual_cash_policy,
)
from pesquisar_controle_strategy30_spyder import localizar_zip_original
from pesquisa.saida_resiliente_motivos_u75 import (
    write_with_parent, check_required_files,
)
from pesquisa.andamento_motivos_u75 import record_progress, last_progress
from pesquisa.validacao_dimensoes_motivos_u75 import validar_dimensoes_reproducao

RUNNER_VERSION = "u75-contextual-cash-advantage-v1.0.0"
OUTPUT = Path(os.environ.get(
    "TCC_CONTEXT_CASH_OUTPUT_DIR", str(ROOT / "output" / "vantagem_cash_contextual_u75"),
)).expanduser()
SNAPSHOT = ROOT / "dados" / "strategy30_snapshot"

# Fixed BEFORE any financial replay with this new risk formulation.
# Same 20-day / 5-day horizon across three models, no sweep on 2026 losses.
SETTINGS = AdvantageSettings(window=20, horizon=5)
EXPERIMENTS = (
    ("contexto_mercado", SETTINGS, "market"),
    ("contexto_mercado_ativo", SETTINGS, "joint"),
    ("contexto_mercado_ativo_forma", SETTINGS, "joint_shape"),
)


def _write_csv(frame, path):
    write_with_parent(
        path, lambda destination: frame.to_csv(
            destination, index=False, float_format="%.17g",
        ),
    )


def _write_json(payload, path):
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2,
        default=str, allow_nan=False,
    )
    write_with_parent(
        path, lambda destination: destination.write_text(
            serialized, encoding="utf-8",
        ),
    )


def _simulate(label, policy, frames, symbols, dates, config, metadata,
              *, benchmark=None):
    return _simular_exato(
        label, policy, frames, symbols, dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=metadata,
        model_label=label,
        benchmark_override=benchmark,
        benchmark_override_name="Fixed Control benchmark" if benchmark is not None else None,
    )


def _risk_metrics(predictions: pd.DataFrame, initial: float) -> dict:
    equity = predictions["strategy_equity"].to_numpy(dtype=float)
    if len(equity) < 2 or not np.isfinite(equity).all() or np.any(equity <= 0):
        raise RuntimeError("Invalid financial equity curve")
    # Include the first day's return relative to initial capital.
    rets = np.diff(np.r_[float(initial), equity]) / np.r_[float(initial), equity[:-1]]
    worst = float(np.quantile(rets, .05))
    es95 = float(np.mean(rets[rets <= worst]))
    return {"ending_capital": float(equity[-1]), "max_drawdown":float(
                np.min(equity/np.maximum.accumulate(np.r_[initial, equity])[1:]-1)),
            "daily_es95":es95, "worst_daily_return":float(np.min(rets)),
            "daily_negative_sessions":int(np.sum(rets < 0))}


def _run(out: Path):
    # %% 1 - Frozen source, identity and exact model reproduction
    contract = carregar_contrato()
    source_zip = localizar_zip_original()
    validation, reference, _ = validar_control_strategy30(
        source_zip, contrato=contract,
    )
    with ZipFile(source_zip) as z:
        strategy = json.loads(z.read("strategy_manifest.json"))
        experiment = json.loads(z.read("experiment_manifest.json"))
    if experiment.get("market_data_signature_sha256") != contract["market_data_signature_sha256"]:
        raise RuntimeError("Market snapshot does not match Strategy #30")
    snapshot_dir = Path(os.environ.get("STRATEGY30_SNAPSHOT_DIR", str(SNAPSHOT))).expanduser()
    snapshot = load_snapshot_strategy30(
        snapshot_dir, contract=contract, experiment_manifest=experiment,
    )
    config = config_from_frozen_manifest(strategy, contract)
    (frames, calendar, calendar_source, symbols, folds, decisions,
     decision_to_fold, metadata) = _construir_contexto_execucao(snapshot.frames, config)
    # A referencia e congelada, mas os tamanhos pertencem ao manifesto:
    # nao repetir no executor as contagens historicas de outra versao.
    dimensions = validar_dimensoes_reproducao(
        symbols, folds, decisions, decision_to_fold, contract,
    )
    _write_csv(snapshot.records, out / "snapshot_identidade.csv")
    record_progress(out, "snapshot_validado", details={
        **dimensions,
        "arquivo": "snapshot_identidade.csv",
    })
    print("[fonte] Control Strategy #30, 75 ativos, 3 folds", flush=True)
    print(f"[saida] {out.resolve()}", flush=True)

    # %% 2 - Refit original models/rotation margin once, no new optimization
    policies = {}
    for fold in folds:
        fid = int(fold["fold_id"])
        train = calendar[:int(fold["train_end_index"])]
        calibration = calendar[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_train = calendar[:int(fold["final_fit_end_index"])]
        if not (train[-1] < calibration[0] and calibration[-1] < fold["test_start"]
                and final_train[-1] < fold["test_start"]):
            raise RuntimeError(f"Temporal violation in fold {fid}")
        record_progress(out, "calibracao_inicio", details={"fold": fid})
        print(f"[modelo] fold {fid}/3 calibrando", flush=True)
        models = _ajustar_modelos_lightgbm(
            frames, symbols, train, config, phase=f"motifs_fold_{fid}_calibration",
        )
        record_progress(out, "calibracao_modelos_concluida", details={"fold": fid})
        candidate_scores = [
            (float(m), _crescimento_politica_simples(
                _politica_utilidade(models, frames, symbols, config, float(m)),
                frames, symbols, calibration, config,
            ))
            for m in config.rotation_switch_margin_candidates
        ]
        choice = _selecionar_switch_margin_fold(config, fid, candidate_scores)
        if choice["selection_source"] != "calibration":
            raise RuntimeError("Control margin override not allowed")
        margin = max(float(config.rotation_switch_margin),
                     float(choice["selected_candidate_margin"]))
        del models
        record_progress(out, "treinamento_final_inicio", details={"fold": fid})
        print(f"[modelo] fold {fid}/3 ajuste final", flush=True)
        trained = _ajustar_modelos_lightgbm(
            frames, symbols, final_train, config, phase=f"motifs_fold_{fid}_final",
        )
        record_progress(out, "treinamento_final_concluido", details={"fold": fid})
        cache, _ = _precalcular_utilidades_modelo(
            trained, frames, symbols, pd.DatetimeIndex(fold["decision_dates"]), config,
        )
        record_progress(out, "utilidades_calculadas", details={"fold": fid})
        policies[fid] = _politica_utilidade(
            trained, frames, symbols, config, margin,
            fold_id=fid, calibrated_switch_margin=float(choice["selected_candidate_margin"]),
            utility_cache=cache,
        )

    record_progress(out, "treinamento_folds_concluido", details={
        "folds":len(folds),
    })
    check_required_files(out, ("snapshot_identidade.csv",))
    base = _politica_agendada(policies, decision_to_fold)
    record_progress(out, "reproducao_control_inicio")
    control = _simulate("Control Strategy #30", base, frames, symbols,
                        decisions, config, metadata)
    record_progress(out, "reproducao_control_concluida")
    parity, detail = compare_control_trajectories(
        reference, control.predictions, contract=contract,
    )
    _write_json(parity, out / "control_paridade.json")
    _write_csv(detail, out / "control_paridade_diaria.csv")
    check_required_files(
        out, ("snapshot_identidade.csv", "control_paridade.json",
              "control_paridade_diaria.csv"),
    )
    if not parity.get("fully_reproduced") or not math.isclose(
        float(control.metrics["strategy_ending_capital"]),
        float(contract["ending_capital"]), rel_tol=0, abs_tol=.01
    ):
        raise RuntimeError("Control mismatch; reject all motif comparisons")
    record_progress(out, "paridade_control_validada")
    print("[control] igualdade de patrimonio e posicoes validada", flush=True)
    _write_csv(control.predictions.reset_index(), out / "control_posicoes.csv")
    baseline_metrics = _risk_metrics(control.predictions, float(config.initial_capital))

    # %% 3 - Observable context: amplitude, volatility, breadth, shape.
    coverage_levels = {setting.min_coverage_fraction for _,setting,_ in EXPERIMENTS}
    if len(coverage_levels) != 1:
        raise RuntimeError("Inconsistent prespecified coverage")
    context = construct_market_context(
        snapshot.frames, symbols, calendar,
        coverage_fraction=next(iter(coverage_levels)),
    )
    _write_csv(context.coverage, out / "contexto_u75_cobertura.csv")
    record_progress(out, "contexto_causal_construido")
    benchmark = control.predictions["buy_hold_equity"]

    # %% 4 - Stateful Control + CASH, the original simulator and fees.
    runs = {}
    full_audit = []
    for label, setting, mode in EXPERIMENTS:
        record_progress(out,"simulacao_cash_inicio",details={
            "cenario":label,"modo":mode,"janela":setting.window,
            "horizonte":setting.horizon,
        })
        print(f"[contexto] {label}, modo={mode}, janela={setting.window}, "
              f"horizonte={setting.horizon}", flush=True)
        detector = ContextualCashRisk(context, setting)
        log = []
        policy = contextual_cash_policy(
            base, symbols, detector, mode=mode, log=log,
        )
        result = _simulate(
            label,policy,frames,symbols,decisions,config,metadata,
            benchmark=benchmark,
        )
        if len(log)!=len(decisions)-1:
            raise RuntimeError(f"{label}: insufficient decision audit")
        signals=pd.DataFrame(log)
        signals["decision_date"]=pd.to_datetime(signals["decision_date"],utc=True)
        cash=result.predictions["selected_asset"].eq("CASH")
        cash_in=cash & ~cash.shift(fill_value=False)
        cash_out=~cash & cash.shift(fill_value=False)
        financial=_risk_metrics(result.predictions,float(config.initial_capital))
        ok=signals["status"].eq("ok")
        flagged=signals["cash_signal"].fillna(False).astype(bool)

        runs[label]={
            **financial,
            "cash_sessions":int(cash.sum()),
            "cash_episodes":int(cash_in.sum()),
            "reentries":int(cash_out.sum()),
            "screened_decisions":int(ok.sum()),
            "insufficient_decisions":int((~ok & signals["candidate_asset"].ne("CASH")).sum()),
            "cash_override_decisions":int(flagged.sum()),
            "total_transaction_fees":float(result.metrics.get("total_transaction_fees") or 0),
            "strategy_sharpe":float(result.metrics.get("strategy_sharpe") or 0),
            "protocol":setting.__dict__,
            "matching_mode":mode,
        }
        _write_csv(signals,out/f"{label}_estimativas_risco.csv")
        predictions=result.predictions.reset_index()
        predictions["decision_date"]=pd.to_datetime(predictions["decision_date"],utc=True)
        joined=predictions.merge(
            signals,on="decision_date",how="left",validate="one_to_one",
        )
        if len(joined)!=len(signals) or joined["status"].isna().any():
            raise RuntimeError(f"{label}: missing causal diagnosis for a position")
        _write_csv(joined,out/f"{label}_decisoes_posicoes.csv")
        _write_csv(result.trades,out/f"{label}_operacoes.csv")
        full_audit.append(pd.DataFrame({
            "date":pd.to_datetime(result.predictions.index,utc=True),
            "scenario":label,
            "equity":result.predictions["strategy_equity"].to_numpy(float),
            "selected_asset":result.predictions["selected_asset"].to_numpy(),
        }))

        # Analysis AFTER policy replay. These future outcomes are never sent
        # to decision(), nor used to change signals or tune a threshold.
        labelled=[]
        index_by_date=detector.index
        h=setting.horizon
        for row in signals.itertuples(index=False):
            if row.candidate_asset=="CASH":
                continue
            pos=index_by_date[pd.Timestamp(row.decision_date)]
            future_end=pos+h+1
            opens=context.asset_log_opens[row.candidate_asset]
            if future_end>=len(opens):
                continue
            future=opens[pos+1:future_end+1]
            if not np.isfinite(future).all():
                continue
            realised=float(future[-1]-future[0])
            labelled.append({
                "decision_date":row.decision_date,
                "candidate_asset":row.candidate_asset,
                "context_signal":bool(row.cash_signal),
                "future_log_return_next_open":realised,
                "future_negative":bool(realised < 0),
                "future_end_date":context.calendar[future_end],
                "purpose":"posthoc_label_not_used_in_policy",
            })
        audit=pd.DataFrame(labelled,columns=[
            "decision_date","candidate_asset","context_signal",
            "future_log_return_next_open","future_negative",
            "future_end_date","purpose",
        ])
        _write_csv(audit,out/f"{label}_avaliacao_futura.csv")
        valid_alarm=audit.loc[audit["context_signal"]]
        runs[label]["labelled_alarms"]=int(len(valid_alarm))
        runs[label]["negative_future_in_alarms"]=int(valid_alarm["future_negative"].sum())
        runs[label]["positive_future_in_alarms"]=int(
            (~valid_alarm["future_negative"]).sum()
        )
        runs[label]["alarm_negative_fraction"]=(
            float(valid_alarm["future_negative"].mean())
            if len(valid_alarm) else None
        )
        record_progress(out,"simulacao_cash_concluida",details={
            "cenario":label,"cash_sessoes":runs[label]["cash_sessions"],
            "alarmes":runs[label]["cash_override_decisions"],
        })
        print(f"[resultado] {label}: CASH={runs[label]['cash_sessions']}, "
              f"MaxDD={runs[label]['max_drawdown']:.3%}, "
              f"ES95={runs[label]['daily_es95']:.3%}",flush=True)

    # %% 5 - Equities, episodes and complete research package
    series = pd.concat(full_audit, ignore_index=True)
    _write_csv(series, out / "patrimonio_cenarios.csv")
    fig, ax = plt.subplots(figsize=(13,5))
    ax.plot(control.predictions.index, control.predictions["strategy_equity"],
            label="Control original", lw=1.8)
    for label in runs:
        df = series.loc[series["scenario"].eq(label)]
        ax.plot(df["date"], df["equity"], lw=1.0, label=label)
    ax.set_yscale("log")
    ax.set_title("TCC: Control versus CASH por contexto causal (piloto)")
    ax.set_ylabel("Patrimonio simulado (log)")
    ax.grid(alpha=.2); ax.legend(loc="best"); fig.tight_layout()
    write_with_parent(
        out / "patrimonio_comparado.png",
        lambda destination: fig.savefig(destination, dpi=140),
    )
    plt.close(fig)
    comparison = pd.DataFrame([
        {"scenario":"Control",**baseline_metrics},
        *[{"scenario":k,**{p:v for p,v in record.items()
                            if p not in ("protocol","matching_mode")}}
          for k,record in runs.items()]
    ])
    _write_csv(comparison,out/"metricas_pareadas.csv")
    payload={
        "version":RUNNER_VERSION, "mathematical_engine_version": CONTEXT_VERSION,
        "status":"completed", "study":"Contextual next-open risk advantage versus CASH",
        "reference":validation,
        "control_parity":parity, "control":baseline_metrics,
        "scenarios":runs, "universe_size":len(symbols),
        "sessions":len(decisions)-1,
        "training_used_only_past":True, "automatic_parameter_search":False,
        "limitations":[
            "Historical exposures are measured from next open to a later open, known before the current decision; no future opening at current t+1 is used in risk scores.",
            "Cash advantage is a fixed-horizon, risk-adjusted proxy (synthetic buffer); actual broker costs and dynamic reentry are only calculated by the unchanged simulator.",
            "The previous Strategy #30 Control has a documented t+1 eligibility issue; parity is NOT a proof of full operational causality.",
            "2026 drawdowns and prior pilot variants were inspected before this hypothesis. Retrospective comparisons are NOT a genuinely new untouched test.",
            "Analogues are temporally separated but economically dependent; Gaussian confidence bounds are screening heuristics, not calibrated probabilistic guarantees.",
            "Both candidate selection and the risk detector inherit historical data constraints; do not claim overfitting is ruled out.",
            "Cash is re-evaluated at each closing, with actual next-open execution and no mandatory cash lock.",
            "The policy is strictly research-only, never promoted to the operational environment. Portfolio protection and missed rallies matter more than terminal capital.",
        ]
    }
    _write_json(payload, out / "resultado.json")
    summary = (
        "# U75 / vantagem contextual de CASH\n\n"+
        "Control foi reproduzido antes de comparar alternativas.\n\n"+
        comparison.to_string(index=False)+"\n\n"+
        "\n".join("- "+str(x) for x in payload["limitations"])+"\n"
    )
    write_with_parent(
        out / "resumo.md",
        lambda destination: destination.write_text(summary, encoding="utf-8"),
    )
    record_progress(out, "relatorio_financeiro_concluido")
    return payload


def main():
    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"[saida] Diretorio da execucao: {output.resolve()}", flush=True)
    success = False
    result = None
    try:
        record_progress(output, "execucao_iniciada", details={
            "versao": RUNNER_VERSION,
            "diretorio_saida": str(output.resolve()),
        })
        result = _run(output)
        expected = (
            "snapshot_identidade.csv",
            "control_paridade.json",
            "control_paridade_diaria.csv",
            "control_posicoes.csv",
            "contexto_u75_cobertura.csv",
            "patrimonio_cenarios.csv",
            "patrimonio_comparado.png",
            "metricas_pareadas.csv",
            "resultado.json",
            "resumo.md",
            "andamento.jsonl",
            "andamento_atual.json",
        )
        for label, _, _ in EXPERIMENTS:
            expected += (f"{label}_decisoes_posicoes.csv",
                         f"{label}_operacoes.csv",
                         f"{label}_estimativas_risco.csv",
                         f"{label}_avaliacao_futura.csv")
        check_required_files(output, expected)
        record_progress(output, "evidencias_verificadas")
        success = True
    except Exception:
        details = traceback.format_exc()
        last_stage = last_progress(output)
        print("[erro] Falha da pesquisa. Traceback:\n" + details, flush=True)
        try:
            record_progress(output, "falha_da_execucao", status="failed", details={
                "etapa_anterior": (
                    last_stage.get("stage") if last_stage else "desconhecida"
                ),
                "tipo_erro": details.splitlines()[-1] if details.splitlines() else "desconhecido",
            })
            write_with_parent(
                output / "ultimo_erro.txt",
                lambda destination: destination.write_text(
                    details, encoding="utf-8",
                ),
            )
        except Exception as log_error:
            # Keep trace outside the timestamp directory if storage keeps
            # removing it. Never silence a failure or declare success.
            fallback = ROOT / "output" / (
                f"ultimo_erro_contexto_cash_{output.name}.txt"
            )
            fallback_text = details + f"\nLog failure: {log_error!r}\n"
            write_with_parent(
                fallback,
                lambda destination: destination.write_text(
                    fallback_text, encoding="utf-8",
                ),
            )
            print(f"[erro] Log alternativo: {fallback}", flush=True)
    pack = output / f"pacote_vantagem_cash_contextual_u75_{output.name}.zip"
    def _package(destination):
        with ZipFile(destination, "w", ZIP_DEFLATED) as z:
            for f in sorted(output.iterdir()):
                if f.is_file() and f != destination and f.suffix in (
                    ".csv", ".md", ".json", ".jsonl", ".png", ".txt",
                ):
                    z.write(f, arcname=f.name)
    try:
        write_with_parent(pack, _package)
        record_progress(
            output,
            "pacote_gerado" if success else "pacote_de_falha_gerado",
            status="completed" if success else "failed",
            details={"arquivo_zip": str(pack)},
        )
        print(f"[pacote] {pack} status={'completed' if success else 'failed'}",
              flush=True)
    except Exception as package_error:
        print(f"[erro] Falha na gravacao do ZIP: {package_error!r}", flush=True)
        raise
    if not success:
        raise RuntimeError(
            f"Pesquisa interrompida; consultar {output / 'ultimo_erro.txt'}"
        )
    return result


if __name__ == "__main__":
    main()
