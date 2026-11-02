"""TCC Spyder F5: historical motifs, DTW and hypothetical CASH vs frozen U75.

Scientific protocol v1.0.0: two prespecified lookback/horizon combinations.
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
from pesquisa.motivos_historicos_cash_u75 import (
    VERSION, PatternSettings, build_series,
    HistoricalMotifRisk, wrap_cash_policy,
)
from pesquisar_controle_strategy30_spyder import localizar_zip_original
from pesquisa.saida_resiliente_motivos_u75 import (
    write_with_parent, check_required_files,
)
from pesquisa.andamento_motivos_u75 import record_progress, last_progress
from pesquisa.validacao_dimensoes_motivos_u75 import validar_dimensoes_reproducao

RUNNER_VERSION = "u75-historical-motifs-cash-v1.0.3"
OUTPUT = Path(os.environ.get(
    "TCC_MOTIFS_OUTPUT_DIR", str(ROOT / "output" / "motivos_historicos_cash_u75"),
)).expanduser()
SNAPSHOT = ROOT / "dados" / "strategy30_snapshot"

# Fixed before looking at this experiment's results. No capital optimization.
EXPERIMENTS = (
    ("market_w20_h5", PatternSettings(window=20, horizon=5), False),
    ("market_or_asset_w20_h5", PatternSettings(window=20, horizon=5), True),
    ("market_or_asset_w40_h10", PatternSettings(window=40, horizon=10), True),
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

    record_progress(out, "treinamento_tres_folds_concluido")
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

    # %% 3 - Build observables at each daily close. Prices AFTER t are not features.
    coverage_levels = {
        setting.minimum_coverage_fraction for _, setting, _ in EXPERIMENTS
    }
    if len(coverage_levels) != 1:
        raise RuntimeError("Scenarios have inconsistent U75 coverage settings")
    market_level, asset_levels, audit = build_series(
        snapshot.frames, symbols, calendar,
        minimum_coverage_fraction=next(iter(coverage_levels)),
    )
    _write_csv(audit, out / "indice_u75_cobertura.csv")
    record_progress(out, "indice_coletivo_construido")
    benchmark = control.predictions["buy_hold_equity"]

    # %% 4 - Full policy replay, once per prespecified scientific hypothesis
    runs = {}
    full_audit = []
    for label, setting, include_asset in EXPERIMENTS:
        record_progress(out, "simulacao_cash_inicio", details={
            "cenario": label, "janela": setting.window,
            "horizonte": setting.horizon,
        })
        print(f"[padroes] iniciando {label}: janela={setting.window} horizonte={setting.horizon}",
              flush=True)
        detector = HistoricalMotifRisk(market_level, asset_levels, calendar, setting)
        log = []
        policy = wrap_cash_policy(
            base, detector, symbols, include_asset=include_asset, log=log,
        )
        result = _simulate(
            label, policy, frames, symbols, decisions, config, metadata,
            benchmark=benchmark,
        )
        if len(log) != len(decisions)-1:
            raise RuntimeError(f"{label}: missing risk decision in portfolio replay")
        signals = pd.DataFrame(log)
        signals["decision_date"] = pd.to_datetime(signals["decision_date"], utc=True)
        cash = result.predictions["selected_asset"].eq("CASH")
        entries = cash & ~cash.shift(fill_value=False)
        reentries = ~cash & cash.shift(fill_value=False)
        results = _risk_metrics(result.predictions, float(config.initial_capital))
        result_record = {
            **results,
            "cash_sessions":int(cash.sum()),
            "cash_episodes":int(entries.sum()),
            "reentries":int(reentries.sum()),
            "overrides":int(signals["override_cash"].sum()),
            "market_alerts":int(signals["market_signal"].sum()),
            "asset_alerts":int(signals["asset_signal"].sum()),
            "total_transaction_fees":float(result.metrics.get("total_transaction_fees") or 0),
            "strategy_sharpe":float(result.metrics.get("strategy_sharpe") or 0),
            "parameter_protocol":setting.__dict__,
            "position_conditioned":include_asset,
        }
        runs[label] = result_record
        pred = result.predictions.reset_index()
        pred["decision_date"] = pd.to_datetime(pred["decision_date"], utc=True)
        joined = pred.merge(signals, on="decision_date", how="left", validate="one_to_one")
        if len(joined) != len(signals) or joined["override_cash"].isna().any():
            raise RuntimeError(f"{label}: position and signal dates misaligned")
        _write_csv(joined, out / f"{label}_decisoes_posicoes.csv")
        _write_csv(result.trades, out / f"{label}_operacoes.csv")
        full_audit.append(pd.DataFrame({
            "date":pd.to_datetime(result.predictions.index, utc=True),
            "scenario":label,
            "equity":result.predictions["strategy_equity"].to_numpy(float),
            "selected_asset":result.predictions["selected_asset"].to_numpy(),
        }))
        record_progress(out, "simulacao_cash_concluida", details={
            "cenario": label, "cash_sessoes": runs[label]["cash_sessions"],
        })
        print(f"[resultado] {label}: cash={runs[label]['cash_sessions']} "
              f"MaxDD={runs[label]['max_drawdown']:.3%} ES95={runs[label]['daily_es95']:.3%}",
              flush=True)

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
    ax.set_title("U75: Control versus CASH por analogos historicos (pesquisa)")
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
                            if p not in ("parameter_protocol","position_conditioned")}}
          for k,record in runs.items()]
    ])
    _write_csv(comparison,out/"metricas_pareadas.csv")
    payload={
        "version":RUNNER_VERSION, "mathematical_engine_version": VERSION,
        "status":"completed", "study":"DTW past-only + CASH",
        "reference":validation,
        "control_parity":parity, "control":baseline_metrics,
        "scenarios":runs, "universe_size":len(symbols),
        "sessions":len(decisions)-1,
        "training_used_only_past":True, "automatic_parameter_search":False,
        "limitations":[
            "This is a causal query-to-past distance profile, not the exact STUMPY full Matrix Profile.",
            "The Control ranking still inherits its separately documented t+1 eligibility limitation.",
            "The original U75 and 2026 drawdowns were already inspected; this replay is retrospective, not an untouched holdout.",
            "Close-to-close five/ten session labels drive the risk estimate, while financial replay uses the original next-open execution and real simulator costs.",
            "Matched analogous windows may be statistically dependent. One-sided Gaussian confidence bounds are descriptive and not a guarantee of nominal coverage.",
            "Analogue ES95 estimated from few episodes is volatile; do not promote signals without nested future validation.",
            "CASH is hypothetical with zero interest in the original simulator, no holding-period lock; no operational code was changed.",
            "Patrimonio terminal e apenas uma metrica secundaria; objetivo e perda evitada, ES95, MaxDD, falsos alertas e retorno sacrificado.",
        ]
    }
    _write_json(payload, out / "resultado.json")
    summary = (
        "# U75 / motivos historicos e CASH\n\n"+
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
            "indice_u75_cobertura.csv",
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
                         f"{label}_operacoes.csv")
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
                f"ultimo_erro_motivos_historicos_{output.name}.txt"
            )
            fallback_text = details + f"\nLog failure: {log_error!r}\n"
            write_with_parent(
                fallback,
                lambda destination: destination.write_text(
                    fallback_text, encoding="utf-8",
                ),
            )
            print(f"[erro] Log alternativo: {fallback}", flush=True)
    pack = output / f"pacote_motivos_historicos_cash_u75_{output.name}.zip"
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
