"""Spyder F5 | U75 rotacao-first + vantagem economica e recuperacao; 3 replays.

Pesquisa offline em snapshots congelados Strategy #30. Nao usa MongoDB,
Alpaca, atualiza MCT ou faz selecao de novos ativos. Nao altera o U75.
O Control original precisa ser reproduzido exatamente ANTES dos overlays.
Os eventos historicos de 2026 ja foram inspecionados: esta e pesquisa
EXPLORATORIA, nao uma nova validacao prospectiva independente.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import traceback
from typing import Any

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
from pesquisa.risco_reversao_antecipado_u75 import (
    EarlyRiskSettings, preparar_painel_supervisionado, estimar_folds,
)
from pesquisa.vantagem_economica_rotacao_u75 import (
    VERSION, EconomicSettings, preparar_resultados_economicos,
    estimar_vantagens_por_fold, politica_vantagem_economica,
)
from pesquisa.validacao_dimensoes_motivos_u75 import validar_dimensoes_reproducao
from pesquisar_controle_strategy30_spyder import localizar_zip_original


# %% Protocolo cientifico congelado ANTES de avaliar resultados deste estudo.
SETTINGS = EarlyRiskSettings()
ECONOMIC_SETTINGS = EconomicSettings()
ROOT_OUTPUT = ROOT / "output" / "vantagem_economica_rotacao_u75"
SNAPSHOT = Path(os.environ.get(
    "STRATEGY30_SNAPSHOT_DIR", str(ROOT / "dados" / "strategy30_snapshot"),
)).expanduser()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                   default=str), encoding="utf-8",
    )


def save_frame(path: Path, data: pd.DataFrame, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=index, index_label="timestamp" if index else None,
                float_format="%.17g")


def _package(out: Path, status: str) -> Path:
    path = out.parent / f"pacote_vantagem_economica_rotacao_u75_{out.name}.zip"
    if path.exists():
        raise RuntimeError("Ja existe um pacote com esse ID")
    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".json", ".csv", ".md", ".png"})
    entries = [
        {"name": p.relative_to(out).as_posix(), "size": p.stat().st_size,
         "sha256": sha256(p.read_bytes()).hexdigest()}
        for p in files
    ]
    manifest = {
        "schema_version": 1, "version": VERSION, "status": status,
        "run_id": out.name, "raw_market_data_included": False,
        "files": entries,
    }
    tmp = path.with_suffix(".tmp")
    try:
        with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("manifesto_pacote.json",
                       json.dumps(manifest, indent=2, ensure_ascii=False))
            for p in files:
                z.write(p, p.relative_to(out).as_posix())
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise RuntimeError("ZIP corrompido")
            for row in entries:
                if (sha256(z.read(row["name"])).hexdigest() != row["sha256"]
                        or len(z.read(row["name"])) != row["size"]):
                    raise RuntimeError(f"Integridade invalida: {row['name']}")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _simulate(label, policies, diagnostics, frames, symbols, dates,
              fold_mapping, metadata, config, *, benchmark=None):
    return _simular_exato(
        label, _politica_agendada(policies, fold_mapping),
        frames, symbols, dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=metadata,
        policy_decision_diagnostics=diagnostics,
        model_label=label,
        benchmark_override=benchmark,
        benchmark_override_name="Control fixed benchmark"
        if benchmark is not None else None,
    )


def _risk_metrics(run, initial: float) -> dict:
    equity = pd.to_numeric(run.predictions["strategy_equity"], errors="coerce")
    if equity.empty or not np.isfinite(equity).all() or (equity <= 0).any():
        raise RuntimeError("Patrimonio nao finito")
    values = np.r_[float(initial), equity.to_numpy(float)]
    daily = np.diff(values) / values[:-1]
    cut = float(np.quantile(daily, .05))
    metrics = run.metrics
    return {
        "ending_capital": float(equity.iloc[-1]),
        "strategy_sharpe": float(metrics["strategy_sharpe"]),
        "maximum_drawdown": float(metrics["strategy_maximum_drawdown"]),
        "daily_es95": float(daily[daily <= cut].mean()),
        "worst_daily_return": float(daily.min()),
        "negative_sessions": int(np.sum(daily < 0)),
        "cash_sessions": int(metrics["cash_days"]),
        "capital_rotations": int(metrics["capital_rotations"]),
        "total_transaction_fees": float(metrics["total_transaction_fees"]),
    }


def _predictive_metrics(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in p.groupby("fold"):
        for name, target in (
            ("downside", "downside_event"),
            ("recovery", "reversal_event"),
        ):
            col = "p_" + name
            if col not in group:
                raise RuntimeError(f"Probabilidade ausente: {col}")
            used = group.dropna(subset=[col, target])
            if name == "recovery":
                used = used[used["recent_drawdown"]]
            if used.empty:
                rows.append({"fold": int(fold), "model": name, "n": 0,
                             "brier": np.nan, "event_frequency": np.nan})
                continue
            actual = used[target].astype(float).to_numpy()
            prob = used[col].astype(float).to_numpy()
            rows.append({
                "fold": int(fold), "model": name,
                "n": int(len(used)),
                "event_frequency": float(actual.mean()),
                "mean_predicted_probability": float(prob.mean()),
                "brier": float(np.mean((prob - actual) ** 2)),
            })
    return pd.DataFrame(rows)


def _plot_curves(runs: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, run in runs.items():
        curve = run.predictions["strategy_equity"].astype(float)
        ax.plot(curve.index, curve, label=label, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title("Patrimonio simulado U75 (escala logaritmica)")
    ax.set_ylabel("USD")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "patrimonio_tres_politicas.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, run in runs.items():
        curve = run.predictions["strategy_equity"].astype(float)
        drawdown = curve / curve.cummax() - 1.0
        ax.plot(curve.index, drawdown * 100.0, label=label, linewidth=1.3)
    ax.set_title("Drawdown da carteira por politica")
    ax.set_ylabel("% abaixo do pico")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "drawdowns_tres_politicas.png", dpi=150)
    plt.close(fig)


def run(out: Path) -> dict:
    # %% 1. Contract and exact data identity, no downloads.
    contract = carregar_contrato()
    archive = localizar_zip_original()
    reference_audit, reference_predictions, _ = validar_control_strategy30(
        archive, contrato=contract,
    )
    with ZipFile(archive) as z:
        strategy = json.loads(z.read("strategy_manifest.json"))
        experiment = json.loads(z.read("experiment_manifest.json"))
    if experiment["market_data_signature_sha256"] != contract["market_data_signature_sha256"]:
        raise RuntimeError("Snapshot do ZIP difere do contrato Control")
    snap = load_snapshot_strategy30(
        SNAPSHOT, contract=contract, experiment_manifest=experiment,
    )
    save_frame(out / "auditoria_snapshot.csv", snap.records)
    config = config_from_frozen_manifest(strategy, contract)
    (frames, calendar, source, symbols, folds, dates,
     fold_by_date, metadata) = _construir_contexto_execucao(snap.frames, config)
    dimensions = validar_dimensoes_reproducao(
        symbols, folds, dates, fold_by_date, contract,
    )
    print(f"[fonte] Control Strategy #30, {len(symbols)} ativos, "
          f"{len(folds)} folds, calendario={source}", flush=True)

    # %% 2. Build two independent prospective targets and train per fold.
    # Labels may use future OHLCV, but only inside completed training records.
    panel = preparar_painel_supervisionado(frames, calendar, SETTINGS)
    signals, folds_audit, predicted = estimar_folds(panel, folds, calendar, SETTINGS)
    save_frame(out / "predicoes_risco_reversao.csv", predicted)
    save_frame(out / "treino_modelos_folds.csv", folds_audit)
    save_frame(out / "metricas_preditivas_por_fold.csv", _predictive_metrics(predicted))
    panel_economic = preparar_resultados_economicos(
        panel, frames, calendar, ECONOMIC_SETTINGS.horizon_days,
        SETTINGS.reversal_horizon,
    )
    economic_signals, economic_stats, economic_forecasts = estimar_vantagens_por_fold(
        panel_economic, signals, folds, calendar,
        ECONOMIC_SETTINGS, SETTINGS.reversal_horizon,
    )
    save_frame(out / "estimativas_economicas_ativos.csv", economic_forecasts)
    save_frame(out / "treino_economico_por_fold.csv", economic_stats)
    save_json(out / "protocolo_cientifico.json", {
        "version": VERSION, "settings": SETTINGS.as_record(),
        "economic_settings": ECONOMIC_SETTINGS.record(),
        "control": reference_audit["id"],
        "snapshot": reference_audit["market_data_signature_sha256"],
        "experimental_scope": "historical_inspected_sample",
        "nested_independent_holdout": False,
        "baseline_next_session_eligibility_issue_inherited": True,
        "base_universe_modified": False,
        "cash_optimized": False,
        "fold_dimensions": dimensions,
    })

    # %% 3. Reuse exact Control training/calibration, same models for all policies.
    scenarios = (
        "Control",
        "Control + vantagem economica",
        "Control + vantagem economica + reversao",
    )
    policies = {name: {} for name in scenarios}
    diagnostics = {name: {} for name in scenarios}
    records = {name: [] for name in scenarios if name != "Control"}
    margins = []
    for fold in folds:
        fid = int(fold["fold_id"])
        train = calendar[:int(fold["train_end_index"])]
        calibration = calendar[
            int(fold["calibration_start_index"]):
            int(fold["calibration_end_index"])
        ]
        final_train = calendar[:int(fold["final_fit_end_index"])]
        if not (train[-1] < calibration[0]
                and calibration[-1] < fold["test_start"]
                and final_train[-1] < fold["test_start"]):
            raise RuntimeError(f"Vazamento temporal em fold {fid}")
        print(f"[modelos] fold {fid}/{len(folds)} calibrando", flush=True)
        trained_calibration = _ajustar_modelos_lightgbm(
            frames, symbols, train, config,
            phase=f"early_risk_fold_{fid}_calibration",
        )
        candidates = [
            (float(margin), _crescimento_politica_simples(
                _politica_utilidade(
                    trained_calibration, frames, symbols, config, float(margin),
                ), frames, symbols, calibration, config,
            ))
            for margin in config.rotation_switch_margin_candidates
        ]
        selected = _selecionar_switch_margin_fold(config, fid, candidates)
        if selected["selection_source"] != "calibration":
            raise RuntimeError("Override de margem proibido")
        effective = max(float(config.rotation_switch_margin),
                        float(selected["selected_candidate_margin"]))
        margins.append({
            "fold": fid, "calibrated_margin": float(selected["selected_candidate_margin"]),
            "effective_margin": effective,
        })
        del trained_calibration
        print(f"[modelos] fold {fid}/{len(folds)} treino final", flush=True)
        trained = _ajustar_modelos_lightgbm(
            frames, symbols, final_train, config,
            phase=f"early_risk_fold_{fid}_final",
        )
        cache, _ = _precalcular_utilidades_modelo(
            trained, frames, symbols,
            pd.DatetimeIndex(fold["decision_dates"]), config,
        )
        for name in scenarios:
            original = _politica_utilidade(
                trained, frames, symbols, config, effective,
                fold_id=fid,
                calibrated_switch_margin=float(selected["selected_candidate_margin"]),
                utility_cache=cache, decision_diagnostics=diagnostics[name],
            )
            if name == "Control":
                policies[name][fid] = original
            else:
                policies[name][fid] = politica_vantagem_economica(
                    original, symbols, cache, economic_signals[fid],
                    ECONOMIC_SETTINGS,
                    use_recovery=(name == scenarios[2]),
                    audit=records[name],
                    diagnostics=diagnostics[name],
                    minimum_score=float(config.rotation_cash_threshold),
                )
    save_frame(out / "margens_control_por_fold.csv", pd.DataFrame(margins))

    # %% 4. Independently simulate Control and reject if a single day differs.
    base = _simulate("Control", policies["Control"], diagnostics["Control"],
                     frames, symbols, dates, fold_by_date, metadata, config)
    parity, detail = compare_control_trajectories(
        reference_predictions, base.predictions, contract=contract,
    )
    save_json(out / "auditoria_paridade.json", parity)
    save_frame(out / "auditoria_paridade_diaria.csv", detail)
    if not parity["fully_reproduced"]:
        raise RuntimeError(
            f"Control diferente: {parity['mismatch_sessions']} sessoes; "
            f"primeira: {parity['first_mismatch_date']}. "
            "Replays experimentais bloqueados."
        )
    print("[control] paridade exata de posicoes e patrimonio aprovada", flush=True)
    runs = {"Control": base}
    benchmark = base.predictions["buy_hold_equity"]
    for name in scenarios[1:]:
        print(f"[replay] {name}", flush=True)
        runs[name] = _simulate(name, policies[name], diagnostics[name],
                               frames, symbols, dates, fold_by_date,
                               metadata, config, benchmark=benchmark)
        if len(records[name]) != len(base.predictions):
            raise RuntimeError(f"{name}: faltam auditorias de decisao")
        slug = "economico_risco" if name == scenarios[1] else "economico_reversao"
        save_frame(out / f"{slug}_decisoes.csv", pd.DataFrame(records[name]))
    # %% 5. Full portfolio evidence, including actual CASH and opportunity cost.
    results = {}
    for name, run in runs.items():
        slug = ("control" if name == "Control" else
                "economico_risco" if name == scenarios[1] else "economico_reversao")
        save_frame(out / f"{slug}_predictions.csv",
                   run.predictions.reset_index())
        save_frame(out / f"{slug}_trades.csv", run.trades)
        results[name] = _risk_metrics(run, float(config.initial_capital))
    comparison = pd.DataFrame({
        "timestamp": base.predictions.index,
        "control_asset": base.predictions["selected_asset"].to_numpy(),
        "control_equity": base.predictions["strategy_equity"].to_numpy(),
        "risk_asset": runs[scenarios[1]].predictions["selected_asset"].to_numpy(),
        "risk_equity": runs[scenarios[1]].predictions["strategy_equity"].to_numpy(),
        "recovery_asset": runs[scenarios[2]].predictions["selected_asset"].to_numpy(),
        "recovery_equity": runs[scenarios[2]].predictions["strategy_equity"].to_numpy(),
    })
    comparison["risk_changed_position"] = comparison["control_asset"] != comparison["risk_asset"]
    comparison["recovery_changed_position"] = comparison["control_asset"] != comparison["recovery_asset"]
    comparison["risk_capital_delta"] = comparison["risk_equity"] - comparison["control_equity"]
    comparison["recovery_capital_delta"] = comparison["recovery_equity"] - comparison["control_equity"]
    initial = float(config.initial_capital)
    for key in ("control", "risk", "recovery"):
        previous = comparison[f"{key}_equity"].shift(1).fillna(initial)
        comparison[f"{key}_return"] = comparison[f"{key}_equity"] / previous - 1.0
    comparison["risk_daily_difference"] = (
        comparison["risk_return"] - comparison["control_return"]
    )
    comparison["recovery_daily_difference"] = (
        comparison["recovery_return"] - comparison["control_return"]
    )
    save_frame(out / "comparacao_dia_a_dia.csv", comparison)
    most_changed = comparison.assign(
        absolute_difference=comparison["recovery_daily_difference"].abs(),
    ).nlargest(30, "absolute_difference")
    save_frame(out / "maiores_diferencas_pregao.csv", most_changed)
    _plot_curves(runs, out)
    payload = {
        "status": "completed", "version": VERSION,
        "research_only": True, "control_parity": parity,
        "control_assets": len(symbols), "fold_count": len(folds),
        "method": "rotation_first_economic_conditional_return_and_recovery",
        "settings": SETTINGS.as_record(),
        "economic_settings": ECONOMIC_SETTINGS.record(), "results": results,
        "overrides": {name: int(sum(bool(r["changed_control_action"])
                                    for r in records[name])) for name in records},
        "cash_last_resort": {name: int(sum(bool(r["cash_last_resort"])
                                         for r in records[name])) for name in records},
        "warning": (
            "Estudo exploratorio em historico inspecionado, sem holdout virgem. "
            "A dependencia de elegibilidade futura do Control permanece. "
            "Nao promover ao MCT nem alterar parametros segundo esta amostra."
        ),
    }
    save_json(out / "resultado.json", payload)
    lines = [
        "# Pesquisa U75: vantagem economica das rotacoes com risco e recuperacao",
        "", "Comparacao integral com calibracao e LightGBM originais.",
        "Control seleciona antes. Novas rotacoes requerem vantagem economica",
        "estimada apos custo adicional e penalizacao de queda de cauda.",
        "Recuperacao nao e bloqueada por uma probabilidade de queda fixa.",
        "Sem novos ativos e sem alteracao do MCT.",
        f"Paridade exata: {parity['fully_reproduced']}", "",
        "| Cenario | Capital final USD | Sharpe | MaxDD | ES95 diario | Cash | Rotacoes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in results.items():
        lines.append(
            f"| {name} | {metric['ending_capital']:,.2f} | "
            f"{metric['strategy_sharpe']:.4f} | {metric['maximum_drawdown']:.2%} | "
            f"{metric['daily_es95']:.2%} | {metric['cash_sessions']} | "
            f"{metric['capital_rotations']} |"
        )
    lines.extend([
        "", "## Interpretacao obrigatoria", "",
        "As quedas historicas, inclusive CAPR e AMS em 2026, ja orientaram "
        "as hipoteses. O teste nao e confirmatorio.",
        "Rotulos futuros maturam antes da primeira data de teste, mas a "
        "elegibilidade do ranking Control historico ainda usa t+1.",
        "CASH apenas quando todos os ativos elegiveis apresentam retorno "
        "economico previsto nao positivo. Caso contrario buscar rotacao.",
        "Registrar perdas evitadas, ganhos sacrificados e reentradas.",
        "Estimativas financeiras de 5 dias sao condicionais ao passado.",
        "Limiares sao exploratorios e nao foram otimizados no historico.",
    ])
    (out / "resumo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> dict:
    out = ROOT_OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    status = "failed"
    try:
        result = run(out)
        status = "completed"
        return result
    except Exception as exc:
        save_json(out / "erro_execucao.json", {
            "status": "failed", "version": VERSION,
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(limit=25),
        })
        raise
    finally:
        path = _package(out, status)
        print(f"[pacote] {path}", flush=True)


if __name__ == "__main__":
    main()
