"""Pesquisa v1: exposicao CASH com Optimal Switching, sem mudar o Control.

Executar no Spyder (F5) ou python pesquisar_optimal_switching_cash_spyder.py.
Usa SOMENTE os CSVs versionados do universo base U67 (65 ativos efetivos).
O checkpoint U67+10 de US$ 587 milhoes nao e reutilizado nem sobrescrito.
"""

# %% 0 - Identidade e parametros definidos antes de ver os resultados
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math

import pandas as pd

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS, U67_EXPECTED_EXCLUSIONS
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import _construir_contexto_execucao
from engine.optimal_switching_cash import (
    SwitchingSettings,
    build_overlay,
    fit_switching_model,
)
from engine.rotacao import _politica_agendada, _simular_exato
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import prepare_model_frames
from pesquisar_expansao_u67_spyder import _train_baseline_states

ROOT = Path(__file__).resolve().parent
INPUT = SnapshotPaths.u67(ROOT)
OUTPUT = ROOT / "output" / "optimal_switching_cash"
CONTRACT = ROOT / "evidencias" / "manifest_universo_congelado.json"
SCRIPT_VERSION = "optimal-switching-cash-reentry-v1.0.0"
SCHEMA_VERSION = "optimal-switching-cash-v1"
SETTINGS = SwitchingSettings(
    lookback=20,
    horizon=10,
    downside_weight=1.0,
    minimum_observations=126,
)


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Tipo de serializacao nao suportado: {type(value).__name__}")


# %% 1 - Validar snapshot congelado, sem chamadas a corretora
def main() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    snapshot = validate_snapshot(INPUT)
    if snapshot["snapshot_sha256"] != contract["snapshot_sha256"]:
        raise RuntimeError("O snapshot U67 nao corresponde ao manifesto historico. Abortado.")
    requested = tuple(U67_REQUESTED_ASSETS)
    if tuple(snapshot["assets"]) != requested:
        raise RuntimeError("Ordem dos ativos diferente do snapshot U67.")
    raw, exclusions, data_diagnostics, data_audit = prepare_model_frames(
        INPUT, assets=requested, csv_float_precision="round_trip",
    )
    if {str(row["symbol"]) for row in exclusions} != set(U67_EXPECTED_EXCLUSIONS):
        raise RuntimeError("Exclusoes estruturais U67 inesperadas.")
    if len(raw) != 65:
        raise RuntimeError("A pesquisa requer 65 ativos efetivos do U67 base.")
    config = build_control_config(raw, CONFIG).copiar_modelo(
        update={"analysis_end_date": contract["analysis_end_date"]}
    )

    # %% 2 - Reaproveitar EXATAMENTE as mesmas previsoes por fold do Control
    (frames, common_dates, calendar, symbols, folds,
     decision_dates, decision_to_fold, decision_metadata) = (
        _construir_contexto_execucao(raw, config)
    )
    print(f"[dados] snapshot={snapshot['snapshot_sha256']} ativos={len(symbols)} folds={len(folds)}", flush=True)
    states, base_policies, calibration = _train_baseline_states(
        frames, common_dates, symbols, folds, config,
    )
    base_policy = _politica_agendada(base_policies, decision_to_fold)

    # %% 3 - Replay Control SEM modicacoes: checkpoint obrigatorio
    control = _simular_exato(
        "tcc_control_frozen",
        base_policy,
        frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="Control U67 intacto",
    )
    control_metrics = summarize_metrics(control, folds, float(config.initial_capital))
    expected = float(contract["baseline_ending_capital"])
    if not math.isclose(control_metrics["ending_capital"], expected, abs_tol=0.01, rel_tol=0):
        raise RuntimeError(
            "Checkpoint Control mudou: observado "
            f"{control_metrics['ending_capital']:.2f}, esperado {expected:.2f}. "
            "Nao confiar na comparacao."
        )
    print(f"[control] US$ {control_metrics['ending_capital']:,.2f} (checkpoint validado)", flush=True)

    # %% 4 - Ajustes estatisticos independentes: fechamento/abertura causais
    models_by_fold = {}
    model_rows = []
    for fold in folds:
        fid = int(fold["fold_id"])
        fit_end = pd.Timestamp(common_dates[int(fold["final_fit_end_index"]) - 1])
        models_by_fold[fid] = {}
        for index, symbol in enumerate(symbols, start=1):
            model = fit_switching_model(
                symbol, frames[symbol], fit_end=fit_end, settings=SETTINGS,
            )
            models_by_fold[fid][symbol] = model
            model_rows.append({
                "fold_id": fid,
                "symbol": symbol,
                "fit_end": fit_end.isoformat(),
                "observations": model.fit_observations,
                "state_lower_boundary": model.boundaries[0],
                "state_upper_boundary": model.boundaries[1],
                **{f"expected_return_state_{i}": model.expected_returns[i] for i in range(3)},
                **{f"downside_state_{i}": model.downside_losses[i] for i in range(3)},
            })
            if index % 15 == 0 or index == len(symbols):
                print(f"[switching-fit] fold={fid} {index}/{len(symbols)}", flush=True)

    # %% 5 - Replay da hipotese: mesmo motor financeiro, novas decisoes CASH
    diagnostics = {}
    overlay = build_overlay(
        base_policy, frames=frames, symbols=symbols,
        decision_to_fold=decision_to_fold,
        models_by_fold=models_by_fold,
        settings=SETTINGS, config=config, diagnostics=diagnostics,
    )
    switching = _simular_exato(
        "tcc_optimal_switching_experimental",
        overlay,
        frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Pesquisa U67 Switching CASH/Reentrada",
        method_line="- Modulo matematico independente; regimes e Bellman ajustados apenas com historico anterior.",
    )
    switching_metrics = summarize_metrics(switching, folds, float(config.initial_capital))

    # %% 6 - Evidencias sem sobrescrever checkpoints ou testes anteriores
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT / created
    output.mkdir(parents=True, exist_ok=False)
    control.predictions.to_csv(output / "control_predictions.csv", float_format="%.17g")
    control.trades.to_csv(output / "control_trades.csv", index=False)
    switching.predictions.to_csv(output / "switching_predictions.csv", float_format="%.17g")
    switching.trades.to_csv(output / "switching_trades.csv", index=False)
    pd.DataFrame(model_rows).to_csv(output / "switching_fold_models.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "control_fold_calibration.csv", index=False)
    decision_rows = [
        {"decision_date": timestamp.isoformat(), **values}
        for timestamp, values in sorted(diagnostics.items())
    ]
    pd.DataFrame(decision_rows).to_csv(output / "switching_decisions.csv", index=False)
    summary = {
        "version": SCRIPT_VERSION,
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "historical_cutoff": contract["analysis_end_date"],
        "source": "frozen_local_alpaca_raw_sip_csv_no_download",
        "research_only": True,
        "data_audit": data_audit,
        "data_diagnostics": data_diagnostics,
        "excluded_assets": exclusions,
        "effective_assets": symbols,
        "calendar": calendar,
        "folds": len(folds),
        "decision_sessions": len(decision_dates) - 1,
        "settings": SETTINGS.__dict__,
        "cost_approximation": "Bellman proportional; exact fees and slippage in common simulator",
        "control_metrics": control_metrics,
        "switching_metrics": switching_metrics,
        "comparison": {
            "delta_ending_capital": switching_metrics["ending_capital"] - control_metrics["ending_capital"],
            "delta_maximum_drawdown": switching_metrics["maximum_drawdown"] - control_metrics["maximum_drawdown"],
            "delta_sharpe": switching_metrics["sharpe"] - control_metrics["sharpe"],
            "cash_days_control": int(control.metrics.get("cash_days", 0)),
            "cash_days_switching": int(switching.metrics.get("cash_days", 0)),
            "fees_control": float(control.metrics.get("total_transaction_fees", 0)),
            "fees_switching": float(switching.metrics.get("total_transaction_fees", 0)),
            "cash_exits": int((switching.trades.get("reason", pd.Series(dtype=str)) == "MOVE_TO_CASH").sum()),
        },
        "scientific_limitations": [
            "Fixed U67 base universe, not historical U67+10 discovery-selected universe.",
            "No guarantee of performance; out-of-sample only within fixed historic universe.",
            "Switching model treats the Control nominee as single risky asset; asset-to-asset costs are accounted in replay, not in two-state Bellman.",
            "Regime signal from trailing close prices; rewards from historical next-open to subsequent-open returns.",
            "Fold parameter training excludes current and future test observations.",
            "Comparisons inherit universe selection and historical source limitations; later independent evaluation required.",
        ],
    }
    (output / "resultado.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[resultado] Control=US$ {control_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Switching=US$ {switching_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] capital_delta=US$ {summary['comparison']['delta_ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Control cash_days={summary['comparison']['cash_days_control']}; Switching cash_days={summary['comparison']['cash_days_switching']}", flush=True)
    print(f"[arquivos] {output}", flush=True)
    return summary


if __name__ == "__main__":
    main()
