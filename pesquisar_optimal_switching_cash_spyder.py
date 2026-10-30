"""Pesquisa v1: exposicao CASH com Optimal Switching, sem mudar o Control.

Executar no Spyder (F5) ou python pesquisar_optimal_switching_cash_spyder.py.
Usa o universo oficial congelado U67+10: 77 solicitados, 75 efetivos.
Requer o cache historico externo dos dez candidatos, com SHA-256 exato.
A main, o Control e o checkpoint de US$ 587.165.941,13 nao mudam.
"""

# %% 0 - Identidade e parametros definidos antes de ver os resultados
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

import pandas as pd

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS, U67_EXPECTED_EXCLUSIONS
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import _construir_contexto_execucao
from pesquisa.optimal_switching_cash import (
    SwitchingSettings,
    build_overlay,
    fit_switching_model,
)
from engine.rotacao import _politica_agendada, _simular_exato, _politica_utilidade
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import prepare_model_frames
from pesquisar_expansao_u67_spyder import (
    _train_baseline_states, _train_candidate_caches,
    _prepare_candidate_frames, _merge_utility_caches, _evaluate_scenario,
)
import pesquisar_expansao_u67_spyder as pesquisa

ROOT = Path(__file__).resolve().parent
INPUT = SnapshotPaths.u67(ROOT)
OUTPUT = ROOT / "output" / "optimal_switching_cash"
CONTRACT = ROOT / "evidencias" / "manifest_universo_congelado.json"
SCRIPT_VERSION = "optimal-switching-cash-reentry-v1.0.1"
SCHEMA_VERSION = "optimal-switching-cash-v1"
SETTINGS = SwitchingSettings(
    lookback=20,
    horizon=10,
    downside_weight=1.0,
    minimum_observations=126,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_selected_cache(contract: dict) -> tuple[str, ...]:
    """Validacao de TODOS os dez candidatos antes do treino, sem downloads."""
    selected = tuple(contract["selected_assets"])
    if selected != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES) or len(selected) != 10:
        raise RuntimeError("Universo selecionado nao corresponde ao Control historico.")
    if len(set(selected)) != 10:
        raise RuntimeError("Ativos selecionados duplicados.")
    for symbol in selected:
        hashes = contract["candidate_file_hashes"][symbol]
        for folder, field in (
            (pesquisa.FULL_DATA.raw_bars, "raw_sha256"),
            (pesquisa.FULL_DATA.corporate_actions, "corporate_actions_sha256"),
        ):
            path = folder / f"{symbol}.csv"
            if not path.is_file():
                raise RuntimeError(
                    f"Cache historico faltando: {path}. "
                    "Restaure o arquivo original com SHA-256; NAO baixe novo historico."
                )
            if _sha256_file(path) != hashes[field]:
                raise RuntimeError(
                    f"Hash divergente em {path}. "
                    "Preserve o snapshot congelado; NAO substitua dados."
                )
    return selected


def _control_policies_75(
    *,
    states: list,
    candidate_caches: dict,
    baseline_symbols: list[str],
    combined_symbols: list[str],
    combined_frames: dict,
    calibration_rows: list[dict],
    config: object,
    selected: tuple[str, ...],
) -> dict:
    """Reconstroi a MESMA politica calibrada de _evaluate_scenario."""
    policies = {}
    for state in states:
        fold_id = int(state.fold_id)
        rows = [
            row for row in calibration_rows
            if row["scenario"] == "TCC_SELECTED_UNIVERSE"
            and int(row["fold_id"]) == fold_id
            and row["selected"]
        ]
        if len(rows) != 1:
            raise RuntimeError(f"Margem de rotacao nao identificada para fold {fold_id}.")
        margin = float(rows[0]["candidate_margin"])
        extra_cache = {
            symbol: candidate_caches[symbol][fold_id].decision_cache
            for symbol in selected
        }
        merged_cache = _merge_utility_caches(
            state.decision_cache, baseline_symbols,
            extra_cache, combined_symbols,
        )
        policies[fold_id] = _politica_utilidade(
            {}, combined_frames, combined_symbols, config,
            max(float(config.rotation_switch_margin), margin),
            fold_id=fold_id, calibrated_switch_margin=margin,
            utility_cache=merged_cache,
        )
    return policies


# %% 1 - Validar snapshot congelado, sem chamadas a corretora
def main() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    snapshot = validate_snapshot(INPUT)
    if snapshot["snapshot_sha256"] != contract["snapshot_sha256"]:
        raise RuntimeError("O snapshot U67 nao corresponde ao manifesto historico. Abortado.")
    requested = tuple(U67_REQUESTED_ASSETS)
    if tuple(snapshot["assets"]) != requested:
        raise RuntimeError("Ordem dos ativos diferente do snapshot U67.")
    selected = _verify_selected_cache(contract)
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

    # %% 3 - Reconstruir exatamente os dez candidatos do Control U67+10.
    # O universo oficial possui 75 ativos efetivos. O cache foi validado
    # por hash antes de qualquer treinamento; nenhum download sera feito.
    candidate_frames, continuity, hashes = _prepare_candidate_frames(
        list(selected), common_dates, decision_dates, config,
    )
    if set(candidate_frames) != set(selected):
        raise RuntimeError("Candidatos ausentes ou inelegiveis no snapshot congelado.")
    for symbol in selected:
        if hashes[symbol] != contract["candidate_file_hashes"][symbol]:
            raise RuntimeError(f"{symbol}: hash divergiu apos carregar o candidato.")
    candidate_caches = {
        symbol: _train_candidate_caches(
            symbol, candidate_frames[symbol], states, config,
        )
        for symbol in selected
    }
    # Mesma avaliacao, margens e contabilizacao do checkpoint historico.
    control, control_metrics, selected_calibration = _evaluate_scenario(
        "TCC_SELECTED_UNIVERSE",
        list(selected),
        frames, candidate_frames, symbols,
        states, candidate_caches, folds,
        decision_dates, decision_to_fold, decision_metadata, config,
    )
    expected = float(contract["expected_ending_capital"])
    if not math.isclose(control_metrics["ending_capital"], expected, abs_tol=0.01, rel_tol=0):
        raise RuntimeError(
            "Checkpoint Control oficial de 75 ativos divergiu: "
            f"observado={control_metrics['ending_capital']:.8f}; "
            f"esperado={expected:.8f}. Comparacao bloqueada."
        )
    combined_symbols = sorted([*symbols, *selected])
    if len(combined_symbols) != 75 or len(set(combined_symbols)) != 75:
        raise RuntimeError("O universo efetivo deveria ter exatamente 75 ativos.")
    combined_frames = {**frames, **candidate_frames}
    combined_config = build_control_config(combined_frames, config)
    policies = _control_policies_75(
        states=states, candidate_caches=candidate_caches,
        baseline_symbols=symbols, combined_symbols=combined_symbols,
        combined_frames=combined_frames,
        calibration_rows=selected_calibration, config=combined_config,
        selected=selected,
    )
    base_policy = _politica_agendada(policies, decision_to_fold)
    print(
        f"[control-75] US$ {control_metrics['ending_capital']:,.2f} "
        f"ativos={len(combined_symbols)} checkpoint=validado", flush=True,
    )

    # %% 4 - Ajustes estatisticos independentes: fechamento/abertura causais
    models_by_fold = {}
    model_rows = []
    for fold in folds:
        fid = int(fold["fold_id"])
        fit_end = pd.Timestamp(common_dates[int(fold["final_fit_end_index"]) - 1])
        models_by_fold[fid] = {}
        for index, symbol in enumerate(combined_symbols, start=1):
            model = fit_switching_model(
                symbol, combined_frames[symbol], fit_end=fit_end, settings=SETTINGS,
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
                print(f"[switching-fit] fold={fid} {index}/{len(combined_symbols)}", flush=True)

    # %% 5 - Replay da hipotese: mesmo motor financeiro, novas decisoes CASH
    diagnostics = {}
    overlay = build_overlay(
        base_policy, frames=combined_frames, symbols=combined_symbols,
        decision_to_fold=decision_to_fold,
        models_by_fold=models_by_fold,
        settings=SETTINGS, config=combined_config, diagnostics=diagnostics,
    )
    switching = _simular_exato(
        "tcc_optimal_switching_experimental",
        overlay,
        combined_frames, combined_symbols, decision_dates, combined_config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Pesquisa U67+10 Switching CASH/Reentrada",
        method_line="- Modulo matematico independente; regimes e Bellman ajustados apenas com historico anterior.",
    )
    switching_metrics = summarize_metrics(switching, folds, float(combined_config.initial_capital))

    # %% 6 - Evidencias sem sobrescrever checkpoints ou testes anteriores
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT / created
    output.mkdir(parents=True, exist_ok=False)
    control.predictions.to_csv(output / "control_predictions.csv", float_format="%.17g")
    control.trades.to_csv(output / "control_trades.csv", index=False)
    switching.predictions.to_csv(output / "switching_predictions.csv", float_format="%.17g")
    switching.trades.to_csv(output / "switching_trades.csv", index=False)
    pd.DataFrame(model_rows).to_csv(output / "switching_fold_models.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "baseline_fold_calibration.csv", index=False)
    pd.DataFrame(selected_calibration).to_csv(output / "control_75_fold_calibration.csv", index=False)
    continuity.to_csv(output / "candidate_continuity.csv", index=False)
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
        "effective_assets": combined_symbols,
        "effective_asset_count": 75,
        "selected_additional_assets": list(selected),
        "candidate_file_hashes": hashes,
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
            "Frozen U67+10 selected universe; candidate choice used historical information and is not a prospective universe-selection validation.",
            "No guarantee of performance; out-of-sample only within fixed historic universe.",
            "Switching model treats the Control nominee as single risky asset; asset-to-asset costs are accounted in replay, not in two-state Bellman.",
            "Regime signal from trailing close prices; rewards from historical next-open to subsequent-open returns.",
            "Fold parameter training excludes current and future test observations.",
            "Comparisons inherit universe selection and historical source limitations; later independent evaluation required.",
        ],
    }
    pesquisa._write_json(output / "resultado.json", summary)
    print(f"[resultado] Control-75=US$ {control_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Switching=US$ {switching_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] capital_delta=US$ {summary['comparison']['delta_ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Control-75 cash_days={summary['comparison']['cash_days_control']}; Switching cash_days={summary['comparison']['cash_days_switching']}", flush=True)
    print(f"[arquivos] {output}", flush=True)
    return summary


if __name__ == "__main__":
    main()
