"""Optimal Switching / CASH: estudo com o universo oficial de 75 ativos.

Uma unica carga e um unico treino do Control sobre TODOS os 75 ativos.
Sem selecao de universo, sem baixas, sem fallback e sem banco de dados.
O checkpoint do Control deve ser reproduzido ao centavo antes de comparar.
"""

# %% 0 - Identidade da pesquisa e parametros fixados
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import traceback

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG, construir_configuracao_controle
from engine.execucao import calcular_taxas_referencia, aplicar_deslizamento
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    selecionar_switch_margin,
)
from engine.rotacao import (
    _precalcular_utilidades_modelo,
    _crescimento_politica_simples,
    _politica_utilidade,
    _politica_agendada,
    _simular_exato,
    construir_quadro_rotacao,
)
from pesquisa.optimal_switching_cash import (
    SwitchingSettings, build_overlay, fit_switching_model,
)
from pesquisa.pacote_analise_switching import gerar_zip_resultados
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import summarize_metrics
from reproducao.preparacao import (
    load_raw_bar_file, load_actions_file, split_normalize,
    structural_identity_issue,
)
from pesquisar_expansao_u67_spyder import _generic_identity_issue, _write_json

ROOT = Path(__file__).resolve().parent
SOURCE = SnapshotPaths.u67(ROOT)
CANDIDATE_SOURCE = SnapshotPaths.from_root(
    ROOT / "dados" / ".u67_expansion_cache" / "full_raw_20261006"
)
UNIVERSE = ROOT / "pesquisa" / "universo_control_75.json"
CALENDAR_CONTRACT = ROOT / "pesquisa" / "contrato_calendario_control_75.json"
CONTRACT = ROOT / "evidencias" / "manifest_universo_congelado.json"
OUTPUT = ROOT / "output" / "optimal_switching_cash"
SCRIPT_VERSION = "optimal-switching-cash-reentry-v1.0.4"
SCHEMA_VERSION = "optimal-switching-cash-v2"
SETTINGS = SwitchingSettings(
    lookback=20,
    horizon=10,
    downside_weight=1.0,
    minimum_observations=126,
)


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


# %% 1 - Carregamento unico do universo oficial
def carregar_universo_oficial() -> tuple[dict[str, pd.DataFrame], dict, dict, dict]:
    """Carrega as 75 series completas sem exclusoes ou universo intermediario."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    definition = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    assets = tuple(definition["assets"])
    if definition["effective_asset_count"] != 75 or len(assets) != 75 or len(set(assets)) != 75:
        raise RuntimeError("Universo cientifico deve conter exatamente 75 ativos distintos.")
    if tuple(sorted(assets)) != assets:
        raise RuntimeError("Manifesto do universo deve estar em ordem canonica.")
    selected = set(contract["selected_assets"])
    if not selected.issubset(assets):
        raise RuntimeError("O universo oficial nao contem todos os ativos do checkpoint.")
    # Verifica a fonte versionada como conjunto fechado (integridade, nao selecao).
    snapshot = validate_snapshot(SOURCE)
    if snapshot["snapshot_sha256"] != contract["snapshot_sha256"]:
        raise RuntimeError("Snapshot congelado diferente do checkpoint, sem substituicao automatica.")

    # Validar disponibilidade e hashes de todo o universo ANTES de processar.
    sources = {}
    for symbol in assets:
        path_group = CANDIDATE_SOURCE if symbol in selected else SOURCE
        raw_path = path_group.raw_bars / f"{symbol}.csv"
        action_path = path_group.corporate_actions / f"{symbol}.csv"
        if not raw_path.is_file() or not action_path.is_file():
            raise RuntimeError(
                f"{symbol}: arquivos congelados ausentes. "
                "Restaure os arquivos originais, sem fazer download."
            )
        if symbol in selected:
            expected = contract["candidate_file_hashes"][symbol]
            if (_file_sha256(raw_path) != expected["raw_sha256"]
                or _file_sha256(action_path) != expected["corporate_actions_sha256"]):
                raise RuntimeError(f"{symbol}: hashes RAW/Corporate Actions divergentes.")
        sources[symbol] = (raw_path, action_path)

    frames = {}
    print(f"[universe] official_assets={len(assets)} status=validated", flush=True)
    for index, symbol in enumerate(assets, start=1):
        raw_path, action_path = sources[symbol]
        bars = load_raw_bar_file(raw_path, float_precision="round_trip")
        corporate_actions = load_actions_file(action_path)
        # A identidade e validada. Falha estrutural ABORTA; nao remove ativos.
        issue = (
            _generic_identity_issue(symbol, corporate_actions)
            if symbol in selected
            else structural_identity_issue(symbol, corporate_actions)
        )
        if issue is not None:
            raise RuntimeError(
                f"{symbol}: identidade estrutural invalida; pesquisa abortada, "
                "sem reconstruir ou alterar o universo."
            )
        normalized, _ = split_normalize(bars, corporate_actions)
        frames[symbol] = normalized
        if index % 15 == 0 or index == 75:
            print(f"[data] {index}/75 assets_loaded", flush=True)
    if len(frames) != 75 or tuple(sorted(frames)) != assets:
        raise RuntimeError("Carga integral dos 75 ativos nao foi concluida.")
    return frames, contract, snapshot, definition


def construir_calendario_control_75(raw: dict[str, pd.DataFrame], config: object) -> pd.DatetimeIndex:
    """Calendario apos os indicadores de 200 sessoes, nunca OHLCV bruto.

    Identico ao calendario AAPL utilizado pela reproducao oficial.
    Exclui do calendario APENAS as sessoes iniciais sem atributos completos;
    nenhum ativo e retirado do universo de 75.
    """
    if "AAPL" not in raw:
        raise RuntimeError("O ativo de referencia AAPL nao consta do universo oficial.")
    prepared = construir_quadro_rotacao(raw["AAPL"], config)
    calendar = pd.DatetimeIndex(prepared.index)
    if calendar.empty or not calendar.is_monotonic_increasing:
        raise RuntimeError("Calendario oficial preparado esta vazio/desordenado.")
    return calendar


def verificar_protocolo_cronologico(
    calendar: pd.DatetimeIndex,
    folds: list[dict],
    decision_dates: pd.DatetimeIndex,
) -> dict:
    """Bloqueia regressao de folds antes de ajustar qualquer LightGBM."""
    contract = json.loads(CALENDAR_CONTRACT.read_text(encoding="utf-8"))
    expected = {
        "prepared_sessions": len(calendar),
        "walk_forward_folds": len(folds),
        "out_of_sample_sessions": len(decision_dates) - 1,
    }
    for key, observed in expected.items():
        if int(contract[key]) != observed:
            raise RuntimeError(
                "Protocolo cronologico do Control de 75 ativos divergente "
                f"({key}): observado={observed} esperado={contract[key]}. "
                "Treinamento bloqueado. Verifique o calendario de atributos."
            )
    first_exec = pd.Timestamp(decision_dates[1]).date().isoformat()
    last_exec = pd.Timestamp(decision_dates[-1]).date().isoformat()
    if (first_exec != contract["first_execution_date"]
        or last_exec != contract["last_execution_date"]):
        raise RuntimeError(
            "Datas cronologicas do checkpoint oficial divergentes: "
            f"observado={first_exec}..{last_exec}, "
            f"esperado={contract['first_execution_date']}..{contract['last_execution_date']}."
        )
    return contract


# %% 2 - Calibracao e treinamento Control sobre os 75 simultaneamente
def ajustar_control_75(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    folds: list[dict],
    config: object,
) -> tuple[dict, list[dict]]:
    policies = {}
    calibration = []
    candidates = tuple(float(v) for v in config.rotation_switch_margin_candidates)
    for idx, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        train = common_dates[:int(fold["train_end_index"])]
        calibrate = common_dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_fit = common_dates[:int(fold["final_fit_end_index"])]
        decisions = pd.DatetimeIndex(fold["decision_dates"])
        print(f"[control-75] fold={idx}/{len(folds)} calibration assets=75", flush=True)
        models = _ajustar_modelos_lightgbm(
            frames, symbols, train, config,
            phase=f"optimal_switching_control75_fold_{fold_id}_calibration",
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            models, frames, symbols, calibrate, config,
        )
        del models
        trial_scores = []
        for margin in candidates:
            policy = _politica_utilidade(
                {}, frames, symbols, config, margin,
                utility_cache=calibration_cache,
            )
            growth = float(_crescimento_politica_simples(
                policy, frames, symbols, calibrate, config
            ))
            trial_scores.append((margin, growth))
        selected = selecionar_switch_margin(trial_scores)
        choice = float(selected["selected_candidate_margin"])
        effective_margin = max(float(config.rotation_switch_margin), choice)
        for margin, growth in trial_scores:
            calibration.append({
                "scenario": "CONTROL_75",
                "fold_id": fold_id,
                "candidate_margin": margin,
                "calibration_score": growth,
                "selected": bool(abs(margin - choice) <= 1e-12),
            })
        print(f"[control-75] fold={idx}/{len(folds)} final_fit assets=75", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames, symbols, final_fit, config,
            phase=f"optimal_switching_control75_fold_{fold_id}_final",
        )
        cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, decisions, config,
        )
        policies[fold_id] = _politica_utilidade(
            {}, frames, symbols, config, effective_margin,
            fold_id=fold_id, calibrated_switch_margin=choice,
            utility_cache=cache,
        )
        del final_models
    return policies, calibration


# %% 3 - Execucao sobre o universo completo
def _executar_experimento(output: Path) -> dict:
    raw, contract, snapshot, universe = carregar_universo_oficial()
    config = construir_configuracao_controle(
        CONFIG, assets=tuple(sorted(raw)),
    ).copiar_modelo(
        update={"analysis_end_date": contract["analysis_end_date"]}
    )
    # A fonte de calendario e AAPL, mas APOS computar os atributos.
    # O calendario bruto adicionava 199 sessoes invalidas ao protocolo,
    # gerando 4 folds e um resultado financeiro nao comparavel.
    reference_calendar = construir_calendario_control_75(raw, config)
    (
        frames, common_dates, calendar, symbols, folds,
        decision_dates, decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao(
        raw, config, calendar_override=reference_calendar,
        calendar_source_label="AAPL",
    )
    if len(symbols) != 75 or tuple(symbols) != tuple(universe["assets"]):
        raise RuntimeError("O motor nao recebeu integralmente os 75 ativos.")
    chronological_contract = verificar_protocolo_cronologico(
        common_dates, folds, decision_dates,
    )
    print(
        "[calendar] reference=AAPL "
        f"raw_sessions={len(raw['AAPL'])} "
        f"prepared_sessions={len(common_dates)} "
        f"folds={len(folds)} oos_sessions={len(decision_dates) - 1}",
        flush=True,
    )
    # Checagem de precos executaveis: falhar, jamais excluir um ativo.
    execution_dates = decision_dates[1:]
    for symbol in symbols:
        prices = frames[symbol].reindex(execution_dates)[["open", "high", "low", "close"]]
        values = prices.to_numpy(dtype=float)
        if not np.isfinite(values).all() or bool((values <= 0).any()):
            raise RuntimeError(f"{symbol}: preco invalido no replay, sem exclusao.")

    print(
        f"[dados] official_assets={len(symbols)} folds={len(folds)} "
        f"snapshot={snapshot['snapshot_sha256']}", flush=True,
    )
    policies, calibration = ajustar_control_75(
        frames, common_dates, symbols, folds, config,
    )
    base_policy = _politica_agendada(policies, decision_to_fold)
    control = _simular_exato(
        "tcc_control_frozen",
        base_policy, frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="Control oficial 75 ativos",
    )
    control_metrics = summarize_metrics(control, folds, float(config.initial_capital))
    expected = float(contract["expected_ending_capital"])
    if not math.isclose(control_metrics["ending_capital"], expected, abs_tol=0.01, rel_tol=0):
        raise RuntimeError(
            "Checkpoint Control oficial divergente. Comparacao bloqueada: "
            f"observado={control_metrics['ending_capital']:.8f}, esperado={expected:.8f}."
        )
    print(
        f"[control-75] capital=US$ {control_metrics['ending_capital']:,.2f} "
        f"status=checkpoint_validado", flush=True,
    )
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
                **{f"adjusted_reward_state_{i}": (
                    model.expected_returns[i] - SETTINGS.downside_weight * model.downside_losses[i]
                ) for i in range(3)},
                **{f"transition_{i}_{j}": float(model.transitions[i, j])
                   for i in range(3) for j in range(3)},
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
        model_label="Pesquisa Control 75 Switching CASH/Reentrada",
        method_line="- Modulo matematico independente; regimes e Bellman ajustados apenas com historico anterior.",
    )
    switching_metrics = summarize_metrics(switching, folds, float(config.initial_capital))

    # %% 6 - Evidencias sem sobrescrever checkpoints ou testes anteriores
    # A pasta da execucao e criada antes do treino para permitir ZIP inclusive em falhas.
    control.predictions.to_csv(output / "control_predictions.csv", float_format="%.17g")
    control.trades.to_csv(output / "control_trades.csv", index=False)
    switching.predictions.to_csv(output / "switching_predictions.csv", float_format="%.17g")
    switching.trades.to_csv(output / "switching_trades.csv", index=False)
    pd.DataFrame(model_rows).to_csv(output / "switching_fold_models.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "control_75_fold_calibration.csv", index=False)
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
        "data_contract": "exact_75_assets_no_exclusions",
        "effective_assets": symbols,
        "effective_asset_count": 75,
        "universe_id": universe["universe_id"],
        "calendar": calendar,
        "calendar_contract": chronological_contract,
        "folds": len(folds),
        "decision_sessions": len(decision_dates) - 1,
        "settings": SETTINGS.__dict__,
        "cost_approximation": "Bellman proportional; exact fees and slippage in common simulator",
        "switching_diagnostics": {
            "all_cash": bool(
                (switching.predictions["selected_asset"] == "CASH").all()
            ),
            "cash_preferred_decisions": sum(
                v.get("switching_reason") == "cash_preferred"
                for v in diagnostics.values()
            ),
            "cash_q_beats_invest_q": sum(
                v.get("switching_q_cash", float("-inf")) >
                v.get("switching_q_invest", float("inf"))
                for v in diagnostics.values()
                if "switching_q_cash" in v
            ),
            "adjusted_reward_positive_states": sum(
                row[f"adjusted_reward_state_{state}"] > 0
                for row in model_rows for state in range(3)
            ),
            "adjusted_reward_state_count": len(model_rows) * 3,
            "note": (
                "All-CASH is a measured research outcome, not a validated "
                "improvement of Control. Do not tune parameters on OOS results."
            ),
        },
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
            "Universe is frozen and historical; asset discovery was retrospective, not prospectively validated.",
            "No guarantee of performance; out-of-sample only within fixed historic universe.",
            "Switching model treats the Control nominee as single risky asset; asset-to-asset costs are accounted in replay, not in two-state Bellman.",
            "Regime signal from trailing close prices; rewards from historical next-open to subsequent-open returns.",
            "Fold parameter training excludes current and future test observations.",
            "Comparisons inherit universe selection and historical source limitations; later independent evaluation required.",
        ],
    }
    _write_json(output / "resultado.json", summary)
    diagnostic = summary["switching_diagnostics"]
    (output / "resumo_para_analise.md").write_text(
        "# Optimal Switching: resumo da execucao\n\n"
        f"Versao: {SCRIPT_VERSION}\n\n"
        f"Universo: {len(symbols)} ativos; dados ate {contract['analysis_end_date']}.\n\n"
        f"Control: US$ {control_metrics['ending_capital']:,.2f}; "
        f"Sharpe={control_metrics['sharpe']:.4f}; "
        f"MaxDD={control_metrics['maximum_drawdown']:.2%}.\n\n"
        f"Switching: US$ {switching_metrics['ending_capital']:,.2f}; "
        f"Sharpe={switching_metrics['sharpe']}; "
        f"MaxDD={switching_metrics['maximum_drawdown']:.2%}.\n\n"
        f"Sessoes em CASH: Control={summary['comparison']['cash_days_control']}; "
        f"Switching={summary['comparison']['cash_days_switching']}.\n\n"
        f"Decisoes CASH preferido: {diagnostic['cash_preferred_decisions']}; "
        f"regimes com recompensa ajustada positiva: "
        f"{diagnostic['adjusted_reward_positive_states']}/"
        f"{diagnostic['adjusted_reward_state_count']}.\n\n"
        "**Interpretacao:** registros historicos simulados. O resultado "
        "pode refletir penalizacao excessiva de risco; nao constitui "
        "validacao prospectiva nem recomendacao de operacao.\n",
        encoding="utf-8",
    )
    print(f"[resultado] Control-75=US$ {control_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Switching=US$ {switching_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] capital_delta=US$ {summary['comparison']['delta_ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Control-75 cash_days={summary['comparison']['cash_days_control']}; Switching cash_days={summary['comparison']['cash_days_switching']}", flush=True)
    print(f"[arquivos] {output}", flush=True)
    return summary


def main() -> dict:
    """Sempre entrega ZIP de auditoria, inclusive para execucao interrompida."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _executar_experimento(output)
    except Exception as exc:
        _write_json(output / "erro_execucao.json", {
            "status": "failed",
            "version": SCRIPT_VERSION,
            "universe_expected_assets": 75,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        })
        try:
            package = gerar_zip_resultados(
                output, versao=SCRIPT_VERSION, status="failed",
            )
            print(f"[pacote] erro registrado, envie o ZIP para analise: {package}", flush=True)
        except Exception as package_error:
            print(f"[pacote] falha ao criar ZIP: {package_error}", flush=True)
        raise
    package = gerar_zip_resultados(
        output, versao=SCRIPT_VERSION, status="completed",
    )
    print(f"[pacote] ZIP de analise pronto: {package}", flush=True)
    return result


if __name__ == "__main__":
    main()
