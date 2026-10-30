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
    SwitchingSettings, build_overlay, fit_switching_model, selecionar_peso_calibracao,
)
from pesquisa.pacote_analise_switching import gerar_zip_resultados
from pesquisa.persistencia_resultados import (
    salvar_csv, salvar_json, salvar_texto,
)
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
SCRIPT_VERSION = "optimal-switching-cash-reentry-v1.0.7"
SCHEMA_VERSION = "optimal-switching-cash-v3"
SETTINGS = SwitchingSettings(
    lookback=20,
    horizon=10,
    downside_weight=0.0,
    minimum_observations=126,
)
RISK_WEIGHT_CANDIDATES = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)
# Os candidatos sao fixados antes desta execucao. A pesquisa anterior (v1.0.4)
# ja foi consultada ao propor esta reformulacao: o periodo de 2020-2026
# nao constitui um conjunto de teste externo virgem nesta nova rodada.



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
) -> tuple[dict, list[dict], dict]:
    policies = {}
    calibration = []
    calibration_policies = {}
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
        calibration_policies[fold_id] = _politica_utilidade(
            {}, frames, symbols, config, effective_margin,
            fold_id=fold_id, calibrated_switch_margin=choice,
            utility_cache=calibration_cache,
        )
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
    return policies, calibration, calibration_policies


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
    policies, calibration, calibration_policies = ajustar_control_75(
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
    # %% 4 - Calibracao do peso de risco: treinar em train, escolher em calibration
    # NUNCA utilizar as sessoes fora da amostra para a escolha do parametro.
    models_by_fold: dict[int, dict] = {}
    model_rows: list[dict] = []
    risk_calibration_rows: list[dict] = []
    risk_settings_by_fold: dict[int, SwitchingSettings] = {}
    for fold in folds:
        fid = int(fold["fold_id"])
        train_end = pd.Timestamp(common_dates[int(fold["train_end_index"]) - 1])
        cal_start = int(fold["calibration_start_index"])
        cal_end = int(fold["calibration_end_index"])
        calibration_dates = common_dates[cal_start:cal_end]
        if not train_end < pd.Timestamp(calibration_dates[0]):
            raise RuntimeError("Treino do Switching nao precede calibracao.")
        if not pd.Timestamp(calibration_dates[-1]) < pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Calibracao usa dados do fold de teste.")
        fold_models_cal = {}
        for symbol in symbols:
            fold_models_cal[symbol] = fit_switching_model(
                symbol, frames[symbol], fit_end=train_end, settings=SETTINGS,
            )
        cal_mapping = {pd.Timestamp(t): fid for t in calibration_dates}
        scores = []
        for weight in RISK_WEIGHT_CANDIDATES:
            fold_settings = SwitchingSettings(
                lookback=SETTINGS.lookback,
                horizon=SETTINGS.horizon,
                minimum_observations=SETTINGS.minimum_observations,
                downside_weight=weight,
            )
            diag: dict[pd.Timestamp, dict] = {}
            calibrated_overlay = build_overlay(
                calibration_policies[fid],
                frames=frames, symbols=symbols,
                decision_to_fold=cal_mapping,
                models_by_fold={fid: fold_models_cal},
                settings=fold_settings, config=config, diagnostics=diag,
            )
            score = float(_crescimento_politica_simples(
                calibrated_overlay, frames, symbols, calibration_dates, config,
            ))
            scores.append((weight, score))
        selected_weight = selecionar_peso_calibracao(scores)
        risk_settings_by_fold[fid] = SwitchingSettings(
            lookback=SETTINGS.lookback,
            horizon=SETTINGS.horizon,
            minimum_observations=SETTINGS.minimum_observations,
            downside_weight=selected_weight,
        )
        for weight, score in scores:
            risk_calibration_rows.append({
                "fold_id": fid,
                "train_end": train_end.isoformat(),
                "calibration_start": pd.Timestamp(calibration_dates[0]).isoformat(),
                "calibration_end": pd.Timestamp(calibration_dates[-1]).isoformat(),
                "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
                "candidate_downside_weight": float(weight),
                "calibration_score": score,
                "selected": bool(weight == selected_weight),
                "model_count": len(fold_models_cal),
                "selection_policy": "Control trained before calibration",
            })
        print(
            f"[risk-calibration] fold={fid}/{len(folds)} "
            f"weight={selected_weight:.4f} training_before_calibration=true",
            flush=True,
        )

        # Ajuste final do modelo estatistico: ate o corte ANTERIOR ao teste.
        # O mesmo modelo final sera usado nas duas politicas fora da amostra.
        fit_end = pd.Timestamp(common_dates[int(fold["final_fit_end_index"]) - 1])
        if not fit_end < pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Ajuste final do Switching invade o teste.")
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
                **{f"expected_return_state_{i}": float(model.expected_returns[i])
                   for i in range(3)},
                **{f"downside_state_{i}": float(model.downside_losses[i])
                   for i in range(3)},
                **{f"adjusted_reward_state_{i}": (
                    float(model.expected_returns[i]) -
                    selected_weight * float(model.downside_losses[i])
                ) for i in range(3)},
                **{f"transition_{i}_{j}": float(model.transitions[i, j])
                   for i in range(3) for j in range(3)},
                "calibrated_downside_weight": selected_weight,
            })
            if index % 15 == 0 or index == len(symbols):
                print(
                    f"[switching-fit] fold={fid} {index}/{len(symbols)}",
                    flush=True,
                )

    # %% 5 - Duas politicas independentes na MESMA validacao financeira
    log_growth_diagnostics: dict[pd.Timestamp, dict] = {}
    log_growth_overlay = build_overlay(
        base_policy,
        frames=frames, symbols=symbols,
        decision_to_fold=decision_to_fold,
        models_by_fold=models_by_fold,
        settings=SETTINGS, config=config, diagnostics=log_growth_diagnostics,
    )
    log_growth = _simular_exato(
        "tcc_switching_expected_log_growth",
        log_growth_overlay,
        frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=log_growth_diagnostics,
        model_label="Pesquisa Control75 + Switching Retorno Esperado",
    )
    log_growth_metrics = summarize_metrics(
        log_growth, folds, float(config.initial_capital),
    )

    diagnostics: dict[pd.Timestamp, dict] = {}
    overlay = build_overlay(
        base_policy,
        frames=frames, symbols=symbols,
        decision_to_fold=decision_to_fold,
        models_by_fold=models_by_fold,
        settings=SETTINGS, settings_by_fold=risk_settings_by_fold,
        config=config, diagnostics=diagnostics,
    )
    switching = _simular_exato(
        "tcc_switching_risk_calibrated",
        overlay,
        frames, symbols, decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="Pesquisa Control75 + Switching Risco Calibrado",
        method_line="- Peso de risco selecionado somente em calibracao anterior ao teste.",
    )
    switching_metrics = summarize_metrics(
        switching, folds, float(config.initial_capital),
    )

    # %% 6 - Evidencias sem sobrescrever checkpoints ou testes anteriores
    # A pasta da execucao e criada antes do treino para permitir ZIP inclusive em falhas.
    salvar_csv(control.predictions, output / "control_predictions.csv", float_format="%.17g")
    salvar_csv(control.trades, output / "control_trades.csv", index=False)
    salvar_csv(switching.predictions, output / "switching_predictions.csv", float_format="%.17g")
    salvar_csv(switching.trades, output / "switching_trades.csv", index=False)
    salvar_csv(log_growth.predictions, output / "log_growth_predictions.csv", float_format="%.17g")
    salvar_csv(log_growth.trades, output / "log_growth_trades.csv", index=False)
    salvar_csv(pd.DataFrame(model_rows), output / "switching_fold_models.csv", index=False)
    salvar_csv(pd.DataFrame(calibration), output / "control_75_fold_calibration.csv", index=False)
    salvar_csv(pd.DataFrame(risk_calibration_rows), output / "switching_risk_calibration.csv", index=False)
    decision_rows = [
        {"decision_date": timestamp.isoformat(), **values}
        for timestamp, values in sorted(diagnostics.items())
    ]
    salvar_csv(pd.DataFrame(decision_rows), output / "switching_decisions.csv", index=False)
    salvar_csv(pd.DataFrame([
        {"decision_date": date.isoformat(), **values}
        for date, values in sorted(log_growth_diagnostics.items())
    ]), output / "log_growth_decisions.csv", index=False)
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
        "risk_calibration_grid": list(RISK_WEIGHT_CANDIDATES),
        "risk_calibrated_weights_by_fold": {
            str(fid): float(v.downside_weight)
            for fid, v in risk_settings_by_fold.items()
        },
        "risk_calibration_method": (
            "Maximize the frozen Control risk-adjusted calibration utility "
            "over predeclared weights. Train models only before calibration; "
            "final fit is before OOS."
        ),
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
        "log_growth_metrics": log_growth_metrics,
        "risk_calibrated_metrics": switching_metrics,
        "switching_metrics": switching_metrics,
        "comparison": {
            "delta_ending_capital": switching_metrics["ending_capital"] - control_metrics["ending_capital"],
            "log_growth_delta_ending_capital": (
                log_growth_metrics["ending_capital"] - control_metrics["ending_capital"]
            ),
            "delta_maximum_drawdown": switching_metrics["maximum_drawdown"] - control_metrics["maximum_drawdown"],
            "delta_sharpe": switching_metrics["sharpe"] - control_metrics["sharpe"],
            "cash_days_control": int(control.metrics.get("cash_days", 0)),
            "cash_days_log_growth": int(log_growth.metrics.get("cash_days", 0)),
            "cash_days_switching": int(switching.metrics.get("cash_days", 0)),
            "fees_control": float(control.metrics.get("total_transaction_fees", 0)),
            "fees_log_growth": float(log_growth.metrics.get("total_transaction_fees", 0)),
            "fees_switching": float(switching.metrics.get("total_transaction_fees", 0)),
            "cash_exits": int((switching.trades.get("reason", pd.Series(dtype=str)) == "MOVE_TO_CASH").sum()),
            "cash_exits_log_growth": int((log_growth.trades.get("reason", pd.Series(dtype=str)) == "MOVE_TO_CASH").sum()),
        },
        "scientific_limitations": [
            "Universe is frozen and historical; asset discovery was retrospective, not prospectively validated.",
            "No guarantee of performance; out-of-sample only within fixed historic universe.",
            "Switching model treats the Control nominee as single risky asset; asset-to-asset costs are accounted in replay, not in two-state Bellman.",
            "Regime signal from trailing close prices; rewards from historical next-open to subsequent-open returns.",
            "Fold parameter training excludes current and future test observations.",
            "The risk weights are selected on calibration performance, not an independent mathematical risk limit or CVaR constraint.",
            "Calibration reuses the fold's Control margin selection, increasing selection-overfitting risk.",
            "This v1.0.5 hypothesis was proposed after inspecting v1.0.4 OOS results: the frozen test is no longer untouched for cross-version claims.",
            "State-conditional asset returns are not conditioned on nomination; the selection-conditioned calibration replay is only a partial remedy.",
            "Comparisons inherit universe selection and historical source limitations; later independent evaluation required.",
        ],
    }
    salvar_json(output / "resultado.json", summary, _write_json)
    diagnostic = summary["switching_diagnostics"]
    salvar_texto(output / "resumo_para_analise.md",
        "# Optimal Switching: resumo da execucao\n\n"
        f"Versao: {SCRIPT_VERSION}\n\n"
        f"Universo: {len(symbols)} ativos; dados ate {contract['analysis_end_date']}.\n\n"
        f"Control: US$ {control_metrics['ending_capital']:,.2f}; "
        f"Sharpe={control_metrics['sharpe']:.4f}; "
        f"MaxDD={control_metrics['maximum_drawdown']:.2%}.\n\n"
        f"Switching log-growth: US$ {log_growth_metrics['ending_capital']:,.2f}; "
        f"Sharpe={log_growth_metrics['sharpe']}; "
        f"MaxDD={log_growth_metrics['maximum_drawdown']:.2%}.\\n\\n"
        f"Switching risco calibrado: US$ {switching_metrics['ending_capital']:,.2f}; "
        f"Sharpe={switching_metrics['sharpe']}; "
        f"MaxDD={switching_metrics['maximum_drawdown']:.2%}.\n\n"
        f"Sessoes em CASH: Control={summary['comparison']['cash_days_control']}; "
        f"Log-growth={summary['comparison']['cash_days_log_growth']}; "
        f"Risco calibrado={summary['comparison']['cash_days_switching']}.\\n\\n"
        f"Pesos por fold: {summary['risk_calibrated_weights_by_fold']}.\\n\\n"
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
    print(f"[resultado] Log-growth=US$ {log_growth_metrics['ending_capital']:,.2f}", flush=True)
    print(f"[resultado] Risk-calibrated=US$ {switching_metrics['ending_capital']:,.2f}", flush=True)
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
        package = gerar_zip_resultados(
            output, versao=SCRIPT_VERSION, status="completed",
        )
        print(f"[pacote] ZIP de analise pronto: {package}", flush=True)
    except Exception as exc:
        # Se o diretorio sumiu durante o processamento, este salvamento
        # o recria antes de gerar o pacote de erro.
        try:
            salvar_json(output / "erro_execucao.json", {
                "status": "failed",
                "version": SCRIPT_VERSION,
                "universe_expected_assets": 75,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }, _write_json)
            package = gerar_zip_resultados(
                output, versao=SCRIPT_VERSION, status="failed",
            )
            print(
                f"[pacote] erro registrado, envie o ZIP para analise: {package}",
                flush=True,
            )
        except Exception as archive_error:
            print(f"[pacote] falha ao registrar erro: {archive_error}", flush=True)
        raise
    return result


if __name__ == "__main__":
    main()
