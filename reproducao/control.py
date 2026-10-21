"""Reprodução Control LightGBM sobre o universo único versionado da pesquisa."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import time

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG, START_DATE, ASSETS
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm, _construir_contexto_execucao,
    diagnosticos_fora_amostra, selecionar_switch_margin,
)
from engine.rotacao import (
    ROTATION_FEATURES, _crescimento_politica_simples,
    _politica_agendada, _politica_utilidade,
    _precalcular_utilidades_modelo, _simular_exato,
    construir_quadro_rotacao,
)
from engine.execucao import calcular_taxas_referencia, aplicar_deslizamento
from engine.configuracao import construir_configuracao_controle
from reproducao.dados import SnapshotPaths, read_snapshot_manifest
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import (
    load_actions_file, load_raw_bar_file, prepare_model_frames,
    split_normalize, structural_identity_issue,
)
from reproducao.artefatos import criar_pacote_analise, sinal_sonoro_conclusao
from reproducao.graficos import gerar_analises_backtest

ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = SnapshotPaths.referencia(ROOT)
SELECTION_FILE = ROOT / "dados" / "selecao_ativos.json"
OUT = ROOT / "output" / "reproducao"
VERSION = "1.22.4-dev.1"
EXECUTION_SCHEMA = "control-selected-universe-swing"
ANALYSIS_END_DATE = "2026-10-16"
HISTORY_START_TOLERANCE_DAYS = 10


def carregar_etapas_treinamento() -> list[str]:
    """Lê a sequência de treinamento preservada da campanha histórica.

    Ela não define outro universo: todos os ativos pertencem à mesma seleção
    para decisão e podem competir pelas mesmas posições na carteira.
    """
    path = ROOT / "dados" / "universo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    staged = [str(x).strip().upper() for x in payload.get("staged_training_assets", [])]
    if len(staged) != len(set(staged)) or set(staged) - set(ASSETS):
        raise ValueError("Sequência de treinamento inválida ou fora do universo.")
    return staged


@dataclass(frozen=True)
class BaselineFoldState:
    fold_id: int
    train_dates: pd.DatetimeIndex
    calibration_dates: pd.DatetimeIndex
    final_fit_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex
    calibration_cache: dict[pd.Timestamp, np.ndarray]
    decision_cache: dict[pd.Timestamp, np.ndarray]
    selected_margin: float
    effective_margin: float
    fit_diagnostics: dict[str, Any] | None = None
    oos_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateFoldCache:
    fold_id: int
    calibration_cache: dict[pd.Timestamp, np.ndarray]
    decision_cache: dict[pd.Timestamp, np.ndarray]
    fit_diagnostics: dict[str, Any] | None = None
    oos_diagnostics: dict[str, Any] | None = None

def _generic_identity_issue(symbol: str, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    known = structural_identity_issue(symbol, actions)
    if known is not None:
        return known
    normalized = symbol.strip().upper()
    for action in actions:
        action_type = str(action.get("action_type") or "").strip().lower()
        old_symbol = str(action.get("old_symbol") or "").strip().upper()
        new_symbol = str(action.get("new_symbol") or "").strip().upper()
        old_cusip = str(action.get("old_cusip") or "").strip().upper()
        new_cusip = str(action.get("new_cusip") or "").strip().upper()
        source_symbol = str(action.get("source_symbol") or "").strip().upper()
        event_symbol = str(action.get("symbol") or "").strip().upper()
        acquiree = str(action.get("acquiree_symbol") or "").strip().upper()
        acquirer = str(action.get("acquirer_symbol") or "").strip().upper()
        aliases = {value for value in (event_symbol, source_symbol, old_symbol, new_symbol, acquiree, acquirer) if value}
        if normalized not in aliases:
            continue
        breaks = False
        if action_type in {"cash_merger", "stock_merger", "stock_and_cash_merger"}:
            breaks = bool(acquiree == normalized and acquirer and acquirer != normalized)
        elif action_type in {"redemption", "worthless_removal"}:
            breaks = normalized in {event_symbol, source_symbol, old_symbol}
        elif action_type == "name_change":
            breaks = bool(
                (old_symbol == normalized and new_symbol and old_symbol != new_symbol)
                or (old_cusip and new_cusip and old_cusip != new_cusip)
            )
        elif action_type == "reverse_split" and old_symbol == normalized and new_symbol and old_symbol != new_symbol:
            breaks = True
        if breaks:
            return {
                "symbol": normalized,
                "reason": "structural_identity_change",
                "action_type": action_type,
                "process_date": action.get("process_date"),
                "effective_date": action.get("effective_date"),
                "old_symbol": old_symbol or None,
                "new_symbol": new_symbol or None,
                "old_cusip": old_cusip or None,
                "new_cusip": new_cusip or None,
            }
    return None


def _continuity_audit(
    symbol: str,
    normalized_raw: pd.DataFrame,
    required_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(pd.to_datetime(normalized_raw.index, utc=True)).normalize().unique().sort_values()
    required = pd.DatetimeIndex(pd.to_datetime(required_sessions, utc=True)).normalize().unique().sort_values()
    if index.empty or required.empty:
        return {"asset": symbol, "status": "rejected", "reason": "empty_history"}
    first = pd.Timestamp(index.min())
    last = pd.Timestamp(index.max())
    latest_start = pd.Timestamp(START_DATE, tz="UTC") + pd.Timedelta(days=HISTORY_START_TOLERANCE_DAYS)
    missing = required.difference(index)
    positions = sorted(int(required.get_loc(item)) for item in missing if item in required)
    longest = 0
    current = 0
    previous = None
    for pos in positions:
        if previous is not None and pos == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = pos
    reason = None
    if first.normalize() > latest_start.normalize():
        reason = "insufficient_history_start"
    elif len(missing):
        reason = "discontinuous_history"
    return {
        "asset": symbol,
        "status": "passed" if reason is None else "rejected",
        "reason": reason,
        "history_actual_start": first.isoformat(),
        "history_actual_end": last.isoformat(),
        "required_session_count": int(len(required)),
        "missing_required_sessions": int(len(missing)),
        "longest_missing_run": int(longest),
    }


def _prepare_candidate_frames(
    symbols: list[str],
    common_dates: pd.DatetimeIndex,
    all_decision_dates: pd.DatetimeIndex,
    config: Any,
    candidate_data: SnapshotPaths,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    audits = []
    for position, symbol in enumerate(symbols, start=1):
        raw_path = candidate_data.raw_bars / f"{symbol}.csv"
        action_path = candidate_data.corporate_actions / f"{symbol}.csv"
        row: dict[str, Any] = {"asset": symbol, "status": "rejected", "reason": None}
        try:
            raw = load_raw_bar_file(raw_path, float_precision="round_trip")
            actions = load_actions_file(action_path)
            issue = _generic_identity_issue(symbol, actions)
            if issue is not None:
                row.update(
                    {
                        "reason": "structural_identity_change",
                        "action_type": issue.get("action_type"),
                        "identity_detail": json.dumps(issue, ensure_ascii=False, default=str),
                    }
                )
                audits.append(row)
                continue
            normalized, applied = split_normalize(raw, actions)
            continuity = _continuity_audit(symbol, normalized, common_dates)
            row.update(continuity)
            row["corporate_actions"] = int(len(actions))
            row["splits_applied"] = int(len(applied))
            if continuity["status"] != "passed":
                audits.append(row)
                continue
            prepared = construir_quadro_rotacao(normalized, config).reindex(common_dates)
            execution = prepared.reindex(all_decision_dates[1:])
            prices = execution[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
            if not np.isfinite(prices).all() or (execution[["open", "high", "low", "close"]] <= 0).any().any():
                row.update({"status": "rejected", "reason": "incomplete_execution_prices"})
                audits.append(row)
                continue
            row.update({"status": "passed", "reason": None, "model_rows": int(len(prepared))})
            frames[symbol] = prepared
        except Exception as exc:
            row.update({"status": "failed", "reason": str(exc)[:500]})
        audits.append(row)
        print(
            f"[continuity] {position}/{len(symbols)} {symbol} status={row['status']} reason={row.get('reason')}",
            flush=True,
        )
    return frames, pd.DataFrame(audits)


def _scenario_metrics(result: Any, folds: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    metrics = summarize_metrics(result, folds, initial_capital)
    raw = result.metrics or {}
    metrics.update(
        {
            "capital_rotations": int(raw.get("capital_rotations") or 0),
            "simulated_buys": int(raw.get("simulated_buys") or 0),
            "simulated_sells": int(raw.get("simulated_sells") or 0),
            "cash_days": int(raw.get("cash_days") or 0),
            "market_exposure": float(raw.get("market_exposure") or 0.0),
            "total_transaction_fees": float(raw.get("total_transaction_fees") or 0.0),
        }
    )
    return metrics


def _train_baseline_states(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    folds: list[dict[str, Any]],
    config: Any,
) -> tuple[list[BaselineFoldState], dict[int, Any], list[dict[str, Any]]]:
    states = []
    policies = {}
    calibration_rows = []
    candidate_margins = tuple(float(value) for value in config.rotation_switch_margin_candidates)
    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]) : int(fold["calibration_end_index"])
        ]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        decision_dates = pd.DatetimeIndex(fold["decision_dates"])
        print(f"[baseline-train] fold={fold_id} {position}/{len(folds)} calibration", flush=True)
        calibration_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config,
            phase=f"reference_expansion_baseline_fold_{fold_id}_calibration",
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_models, frames, symbols, calibration_dates, config
        )
        scores = []
        for margin in candidate_margins:
            policy = _politica_utilidade(
                {}, frames, symbols, config, margin, utility_cache=calibration_cache
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, calibration_dates, config
            )
            scores.append((margin, float(score)))
        selection = selecionar_switch_margin(scores)
        selected_margin = float(selection["selected_candidate_margin"])
        effective_margin = max(float(config.rotation_switch_margin), selected_margin)
        for margin, score in scores:
            calibration_rows.append(
                {
                    "scenario": "REFERENCE_BASELINE",
                    "fold_id": fold_id,
                    "candidate_margin": margin,
                    "calibration_score": score,
                    "selected": bool(abs(margin - selected_margin) <= 1e-12),
                }
            )
        print(f"[baseline-train] fold={fold_id} final", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"reference_expansion_baseline_fold_{fold_id}_final",
        )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, decision_dates, config
        )
        fit_diagnostics = {
            symbol: dict(getattr(model, "_fit_diagnostics", {}) or {})
            for symbol, model in final_models.items()
        }
        oos_diagnostics = diagnosticos_fora_amostra(
            final_models,
            frames,
            symbols,
            decision_dates,
            decision_cache,
            fold_id=fold_id,
        )
        policies[fold_id] = _politica_utilidade(
            {},
            frames,
            symbols,
            config,
            effective_margin,
            fold_id=fold_id,
            calibrated_switch_margin=selected_margin,
            utility_cache=decision_cache,
        )
        states.append(
            BaselineFoldState(
                fold_id=fold_id,
                train_dates=train_dates,
                calibration_dates=calibration_dates,
                final_fit_dates=final_fit_dates,
                decision_dates=decision_dates,
                calibration_cache=calibration_cache,
                decision_cache=decision_cache,
                selected_margin=selected_margin,
                effective_margin=effective_margin,
                fit_diagnostics=fit_diagnostics,
                oos_diagnostics=oos_diagnostics,
            )
        )
        del calibration_models, final_models
    return states, policies, calibration_rows


def _train_candidate_caches(
    symbol: str,
    frame: pd.DataFrame,
    states: list[BaselineFoldState],
    config: Any,
) -> dict[int, CandidateFoldCache]:
    output = {}
    for position, state in enumerate(states, start=1):
        training = frame.reindex(state.train_dates)
        valid_train = training[[*ROTATION_FEATURES, "forward_risk_adjusted_utility"]].dropna()
        if len(valid_train) < int(config.rotation_minimum_training_rows):
            raise RuntimeError(
                f"{symbol}: insufficient_model_training_rows fold={state.fold_id} rows={len(valid_train)}"
            )
        one_frame = {symbol: frame}
        print(
            f"[candidate-train] {symbol} fold={state.fold_id} {position}/{len(states)} calibration",
            flush=True,
        )
        calibration_model = _ajustar_modelos_lightgbm(
            one_frame,
            [symbol],
            state.train_dates,
            config,
            phase=f"reference_expansion_{symbol}_fold_{state.fold_id}_calibration",
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_model,
            one_frame,
            [symbol],
            state.calibration_dates,
            config,
        )
        print(f"[candidate-train] {symbol} fold={state.fold_id} final", flush=True)
        final_model = _ajustar_modelos_lightgbm(
            one_frame,
            [symbol],
            state.final_fit_dates,
            config,
            phase=f"reference_expansion_{symbol}_fold_{state.fold_id}_final",
        )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_model,
            one_frame,
            [symbol],
            state.decision_dates,
            config,
        )
        fit_diagnostics = dict(
            getattr(final_model.get(symbol), "_fit_diagnostics", {}) or {}
        )
        oos_diagnostics = diagnosticos_fora_amostra(
            final_model,
            one_frame,
            [symbol],
            state.decision_dates,
            decision_cache,
            fold_id=state.fold_id,
        )
        output[state.fold_id] = CandidateFoldCache(
            fold_id=state.fold_id,
            calibration_cache=calibration_cache,
            decision_cache=decision_cache,
            fit_diagnostics=fit_diagnostics,
            oos_diagnostics=oos_diagnostics,
        )
        del calibration_model, final_model
    return output


def _merge_utility_caches(
    baseline_cache: dict[pd.Timestamp, np.ndarray],
    baseline_symbols: list[str],
    candidate_caches: dict[str, dict[pd.Timestamp, np.ndarray]],
    combined_symbols: list[str],
) -> dict[pd.Timestamp, np.ndarray]:
    """Combina caches mantendo a posicao 0 reservada para CASH.

    Cada cache de candidato foi treinado isoladamente e possui vetor [CASH, score].
    Isso e exato porque o Control atual ajusta um LightGBM independente por ativo.
    """
    combined_position = {symbol: index + 1 for index, symbol in enumerate(combined_symbols)}
    base_position = {symbol: index + 1 for index, symbol in enumerate(baseline_symbols)}
    output: dict[pd.Timestamp, np.ndarray] = {}
    for timestamp, base_values in baseline_cache.items():
        values = np.full(len(combined_symbols) + 1, float("-inf"), dtype=np.float64)
        values[0] = 0.0
        for symbol, source_position in base_position.items():
            values[combined_position[symbol]] = float(base_values[source_position])
        for symbol, cache in candidate_caches.items():
            candidate_values = cache.get(pd.Timestamp(timestamp))
            if candidate_values is not None and len(candidate_values) >= 2:
                values[combined_position[symbol]] = float(candidate_values[1])
        output[pd.Timestamp(timestamp)] = values
    return output



def executar_control() -> dict[str, Any]:
    """Reproduz o Control com a seleção de ativos registrada pela pesquisa."""

    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    needed = carregar_etapas_treinamento()
    historical_reference_capital = None
    print(f"[TCC] versão={VERSION} modo=Control", flush=True)
    print(f"[TCC] dados={BASE_DATA.root} universo_solicitado={len(ASSETS)}", flush=True)

    # A leitura do manifesto não faz comparações por hash nem restringe
    # a execução a uma identidade/versionamento antigos.
    metadata = read_snapshot_manifest(BASE_DATA)
    # A data do experimento deve corresponder ao snapshot efetivamente coletado.
    # Nunca ampliar apenas o cutoff do modelo sobre CSVs antigos.
    bar_cutoff = str((metadata.get("bars") or {}).get("bar_snapshot_as_of_end") or "")
    actions_cutoff = str((metadata.get("corporate_actions") or {}).get("query_end") or "")
    if bar_cutoff != ANALYSIS_END_DATE or actions_cutoff != ANALYSIS_END_DATE:
        raise RuntimeError(
            "Snapshot da pesquisa desatualizado para a data de corte selecionada: "
            f"esperado={ANALYSIS_END_DATE}, barras={bar_cutoff or 'ausente'}, "
            f"corporate_actions={actions_cutoff or 'ausente'}. "
            "Atualize os CSVs pela Alpaca e reconstrua o manifesto antes da reproducao; "
            "nao modifique apenas as datas do manifesto."
        )
    collected = tuple(str(x).strip().upper() for x in metadata.get("assets", []))
    if set(collected) != set(ASSETS) or len(collected) != len(ASSETS):
        raise RuntimeError(
            "Universo coletado diferente do universo declarado. "
            f"declarados={len(ASSETS)} coletados={len(collected)} "
            f"faltantes={sorted(set(ASSETS) - set(collected))} "
            f"extras={sorted(set(collected) - set(ASSETS))}. "
            "Atualize o snapshot completo antes da reprodução."
        )
    baseline_requested = [symbol for symbol in ASSETS if symbol not in needed]
    base_raw, exclusions, preparation, data_audit = prepare_model_frames(
        BASE_DATA,
        assets=tuple(baseline_requested),
        csv_float_precision="round_trip",
    )
    baseline = [symbol for symbol in baseline_requested if symbol in base_raw]
    configuration_reference = build_control_config(
        {symbol: base_raw[symbol] for symbol in baseline}, CONFIG
    ).copiar_modelo(update={"analysis_end_date": ANALYSIS_END_DATE})
    (
        baseline_frames, common_dates, source_calendar, baseline_symbols,
        folds, all_decision_dates, decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao(
        {symbol: base_raw[symbol] for symbol in baseline},
        configuration_reference,
    )
    if needed:
        candidate_frames, continuity = _prepare_candidate_frames(
            needed, common_dates, all_decision_dates, configuration_reference,
            BASE_DATA,
        )
    else:
        candidate_frames, continuity = {}, pd.DataFrame(
            columns=["asset", "status", "reason"]
        )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    rejected = [s for s in needed if s not in candidate_frames]
    if rejected:
        raise ValueError(
            "Os seguintes ativos não produziram séries utilizáveis: "
            + ", ".join(rejected)
            + ". Confira continuity_audit.csv para conhecer a causa."
        )

    # Método numérico histórico:
    # treina modelos por ativo e por fold separados, funde caches na
    # ordem CASH + todos os símbolos e calibra margens só no treino.
    states, _, baseline_margins = _train_baseline_states(
        baseline_frames, common_dates, baseline_symbols,
        folds, configuration_reference,
    )
    candidates = {
        symbol: _train_candidate_caches(
            symbol, candidate_frames[symbol], states, configuration_reference
        ) for symbol in needed
    }
    symbols = sorted([*baseline_symbols, *needed])
    frames = {**baseline_frames, **candidate_frames}
    config = construir_configuracao_controle(
        configuration_reference, assets=symbols,
    )
    policies: dict[int, Any] = {}
    decisions: dict[pd.Timestamp, dict[str, Any]] = {}
    fold_margins: list[dict[str, Any]] = []

    for state in states:
        calibration_cache = _merge_utility_caches(
            state.calibration_cache,
            baseline_symbols,
            {
                symbol: candidates[symbol][state.fold_id].calibration_cache
                for symbol in needed
            },
            symbols,
        )
        decision_cache = _merge_utility_caches(
            state.decision_cache,
            baseline_symbols,
            {
                symbol: candidates[symbol][state.fold_id].decision_cache
                for symbol in needed
            },
            symbols,
        )
        scores = []
        for proposed_margin in config.rotation_switch_margin_candidates:
            policy = _politica_utilidade(
                {}, frames, symbols, config,
                float(proposed_margin),
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, state.calibration_dates, config
            )
            scores.append((float(proposed_margin), float(score)))
        selected = selecionar_switch_margin(scores)
        margin = max(
            float(config.rotation_switch_margin),
            float(selected["selected_candidate_margin"]),
        )
        policies[state.fold_id] = _politica_utilidade(
            {}, frames, symbols, config, margin,
            utility_cache=decision_cache,
            decision_diagnostics=decisions,
            fold_id=state.fold_id,
        )
        fold_margins.append({
            "fold_id": state.fold_id,
            "margin": margin,
            "selected_historical_margin": float(
                selected["selected_candidate_margin"]
            ),
            "candidate_scores": scores,
        })
        print(
            f"[TCC] fold={state.fold_id} margin={margin:.6f} "
            f"hold_min={config.rotation_min_holding_days}",
            flush=True,
        )
    result = _simular_exato(
        "tcc_control_swing",
        _politica_agendada(policies, decision_to_fold),
        frames, symbols, all_decision_dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label=f"TCC Control: {len(symbols)} ativos, swing trade",
    )
    metrics = _scenario_metrics(
        result, folds, float(config.initial_capital)
    )
    observed = float(metrics["ending_capital"])
    if historical_reference_capital is not None:
        print(
            f"[TCC] Control capital=US$ {observed:,.2f}; "
            f"referencia_historica=US$ {historical_reference_capital:,.2f}; "
            f"diferenca=US$ {observed-historical_reference_capital:+,.2f}",
            flush=True,
        )
    else:
        print(f"[TCC] Control capital=US$ {observed:,.2f}", flush=True)
    print(
        f"[TCC] Sharpe={metrics['sharpe']:.4f} "
        f"MaxDD={metrics['maximum_drawdown']:.3%} "
        f"rotacoes={metrics['capital_rotations']}",
        flush=True,
    )

    # A pasta pode ter sido removida por uma limpeza externa durante o treino.
    # Revalidar imediatamente antes da gravação, sem alterar a simulação.
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[TCC] gravando artefatos em {OUT}", flush=True)
    # Outputs funcionais, não mecanismos para vetar a execução.
    result.predictions.to_csv(OUT / "control_equity.csv", index_label="timestamp")
    result.trades.to_csv(OUT / "control_trades.csv", index=False)
    pd.DataFrame(fold_margins).to_csv(OUT / "fold_margins.csv", index=False)
    pd.DataFrame([
        {"decision_date": pd.Timestamp(d).isoformat(), **row}
        for d, row in sorted(decisions.items())
    ]).to_csv(OUT / "control_decisions.csv", index=False)
    pd.DataFrame([{"asset": symbol} for symbol in symbols]).to_csv(
        OUT / "universe.csv", index=False
    )
    payload = {
        "execution_schema": EXECUTION_SCHEMA,
        "version": VERSION,
        "status": "completed",
        "strategy": "Control",
        "target": "forward_risk_adjusted_utility",
        "minimum_holding_sessions": config.rotation_min_holding_days,
        "margin_calibration": "past_only_by_fold",
        "assets": list(symbols),
        "total_assets": len(symbols),
        "initial_capital": float(config.initial_capital),
        "historical_reference_capital": historical_reference_capital,
        "actual_ending_capital": observed,
        "difference_from_reference": (observed - historical_reference_capital) if historical_reference_capital is not None else None,
        "metrics": metrics,
        "fold_margins": fold_margins,
        "snapshot_metadata": metadata,
        "structural_exclusions": exclusions,
        "runtime_seconds": time.perf_counter() - started,
        "origin": "single_universe_with_historical_training_sequence",
    }
    (OUT / "reproducao_control.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    gerar_analises_backtest(OUT, manifest=metadata, result=result)
    package = criar_pacote_analise(
        OUT, comparison_file="reproducao_control.json",
        execution_schema=EXECUTION_SCHEMA,
        archive_name="pacote_reproducao_control.zip",
    )
    print(f"[TCC] pacote={package}", flush=True)
    sinal_sonoro_conclusao()
    return payload
