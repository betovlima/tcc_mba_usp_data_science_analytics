"""Experimento isolado U67+10: holding=2 vs holding=0 vs evidencia por horizonte.

Sem download de dados, sem mutacao da reproducao oficial ou da aplicação operacional.
Uso no Spyder: execute este arquivo (F5), com os caches U67+10 presentes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from engine.configuracao import (
    CONFIG,
    U67_REQUESTED_ASSETS,
    U67_EXPECTED_EXCLUSIONS,
    construir_configuracao_controle,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    selecionar_switch_margin,
)
from engine.rotacao import (
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import prepare_model_frames
from pesquisar_expansao_u67_spyder import (
    ANALYSIS_END_DATE,
    DISCOVERY_SELECTED_CANDIDATES,
    EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL,
    FULL_DATA,
    _merge_utility_caches,
    _prepare_candidate_frames,
    _train_baseline_states,
    _train_candidate_caches,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "persistencia_horizontes_u77"
VERSION = "horizon-persistence-oos-v1.0.0"
CONFIDENCE_MULTIPLIERS = (0.0, 0.5, 1.0)
ANCHOR_BLEND = 0.5
CHECKPOINT_TOLERANCE_DOLLARS = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return safe_json(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(safe_json(obj), ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def uncertainty_from_matured_calibration(
    *,
    h: int,
    timestamps: pd.DatetimeIndex,
    full_dates: pd.DatetimeIndex,
    latest_label_date_index: int,
    symbols: list[str],
    frames: dict[str, pd.DataFrame],
    predictions: dict[pd.Timestamp, np.ndarray],
) -> tuple[float, int]:
    """Erro absoluto p75 somente de alvos totalmente maturados antes do OOS."""
    lookup = {pd.Timestamp(date): i for i, date in enumerate(full_dates)}
    errors: list[float] = []
    for day in timestamps:
        timestamp = pd.Timestamp(day)
        index = lookup.get(timestamp)
        if index is None or index + h >= latest_label_date_index:
            continue
        predicted = predictions.get(timestamp)
        if predicted is None:
            continue
        for column, symbol in enumerate(symbols, start=1):
            actual = frames[symbol].loc[timestamp, f"forward_horizon_utility_{h}"]
            forecast = float(predicted[column])
            if np.isfinite(actual) and np.isfinite(forecast):
                errors.append(abs(float(actual) - forecast))
    if len(errors) < 30:
        raise RuntimeError(
            f"H{h}: insuficientes residuos maturados para medir a incerteza "
            f"antes do OOS: {len(errors)}. Nao usar rotulos futuros."
        )
    return float(np.quantile(np.asarray(errors), 0.75)), len(errors)


class HorizonPersistencePolicy:
    """Politica cronologica com memoria da previsao na entrada.

    A incerteza e um proxy empirico, NAO probabilidade calibrada.
    A regra de margem financeira original continua ativa; a confianca
    acrescenta evidencia por horizonte, sem intervalo de holding fixo.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        aggregate: dict[pd.Timestamp, np.ndarray],
        per_horizon: dict[int, dict[pd.Timestamp, np.ndarray]],
        uncertainties: dict[int, float],
        weights: dict[int, float],
        margin: float,
        cash_threshold: float,
        entry_threshold: float,
        confidence_multiplier: float,
        audit: list[dict[str, Any]] | None = None,
        fold_id: int | None = None,
    ) -> None:
        self.symbols = list(symbols)
        self.date_to_index = {pd.Timestamp(day): i for i, day in enumerate(dates)}
        self.aggregate = aggregate
        self.per_horizon = per_horizon
        self.uncertainties = uncertainties
        self.weights = weights
        self.margin = float(margin)
        self.cash_threshold = float(cash_threshold)
        self.entry_threshold = float(entry_threshold)
        self.confidence_multiplier = float(confidence_multiplier)
        self.audit = audit
        self.fold_id = fold_id
        self.anchor: dict[str, Any] | None = None

    def __call__(self, day: pd.Timestamp, current: int, holding: int) -> tuple[int, float]:
        timestamp = pd.Timestamp(day)
        index = self.date_to_index[timestamp]
        values = self.aggregate[timestamp]
        if not np.isfinite(values[1:]).any():
            return 0, 0.0
        best = int(np.nanargmax(values))
        q_best = float(values[best])
        q_current = float(values[current])
        scores = {
            h: self.per_horizon[h].get(timestamp)
            for h in self.weights
        }
        if current > 0 and (
            self.anchor is None or self.anchor["position"] != current
        ):
            # Crossing a fold can inherit a live position whose original
            # entry forecast is not carried by this fold-local policy.
            self.anchor = {
                "position": current,
                "date_index": index,
                "scores": {
                    h: float(row[current]) if row is not None else float("nan")
                    for h, row in scores.items()
                },
                "observed_at_entry": False,
            }

        target = current
        reason = "HOLD_CURRENT"
        horizon_advantage: float | None = None
        uncertainty: float | None = None
        age: int | None = None
        if best == 0 or q_best <= self.cash_threshold:
            target = 0
            reason = "CASH_THRESHOLD"
        elif current == 0:
            target = best if q_best >= self.entry_threshold else 0
            reason = "ENTER_BEST" if target else "MIN_EXPECTED_EDGE"
        elif best == current:
            reason = "CURRENT_BEST"
        elif q_best < q_current + self.margin:
            reason = "SWITCH_MARGIN"
        else:
            assert self.anchor is not None
            age = max(0, index - int(self.anchor["date_index"]))
            available = []
            for h, weight in self.weights.items():
                row = scores[h]
                old = float(self.anchor["scores"].get(h, float("nan")))
                if row is None or not (np.isfinite(row[current]) and np.isfinite(row[best])):
                    continue
                remaining = max(0.0, 1.0 - age / float(h))
                if remaining > 0 and np.isfinite(old):
                    available.append((h, weight * remaining, old))
            if not available:
                # No non-expired entry horizon (or inherited fold boundary):
                # compare fresh horizon scores rather than manufacture an old one.
                available = [
                    (h, weight, float(scores[h][current]))
                    for h, weight in self.weights.items()
                    if scores[h] is not None
                    and np.isfinite(scores[h][current])
                    and np.isfinite(scores[h][best])
                ]
            total_weight = sum(w for _, w, _ in available)
            if total_weight <= 0:
                reason = "INSUFFICIENT_HORIZON_EVIDENCE"
            else:
                new_current = sum(
                    w * float(scores[h][current]) for h, w, _ in available
                ) / total_weight
                old_current = sum(w * old for _, w, old in available) / total_weight
                new_best = sum(
                    w * float(scores[h][best]) for h, w, _ in available
                ) / total_weight
                horizon_advantage = new_best - (
                    ANCHOR_BLEND * old_current
                    + (1.0 - ANCHOR_BLEND) * new_current
                )
                uncertainty = sum(
                    w * float(self.uncertainties[h]) for h, w, _ in available
                ) / total_weight
                if horizon_advantage > self.confidence_multiplier * uncertainty:
                    target = best
                    reason = "ROTATE_HORIZON_CONFIDENT"
                else:
                    reason = "HORIZON_CONFIDENCE_NOT_MET"

        if self.audit is not None:
            row: dict[str, Any] = {
                "fold_id": self.fold_id,
                "decision_date": timestamp.isoformat(),
                "current_asset": "CASH" if current == 0 else self.symbols[current - 1],
                "raw_best_asset": "CASH" if best == 0 else self.symbols[best - 1],
                "target_asset": "CASH" if target == 0 else self.symbols[target - 1],
                "reason": reason,
                "holding_days_simulator": holding,
                "forecast_age_sessions": age,
                "q_best": q_best,
                "q_current": q_current,
                "q_margin": self.margin,
                "horizon_advantage": horizon_advantage,
                "horizon_uncertainty_p75": uncertainty,
                "confidence_multiplier": self.confidence_multiplier,
            }
            for h in self.weights:
                pred = scores[h]
                row[f"H{h}_current_today"] = (
                    float(pred[current]) if pred is not None and current > 0 else None
                )
                row[f"H{h}_best_today"] = (
                    float(pred[best]) if pred is not None and best > 0 else None
                )
                row[f"H{h}_entry_forecast"] = (
                    self.anchor["scores"].get(h)
                    if self.anchor is not None and current > 0
                    else None
                )
            self.audit.append(row)

        if target == 0:
            self.anchor = None
        elif target != current:
            self.anchor = {
                "position": target,
                "date_index": index,
                "scores": {
                    h: float(row[target]) if row is not None else float("nan")
                    for h, row in scores.items()
                },
                "observed_at_entry": True,
            }
        return target, float(values[target])


def make_policy(
    *,
    symbols: list[str],
    dates: pd.DatetimeIndex,
    aggregate: dict[pd.Timestamp, np.ndarray],
    per_horizon: dict[int, dict[pd.Timestamp, np.ndarray]],
    uncertainties: dict[int, float],
    config: Any,
    margin: float,
    multiplier: float,
    audit: list[dict[str, Any]] | None = None,
    fold_id: int | None = None,
) -> HorizonPersistencePolicy:
    weights = np.asarray(config.rotation_target_horizon_weights, dtype=float)
    weights /= weights.sum()
    return HorizonPersistencePolicy(
        symbols=symbols,
        dates=dates,
        aggregate=aggregate,
        per_horizon=per_horizon,
        uncertainties=uncertainties,
        weights=dict(zip(config.rotation_target_horizons, weights)),
        margin=margin,
        cash_threshold=float(config.rotation_cash_threshold),
        entry_threshold=(
            float(config.rotation_cash_threshold)
            + float(config.rotation_min_expected_edge)
        ),
        confidence_multiplier=multiplier,
        audit=audit,
        fold_id=fold_id,
    )


def scenario_metrics(result: Any, folds: list[dict[str, Any]], capital: float) -> dict[str, Any]:
    m = summarize_metrics(result, folds, capital)
    m.update({
        "capital_rotations": int(result.metrics.get("capital_rotations") or 0),
        "total_transaction_fees": float(result.metrics.get("total_transaction_fees") or 0.0),
        "simulated_buys": int(result.metrics.get("simulated_buys") or 0),
        "simulated_sells": int(result.metrics.get("simulated_sells") or 0),
    })
    return m


def main() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    data = SnapshotPaths.u67(ROOT)
    snapshot = validate_snapshot(data)
    if tuple(snapshot.get("assets") or ()) != tuple(U67_REQUESTED_ASSETS):
        raise RuntimeError("Snapshot congelado U67 divergente; pesquisa interrompida.")
    needed = tuple(DISCOVERY_SELECTED_CANDIDATES)
    absent = [
        symbol for symbol in needed
        if not (FULL_DATA.raw_bars / f"{symbol}.csv").exists()
        or not (FULL_DATA.corporate_actions / f"{symbol}.csv").exists()
    ]
    if absent:
        raise RuntimeError(
            "Histórico U67+10 nao esta congelado no cache local: "
            + ", ".join(absent)
            + ". Reutilize o cache da pesquisa anterior. Nenhum download foi tentado."
        )

    base_raw, exclusions, _, _ = prepare_model_frames(
        data, assets=U67_REQUESTED_ASSETS, csv_float_precision="round_trip"
    )
    if frozenset(str(e.get("symbol") or "") for e in exclusions) != U67_EXPECTED_EXCLUSIONS:
        raise RuntimeError("Exclusões estruturais inesperadas na reproducao U67.")
    baseline = [s for s in U67_REQUESTED_ASSETS if s in base_raw]
    config = build_control_config({s: base_raw[s] for s in baseline}, CONFIG)
    config = config.copiar_modelo(update={"analysis_end_date": ANALYSIS_END_DATE})
    (
        baseline_frames, common_dates, _, baseline_symbols, folds,
        all_decision_dates, decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao({s: base_raw[s] for s in baseline}, config)

    candidate_frames, continuity, hashes = _prepare_candidate_frames(
        list(needed), common_dates, all_decision_dates, config
    )
    missing = [s for s in needed if s not in candidate_frames]
    if missing:
        continuity.to_csv(OUT / "continuity_audit.csv", index=False)
        raise RuntimeError(
            "Ativos externos sem continuidade/identidade de snapshot: "
            + ", ".join(missing)
        )
    continuity.to_csv(OUT / "continuity_audit.csv", index=False)
    print("[training] Baseline U67 agregado e 10 ativos adicionais", flush=True)
    states, _, _ = _train_baseline_states(
        baseline_frames, common_dates, baseline_symbols, folds, config
    )
    candidates = {
        s: _train_candidate_caches(s, candidate_frames[s], states, config)
        for s in needed
    }
    symbols = sorted([*baseline_symbols, *needed])
    frames = {**baseline_frames, **candidate_frames}
    configuration = construir_configuracao_controle(config, assets=symbols)
    configuration_without_hold = configuration.copiar_modelo(
        update={"rotation_min_holding_days": 0}
    )
    policies_2: dict[int, Any] = {}
    policies_0: dict[int, Any] = {}
    control_guard_audit: dict[pd.Timestamp, dict[str, Any]] = {}
    no_hold_guard_audit: dict[pd.Timestamp, dict[str, Any]] = {}
    fold_caches: dict[int, dict[str, Any]] = {}
    for state in states:
        cal = _merge_utility_caches(
            state.calibration_cache,
            baseline_symbols,
            {s: candidates[s][state.fold_id].calibration_cache for s in needed},
            symbols,
        )
        oos = _merge_utility_caches(
            state.decision_cache,
            baseline_symbols,
            {s: candidates[s][state.fold_id].decision_cache for s in needed},
            symbols,
        )
        scores = []
        for candidate_margin in configuration.rotation_switch_margin_candidates:
            policy = _politica_utilidade(
                {}, frames, symbols, configuration,
                float(candidate_margin), utility_cache=cal
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, state.calibration_dates, configuration
            )
            scores.append((float(candidate_margin), float(score)))
        selection = selecionar_switch_margin(scores)
        margin = max(
            float(configuration.rotation_switch_margin),
            float(selection["selected_candidate_margin"]),
        )
        policies_2[state.fold_id] = _politica_utilidade(
            {}, frames, symbols, configuration, margin,
            utility_cache=oos, decision_diagnostics=control_guard_audit,
            fold_id=state.fold_id,
        )
        policies_0[state.fold_id] = _politica_utilidade(
            {}, frames, symbols, configuration_without_hold, margin,
            utility_cache=oos, decision_diagnostics=no_hold_guard_audit,
            fold_id=state.fold_id,
        )
        fold_caches[state.fold_id] = {
            "calibration": cal,
            "oos": oos,
            "margin": margin,
            "calibration_margin_scores": scores,
        }

    result_2 = _simular_exato(
        "horizon_research_control_holding_2",
        _politica_agendada(policies_2, decision_to_fold),
        frames, symbols, all_decision_dates, configuration,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="U67+10 hold=2 (frozen control)",
    )
    metrics_2 = scenario_metrics(result_2, folds, float(configuration.initial_capital))
    actual = metrics_2["ending_capital"]
    expected = EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL
    if abs(actual - expected) > CHECKPOINT_TOLERANCE_DOLLARS:
        raise RuntimeError(
            f"Controle U67+10 nao reproduz o checkpoint: {actual:.8f} vs {expected:.8f}"
        )
    print(f"[control] US$ {actual:,.2f} reproduzido ao centavo", flush=True)

    result_0 = _simular_exato(
        "horizon_research_holding_0",
        _politica_agendada(policies_0, decision_to_fold),
        frames, symbols, all_decision_dates, configuration_without_hold,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="U67+10 hold=0 (ablation)",
    )
    metrics_0 = scenario_metrics(result_0, folds, float(configuration.initial_capital))
    print(f"[ablation] US$ {metrics_0['ending_capital']:,.2f}", flush=True)

    # Horizonte individual e estimativa de incerteza calibrados ANTES do OOS.
    policy_h: dict[int, Any] = {}
    model_audits: list[dict[str, Any]] = []
    decision_audit: list[dict[str, Any]] = []
    confidence_selection: list[dict[str, Any]] = []
    for state in states:
        horizon_cal = {}
        horizon_oos = {}
        uncertainty = {}
        for h in configuration.rotation_target_horizons:
            target_column = f"forward_horizon_utility_{h}"
            # Evita utilizar labels de horizonte que alcancariam a janela OOS.
            safe_final_dates = common_dates[
                :max(0, int(folds[state.fold_id - 1]["final_fit_end_index"]) - int(h))
            ]
            # The training label at date t uses observations until t+h.
            # Purge that tail before the first calibration observation.
            safe_train_dates = state.train_dates[:-int(h)]
            cal_models = _ajustar_modelos_lightgbm(
                frames, symbols, safe_train_dates, configuration,
                phase=f"horizon_{h}_fold_{state.fold_id}_cal",
                target_column=target_column,
            )
            cal_cache, _ = _precalcular_utilidades_modelo(
                cal_models, frames, symbols, state.calibration_dates, configuration
            )
            error, samples = uncertainty_from_matured_calibration(
                h=int(h),
                timestamps=state.calibration_dates,
                full_dates=common_dates,
                latest_label_date_index=int(
                    folds[state.fold_id - 1]["final_fit_end_index"]
                ),
                symbols=symbols,
                frames=frames,
                predictions=cal_cache,
            )
            final_models = _ajustar_modelos_lightgbm(
                frames, symbols, safe_final_dates, configuration,
                phase=f"horizon_{h}_fold_{state.fold_id}_final",
                target_column=target_column,
            )
            oos_cache, _ = _precalcular_utilidades_modelo(
                final_models, frames, symbols, state.decision_dates, configuration
            )
            horizon_cal[int(h)] = cal_cache
            horizon_oos[int(h)] = oos_cache
            uncertainty[int(h)] = error
            model_audits.append({
                "fold_id": state.fold_id,
                "horizon": int(h),
                "matured_calibration_errors": samples,
                "absolute_error_p75": error,
                "calibration_models": len(cal_models),
                "calibration_training_end": safe_train_dates[-1].isoformat(),
                "oos_models": len(final_models),
                "final_training_end": safe_final_dates[-1].isoformat(),
            })
            print(
                f"[horizon] fold={state.fold_id} H{h} "
                f"sigma_p75={error:.6f} models={len(final_models)}",
                flush=True,
            )

        calibrated: list[tuple[float, float]] = []
        for multiplier in CONFIDENCE_MULTIPLIERS:
            policy = make_policy(
                symbols=symbols,
                dates=state.calibration_dates,
                aggregate=fold_caches[state.fold_id]["calibration"],
                per_horizon=horizon_cal,
                uncertainties=uncertainty,
                config=configuration_without_hold,
                margin=fold_caches[state.fold_id]["margin"],
                multiplier=multiplier,
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, state.calibration_dates,
                configuration_without_hold,
            )
            calibrated.append((multiplier, float(score)))
        # Calibra o multiplicador apenas com as sessoes anteriores ao teste.
        selected = sorted(calibrated, key=lambda item: (-item[1], item[0]))[0][0]
        confidence_selection.append({
            "fold_id": state.fold_id,
            "candidates": [
                {"multiplier": x, "calibration_score": y}
                for x, y in calibrated
            ],
            "selected": selected,
        })
        policy_h[state.fold_id] = make_policy(
            symbols=symbols,
            dates=state.decision_dates,
            aggregate=fold_caches[state.fold_id]["oos"],
            per_horizon=horizon_oos,
            uncertainties=uncertainty,
            config=configuration_without_hold,
            margin=fold_caches[state.fold_id]["margin"],
            multiplier=selected,
            audit=decision_audit,
            fold_id=state.fold_id,
        )
    result_h = _simular_exato(
        "horizon_research_confidence",
        _politica_agendada(policy_h, decision_to_fold),
        frames, symbols, all_decision_dates, configuration_without_hold,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="U67+10 horizon-persistence experimental",
    )
    metrics_h = scenario_metrics(result_h, folds, float(configuration.initial_capital))
    variants = {
        "control_holding_2": metrics_2,
        "ablation_holding_0": metrics_0,
        "horizon_confidence": metrics_h,
    }
    for variant, result in (
        ("control_holding_2", result_2),
        ("ablation_holding_0", result_0),
        ("horizon_confidence", result_h),
    ):
        result.predictions.to_csv(
            OUT / f"{variant}_equity.csv", index_label="timestamp"
        )
        result.trades.to_csv(OUT / f"{variant}_trades.csv", index=False)
    pd.DataFrame(model_audits).to_csv(OUT / "horizon_calibration.csv", index=False)
    pd.DataFrame([
        {"decision_date": date.isoformat(), **row}
        for date, row in sorted(control_guard_audit.items())
    ]).to_csv(OUT / "control_decision_reasons.csv", index=False)
    pd.DataFrame([
        {"decision_date": date.isoformat(), **row}
        for date, row in sorted(no_hold_guard_audit.items())
    ]).to_csv(OUT / "no_hold_decision_reasons.csv", index=False)
    pd.DataFrame(decision_audit).to_csv(OUT / "horizon_decisions.csv", index=False)
    pd.DataFrame([
        {"variant": name, **{k: values.get(k) for k in (
            "ending_capital", "cagr", "sharpe", "maximum_drawdown",
            "worst_fold_return", "capital_rotations", "total_transaction_fees",
        )}} for name, values in variants.items()
    ]).to_csv(OUT / "comparison.csv", index=False)

    hashes = {
        symbol: {
            "raw_sha256": sha256_file(FULL_DATA.raw_bars / f"{symbol}.csv"),
            "actions_sha256": sha256_file(FULL_DATA.corporate_actions / f"{symbol}.csv"),
        }
        for symbol in needed
    }
    payload = {
        "script_version": VERSION,
        "status": "completed",
        "analysis_end": ANALYSIS_END_DATE,
        "snapshot_u67_sha256": snapshot.get("snapshot_sha256"),
        "external_asset_hashes": hashes,
        "requested_asset_count": len(U67_REQUESTED_ASSETS) + len(needed),
        "effective_asset_count": len(symbols),
        "structural_exclusions": sorted(U67_EXPECTED_EXCLUSIONS),
        "horizons": list(configuration.rotation_target_horizons),
        "horizon_weights": list(configuration.rotation_target_horizon_weights),
        "confidence_multiplier_candidates": list(CONFIDENCE_MULTIPLIERS),
        "confidence_note": (
            "Proxy using matured pre-OOS prediction errors; not a calibrated "
            "probability. Full impact requires independent OOS confirmation."
        ),
        "confidence_selections": confidence_selection,
        "metrics": variants,
        "checkpoint_reference": expected,
        "reproduces_checkpoint_to_cent": True,
        "export_to_operational": False,
    }
    write_json(OUT / "experiment.json", payload)
    package_path = OUT / "pacote_persistencia_horizontes_u77.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(OUT.iterdir()):
            if artifact.is_file() and artifact != package_path:
                archive.write(artifact, arcname=artifact.name)
    print(f"[package] {package_path}", flush=True)
    print("[final] " + json.dumps(
        {name: values["ending_capital"] for name, values in variants.items()}
    ), flush=True)
    print(f"[output] {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    main()
