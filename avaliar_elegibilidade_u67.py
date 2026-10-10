"""Mede o efeito da elegibilidade causal com os mesmos dados e modelos.

Execute no Spyder com F5 ou no terminal: python avaliar_elegibilidade_u67.py.
Não baixa dados nem altera parâmetros. A máscara antiga existe somente neste
controle histórico; o fluxo oficial utiliza sempre a elegibilidade corrigida.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from engine.configuracao import (
    CONFIG, EXPERIMENT_VERSION, U67_REQUESTED_ASSETS,
    U67_EXPECTED_EFFECTIVE_COUNT, U67_EXPECTED_EXCLUSIONS,
)
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm, _construir_contexto_execucao,
    diagnosticos_fora_amostra, selecionar_switch_margin,
)
from engine.rotacao import (
    _crescimento_politica_simples, _politica_agendada, _politica_utilidade,
    _precalcular_utilidades_modelo, _simular_exato,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config, summarize_metrics
from reproducao.preparacao import prepare_model_frames

ROOT = Path(__file__).resolve().parent
BASELINE_COMMIT = "192dc7bd457beed66a62a13c368b20865da748c9"
SNAPSHOT_SHA256 = "e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575"
HISTORICAL_ENDING_CAPITAL = 76_927_051.38897176


def mascara_historica(cache, frames, symbols, dates):
    """Reconstrói o filtro v1.22.0-dev.9 sobre previsões dos mesmos modelos."""
    legacy = {pd.Timestamp(t): values.copy() for t, values in cache.items()}
    removed = 0
    for column, symbol in enumerate(symbols, 1):
        frame = frames[symbol]
        for date in dates:
            values = legacy[pd.Timestamp(date)]
            if not np.isfinite(values[column]):
                continue
            location = frame.index.get_indexer([date])[0]
            valid = location >= 0 and location + 1 < len(frame)
            if valid:
                prices = frame.iloc[location + 1][["open", "close"]].to_numpy(dtype=float)
                valid = bool(np.isfinite(prices).all() and (prices > 0).all())
            if not valid:
                values[column] = -np.inf
                removed += 1
    return legacy, removed


def calibrar(models, frames, symbols, dates, config, cache):
    candidates = []
    for margin in config.rotation_switch_margin_candidates:
        policy = _politica_utilidade(models, frames, symbols, config, float(margin), utility_cache=cache)
        score = _crescimento_politica_simples(policy, frames, symbols, dates, config)
        candidates.append((float(margin), float(score)))
    selected = selecionar_switch_margin(candidates)
    selected["effective_margin"] = max(config.rotation_switch_margin, selected["selected_candidate_margin"])
    selected["candidates"] = candidates
    return selected


def avaliar(data_root=None, output=None, historical=None):
    started = time.perf_counter()
    out = Path(output) if output else ROOT / "output" / "elegibilidade"
    out.mkdir(parents=True, exist_ok=True)
    paths = SnapshotPaths.from_root(Path(data_root)) if data_root else SnapshotPaths.u67(ROOT)
    snapshot = validate_snapshot(paths)
    if snapshot["snapshot_sha256"] != SNAPSHOT_SHA256:
        raise ValueError("Esta comparação exige o mesmo snapshot do experimento registrado.")
    raw, exclusions, _, audit = prepare_model_frames(
        paths, assets=U67_REQUESTED_ASSETS, csv_float_precision="round_trip"
    )
    if len(raw) != U67_EXPECTED_EFFECTIVE_COUNT or frozenset(e["symbol"] for e in exclusions) != U67_EXPECTED_EXCLUSIONS:
        raise ValueError("O processamento não preservou o contrato do universo fixo.")
    config = build_control_config(raw, CONFIG).copiar_modelo(update={"analysis_end_date": "2026-10-06"})
    frames, common_dates, calendar, symbols, folds, dates, mapping, metadata = _construir_contexto_execucao(raw, config)
    policies = {"legado": {}, "corrigido": {}}
    calibrations = []
    predictive = []
    availability = []
    fits = []

    def progress(phase):
        def report(done, total, device):
            if done % 5 == 0 or done == total:
                print(f"[train] {phase} {done}/{total} device={device}", flush=True)
        return report

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[int(fold["calibration_start_index"]):int(fold["calibration_end_index"])]
        final_dates = common_dates[:int(fold["final_fit_end_index"])]
        decision_dates = pd.DatetimeIndex(fold["decision_dates"])
        print(f"[fold] {fold_id}/{len(folds)} calibration", flush=True)
        models = _ajustar_modelos_lightgbm(
            frames, symbols, train_dates, config, phase=f"eligibility_fold_{fold_id}_calibration",
            progress_callback=progress(f"fold={fold_id} calibration"),
        )
        fits.append({"fold_id": fold_id, "phase": "calibration", "model_count": len(models)})
        causal, _ = _precalcular_utilidades_modelo(models, frames, symbols, calibration_dates, config)
        old, removed_calibration = mascara_historica(causal, frames, symbols, calibration_dates[:-1])
        selections = {}
        for variant, cache in [("legado", old), ("corrigido", causal)]:
            selections[variant] = calibrar(models, frames, symbols, calibration_dates, config, cache)
        calibrations.append({"fold_id": fold_id, **selections})
        del models, causal, old

        print(f"[fold] {fold_id}/{len(folds)} final_fit", flush=True)
        models = _ajustar_modelos_lightgbm(
            frames, symbols, final_dates, config, phase=f"eligibility_fold_{fold_id}_final",
            progress_callback=progress(f"fold={fold_id} final"),
        )
        fits.append({"fold_id": fold_id, "phase": "final_fit", "model_count": len(models)})
        causal, _ = _precalcular_utilidades_modelo(models, frames, symbols, decision_dates, config)
        old, removed_test = mascara_historica(causal, frames, symbols, decision_dates[:-1])
        predictive.append(diagnosticos_fora_amostra(
            models, frames, symbols, decision_dates[:-1], causal, fold_id=fold_id,
        ))
        availability.append({"fold_id": fold_id, "removed_by_legacy_filter_calibration": removed_calibration,
                             "removed_by_legacy_filter_test": removed_test,
                             "calibration_asset_decisions": len(calibration_dates[:-1]) * len(symbols),
                             "test_asset_decisions": len(decision_dates[:-1]) * len(symbols)})
        for variant, cache in [("legado", old), ("corrigido", causal)]:
            selected = selections[variant]
            # Todos os escores necessários já estão no cache. Os dois braços
            # compartilham os ajustes; nenhum ajuste adicional muda a comparação.
            policies[variant][fold_id] = _politica_utilidade(
                {}, frames, symbols, config, selected["effective_margin"], fold_id=fold_id,
                calibrated_switch_margin=selected["selected_candidate_margin"], utility_cache=cache,
            )
        del models
        print(f"[eligibility] fold={fold_id} calibration_changes={removed_calibration} test_changes={removed_test}", flush=True)

    results = {}
    metrics = {}
    for variant in policies:
        policy = _politica_agendada(policies[variant], mapping)
        results[variant] = _simular_exato(
            "eligibility_audit", policy, frames, symbols, dates, config,
            calcular_taxas_referencia, aplicar_deslizamento, decision_metadata=metadata,
            model_label=f"Control eligibility {variant}",
        )
        metrics[variant] = summarize_metrics(results[variant], folds, config.initial_capital)
        metrics[variant]["calendar_source_asset"] = calendar
        metrics[variant]["effective_compute_device"] = "cpu"
        metrics[variant]["eligibility_information"] = (
            "decision_session_features_and_next_session_open_close" if variant == "legado"
            else "decision_session_features_only"
        )
        metrics[variant]["predictive_diagnostics"] = {
            "evaluation": "out_of_sample_decision_sessions",
            "folds": predictive,
        }
        results[variant].predictions.to_csv(out / f"{variant}_predictions.csv", float_format="%.17g")
        results[variant].trades.to_csv(out / f"{variant}_trades.csv", index=False, float_format="%.17g")

    old, new = results["legado"], results["corrigido"]
    if not old.predictions.index.equals(new.predictions.index):
        raise AssertionError("Os calendários dos dois braços divergiram.")
    curve_delta = new.predictions["strategy_equity"] - old.predictions["strategy_equity"]
    decisions_changed = int((new.predictions["selected_asset"] != old.predictions["selected_asset"]).sum())
    historical_metrics = None
    if historical:
        hpath = Path(historical)
        record = json.loads(hpath.read_text(encoding="utf-8"))
        historical_metrics = {"source_sha256": hashlib.sha256(hpath.read_bytes()).hexdigest(),
                              "reproduction_version": record.get("reproduction_version"),
                              "ending_capital": record["metrics"]["ending_capital"]}
    cfg = asdict(config)
    payload = {
        "version": EXPERIMENT_VERSION, "status": "completed", "baseline_commit": BASELINE_COMMIT,
        "comparison_design": "paired_same_fitted_models_future_availability_filter_only",
        "snapshot_sha256": snapshot["snapshot_sha256"], "verified_file_hashes": len(snapshot["file_hashes"]),
        "configuration": cfg,
        "configuration_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "data_audit": audit, "calendar_source": calendar,
        "execution_start": new.predictions.index[0].isoformat(),
        "execution_end": new.predictions.index[-1].isoformat(),
        "sessions": len(new.predictions), "model_fits": fits,
        "calibrations": calibrations, "eligibility_audit": availability,
        "metrics": metrics, "predictive_diagnostics": {"folds": predictive},
        "comparison": {"ending_capital_delta": metrics["corrigido"]["ending_capital"] - metrics["legado"]["ending_capital"],
                       "curve_max_abs_delta": float(curve_delta.abs().max()),
                       "decision_sessions_changed": decisions_changed,
                       "trades_identical": old.trades.equals(new.trades),
                       "recorded_ending_capital_reference": HISTORICAL_ENDING_CAPITAL,
                       "legacy_run_delta_from_recorded": metrics["legado"]["ending_capital"] - HISTORICAL_ENDING_CAPITAL},
        "historical_record": historical_metrics,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {p: version(p) for p in ["numpy", "pandas", "lightgbm", "scikit-learn", "threadpoolctl"]},
                    "threadpools": threadpool_info()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (out / "comparacao_elegibilidade.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps({"comparison": payload["comparison"],
                      "ending_capital": {v: m["ending_capital"] for v, m in metrics.items()},
                      "buy_hold": metrics["corrigido"]["buy_hold_ending_capital"],
                      "output": str(out)}, ensure_ascii=False, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", help="Diretório do snapshot U67 congelado")
    parser.add_argument("--output", help="Diretório de resultados")
    parser.add_argument("--historico", help="JSON original preservado, opcional")
    args = parser.parse_args()
    avaliar(args.dados, args.output, args.historico)
