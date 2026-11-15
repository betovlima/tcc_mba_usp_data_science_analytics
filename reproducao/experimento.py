"""Apoio da reproducao oficial do universo completo de 77 ativos."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from engine.configuracao import (
    CONFIG,
    StandaloneBacktestConfig,
    construir_configuracao_controle,
)
from engine.rotacao import _desempenho_folds


def build_control_config(
    frames: dict[str, pd.DataFrame],
    base: StandaloneBacktestConfig = CONFIG,
) -> StandaloneBacktestConfig:
    """Cria a configuracao Control somente com os ativos elegiveis."""
    eligible = tuple(frames)
    prepared = base.copiar_modelo(update={"assets": eligible})
    return construir_configuracao_controle(
        prepared,
        assets=eligible,
    )


def summarize_metrics(
    result: Any,
    folds: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    """Extrai as metricas usadas no artefato final da reproducao."""
    fold_rows = _desempenho_folds(
        result.predictions,
        folds,
        initial_capital,
    )
    worst_fold = (
        min(float(row["strategy_return"]) for row in fold_rows)
        if fold_rows
        else None
    )
    simulation = deepcopy(result.metrics.get("simulation_profile") or {})
    return {
        "ending_capital": float(
            result.metrics.get("strategy_ending_capital") or 0.0
        ),
        "strategy_return": float(
            result.metrics.get("strategy_return") or 0.0
        ),
        "cagr": float(result.metrics.get("strategy_cagr") or 0.0),
        "sharpe": float(result.metrics.get("strategy_sharpe") or 0.0),
        "maximum_drawdown": float(
            result.metrics.get("strategy_maximum_drawdown") or 0.0
        ),
        "worst_fold_return": worst_fold,
        "folds": fold_rows,
        "buy_hold_ending_capital": float(
            result.metrics.get("buy_hold_ending_capital") or 0.0
        ),
        "buy_hold_return": float(
            result.metrics.get("buy_hold_return") or 0.0
        ),
        "buy_hold_cagr": float(
            result.metrics.get("buy_hold_cagr") or 0.0
        ),
        "buy_hold_sharpe": float(
            result.metrics.get("buy_hold_sharpe") or 0.0
        ),
        "buy_hold_maximum_drawdown": float(
            result.metrics.get("buy_hold_maximum_drawdown") or 0.0
        ),
        "benchmark_name": result.metrics.get("benchmark_name"),
        "benchmark_assets": list(result.metrics.get("benchmark_assets") or []),
        "benchmark_asset_count": result.metrics.get("benchmark_asset_count"),
        "benchmark_same_universe": result.metrics.get("benchmark_same_universe"),
        "calendar_source_asset": result.metrics.get(
            "calendar_source_asset"
        ),
        "requested_compute_device": result.metrics.get(
            "requested_compute_device"
        ),
        "effective_compute_device": result.metrics.get(
            "effective_compute_device"
        ),
        "predictive_diagnostics": deepcopy(
            result.metrics.get("lightgbm_predictive_diagnostics") or {}
        ),
        "simulation_profile": simulation,
        "simulation_session_count": simulation.get("session_count"),
    }



def reproduzir_universo_congelado(
    *,
    root: Path,
    output: Path,
    version: str,
    execution_schema: str,
) -> dict[str, Any]:
    """Treina, calibra e simula os 77 ativos conjuntamente em cada fold.

    Os dois diretorios de entrada existem apenas para preservar os snapshots
    congelados. Nao existem universos de treino separados, acrescentos de
    modelos a posteriori nem checkpoints de capital da campanha anterior.
    """
    import hashlib
    import json
    import math
    import shutil
    import time

    import numpy as np

    from engine.configuracao import U67_REQUESTED_ASSETS
    from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
    from engine.modelo_lightgbm import (
        _ajustar_modelos_lightgbm,
        _construir_contexto_execucao,
        diagnosticos_fora_amostra,
        selecionar_switch_margin,
    )
    from engine.rotacao import (
        _crescimento_politica_simples,
        _politica_agendada,
        _politica_utilidade,
        _precalcular_utilidades_modelo,
        _simular_exato,
    )
    from reproducao.artefatos import criar_pacote_analise, sinal_sonoro_conclusao
    from reproducao.dados import SnapshotPaths, validate_snapshot
    from reproducao.graficos import gerar_analises_backtest
    from reproducao.preparacao import prepare_model_frames

    started = time.perf_counter()
    root = Path(root).resolve()
    output = Path(output).resolve()

    contract = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json")
        .read_text(encoding="utf-8")
    )
    selected = tuple(contract["selected_assets"])
    base_assets = tuple(U67_REQUESTED_ASSETS)
    requested = (*base_assets, *selected)
    if len(requested) != 77 or len(set(requested)) != 77:
        raise RuntimeError(
            "O experimento exige exatamente 77 ativos distintos, sem exclusoes."
        )
    print("[universe] requested=77 preparation=single_pass folds=joint", flush=True)

    base_data = SnapshotPaths.u67(root)
    candidate_data = SnapshotPaths.from_root(
        root / "dados" / ".u67_expansion_cache" / "full_raw_20261006"
    )
    snapshot = validate_snapshot(base_data)
    if snapshot.get("snapshot_sha256") != contract["snapshot_sha256"]:
        raise RuntimeError("Identidade do snapshot congelado divergente.")
    if tuple(snapshot.get("assets") or ()) != base_assets:
        raise RuntimeError("Lista do snapshot base divergente da configuracao.")

    def digest(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    # A origem dos arquivos e dupla por razoes historicas, mas o painel de
    # dados, o treino, a calibracao e o replay sao UNICOS para os 77.
    candidate_hashes: dict[str, dict[str, str]] = {}
    if len(selected) != 10 or set(contract["candidate_file_hashes"]) != set(selected):
        raise RuntimeError("Manifesto dos dez ativos adicionais incompleto.")
    for symbol in selected:
        expected = contract["candidate_file_hashes"][symbol]
        paths = {
            "raw_sha256": candidate_data.raw_bars / f"{symbol}.csv",
            "corporate_actions_sha256": (
                candidate_data.corporate_actions / f"{symbol}.csv"
            ),
        }
        observed = {}
        for name, file_path in paths.items():
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Arquivo congelado ausente: {file_path}. "
                    "Restaure o cache historico; nao execute downloads."
                )
            observed[name] = digest(file_path)
            if observed[name] != expected[name]:
                raise RuntimeError(
                    f"{symbol}: SHA-256 divergente em {name}; "
                    "nao modificar o historico para obter um resultado."
                )
        candidate_hashes[symbol] = observed

    raw, exclusions, diagnostics, data_audit = prepare_model_frames(
        base_data,
        assets=requested,
        source_paths={symbol: candidate_data for symbol in selected},
        csv_float_precision="round_trip",
    )
    if exclusions or set(raw) != set(requested) or len(raw) != 77:
        raise RuntimeError(
            "Universo de entrada incompleto: "
            f"esperados=77 preparados={len(raw)} "
            f"ausentes={sorted(set(requested) - set(raw))} "
            f"exclusoes={exclusions}"
        )
    print("[universe] prepared=77/77 excluded=0", flush=True)

    config = build_control_config(
        {symbol: raw[symbol] for symbol in requested}
    ).copiar_modelo(update={"analysis_end_date": contract["analysis_end_date"]})
    (
        frames,
        common_dates,
        calendar_source,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _construir_contexto_execucao(
        {symbol: raw[symbol] for symbol in requested},
        config,
    )
    if (
        len(frames) != 77
        or len(symbols) != 77
        or set(symbols) != set(requested)
    ):
        raise RuntimeError(
            "Painel unificado nao possui 77 ativos: "
            f"frames={len(frames)} modelos={len(symbols)} "
            f"ausentes={sorted(set(requested) - set(symbols))}"
        )
    if len(folds) != 3:
        raise RuntimeError(
            f"Protocolo temporal inesperado: esperados=3 folds, obtidos={len(folds)}"
        )

    # Conferencia causal e de precos de execucao ANTES de qualquer modelo.
    required_dates = pd.DatetimeIndex(common_dates).normalize()
    execution_dates = pd.DatetimeIndex(all_decision_dates[1:])
    continuity_rows = []
    for symbol in requested:
        available_dates = pd.DatetimeIndex(raw[symbol].index).normalize()
        missing = required_dates.difference(available_dates)
        execution = frames[symbol].reindex(execution_dates)
        bars = execution[["open", "high", "low", "close", "volume"]].to_numpy(
            dtype=np.float64
        )
        valid = bool(np.isfinite(bars).all() and (bars[:, :4] > 0).all())
        row = {
            "asset": symbol,
            "status": "passed" if not len(missing) and valid else "rejected",
            "missing_required_sessions": int(len(missing)),
            "execution_prices_valid": valid,
            "required_session_count": int(len(required_dates)),
        }
        continuity_rows.append(row)
        if row["status"] != "passed":
            raise RuntimeError(
                f"{symbol}: historico incompleto para o painel unico. "
                f"missing_sessions={len(missing)} execution_prices_valid={valid}"
            )
    continuity = pd.DataFrame(continuity_rows)

    print(
        f"[universe] context=77/77 folds={len(folds)} "
        f"decision_sessions={len(all_decision_dates)-1} "
        "train_calibration_replay=joint",
        flush=True,
    )
    policies = {}
    calibration_rows = []
    model_fits = []
    predictive = []

    def on_fit(fold_id: int, phase: str):
        def report(done: int, total: int, device: str) -> None:
            if done % 10 == 0 or done == total:
                print(
                    f"[train] fold={fold_id}/{len(folds)} phase={phase} "
                    f"models={done}/{total} device={device}",
                    flush=True,
                )
        return report

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_fit_dates = common_dates[:int(fold["final_fit_end_index"])]
        decision_dates = pd.DatetimeIndex(fold["decision_dates"])

        print(
            f"[fold] {fold_id}/{len(folds)} calibration assets=77",
            flush=True,
        )
        calibration_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config,
            phase=f"universe77_fold_{fold_id}_calibration",
            progress_callback=on_fit(fold_id, "calibration"),
        )
        if set(calibration_models) != set(requested):
            raise RuntimeError(
                f"fold={fold_id}: modelos de calibracao incompletos "
                f"({len(calibration_models)}/77), "
                f"ausentes={sorted(set(requested)-set(calibration_models))}"
            )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_models, frames, symbols, calibration_dates, config
        )
        scored_margins = []
        for margin in config.rotation_switch_margin_candidates:
            policy = _politica_utilidade(
                {},
                frames,
                symbols,
                config,
                float(margin),
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, calibration_dates, config
            )
            scored_margins.append((float(margin), float(score)))
        choice = selecionar_switch_margin(scored_margins)
        chosen_margin = float(choice["selected_candidate_margin"])
        effective_margin = max(float(config.rotation_switch_margin), chosen_margin)
        for margin, score in scored_margins:
            calibration_rows.append({
                "fold_id": fold_id,
                "asset_count": 77,
                "candidate_margin": margin,
                "calibration_score": score,
                "selected": bool(abs(margin - chosen_margin) <= 1e-12),
            })
        model_fits.append({
            "fold_id": fold_id, "phase": "calibration",
            "model_count": len(calibration_models),
        })
        del calibration_models

        print(f"[fold] {fold_id}/{len(folds)} final_fit assets=77", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config,
            phase=f"universe77_fold_{fold_id}_final",
            progress_callback=on_fit(fold_id, "final_fit"),
        )
        if set(final_models) != set(requested):
            raise RuntimeError(
                f"fold={fold_id}: modelos finais incompletos "
                f"({len(final_models)}/77), "
                f"ausentes={sorted(set(requested)-set(final_models))}"
            )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, decision_dates, config
        )
        predictive.append(diagnosticos_fora_amostra(
            final_models,
            frames,
            symbols,
            decision_dates,
            decision_cache,
            fold_id=fold_id,
        ))
        policies[fold_id] = _politica_utilidade(
            {},
            frames,
            symbols,
            config,
            effective_margin,
            fold_id=fold_id,
            calibrated_switch_margin=chosen_margin,
            utility_cache=decision_cache,
        )
        model_fits.append({
            "fold_id": fold_id, "phase": "final_fit",
            "model_count": len(final_models),
        })
        del final_models
        print(
            f"[fold] {fold_id}/{len(folds)} completed "
            "calibration_models=77 final_models=77",
            flush=True,
        )

    print("[replay] assets=77 policy=single_joint_rotation", flush=True)
    result = _simular_exato(
        "tcc_unified_77",
        _politica_agendada(policies, decision_to_fold),
        frames,
        symbols,
        all_decision_dates,
        config,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="TCC Winner 77 - LightGBM Control",
        method_line="- Os 77 ativos foram treinados e calibrados conjuntamente por fold.",
    )
    metrics = summarize_metrics(result, folds, float(config.initial_capital))
    metrics.update({
        "capital_rotations": int(result.metrics.get("capital_rotations") or 0),
        "simulated_buys": int(result.metrics.get("simulated_buys") or 0),
        "simulated_sells": int(result.metrics.get("simulated_sells") or 0),
        "cash_days": int(result.metrics.get("cash_days") or 0),
        "market_exposure": float(result.metrics.get("market_exposure") or 0),
        "predictive_diagnostics": {"folds": predictive},
    })
    capital = float(metrics["ending_capital"])

    # Nenhuma comparacao com um capital calculado em outro universo.
    exact = None
    expected_capital = None

    # Apenas o replay completo e validado pode substituir o ZIP anterior.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"asset": requested}).to_csv(
        output / "requested_assets.csv", index=False
    )
    pd.DataFrame({"asset": requested}).to_csv(
        output / "effective_assets.csv", index=False
    )
    pd.DataFrame(columns=["symbol", "reason"]).to_csv(
        output / "excluded_assets.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        output / "fold_calibration.csv", index=False
    )
    pd.DataFrame(model_fits).to_csv(
        output / "model_fits.csv", index=False
    )
    continuity.to_csv(output / "asset_continuity.csv", index=False)
    result.predictions.to_csv(
        output / "predictions.csv", index=True, float_format="%.17g"
    )
    result.trades.to_csv(
        output / "trades.csv", index=False
    )

    payload = {
        "reproduction_version": version,
        "execution_schema": execution_schema,
        "status": "completed",
        "purpose": "reproducao_conjunta_77_ativos",
        "reproduced_exact_to_cent": exact,
        "checkpoint_validation": "awaiting_first_77_asset_baseline",
        "analysis_end_date": contract["analysis_end_date"],
        "checkpoint": {
            "observed_capital": capital,
            "expected_capital": expected_capital,
            "delta": None,
            "status": "new_77_asset_reference",
        },
        "data": {
            "source": "alpaca",
            "feed": "sip",
            "adjustment": "raw",
            "timeframe": "1Day",
            "full_refresh": False,
            "database_used": False,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "candidate_file_hashes": candidate_hashes,
            "candidate_cache_committed_to_git": False,
            "data_audit": data_audit,
            "diagnostics": diagnostics,
        },
        "universe": {
            "requested_assets": list(requested),
            "requested_count": 77,
            "effective_assets": list(requested),
            "effective_count": 77,
            "trained_assets": symbols,
            "trained_count_each_fold": 77,
            "excluded_assets": [],
        },
        "calibration": {
            "folds": len(folds),
            "asset_count_each_fold": 77,
            "fit_counts": model_fits,
        },
        "calendar": {
            "calendar_source": calendar_source,
            "decision_sessions": len(all_decision_dates) - 1,
        },
        "metrics": metrics,
        "baseline_reference": None,
        "runtime_seconds": time.perf_counter() - started,
    }

    def make_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [make_json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return make_json_safe(value.item())
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    (output / "reproducao_universo.json").write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    gerar_analises_backtest(output, manifest=snapshot, result=result)
    package = criar_pacote_analise(
        output,
        comparison_file="reproducao_universo.json",
        execution_schema=execution_schema,
        archive_name="pacote_reproducao_universo.zip",
    )
    print(
        f"[done] capital=US$ {capital:,.2f} effective_assets=77/77 "
        f"folds={len(folds)} zip={package}",
        flush=True,
    )
    sinal_sonoro_conclusao()
    return payload
