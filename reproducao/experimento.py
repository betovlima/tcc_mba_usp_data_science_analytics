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
    """Reproduz o universo selecionado pela mesma rotina financeira da pesquisa.

    O modulo historico continua sendo a unica fonte do procedimento de treino,
    da calibracao, da composicao dos caches e do replay. Nenhum download e
    permitido neste caminho; os dados dos candidatos precisam corresponder
    exatamente aos hashes versionados em evidencias/.
    """
    import hashlib
    import json
    import shutil
    import time

    from reproducao.dados import validate_snapshot
    from reproducao.preparacao import prepare_model_frames
    from reproducao.artefatos import criar_pacote_analise, sinal_sonoro_conclusao
    from reproducao.graficos import gerar_analises_backtest

    # Dependencia cientifica legada mantida por identidade de calculo.
    import pesquisar_expansao_u67_spyder as pesquisa

    started = time.perf_counter()
    root = Path(root).resolve()
    output = Path(output).resolve()

    manifest_path = root / "evidencias" / "manifest_universo_congelado.json"
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = tuple(contract["selected_assets"])
    if selected != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES):
        raise RuntimeError("Composicao selecionada divergiu da pesquisa congelada.")
    if len(set(selected)) != len(selected):
        raise RuntimeError("Ativos duplicados no manifesto do universo.")

    snapshot = validate_snapshot(pesquisa.BASE_DATA)
    if snapshot.get("snapshot_sha256") != contract["snapshot_sha256"]:
        raise RuntimeError(
            "Snapshot diferente do congelado. Preserve o historico; "
            "nao atualize os dados para ajustar o resultado."
        )

    def digest(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    # Verificar as duas fontes locais de cada candidato ANTES de treinar.
    file_hashes: dict[str, dict[str, str]] = {}
    for symbol in selected:
        expected = contract["candidate_file_hashes"][symbol]
        paths = {
            "raw_sha256": pesquisa.FULL_DATA.raw_bars / f"{symbol}.csv",
            "corporate_actions_sha256": (
                pesquisa.FULL_DATA.corporate_actions / f"{symbol}.csv"
            ),
        }
        observed = {}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Arquivo congelado ausente: {path}. "
                    "Restaure o cache historico; nao faca download."
                )
            observed[key] = digest(path)
            if observed[key] != expected[key]:
                raise RuntimeError(
                    f"Hash divergente para {symbol} ({key}). "
                    "O arquivo foi alterado; nenhum download foi realizado."
                )
        file_hashes[symbol] = observed
    print(f"[input] snapshot validado; {len(selected)} candidatos com SHA-256 exato",
          flush=True)

    requested = tuple(pesquisa.U67_REQUESTED_ASSETS)
    if tuple(snapshot.get("assets") or ()) != requested:
        raise RuntimeError("Ordem/identidade dos ativos base divergente.")

    raw, exclusions, diagnostics, data_audit = prepare_model_frames(
        pesquisa.BASE_DATA,
        assets=requested,
        csv_float_precision="round_trip",
    )
    excluded = frozenset(str(item.get("symbol") or "").upper() for item in exclusions)
    if excluded or exclusions:
        raise RuntimeError(
            "Universo incompleto: nenhuma exclusao e permitida na reproducao "
            f"de 77 ativos: {sorted(excluded)}."
        )
    baseline_order = tuple(symbol for symbol in requested if symbol in raw)
    if baseline_order != requested or len(baseline_order) != pesquisa.U67_EXPECTED_REQUESTED_COUNT:
        raise RuntimeError(
            "Universo base incompleto: todos os 67 ativos devem ser "
            "efetivamente preparados."
        )

    config = build_control_config(
        {symbol: raw[symbol] for symbol in baseline_order},
        pesquisa.CONFIG,
    ).copiar_modelo(update={"analysis_end_date": contract["analysis_end_date"]})
    (
        baseline_frames, common_dates, calendar_source, baseline_symbols,
        folds, all_decision_dates, decision_to_fold, decision_metadata,
    ) = pesquisa._construir_contexto_execucao(
        {symbol: raw[symbol] for symbol in baseline_order}, config
    )
    states, baseline_policies, baseline_calibration = (
        pesquisa._train_baseline_states(
            baseline_frames, common_dates, baseline_symbols, folds, config
        )
    )
    baseline_result = pesquisa._simular_exato(
        "tcc_base_reference",
        pesquisa._politica_agendada(baseline_policies, decision_to_fold),
        baseline_frames,
        baseline_symbols,
        all_decision_dates,
        config,
        pesquisa.calcular_taxas_referencia,
        pesquisa.aplicar_deslizamento,
        decision_metadata=decision_metadata,
        model_label="TCC Control - referencia historica",
        method_line="- Universo base congelado; nenhuma substituicao.",
    )
    baseline_metrics = pesquisa._scenario_metrics(
        baseline_result, folds, float(config.initial_capital)
    )
    # O checkpoint de origem foi obtido com outro universo efetivo. Ele NAO
    # pode ser usado como assert nem como meta financeira nesta nova execucao.
    print(
        f"[base] ativos={len(baseline_symbols)} "
        f"capital=US$ {baseline_metrics['ending_capital']:,.2f} "
        "checkpoint=pendente_77_ativos",
        flush=True,
    )

    frames, continuity, verified_hashes = pesquisa._prepare_candidate_frames(
        list(selected), common_dates, all_decision_dates, config
    )
    missing = [symbol for symbol in selected if symbol not in frames]
    if missing:
        raise RuntimeError(
            "Candidatos incompativeis com as series congeladas: "
            + ", ".join(missing)
        )
    if verified_hashes != file_hashes:
        raise RuntimeError("Dados dos candidatos foram alterados durante a leitura.")

    caches = {}
    for position, symbol in enumerate(selected, start=1):
        caches[symbol] = pesquisa._train_candidate_caches(
            symbol, frames[symbol], states, config
        )
        print(f"[model] {position}/{len(selected)} {symbol} pronto", flush=True)

    result, metrics, calibration_rows = pesquisa._evaluate_scenario(
        "TCC_SELECTED_UNIVERSE",
        list(selected),
        baseline_frames,
        frames,
        baseline_symbols,
        states,
        caches,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
        config,
    )
    capital = float(metrics["ending_capital"])
    effective = [*baseline_symbols, *selected]
    expected_universe = [*requested, *selected]
    if len(expected_universe) != 77 or len(set(expected_universe)) != 77:
        raise RuntimeError("Contrato Winner exige exatamente 77 ativos distintos.")
    if effective != expected_universe:
        raise RuntimeError(
            "Os 77 ativos nao participaram integralmente do replay. "
            f"esperados={len(expected_universe)} efetivos={len(effective)} "
            f"ausentes={sorted(set(expected_universe) - set(effective))}"
        )

    # Sem um primeiro checkpoint validado com estes 77 ativos, nao se pode
    # certificar reproducao monetaria exata contra o resultado anterior.
    exact = None
    expected_capital = None

    # So apos replay concluido, sobrescrever o MESMO diretorio e ZIP.
    # Nao criar variantes ou pastas versionadas entre execucoes de teste.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"asset": [*requested, *selected]}).to_csv(
        output / "requested_assets.csv", index=False
    )
    pd.DataFrame({"asset": effective}).to_csv(
        output / "effective_assets.csv", index=False
    )
    pd.DataFrame(exclusions).to_csv(output / "excluded_assets.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output / "fold_calibration.csv", index=False
    )
    pd.DataFrame(baseline_calibration).to_csv(
        output / "baseline_fold_calibration.csv", index=False
    )
    continuity.to_csv(output / "candidate_continuity.csv", index=False)
    result.predictions.to_csv(
        output / "predictions.csv", index=True, float_format="%.17g"
    )
    if isinstance(getattr(result, "trades", None), pd.DataFrame):
        result.trades.to_csv(output / "trades.csv", index=False)

    payload = {
        "reproduction_version": version,
        "execution_schema": execution_schema,
        "status": "completed",
        "purpose": "reproducao_universo_selecionado",
        "reproduced_exact_to_cent": exact,
        "checkpoint_validation": "awaiting_first_77_asset_baseline",
        "analysis_end_date": contract["analysis_end_date"],
        "checkpoint": {
            "observed_capital": capital,
            "expected_capital": expected_capital,
            "delta": None,
            "status": "not_comparable_to_previous_universe",
        },
        "data": {
            "source": "alpaca",
            "feed": "sip",
            "adjustment": "raw",
            "timeframe": "1Day",
            "full_refresh": False,
            "database_used": False,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "candidate_file_hashes": verified_hashes,
            "candidate_cache_committed_to_git": False,
            "data_audit": data_audit,
            "diagnostics": diagnostics,
        },
        "universe": {
            "requested_assets": list(requested) + list(selected),
            "requested_count": len(requested) + len(selected),
            "effective_assets": effective,
            "effective_count": len(effective),
            "excluded_assets": exclusions,
        },
        "calendar": {
            "calendar_source": calendar_source,
            "decision_sessions": len(all_decision_dates) - 1,
        },
        "metrics": metrics,
        "baseline_reference": baseline_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    pesquisa._write_json(output / "reproducao_universo.json", payload)
    gerar_analises_backtest(
        output, manifest=snapshot, result=result
    )
    package = criar_pacote_analise(
        output,
        comparison_file="reproducao_universo.json",
        execution_schema=execution_schema,
        archive_name="pacote_reproducao_universo.zip",
    )
    print(
        f"[done] capital=US$ {capital:,.2f} effective_assets={len(effective)} "
        f"checkpoint=pending "
        f"package={package}",
        flush=True,
    )
    sinal_sonoro_conclusao()
    return payload
