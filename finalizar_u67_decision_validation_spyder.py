"""Finaliza uma v1.4.0 concluida que falhou apenas na serializacao JSON.

Use este arquivo somente quando pesquisar_expansao_u67_spyder.py ja terminou
todos os cenarios da decision validation e falhou em
pesquisa_u67_decision_validation.json por conter NaN.

O script nao treina modelos, nao repete backtests e nao apaga output.
Ele valida os artefatos parciais existentes, recompõe o JSON final com null
para valores nao finitos e cria o pacote de analise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from pesquisar_expansao_u67_spyder import (
    ANALYSIS_END_DATE,
    EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
    EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL,
    FULL_DATA,
    SOURCE_SUBSET_UNIQUE_SCENARIOS,
    SUBSET_SOURCE_CANDIDATES,
    DISCOVERY_SELECTED_CANDIDATES,
    _json_safe,
    _sha256_file,
    _write_json,
)
from reproducao.artefatos import criar_pacote_analise, sinal_sonoro_conclusao


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "u67_decision_validation"
RECOVERY_VERSION = "u67-asset-expansion-v1.4.1"
RECOVERED_EXECUTION_VERSION = "u67-asset-expansion-v1.4.0"
EXECUTION_SCHEMA = "u67-asset-expansion-v5"
BASELINE_ABS_TOL = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    return _json_safe(frame.to_dict(orient="records"))


def _capital_for_assets(frame: pd.DataFrame, assets: list[str]) -> float:
    expected = set(assets)
    for row in frame.itertuples(index=False):
        observed = {
            item.strip()
            for item in str(row.assets).split(",")
            if item.strip()
        }
        if observed == expected:
            return float(row.ending_capital)
    raise RuntimeError(
        "Cenario esperado nao foi encontrado em multiple_testing_scenarios.csv: "
        + ",".join(assets)
    )


def _candidate_hashes() -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for symbol in SUBSET_SOURCE_CANDIDATES:
        raw_path = FULL_DATA.raw_bars / f"{symbol}.csv"
        action_path = FULL_DATA.corporate_actions / f"{symbol}.csv"
        if not raw_path.exists() or not action_path.exists():
            raise RuntimeError(
                f"Cache local incompleto para {symbol}; recovery nao pode "
                "preservar os hashes da execucao."
            )
        hashes[symbol] = {
            "raw_sha256": _sha256_file(raw_path),
            "corporate_actions_sha256": _sha256_file(action_path),
        }
    return hashes


def main() -> dict[str, Any]:
    required = [
        "u67_baseline_metrics.json",
        "candidate_download.csv",
        "continuity_audit.csv",
        "full_window_subset_search.csv",
        "multiple_testing_scenarios.csv",
        "multiple_testing_daily_returns.csv",
        "multiple_testing_summary.json",
        "pbo_log_growth.csv",
        "pbo_sharpe.csv",
        "decision_rank_sessions.csv",
        "decision_rank_summary.csv",
        "dynamic_universe_temporal_validation.csv",
        "dynamic_universe_transitions.csv",
        "dynamic_universe_search.csv",
        "scenario_fold_calibration.csv",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise RuntimeError(
            "Recovery exige a execucao v1.4.0 ja concluida. Faltam: "
            + ", ".join(missing)
        )

    baseline = _read_json(OUT / "u67_baseline_metrics.json")
    multiple_testing = _read_json(OUT / "multiple_testing_summary.json")
    scenarios = pd.read_csv(OUT / "multiple_testing_scenarios.csv")
    if len(scenarios) != SOURCE_SUBSET_UNIQUE_SCENARIOS:
        raise RuntimeError(
            "Quantidade de cenarios divergente no recovery: "
            f"{len(scenarios)} != {SOURCE_SUBSET_UNIQUE_SCENARIOS}"
        )

    all15_capital = _capital_for_assets(
        scenarios,
        list(SUBSET_SOURCE_CANDIDATES),
    )
    selected10_capital = _capital_for_assets(
        scenarios,
        list(DISCOVERY_SELECTED_CANDIDATES),
    )
    if not math.isclose(
        all15_capital,
        EXPECTED_ALL_POSITIVE_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    ):
        raise RuntimeError("Checkpoint U67+15 divergiu no recovery.")
    if not math.isclose(
        selected10_capital,
        EXPECTED_DISCOVERY_SELECTED_ENDING_CAPITAL,
        rel_tol=0.0,
        abs_tol=BASELINE_ABS_TOL,
    ):
        raise RuntimeError("Checkpoint U67+10 divergiu no recovery.")

    rank_summary = pd.read_csv(OUT / "decision_rank_summary.csv")
    rank_overall = _json_safe(
        rank_summary.loc[
            rank_summary["fold_id"].astype(str).eq("ALL")
        ].to_dict(orient="records")
    )
    temporal_rows = _records(
        OUT / "dynamic_universe_temporal_validation.csv"
    )
    transition_frame = pd.read_csv(
        OUT / "dynamic_universe_transitions.csv"
    )
    transition_actions = _json_safe(
        transition_frame.groupby(
            ["stage", "action"],
            sort=False,
        ).size().reset_index(name="count").to_dict(orient="records")
    )

    payload = {
        "execution_schema": EXECUTION_SCHEMA,
        "version": RECOVERED_EXECUTION_VERSION,
        "status": "completed",
        "purpose": "u67_decision_and_selection_validation",
        "run_mode": "decision_validation",
        "recovery": {
            "applied": True,
            "recovery_version": RECOVERY_VERSION,
            "reason": (
                "A v1.4.0 concluiu os calculos mas falhou ao serializar "
                "NaN no JSON final. Nenhum modelo/backtest foi repetido."
            ),
            "analysis_end_date": ANALYSIS_END_DATE,
        },
        "baseline": baseline,
        "checkpoints": {
            "u67_exact_to_cent": bool(baseline.get("exact_to_cent")),
            "all_15_exact_to_cent": True,
            "selected_10_exact_to_cent": True,
            "all_15_ending_capital": all15_capital,
            "selected_10_ending_capital": selected10_capital,
            "full_window_subset_scenarios_reproduced": int(len(scenarios)),
        },
        "decision_ranking": {
            "overall": rank_overall,
            "interpretation": (
                "Ranking diagnostics compare predictions available at the "
                "decision session with forward utility realized afterward. "
                "They are evaluation diagnostics and are not used to make "
                "historical decisions."
            ),
        },
        "multiple_testing": multiple_testing,
        "dynamic_universe": {
            "asset_list_is_permanent": False,
            "stages": temporal_rows,
            "transition_actions": transition_actions,
            "current_scope": (
                "Fold-boundary pilot over the fixed 15-candidate discovery "
                "pool. It does not yet rediscover the external universe daily."
            ),
        },
        "data": {
            "baseline_snapshot_sha256": baseline.get("snapshot_sha256"),
            "candidate_file_hashes": _candidate_hashes(),
        },
        "runtime_seconds": None,
    }

    final_path = OUT / "pesquisa_u67_decision_validation.json"
    _write_json(final_path, payload)
    package = criar_pacote_analise(
        OUT,
        comparison_file=final_path.name,
        execution_schema=EXECUTION_SCHEMA,
        archive_name="pacote_analise_u67_decision_validation.zip",
    )
    print("=" * 78, flush=True)
    print("[recovery] calculos existentes validados; nenhum backtest repetido", flush=True)
    print(f"[recovery] json={final_path}", flush=True)
    print(f"[recovery] package={package}", flush=True)
    print("=" * 78, flush=True)
    sinal_sonoro_conclusao()
    return payload


if __name__ == "__main__":
    main()
