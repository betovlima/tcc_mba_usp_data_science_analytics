"""Atualizacao EXPERIMENTAL do universo fixo TCC v1.23.0 (Spyder: F5).

Uma nova fotografia RAW/SIP da Alpaca e criada sob dados/.u67_fresh_runs/.
A reproducao congelada de US$ 587 milhoes em dados/u67 e evidencias/ NAO
e reescrita. Nenhuma ordem e enviada a Alpaca e nenhum banco e utilizado.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from engine.configuracao import U67_REQUESTED_ASSETS
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.experimento import reproduzir_universo_congelado

ROOT = Path(__file__).resolve().parent
TARGET_CUTOFF = "2026-10-22"  # Ultimo pregao fechado a avaliar (XNYS).
EXPERIMENT_VERSION = "1.23.1"
EXECUTION_SCHEMA = "fresh-fixed-universe-v1"
EXPERIMENT_DATA_ROOT = ROOT / "dados" / ".u67_fresh_runs"
EXPERIMENT_OUTPUT_ROOT = ROOT / "output" / "fresh_universe"


def assert_completed_session(cutoff: str, *, now: datetime | None = None) -> None:
    """Nao permita dados da sessao ainda aberta ou uma data futura."""
    target = pd.Timestamp(cutoff)
    if target.tzinfo is not None or target.strftime("%Y-%m-%d") != cutoff:
        raise ValueError("TARGET_CUTOFF precisa ter formato YYYY-MM-DD.")
    eastern = now or datetime.now(ZoneInfo("America/New_York"))
    eastern = eastern.astimezone(ZoneInfo("America/New_York"))
    if target.date() > eastern.date():
        raise ValueError("Cutoff no futuro: dados RAW/SIP nao estao fechados.")
    if target.date() == eastern.date() and (eastern.hour, eastern.minute) < (17, 0):
        raise ValueError(
            "Aguarde o fechamento e a consolidacao da barra diaria em Nova York (17:00 ET)."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_session(paths: SnapshotPaths, cutoff: str) -> None:
    """AAPL precisa ter uma barra diaria exatamente no ultimo pregao solicitado."""
    path = paths.raw_bars / "AAPL.csv"
    bars = pd.read_csv(path, usecols=["timestamp"])
    observed = pd.to_datetime(bars["timestamp"], utc=True).max().date().isoformat()
    if observed != cutoff:
        raise RuntimeError(
            f"AAPL nao tem barra fechada para {cutoff} (ultima={observed}). "
            "Snapshot recusado; nao usar data parcial ou um cutoff ficticio."
        )


def main(*, cutoff: str = TARGET_CUTOFF) -> dict:
    assert_completed_session(cutoff)
    historical = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(encoding="utf-8")
    )
    selected = tuple(historical["selected_assets"])
    if len(selected) != 10 or len(U67_REQUESTED_ASSETS) != 67:
        raise RuntimeError("Universo historico nao corresponde a 67 + 10 ativos.")
    if len(set((*U67_REQUESTED_ASSETS, *selected))) != 77:
        raise RuntimeError("Universo fixo U67+10 contem simbolos repetidos.")

    # Uma pasta diferente em cada execucao: jamais apagar dados oficiais ou anteriores.
    run_id = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    run_root = EXPERIMENT_DATA_ROOT / f"asof_{cutoff}_{run_id}"
    if run_root.exists():
        raise FileExistsError(run_root)
    base = SnapshotPaths.from_root(run_root / "base")
    candidates = SnapshotPaths.from_root(run_root / "candidates")
    output = EXPERIMENT_OUTPUT_ROOT / f"asof_{cutoff}_{run_id}"
    credentials = load_alpaca_credentials(ROOT)

    print(f"[fresh] cutoff={cutoff} base=67 adicionais=10 output={output}", flush=True)
    print("[fresh] fonte: Alpaca RAW/SIP 1Day; sem Mongo e sem ordens", flush=True)
    print("[fresh] NAO modifica dados/u67 nem evidencias/", flush=True)

    base_bars = download_raw_bars(
        credentials, base, assets=tuple(U67_REQUESTED_ASSETS),
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
    )
    base_actions = download_corporate_actions(
        credentials, base, assets=tuple(U67_REQUESTED_ASSETS), query_end=cutoff,
    )
    base_manifest = build_snapshot_manifest(
        base, base_bars, base_actions,
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
        assets=tuple(U67_REQUESTED_ASSETS), snapshot_name=f"fresh-u67-{cutoff}",
    )
    validated_base = validate_snapshot(base)
    verify_reference_session(base, cutoff)
    if validated_base["snapshot_sha256"] != base_manifest["snapshot_sha256"]:
        raise RuntimeError("Hash do snapshot base alterado durante a coleta.")

    candidate_bars = download_raw_bars(
        credentials, candidates, assets=selected,
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
    )
    candidate_actions = download_corporate_actions(
        credentials, candidates, assets=selected, query_end=cutoff,
    )
    build_snapshot_manifest(
        candidates, candidate_bars, candidate_actions,
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
        assets=selected, snapshot_name=f"fresh-selected10-{cutoff}",
    )
    validate_snapshot(candidates)
    file_hashes = {
        symbol: {
            "raw_sha256": sha256_file(candidate_bars[symbol]),
            "corporate_actions_sha256": sha256_file(candidate_actions[symbol]),
        }
        for symbol in selected
    }
    contract = {
        "schema_version": 1,
        "scope": "fresh_fixed_universe_experimental",
        "analysis_end_date": cutoff,
        "snapshot_sha256": validated_base["snapshot_sha256"],
        "selected_assets": list(selected),
        "candidate_file_hashes": file_hashes,
        "baseline_ending_capital": None,
        "expected_ending_capital": None,
        "historical_reference_capital": historical["expected_ending_capital"],
        "historical_reference_end_date": historical["analysis_end_date"],
        "full_refresh": True,
    }
    contract_path = run_root / "experimental_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    result = reproduzir_universo_congelado(
        root=ROOT,
        output=output,
        version=EXPERIMENT_VERSION,
        execution_schema=EXECUTION_SCHEMA,
        experimental={
            "manifest_path": str(contract_path),
            "base_root": str(base.root),
            "candidate_root": str(candidates.root),
        },
    )
    print(
        f"[fresh] concluido capital=US$ {result['checkpoint']['observed_capital']:,.2f}; "
        f"zip={output / 'pacote_atualizacao_series.zip'}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
