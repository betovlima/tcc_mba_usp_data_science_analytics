"""V1.23.2 — replay Control com o universo vencedor inteiro e novos dados Alpaca.

Execucao unica no Spyder (F5): coleta os 77 simbolos fixos UMA VEZ e roda
os cortes de 06 e 21 de outubro sobre os MESMOS arquivos atualizados.
O motor matematico, parametros e universo do TCC v1.23.0 permanecem iguais.
Sem busca de ativos, sem alteracao do snapshot oficial, sem ordens ou Mongo.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile
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
TARGET_CUTOFF = "2026-10-22"  # Data ate a qual as series sao atualizadas.
EVALUATION_CUTOFFS = ("2026-10-06", "2026-10-21")
EXPERIMENT_VERSION = "1.23.2"
EXECUTION_SCHEMA = "fresh-fixed-universe-v2"
EXPERIMENT_DATA_ROOT = ROOT / "dados" / ".u67_fresh_runs"
EXPERIMENT_OUTPUT_ROOT = ROOT / "output" / "fresh_universe"


def assert_completed_session(cutoff: str, *, now: datetime | None = None) -> None:
    """Nao utilizar um dia futuro nem a barra da sessao atual antes das 17h ET."""
    target = pd.Timestamp(cutoff)
    if target.tzinfo is not None or target.strftime("%Y-%m-%d") != cutoff:
        raise ValueError("A data deve ter formato YYYY-MM-DD.")
    eastern = now or datetime.now(ZoneInfo("America/New_York"))
    eastern = eastern.astimezone(ZoneInfo("America/New_York"))
    if target.date() > eastern.date():
        raise ValueError("Cutoff no futuro: dados RAW/SIP nao estao fechados.")
    if target.date() == eastern.date() and (eastern.hour, eastern.minute) < (17, 0):
        raise ValueError("Aguarde a consolidacao da barra diaria (17:00 ET).")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_session(paths: SnapshotPaths, cutoff: str) -> None:
    """Exige barra completa do pregao de referencia, sem aproximar datas."""
    bars = pd.read_csv(paths.raw_bars / "AAPL.csv", usecols=["timestamp"])
    observed = pd.to_datetime(bars["timestamp"], utc=True).max().date().isoformat()
    if observed != cutoff:
        raise RuntimeError(
            f"AAPL nao tem barra fechada para {cutoff} (ultima={observed}). "
            "Refaca a coleta com o ultimo pregao completo."
        )


def validate_cutoffs(snapshot_end: str, evaluation_cutoffs: tuple[str, ...]) -> None:
    if len(evaluation_cutoffs) != 2 or evaluation_cutoffs != (
        "2026-10-06", "2026-10-21"
    ):
        raise ValueError(
            "Esta campanha exige os dois cortes fixos 2026-10-06 e 2026-10-21."
        )
    if any(cutoff > snapshot_end for cutoff in evaluation_cutoffs):
        raise ValueError("Um dos cortes de avaliacao ultrapassa o snapshot RAW.")


def main(*, cutoff: str = TARGET_CUTOFF) -> dict:
    assert_completed_session(cutoff)
    validate_cutoffs(cutoff, EVALUATION_CUTOFFS)
    historical = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    selected = tuple(historical["selected_assets"])
    full_universe = (*U67_REQUESTED_ASSETS, *selected)
    if len(U67_REQUESTED_ASSETS) != 67 or len(selected) != 10:
        raise RuntimeError("O universo historico precisa conter 67+10 simbolos.")
    if len(full_universe) != 77 or len(set(full_universe)) != 77:
        raise RuntimeError("Os 77 ativos fixos precisam ser unicos.")

    # A coleta e indivisivel: 77 simbolos no mesmo snapshot, sem descoberta.
    run_id = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = EXPERIMENT_DATA_ROOT / f"asof_{cutoff}_{run_id}"
    output_root = EXPERIMENT_OUTPUT_ROOT / f"asof_{cutoff}_{run_id}"
    if run_root.exists() or output_root.exists():
        raise FileExistsError("O identificador da execucao ja existe.")
    source = SnapshotPaths.from_root(run_root / "universe_77")
    credentials = load_alpaca_credentials(ROOT)

    print(
        f"[fresh] versao={EXPERIMENT_VERSION} cutoff_coleta={cutoff} "
        f"universo_fixo={len(full_universe)} cortes={EVALUATION_CUTOFFS}",
        flush=True,
    )
    print("[fresh] coleta RAW/SIP/1Day; sem atualizacao dos arquivos oficiais", flush=True)
    bars = download_raw_bars(
        credentials, source, assets=full_universe,
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
    )
    actions = download_corporate_actions(
        credentials, source, assets=full_universe, query_end=cutoff,
    )
    manifest = build_snapshot_manifest(
        source, bars, actions, bar_snapshot_as_of_end=cutoff,
        analysis_end_date=cutoff, assets=full_universe,
        snapshot_name=f"fixed-77-fresh-{cutoff}",
    )
    validated = validate_snapshot(source)
    verify_reference_session(source, cutoff)
    if tuple(validated["assets"]) != full_universe:
        raise RuntimeError("O snapshot nao contem exatamente os 77 simbolos fixos.")
    if validated["snapshot_sha256"] != manifest["snapshot_sha256"]:
        raise RuntimeError("Hash da coleta alterado durante a validacao.")

    candidate_hashes = {
        symbol: {
            "raw_sha256": sha256_file(bars[symbol]),
            "corporate_actions_sha256": sha256_file(actions[symbol]),
        }
        for symbol in selected
    }
    results: dict[str, dict] = {}
    output_root.mkdir(parents=True)

    # Cada replay utiliza o mesmo snapshot e os mesmos modelos/regras do Control.
    for evaluation_date in EVALUATION_CUTOFFS:
        contract = {
            "schema_version": 2,
            "scope": "fixed_77_updated_snapshot",
            "analysis_end_date": evaluation_date,
            "snapshot_sha256": validated["snapshot_sha256"],
            "snapshot_data_end_date": cutoff,
            "selected_assets": list(selected),
            "requested_assets": list(full_universe),
            "candidate_file_hashes": candidate_hashes,
            "baseline_ending_capital": None,
            "expected_ending_capital": None,
            "historical_reference_capital": historical["expected_ending_capital"],
            "historical_reference_end_date": historical["analysis_end_date"],
            "full_refresh": True,
        }
        contract_path = run_root / f"contract_{evaluation_date}.json"
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        result = reproduzir_universo_congelado(
            root=ROOT,
            output=output_root / f"cutoff_{evaluation_date}",
            version=EXPERIMENT_VERSION,
            execution_schema=EXECUTION_SCHEMA,
            experimental={
                "manifest_path": str(contract_path),
                "snapshot_root": str(source.root),
            },
        )
        if result["universe"]["effective_count"] != 75:
            raise RuntimeError("O experimento nao utilizou os 75 ativos efetivos.")
        results[evaluation_date] = result
        print(
            f"[fresh] corte={evaluation_date} "
            f"capital=US$ {result['checkpoint']['observed_capital']:,.2f}",
            flush=True,
        )

    summary = {
        "version": EXPERIMENT_VERSION,
        "same_snapshot_both_cutoffs": True,
        "snapshot_sha256": validated["snapshot_sha256"],
        "snapshot_data_end_date": cutoff,
        "requested_assets": list(full_universe),
        "effective_asset_count": 75,
        "official_historical_capital": historical["expected_ending_capital"],
        "official_historical_end_date": historical["analysis_end_date"],
        "cutoffs": {
            date: {
                "ending_capital": result["checkpoint"]["observed_capital"],
                "folds": result["metrics"]["folds"],
                "decision_sessions": result["calendar"]["decision_sessions"],
                "excluded_assets": result["universe"]["excluded_assets"],
            }
            for date, result in results.items()
        },
    }
    compare_path = output_root / "comparacao_cutoffs.json"
    compare_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Um ZIP compartilhavel inclui os dois replays e o manifesto, nao os RAW.
    zip_path = output_root / "pacote_comparacao_cutoffs.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(compare_path, arcname="comparacao_cutoffs.json")
        archive.write(source.manifest, arcname="snapshot_77_manifest.json")
        for contract_path in sorted(run_root.glob("contract_*.json")):
            archive.write(contract_path, arcname=f"contratos/{contract_path.name}")
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != zip_path and path != compare_path:
                archive.write(path, arcname=str(path.relative_to(output_root)))
    print(f"[fresh] comparacao gerada: {zip_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
