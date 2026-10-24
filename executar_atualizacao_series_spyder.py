"""v1.23.7 — Control em um unico universo fixo de 75 ativos.

No Spyder: F5. Usa as mesmas fontes Alpaca SIP/RAW, sem DOC ou CLMT.
Se existir um snapshot legado de 77 integro, extrai apenas os 75 CSVs
e cria um NOVO manifesto U75, sem baixar novamente nem modificar a fonte.
Os 75 ativos sao treinados e calibrados juntos, nao como 65+10.
O snapshot congelado e o replay cientifico v1.23.0 permanecem intocados.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import zipfile
from zoneinfo import ZoneInfo

import pandas as pd

from engine.configuracao import U67_EXPECTED_EXCLUSIONS, U67_REQUESTED_ASSETS
from reproducao.dados import (
    SnapshotPaths, build_snapshot_manifest, download_corporate_actions,
    download_raw_bars, load_alpaca_credentials, validate_snapshot,
)
from reproducao.universo_fixo75 import (
    reproduzir_universo_fixo75, universo_75,
)

ROOT = Path(__file__).resolve().parent
TARGET_CUTOFF = "2026-10-22"
EVALUATION_CUTOFFS = ("2026-10-06", "2026-10-21")
EXPERIMENT_VERSION = "1.23.7"
EXECUTION_SCHEMA = "single-fixed-75-v1"
EXPERIMENT_DATA_ROOT = ROOT / "dados" / ".u75_fresh_runs"
LEGACY_77_DATA_ROOT = ROOT / "dados" / ".u67_fresh_runs"
EXPERIMENT_OUTPUT_ROOT = ROOT / "output" / "fresh_universe_75"
REUSE_LAST_COMPLETE_SNAPSHOT = True


def assert_completed_session(cutoff: str, *, now: datetime | None = None) -> None:
    target = pd.Timestamp(cutoff)
    if target.tzinfo is not None or target.strftime("%Y-%m-%d") != cutoff:
        raise ValueError("O cutoff deve estar no formato YYYY-MM-DD.")
    eastern = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(
        ZoneInfo("America/New_York")
    )
    if target.date() > eastern.date():
        raise ValueError("Cutoff no futuro.")
    if target.date() == eastern.date() and (eastern.hour, eastern.minute) < (17, 0):
        raise ValueError("Aguarde consolidacao de barras diarias (17:00 ET).")


def verify_reference_session(paths: SnapshotPaths, cutoff: str) -> None:
    bars = pd.read_csv(paths.raw_bars / "AAPL.csv", usecols=["timestamp"])
    last = pd.to_datetime(bars["timestamp"], utc=True).max().date().isoformat()
    if last != cutoff:
        raise RuntimeError(
            f"AAPL nao tem barra do ultimo pregao completo: {cutoff}, ultima={last}."
        )


def validate_cutoffs(snapshot_end: str, evaluation_cutoffs: tuple[str, ...]) -> None:
    if evaluation_cutoffs != ("2026-10-06", "2026-10-21"):
        raise ValueError("Os cortes fixos desta campanha foram alterados.")
    if any(d > snapshot_end for d in evaluation_cutoffs):
        raise ValueError("A data de avaliacao ultrapassa o snapshot.")


def check_snapshot(
    paths: SnapshotPaths, cutoff: str, assets: tuple[str, ...]
) -> dict:
    manifest = validate_snapshot(paths)
    if tuple(manifest.get("assets") or ()) != assets or len(assets) != 75:
        raise RuntimeError("Snapshot precisa conter exatamente os 75 ativos vencedores.")
    if manifest["bars"]["bar_snapshot_as_of_end"] != cutoff:
        raise RuntimeError("Snapshot reutilizado possui cutoff divergente.")
    verify_reference_session(paths, cutoff)
    return manifest


def find_reusable_snapshot(
    cutoff: str, assets: tuple[str, ...]
) -> tuple[Path, SnapshotPaths, dict] | None:
    for run_root in sorted(
        EXPERIMENT_DATA_ROOT.glob(f"asof_{cutoff}_*"), reverse=True
    ):
        if not run_root.is_dir():
            continue
        paths = SnapshotPaths.from_root(run_root / "universe_75")
        if not paths.manifest.is_file():
            continue
        manifest = check_snapshot(paths, cutoff, assets)
        print(f"[resume75] snapshot SHA-256 validado: {paths.root}", flush=True)
        return run_root, paths, manifest
    return None


def reuse_legacy_snapshot_as_75(
    cutoff: str, assets: tuple[str, ...], output_dir: Path
) -> tuple[SnapshotPaths, dict, str] | None:
    """Migra uma fotografia legada em modo leitura, filtrando DOC e CLMT."""
    original = tuple(U67_REQUESTED_ASSETS)
    reference = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    historical77 = (*original, *tuple(reference["selected_assets"]))
    if tuple(x for x in historical77 if x not in U67_EXPECTED_EXCLUSIONS) != assets:
        raise RuntimeError("A composicao dos 75 ativos divergiu da referencia.")
    for candidate in sorted(
        LEGACY_77_DATA_ROOT.glob(f"asof_{cutoff}_*/universe_77"), reverse=True
    ):
        legacy = SnapshotPaths.from_root(candidate)
        if not legacy.manifest.is_file():
            continue
        old = validate_snapshot(legacy)
        if tuple(old.get("assets") or ()) != historical77:
            raise RuntimeError("Snapshot legado nao contem os 77 simbolos originais.")
        if old["bars"]["bar_snapshot_as_of_end"] != cutoff:
            raise RuntimeError("Data divergente no snapshot legado.")
        verify_reference_session(legacy, cutoff)
        target = SnapshotPaths.from_root(output_dir / "universe_75")
        if target.root.exists():
            raise FileExistsError(target.root)
        target.ensure()
        bars = {}
        actions = {}
        for symbol in assets:
            dest_raw = target.raw_bars / f"{symbol}.csv"
            dest_actions = target.corporate_actions / f"{symbol}.csv"
            shutil.copy2(legacy.raw_bars / f"{symbol}.csv", dest_raw)
            shutil.copy2(
                legacy.corporate_actions / f"{symbol}.csv", dest_actions
            )
            bars[symbol] = dest_raw
            actions[symbol] = dest_actions
        new = build_snapshot_manifest(
            target, bars, actions, bar_snapshot_as_of_end=cutoff,
            analysis_end_date=cutoff, assets=assets,
            snapshot_name=f"fixed75-filtered-from-{old['snapshot_sha256'][:12]}",
        )
        check_snapshot(target, cutoff, assets)
        print(
            f"[migrate75] original77={old['snapshot_sha256']} "
            f"novo75={new['snapshot_sha256']}; nenhuma requisicao Alpaca",
            flush=True,
        )
        return target, new, old["snapshot_sha256"]
    return None


def main(*, cutoff: str = TARGET_CUTOFF) -> dict:
    assert_completed_session(cutoff)
    validate_cutoffs(cutoff, EVALUATION_CUTOFFS)
    historical = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(historical, ())
    reusable = (
        find_reusable_snapshot(cutoff, assets)
        if REUSE_LAST_COMPLETE_SNAPSHOT else None
    )
    provenance = None
    if reusable is not None:
        run_root, source, manifest = reusable
    else:
        run_id = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%S%fZ")
        run_root = EXPERIMENT_DATA_ROOT / f"asof_{cutoff}_{run_id}"
        if run_root.exists():
            raise FileExistsError(run_root)
        migrated = (
            reuse_legacy_snapshot_as_75(cutoff, assets, run_root)
            if REUSE_LAST_COMPLETE_SNAPSHOT else None
        )
        if migrated is not None:
            source, manifest, provenance = migrated
        else:
            source = SnapshotPaths.from_root(run_root / "universe_75")
            credentials = load_alpaca_credentials(ROOT)
            print(f"[fixed75] download RAW/SIP unicamente {len(assets)} ativos", flush=True)
            bars = download_raw_bars(
                credentials, source, assets=assets,
                bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
                sip_delay_guard_minutes=20,
            )
            actions = download_corporate_actions(
                credentials, source, assets=assets, query_end=cutoff,
            )
            manifest = build_snapshot_manifest(
                source, bars, actions, bar_snapshot_as_of_end=cutoff,
                analysis_end_date=cutoff, assets=assets,
                snapshot_name=f"single-fixed-75-{cutoff}",
            )
            check_snapshot(source, cutoff, assets)

    output_root = EXPERIMENT_OUTPUT_ROOT / run_root.name
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[fixed75] versao={EXPERIMENT_VERSION} 75 ativos, "
        f"snapshot={manifest['snapshot_sha256']} cortes={EVALUATION_CUTOFFS}",
        flush=True,
    )
    results = {}
    for evaluation_date in EVALUATION_CUTOFFS:
        result = reproduzir_universo_fixo75(
            root=ROOT, output=output_root / f"cutoff_{evaluation_date}",
            snapshot_root=source.root, analysis_end_date=evaluation_date,
            version=EXPERIMENT_VERSION, execution_schema=EXECUTION_SCHEMA,
        )
        if (
            result["universe"]["effective_count"] != 75
            or result["universe"]["requested_count"] != 75
        ):
            raise RuntimeError("O replay nao empregou exclusivamente 75 ativos.")
        results[evaluation_date] = result

    summary = {
        "version": EXPERIMENT_VERSION,
        "execution_schema": EXECUTION_SCHEMA,
        "single_universe": True,
        "trained_together": True,
        "same_snapshot_both_cutoffs": True,
        "snapshot_sha256": manifest["snapshot_sha256"],
        "snapshot_origin_legacy77_sha256": provenance,
        "snapshot_data_end_date": cutoff,
        "requested_assets": list(assets),
        "requested_count": 75,
        "effective_asset_count": 75,
        "disallowed_assets": sorted(U67_EXPECTED_EXCLUSIONS),
        "official_historical_capital": historical["expected_ending_capital"],
        "official_historical_end_date": historical["analysis_end_date"],
        "scientific_note": (
            "Treino conjunto 75 nao deve ser tratado como mesma implementacao "
            "do replay legado 65+10; comparar resultados, nao assumir paridade."
        ),
        "cutoffs": {
            date: {
                "ending_capital": result["checkpoint"]["observed_capital"],
                "folds": result["metrics"]["folds"],
                "decision_sessions": result["calendar"]["decision_sessions"],
            }
            for date, result in results.items()
        },
    }
    compare_path = output_root / "comparacao_cutoffs_75.json"
    compare_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    archive = output_root / "pacote_comparacao_cutoffs_75.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as dest:
        dest.write(compare_path, arcname=compare_path.name)
        dest.write(source.manifest, arcname="snapshot_75_manifest.json")
        for file in sorted(output_root.rglob("*")):
            if file.is_file() and file not in (archive, compare_path):
                dest.write(file, arcname=file.relative_to(output_root).as_posix())
    print(f"[fixed75] pacote final: {archive}", flush=True)
    return summary


if __name__ == "__main__":
    main()
