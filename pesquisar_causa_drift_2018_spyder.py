"""Teste de sensibilidade: 73 barras de 31 ativos efetivos, sem alterar fontes.

Dois replays CONTROL ate 06/10/2026, mesmo cutoff e mesmas Corporate Actions:
1. snapshot Alpaca atual sem alteracoes;
2. COPIA ISOLADA desse snapshot com as 73 observacoes divergentes de maio/2018
   restauradas exclusivamente dos CSVs historicos com hash verificado.

Nao e correcao nem substituicao dos dados oficiais: e intervencao experimental
rastreavel. Nao baixa dados, nao transmite ordens, nao acessa banco.
No Spyder: executar com F5.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from auditar_drift_historico_spyder import (
    ANALYSIS_CUTOFF, FRESH_CUTOFF, BARS,
    compare_bar_frames, file_sha256, original_extra_paths,
)
from engine.configuracao import U67_EXPECTED_EXCLUSIONS, U67_REQUESTED_ASSETS
from executar_atualizacao_series_spyder import (
    find_reusable_snapshot, reuse_legacy_snapshot_as_75,
)
from reproducao.dados import (
    SnapshotPaths, build_snapshot_manifest, validate_snapshot,
)
from reproducao.universo_fixo75 import reproduzir_universo_fixo75, universo_75
from reproducao.preparacao import load_raw_bar_file

ROOT = Path(__file__).resolve().parent
EXPERIMENT_VERSION = "1.23.7"
EXECUTION_SCHEMA = "causal-2018-single-75-v2"
COMPARE_CUTOFF = ANALYSIS_CUTOFF  # 06/10/2026: mesma data do checkpoint.
SNAPSHOT_CUTOFF = FRESH_CUTOFF    # 22/10/2026: mesma fotografia Alpaca.
EXPECTED_CHANGED_ROWS = 73
EXPECTED_CHANGED_ASSETS = 31
ALLOWED_DATES = frozenset({"2018-05-01", "2018-05-02", "2018-05-03"})
WORK_ROOT = ROOT / "dados" / ".u67_fresh_runs"
OUTPUT_ROOT = ROOT / "output" / "causal_drift_2018"


def replace_verified_2018_rows(
    original_raw: Path,
    updated_raw: Path,
    destination_raw: Path,
    *,
    symbol: str,
    cutoff: str = COMPARE_CUTOFF,
) -> list[dict]:
    """Restaura somente linhas divergentes de 2018 de um RAW validado.

    Nunca escreve nos dois arquivos de origem. A serializacao do clone e validada
    por float64 de round-trip antes e depois, inclusive nas datas nao trocadas.
    """
    historical = load_raw_bar_file(original_raw, float_precision="round_trip")
    fresh = load_raw_bar_file(updated_raw, float_precision="round_trip")
    report, changes = compare_bar_frames(symbol, historical, fresh, cutoff)
    if report["old_only_rows"] or report["new_only_rows"]:
        raise RuntimeError(f"{symbol}: calendario original e atualizado divergem.")
    for record in changes:
        day = pd.Timestamp(record["timestamp"]).date().isoformat()
        if day not in ALLOWED_DATES:
            raise RuntimeError(f"{symbol}: nova divergencia inesperada em {day}.")
    if not changes:
        shutil.copy2(updated_raw, destination_raw)
        return []

    # Preserva timestamp, OHLCV, contagem de negocios e VWAP dos outros dias.
    source = pd.read_csv(updated_raw, float_precision="round_trip")
    source_times = pd.to_datetime(source["timestamp"], utc=True)
    old = historical.loc[:, list(BARS)]
    for record in changes:
        stamp = pd.Timestamp(record["timestamp"])
        matching = source_times == stamp
        if int(matching.sum()) != 1:
            raise RuntimeError(f"{symbol}: timestamp duplicado/ausente: {stamp}.")
        for field in BARS:
            source.loc[matching, field] = float(old.loc[stamp, field])
    source.to_csv(destination_raw, index=False, float_format="%.17g")

    restored = load_raw_bar_file(destination_raw, float_precision="round_trip")
    after, remaining = compare_bar_frames(symbol, historical, restored, cutoff)
    if remaining or after["changed_material_rows"]:
        raise RuntimeError(f"{symbol}: clone nao restaurou todas as 75 linhas.")
    for field in BARS:
        orig_values = fresh[field].to_numpy(dtype=np.float64)
        new_values = restored[field].to_numpy(dtype=np.float64)
        allowed = fresh.index.isin(
            pd.to_datetime([x["timestamp"] for x in changes], utc=True)
        )
        if not np.array_equal(orig_values[~allowed], new_values[~allowed]):
            raise RuntimeError(
                f"{symbol}: clone alterou {field} fora das barras autorizadas."
            )
    return changes


def verify_historical_sources(root: Path, historical: dict) -> SnapshotPaths:
    """Confere o universo antigo sem alterar nem rebaixar as series."""
    base = SnapshotPaths.u67(root)
    verified = validate_snapshot(base)
    if verified["snapshot_sha256"] != historical["snapshot_sha256"]:
        raise RuntimeError("O snapshot historico original mudou.")
    return base


def make_counterfactual_snapshot(
    *, root: Path, fresh: SnapshotPaths, destination: SnapshotPaths,
    historical: dict, cutoff: str = COMPARE_CUTOFF,
    snapshot_end: str = SNAPSHOT_CUTOFF,
) -> tuple[dict, dict]:
    """Cria nova identidade SHA-256 para copia atualizada com barras isoladas."""
    original = verify_historical_sources(root, historical)
    universe = universo_75(historical, ())
    selected = tuple(historical["selected_assets"])
    original_fresh = validate_snapshot(fresh)
    if tuple(original_fresh["assets"]) != universe:
        raise RuntimeError("Snapshot atual nao contem os 75 ativos elegiveis.")
    if original_fresh["bars"]["bar_snapshot_as_of_end"] != snapshot_end:
        raise RuntimeError("O snapshot de origem possui data de coleta diferente.")
    if destination.root.exists():
        raise FileExistsError(destination.root)
    destination.ensure()
    changes_by_symbol: list[dict] = []
    raw_files: dict[str, Path] = {}
    action_files: dict[str, Path] = {}
    for symbol in universe:
        old = (
            original if symbol in U67_REQUESTED_ASSETS
            else original_extra_paths(root, symbol)
        )
        old_raw = old.raw_bars / f"{symbol}.csv"
        if symbol in selected:
            expected = historical["candidate_file_hashes"][symbol]
            if file_sha256(old_raw) != expected["raw_sha256"]:
                raise RuntimeError(f"Hash historico RAW divergente: {symbol}")
            if file_sha256(old.corporate_actions / f"{symbol}.csv") != expected[
                "corporate_actions_sha256"
            ]:
                raise RuntimeError(f"Hash historico de eventos divergente: {symbol}")
        dest_bars = destination.raw_bars / f"{symbol}.csv"
        changes = replace_verified_2018_rows(
            old_raw, fresh.raw_bars / f"{symbol}.csv", dest_bars,
            symbol=symbol, cutoff=cutoff,
        )
        changes_by_symbol.extend(changes)
        raw_files[symbol] = dest_bars
        dest_actions = destination.corporate_actions / f"{symbol}.csv"
        shutil.copy2(fresh.corporate_actions / f"{symbol}.csv", dest_actions)
        action_files[symbol] = dest_actions
        if file_sha256(dest_actions) != file_sha256(
            fresh.corporate_actions / f"{symbol}.csv"
        ):
            raise RuntimeError(f"{symbol}: Corporate Actions do clone mudaram.")

    if (len(changes_by_symbol) != EXPECTED_CHANGED_ROWS
            or len({x["symbol"] for x in changes_by_symbol}) != EXPECTED_CHANGED_ASSETS):
        raise RuntimeError(
            "Quantidade das divergencias nos 75 ativos inesperada: "
            f"{len(changes_by_symbol)} registros, "
            f"{len({x['symbol'] for x in changes_by_symbol})} ativos. "
            "Nao executar um contrafactual diferente do auditado."
        )
    manifest = build_snapshot_manifest(
        destination, raw_files, action_files,
        bar_snapshot_as_of_end=snapshot_end,
        analysis_end_date=snapshot_end,
        assets=universe,
        snapshot_name=f"research-counterfactual-2018-from-{original_fresh['snapshot_sha256'][:12]}",
    )
    validated = validate_snapshot(destination)
    if manifest["snapshot_sha256"] != validated["snapshot_sha256"]:
        raise RuntimeError("Hash contrafactual nao reproduzido.")
    evidence = {
        "version": EXPERIMENT_VERSION,
        "analysis_end_date": cutoff,
        "source_fresh_snapshot_sha256": original_fresh["snapshot_sha256"],
        "source_historical_base_sha256": historical["snapshot_sha256"],
        "counterfactual_snapshot_sha256": validated["snapshot_sha256"],
        "intervention": "restore_73_effective_OHLCV_rows_from_2018_on_temporary_copy",
        "fixed_asset_count": 75,
        "structurally_excluded_from_research": sorted(U67_EXPECTED_EXCLUSIONS),
        "dates_permitted": sorted(ALLOWED_DATES),
        "modified_material_bars": len(changes_by_symbol),
        "modified_asset_count": len({x["symbol"] for x in changes_by_symbol}),
        "changed_bar_records": changes_by_symbol,
        "corporate_actions_unchanged": True,
        "original_files_modified": False,
    }
    return validated, evidence


def run(*, root: Path = ROOT, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    universe = universo_75(historical, ())
    reused = find_reusable_snapshot(SNAPSHOT_CUTOFF, universe)
    if reused is not None:
        _, fresh, original_manifest = reused
    else:
        migration_dir = root / "dados" / ".u75_fresh_runs" / (
            "asof_2026-10-22_migrated_for_causal_" +
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        migrated = reuse_legacy_snapshot_as_75(
            SNAPSHOT_CUTOFF, universe, migration_dir
        )
        if migrated is None:
            raise FileNotFoundError(
                "Nao existe snapshot 75 pronto nem snapshot legado validado; "
                "nao baixar nem reconstruir series originais."
            )
        fresh, original_manifest, _ = migrated
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    temp_root = root / "dados" / ".u75_fresh_runs"
    scenario_dir = temp_root / f"counterfactual_2018_{stamp}"
    clone = SnapshotPaths.from_root(scenario_dir / "universe_75")
    out = (Path(output) if output else root / "output" / "causal_drift_2018" / stamp)
    if out.exists():
        raise FileExistsError(out)
    manifest, evidence = make_counterfactual_snapshot(
        root=root, fresh=fresh, destination=clone, historical=historical,
        cutoff=COMPARE_CUTOFF, snapshot_end=SNAPSHOT_CUTOFF,
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "intervencao_controlada.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    decisions: dict[str, dict] = {}
    for label, paths, manifest_payload in (
        ("fresh_reference", fresh, original_manifest),
        ("fresh_with_original_2018_bars", clone, manifest),
    ):
        contract = {
            "schema_version": 2,
            "scope": "fixed_77_input_intervention_research",
            "analysis_end_date": COMPARE_CUTOFF,
            "snapshot_data_end_date": SNAPSHOT_CUTOFF,
            "snapshot_sha256": manifest_payload["snapshot_sha256"],
            "selected_assets": historical["selected_assets"],
            "candidate_file_hashes": {
                asset: {
                    "raw_sha256": file_sha256(paths.raw_bars / f"{asset}.csv"),
                    "corporate_actions_sha256": file_sha256(
                        paths.corporate_actions / f"{asset}.csv"
                    ),
                }
                for asset in historical["selected_assets"]
            },
            "baseline_ending_capital": None,
            "expected_ending_capital": None,
            "full_refresh": False,
        }
        contract_path = out / f"contract_{label}.json"
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = reproduzir_universo_fixo75(
            root=root,
            output=out / label,
            snapshot_root=paths.root,
            analysis_end_date=COMPARE_CUTOFF,
            version=EXPERIMENT_VERSION,
            execution_schema=EXECUTION_SCHEMA,
        )
        if result["universe"]["effective_count"] != 75:
            raise RuntimeError(f"{label}: universo efetivo diferente de 75.")
        decisions[label] = {
            "capital": result["checkpoint"]["observed_capital"],
            "metrics": result["metrics"],
            "snapshot_sha256": manifest_payload["snapshot_sha256"],
        }
        print(
            f"[counterfactual] {label}: "
            f"US$ {decisions[label]['capital']:,.2f}",
            flush=True,
        )

    report = {
        **{key: value for key, value in evidence.items() if key != "changed_bar_records"},
        "scenario_results": decisions,
        "difference_due_to_2018_bar_values": (
            decisions["fresh_with_original_2018_bars"]["capital"]
            - decisions["fresh_reference"]["capital"]
        ),
        "historical_checkpoint_at_2026_10_06": historical["expected_ending_capital"],
        "model_design": "all_75_models_trained_in_same_fold_no_65_plus_10",
        "warning": (
            "Comparacao contrafactual de sensibilidade, nao correcao oficial "
            "nem prova isolada da causa de cada ordem. Os dados imutaveis "
            "das duas origens permanecem preservados."
        ),
    }
    report_path = out / "comparacao_contrafactual_2018.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    archive_path = out / "pacote_contrafactual_2018.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(out.rglob("*")):
            if item.is_file() and item != archive_path:
                archive.write(item, arcname=item.relative_to(out).as_posix())
        archive.write(clone.manifest, arcname="counterfactual_snapshot_manifest.json")
    print(f"[counterfactual] pacote: {archive_path}", flush=True)
    return report


if __name__ == "__main__":
    run()
