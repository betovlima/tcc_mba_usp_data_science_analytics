"""Pesquisa v1.23.8: decompor o efeito das 73 barras em volume e OHLC.

Controle e intervencao completa provem do pacote local v1.23.7 ja executado.
Esta rotina calcula SOMENTE os dois cenarios restantes do fatorial 2x2:
(1) repor apenas volume, (2) repor apenas OHLC. Sem Alpaca, Mongo ou ordens.
Nunca altera snapshots de origem. No Spyder executar com F5.
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
    BARS, compare_bar_frames, file_sha256, original_extra_paths,
)
from engine.configuracao import U67_REQUESTED_ASSETS
from executar_atualizacao_series_spyder import (
    find_reusable_snapshot, reuse_legacy_snapshot_as_75,
)
from reproducao.dados import SnapshotPaths, build_snapshot_manifest, validate_snapshot
from reproducao.preparacao import load_raw_bar_file
from reproducao.universo_fixo75 import reproduzir_universo_fixo75, universo_75
from pesquisar_causa_drift_2018_spyder import (
    EXPECTED_CHANGED_ROWS, EXPECTED_CHANGED_ASSETS, ALLOWED_DATES,
    verify_historical_sources,
)

ROOT = Path(__file__).resolve().parent
VERSION = "1.23.8"
SCHEMA = "factorial-ohlc-volume-v1"
CUTOFF = "2026-10-06"
SNAPSHOT_END = "2026-10-22"
OUTPUT_ROOT = ROOT / "output" / "fatorial_drift_2018"
GROUPS = {
    "volume_only": ("volume",),
    "ohlc_only": ("open", "high", "low", "close"),
}


def restore_selected_fields(
    original_raw: Path, fresh_raw: Path, destination_raw: Path,
    *, symbol: str, restore_fields: tuple[str, ...],
    cutoff: str = CUTOFF,
) -> list[dict]:
    """Restaura apenas campos selecionados de barras comprovadamente divergentes."""
    fields = set(restore_fields)
    if not fields or not fields.issubset(BARS):
        raise ValueError("Os campos permitidos sao OHLC e/ou volume.")
    old = load_raw_bar_file(original_raw, float_precision="round_trip")
    fresh = load_raw_bar_file(fresh_raw, float_precision="round_trip")
    summary, changes = compare_bar_frames(symbol, old, fresh, cutoff)
    if summary["old_only_rows"] or summary["new_only_rows"]:
        raise RuntimeError(f"{symbol}: calendario de dados divergente.")
    dates = [pd.Timestamp(record["timestamp"]) for record in changes]
    if any(stamp.date().isoformat() not in ALLOWED_DATES for stamp in dates):
        raise RuntimeError(f"{symbol}: nova divergencia fora de maio/2018.")
    if not changes:
        shutil.copy2(fresh_raw, destination_raw)
        return []
    new = pd.read_csv(fresh_raw, float_precision="round_trip")
    timestamps = pd.to_datetime(new["timestamp"], utc=True)
    affected: list[dict] = []
    for record in changes:
        stamp = pd.Timestamp(record["timestamp"])
        matching = timestamps.eq(stamp)
        if int(matching.sum()) != 1:
            raise RuntimeError(f"{symbol}: barra duplicada ou ausente {stamp}.")
        allowed_changes = fields.intersection(record["fields"].split(","))
        if not allowed_changes:
            continue
        for field in allowed_changes:
            new.loc[matching, field] = float(old.loc[stamp, field])
        affected.append({
            "symbol": symbol, "timestamp": stamp.isoformat(),
            "restored_fields": sorted(allowed_changes),
        })
    if not affected:
        shutil.copy2(fresh_raw, destination_raw)
        return []
    new.to_csv(destination_raw, index=False, float_format="%.17g")
    verified = load_raw_bar_file(destination_raw, float_precision="round_trip")
    # Cada campo pode mudar somente nos registros autorizados e restaurados.
    allowed_timestamps = {
        field: {
            pd.Timestamp(item["timestamp"])
            for item in affected if field in item["restored_fields"]
        }
        for field in BARS
    }
    for field in BARS:
        before_values = fresh[field].to_numpy(dtype=np.float64)
        after_values = verified[field].to_numpy(dtype=np.float64)
        outside = ~fresh.index.isin(list(allowed_timestamps[field]))
        if not np.array_equal(before_values[outside], after_values[outside]):
            raise RuntimeError(f"{symbol}: coluna {field} alterada fora da intervencao.")
        if allowed_timestamps[field]:
            chosen = verified.index.isin(list(allowed_timestamps[field]))
            if not np.array_equal(
                verified[field].to_numpy(dtype=np.float64)[chosen],
                old[field].to_numpy(dtype=np.float64)[chosen],
            ):
                raise RuntimeError(f"{symbol}: coluna {field} nao restaurada exatamente.")
    return affected


def latest_original_factorial_controls(
    root: Path, original_fresh_sha: str, historical_sha: str,
) -> tuple[Path, dict]:
    """Reutiliza dois controles reais previamente calculados no contrafactual."""
    for report in sorted(
        (root / "output" / "causal_drift_2018").glob(
            "*/comparacao_contrafactual_2018.json"
        ), reverse=True,
    ):
        result = json.loads(report.read_text(encoding="utf-8"))
        if (
            result.get("source_fresh_snapshot_sha256") != original_fresh_sha
            or result.get("source_historical_base_sha256") != historical_sha
            or result.get("analysis_end_date") != CUTOFF
        ):
            continue
        if (
            result.get("modified_material_bars") != EXPECTED_CHANGED_ROWS
            or result.get("modified_asset_count") != EXPECTED_CHANGED_ASSETS
        ):
            continue
        base = report.parent
        required = (
            base / "fresh_reference" / "predictions.csv",
            base / "fresh_with_original_2018_bars" / "predictions.csv",
            base / "fresh_reference" / "trades.csv",
            base / "fresh_with_original_2018_bars" / "trades.csv",
        )
        if all(path.is_file() for path in required):
            return base, result
    raise FileNotFoundError(
        "Contrafactual v1.23.7 com seus dois controles nao encontrado. "
        "Execute primeiro pesquisar_causa_drift_2018_spyder.py (F5)."
    )


def run(*, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    old_base = verify_historical_sources(root, historical)
    selected = tuple(historical["selected_assets"])
    assets = universo_75(historical, ())
    reusable = find_reusable_snapshot(SNAPSHOT_END, assets)
    if reusable is None:
        target = root / "dados" / ".u75_fresh_runs" / (
            "asof_2026-10-22_factorial_source_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        migrated = reuse_legacy_snapshot_as_75(SNAPSHOT_END, assets, target)
        if migrated is None:
            raise FileNotFoundError("Snapshot novo de 75 ativos indisponivel.")
        fresh, fresh_manifest, _ = migrated
    else:
        _, fresh, fresh_manifest = reusable

    prior_folder, prior = latest_original_factorial_controls(
        root, fresh_manifest["snapshot_sha256"], historical["snapshot_sha256"]
    )
    recorded = prior["scenario_results"]
    capitals = {
        "fresh_reference": float(recorded["fresh_reference"]["capital"]),
        "full_2018_restore": float(
            recorded["fresh_with_original_2018_bars"]["capital"]
        ),
    }
    if capitals["full_2018_restore"] != float(historical["expected_ending_capital"]):
        raise RuntimeError("O controle historico nao reproduz o checkpoint esperado.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    storage = root / "dados" / ".u75_fresh_runs" / f"factorial_2018_{stamp}"
    out = root / "output" / "fatorial_drift_2018" / stamp
    if storage.exists() or out.exists():
        raise FileExistsError("A execucao fatorial nao pode sobrescrever arquivos.")
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for label, fields in GROUPS.items():
        dest = SnapshotPaths.from_root(storage / label / "universe_75")
        dest.ensure()
        raw_files, action_files, modifications = {}, {}, []
        original_changes = []
        for symbol in assets:
            old_paths = (
                old_base if symbol in U67_REQUESTED_ASSETS
                else original_extra_paths(root, symbol)
            )
            old_raw = old_paths.raw_bars / f"{symbol}.csv"
            if symbol in selected:
                expected = historical["candidate_file_hashes"][symbol]
                if (
                    file_sha256(old_raw) != expected["raw_sha256"]
                    or file_sha256(
                        old_paths.corporate_actions / f"{symbol}.csv"
                    ) != expected["corporate_actions_sha256"]
                ):
                    raise RuntimeError(f"{symbol}: historico original divergente.")
            source_raw = fresh.raw_bars / f"{symbol}.csv"
            _, changes = compare_bar_frames(
                symbol,
                load_raw_bar_file(old_raw, float_precision="round_trip"),
                load_raw_bar_file(source_raw, float_precision="round_trip"),
                CUTOFF,
            )
            original_changes.extend(changes)
            raw = dest.raw_bars / f"{symbol}.csv"
            restored = restore_selected_fields(
                old_raw, source_raw, raw,
                symbol=symbol, restore_fields=fields,
            )
            modifications.extend(restored)
            raw_files[symbol] = raw
            action = dest.corporate_actions / f"{symbol}.csv"
            original_actions = fresh.corporate_actions / f"{symbol}.csv"
            shutil.copy2(original_actions, action)
            if file_sha256(action) != file_sha256(original_actions):
                raise RuntimeError(f"{symbol}: eventos corporativos alterados.")
            action_files[symbol] = action
        if (
            len(original_changes) != EXPECTED_CHANGED_ROWS
            or len({x["symbol"] for x in original_changes}) != EXPECTED_CHANGED_ASSETS
        ):
            raise RuntimeError("As 73 barras originais nao foram conferidas.")
        manifest = build_snapshot_manifest(
            dest, raw_files, action_files,
            bar_snapshot_as_of_end=SNAPSHOT_END,
            analysis_end_date=SNAPSHOT_END,
            assets=assets, snapshot_name=f"factorial-{label}-{stamp}",
        )
        validated = validate_snapshot(dest)
        if manifest["snapshot_sha256"] != validated["snapshot_sha256"]:
            raise RuntimeError("Falha no hash do snapshot fatorial.")
        result = reproduzir_universo_fixo75(
            root=root, output=out / label, snapshot_root=dest.root,
            analysis_end_date=CUTOFF, version=VERSION, execution_schema=SCHEMA,
        )
        if result["universe"]["effective_count"] != 75:
            raise RuntimeError("O replay considerou menos de 75 ativos.")
        capitals[label] = float(result["checkpoint"]["observed_capital"])
        evidence[label] = {
            "restored_fields": list(fields),
            "modified_bars": len(modifications),
            "affected_assets": len({x["symbol"] for x in modifications}),
            "snapshot_sha256": manifest["snapshot_sha256"],
            "ending_capital": capitals[label],
        }
        print(f"[factorial] {label} US$ {capitals[label]:,.2f}", flush=True)
    fresh_capital = capitals["fresh_reference"]
    all_capital = capitals["full_2018_restore"]
    vol_capital = capitals["volume_only"]
    price_capital = capitals["ohlc_only"]
    interaction = all_capital - vol_capital - price_capital + fresh_capital
    report = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "scope": "descriptive_2x2_field_intervention_not_strategy_optimization",
        "cutoff": CUTOFF,
        "universe_count": len(assets),
        "excluded_assets": ["CLMT", "DOC"],
        "source_snapshot_sha256": fresh_manifest["snapshot_sha256"],
        "historical_snapshot_sha256": historical["snapshot_sha256"],
        "prior_control_report": str(prior_folder),
        "prior_control_full_snapshot_sha256": prior["counterfactual_snapshot_sha256"],
        "four_capitals": capitals,
        "interventions": evidence,
        "volume_only_delta_usd": vol_capital - fresh_capital,
        "ohlc_only_delta_usd": price_capital - fresh_capital,
        "joint_2018_delta_usd": all_capital - fresh_capital,
        "interaction_usd": interaction,
        "interaction_definition": (
            "joint_delta - volume_only_delta - ohlc_only_delta; "
            "nonlinear re-training and rotation make individual effects nonadditive"
        ),
        "interpretation_limit": (
            "Atribuicao fatorial condicional aos CSVs e algoritmo analisados; "
            "nao corrige, promove ou valida a veracidade do snapshot antigo."
        ),
    }
    summary = out / "comparacao_fatorial_ohlc_volume.json"
    summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    archive = out / "pacote_fatorial_ohlc_volume.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.write(summary, summary.name)
        for folder in ("volume_only", "ohlc_only"):
            for path in sorted((out / folder).rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(out).as_posix())
        for folder in GROUPS:
            handle.write(
                storage / folder / "universe_75" / "manifest.json",
                arcname=f"manifest_{folder}.json",
            )
    print(f"[factorial] relatorio: {archive}", flush=True)
    return report


if __name__ == "__main__":
    run()
