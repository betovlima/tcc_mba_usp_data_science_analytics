"""Audita revisoes RAW e eventos somente nos 75 ativos efetivos do TCC.

Somente leitura dos snapshots (nenhuma chamada Alpaca, nenhuma ordem ou treino).
No Spyder: abra este arquivo e use F5. Saidas somente em output/.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from engine.configuracao import U67_REQUESTED_ASSETS
from reproducao.universo_fixo75 import universo_75
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import (
    load_actions_file, load_raw_bar_file, split_normalize,
)

ROOT = Path(__file__).resolve().parent
ANALYSIS_CUTOFF = "2026-10-06"
FIRST_DIFFERENT_DECISION = "2020-07-28"
FRESH_CUTOFF = "2026-10-22"
AUDIT_VERSION = "historical-drift-v1.0.1-fixed75"
OUTPUT = ROOT / "output" / "auditoria_drift_historico"
BARS = ("open", "high", "low", "close", "volume")
ECONOMIC_FIELDS = (
    "action_type", "symbol", "ex_date", "effective_date", "old_rate",
    "new_rate", "rate", "acquirer_symbol", "acquiree_symbol",
    "old_cusip", "new_cusip", "acquirer_cusip", "acquiree_cusip",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restrict_bars(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    exclusive = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[frame.index < exclusive, list(BARS)].copy()


def compare_bar_frames(
    symbol: str,
    old: pd.DataFrame,
    new: pd.DataFrame,
    cutoff: str = ANALYSIS_CUTOFF,
    *,
    kind: str = "raw",
) -> tuple[dict, list[dict]]:
    """Separa novas linhas de revisoes em datas comuns, sem uniao por posicao."""
    before = restrict_bars(old, cutoff)
    after = restrict_bars(new, cutoff)
    common = before.index.intersection(after.index).sort_values()
    old_only = before.index.difference(after.index)
    new_only = after.index.difference(before.index)
    left = before.reindex(common)
    right = after.reindex(common)
    exact = np.column_stack([
        left[column].to_numpy(dtype=float) != right[column].to_numpy(dtype=float)
        for column in BARS
    ])
    material = np.column_stack([
        ~np.isclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=1e-12, atol=1e-10, equal_nan=True,
        )
        for column in BARS
    ])
    rows: list[dict] = []
    changed_index = np.flatnonzero(material.any(axis=1))
    for index in changed_index:
        stamp = common[index]
        changed = [name for pos, name in enumerate(BARS) if material[index, pos]]
        record = {
            "symbol": symbol,
            "type": kind,
            "timestamp": stamp.isoformat(),
            "fields": ",".join(changed),
        }
        for field in changed:
            record[f"{field}_old"] = float(left.iloc[index][field])
            record[f"{field}_fresh"] = float(right.iloc[index][field])
        rows.append(record)
    decision = pd.Timestamp(FIRST_DIFFERENT_DECISION, tz="UTC") + pd.Timedelta(days=1)
    before_decision = int((common[material.any(axis=1)] < decision).sum())
    return {
        "symbol": symbol,
        "type": kind,
        "old_rows_to_cutoff": len(before),
        "new_rows_to_cutoff": len(after),
        "common_rows": len(common),
        "old_only_rows": len(old_only),
        "new_only_rows": len(new_only),
        "first_old_only": old_only.min().isoformat() if len(old_only) else None,
        "first_new_only": new_only.min().isoformat() if len(new_only) else None,
        "changed_exact_rows": int(exact.any(axis=1).sum()),
        "changed_material_rows": int(len(changed_index)),
        "first_material_change": (
            common[changed_index[0]].isoformat() if len(changed_index) else None
        ),
        "material_change_through_first_divergence": before_decision,
    }, rows


def event_date(action: dict) -> str | None:
    for field in ("ex_date", "effective_date", "process_date"):
        value = action.get(field)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value)[:10]
    return None


def fingerprint_event(action: dict) -> tuple[str, ...]:
    """Ignora apenas IDs/metadata da API; compara propriedades economicas."""
    values = []
    for field in ECONOMIC_FIELDS:
        value = action.get(field)
        values.append(
            "" if value is None or pd.isna(value) else str(value).strip()
        )
    return tuple(values)


def compare_action_records(
    symbol: str,
    old: list[dict],
    new: list[dict],
    cutoff: str = ANALYSIS_CUTOFF,
) -> tuple[dict, list[dict]]:
    def partition(actions: list[dict]) -> tuple[Counter, int, int]:
        earlier = Counter()
        later = 0
        unknown = 0
        for action in actions:
            date_value = event_date(action)
            if date_value is None:
                unknown += 1
            elif date_value > cutoff:
                later += 1
            else:
                earlier[fingerprint_event(action)] += 1
        return earlier, later, unknown

    before, old_later, old_unknown = partition(old)
    after, new_later, new_unknown = partition(new)
    removed = before - after
    added = after - before
    changes = []
    for direction, items in (("removed", removed), ("added", added)):
        for item, count in sorted(items.items()):
            changes.append({
                "symbol": symbol,
                "direction": direction,
                "event_type": item[0],
                "ex_date": item[2],
                "effective_date": item[3],
                "fingerprint": json.dumps(dict(zip(ECONOMIC_FIELDS, item))),
                "count": count,
            })

    def later_splits(actions: list[dict]) -> int:
        return sum(
            action.get("action_type") in ("forward_split", "reverse_split")
            and event_date(action) is not None
            and event_date(action) > cutoff
            for action in actions
        )

    return {
        "symbol": symbol,
        "action_old_to_cutoff": sum(before.values()),
        "action_new_to_cutoff": sum(after.values()),
        "action_removed_before_cutoff": sum(removed.values()),
        "action_added_before_cutoff": sum(added.values()),
        "action_old_after_cutoff": old_later,
        "action_new_after_cutoff": new_later,
        "action_undated_old": old_unknown,
        "action_undated_new": new_unknown,
        "new_post_cutoff_splits": later_splits(new),
    }, changes


def original_extra_paths(root: Path, symbol: str) -> SnapshotPaths:
    selected = SnapshotPaths.from_root(
        root / "dados" / ".u67_expansion_cache" / "full_raw_20261006"
    )
    if not (selected.raw_bars / f"{symbol}.csv").is_file():
        raise FileNotFoundError(
            f"Cache historico obrigatorio ausente para {symbol}: "
            f"{selected.raw_bars}. Restaurar exatamente o cache antigo; nao baixar."
        )
    return selected


def latest_valid_fresh_snapshot(
    root: Path, *, cutoff: str = FRESH_CUTOFF,
) -> tuple[SnapshotPaths, dict]:
    base = root / "dados" / ".u75_fresh_runs"
    candidates = sorted(base.glob(f"asof_{cutoff}_*/universe_75"), reverse=True)
    for item in candidates:
        paths = SnapshotPaths.from_root(item)
        if not paths.manifest.is_file():
            continue
        manifest = validate_snapshot(paths)
        if len(manifest.get("assets") or []) != 75:
            raise RuntimeError("Auditoria so permite snapshot de 75 ativos.")
        return paths, manifest
    raise FileNotFoundError(
        f"Nenhum snapshot de 75 ativos completo em {base}; "
        "o ZIP do replay nao contem CSVs brutos."
    )


def audit(
    *,
    root: Path = ROOT,
    output: Path | None = None,
    cutoff: str = ANALYSIS_CUTOFF,
    fresh_cutoff: str = FRESH_CUTOFF,
) -> dict:
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "output" / "auditoria_drift_historico"
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(encoding="utf-8")
    )
    originals = SnapshotPaths.u67(root)
    old_manifest = validate_snapshot(originals)
    if old_manifest["snapshot_sha256"] != historical["snapshot_sha256"]:
        raise RuntimeError("Base historica nao corresponde ao snapshot da pesquisa.")
    fresh, new_manifest = latest_valid_fresh_snapshot(root, cutoff=fresh_cutoff)

    selected = tuple(historical["selected_assets"])
    requested = universo_75(historical, ())
    if tuple(new_manifest.get("assets") or ()) != requested:
        raise RuntimeError("Snapshot renovado nao corresponde aos 75 ativos fixos.")

    summaries: list[dict] = []
    raw_changes: list[dict] = []
    adjusted_changes: list[dict] = []
    action_changes: list[dict] = []
    for number, symbol in enumerate(requested, 1):
        old = (
            originals if symbol in U67_REQUESTED_ASSETS
            else original_extra_paths(root, symbol)
        )
        old_raw = old.raw_bars / f"{symbol}.csv"
        old_act = old.corporate_actions / f"{symbol}.csv"
        new_raw = fresh.raw_bars / f"{symbol}.csv"
        new_act = fresh.corporate_actions / f"{symbol}.csv"
        if symbol in selected:
            expected = historical["candidate_file_hashes"][symbol]
            if file_sha256(old_raw) != expected["raw_sha256"]:
                raise RuntimeError(f"Hash RAW historico incorreto: {symbol}")
            if file_sha256(old_act) != expected["corporate_actions_sha256"]:
                raise RuntimeError(f"Hash action historico incorreto: {symbol}")

        old_bars = load_raw_bar_file(old_raw, float_precision="round_trip")
        fresh_bars = load_raw_bar_file(new_raw, float_precision="round_trip")
        old_actions = load_actions_file(old_act)
        new_actions = load_actions_file(new_act)
        bars_summary, bars_changes = compare_bar_frames(
            symbol, old_bars, fresh_bars, cutoff
        )
        old_adjusted, _ = split_normalize(old_bars, old_actions)
        new_adjusted, _ = split_normalize(fresh_bars, new_actions)
        adjusted_summary, adjusted_diffs = compare_bar_frames(
            symbol, old_adjusted, new_adjusted, cutoff, kind="split_adjusted"
        )
        event_summary, event_diffs = compare_action_records(
            symbol, old_actions, new_actions, cutoff
        )
        bars_summary["new_rows_after_cutoff"] = int(
            (fresh_bars.index >= pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)).sum()
        )
        bars_summary.update({
            "raw_sha_old": file_sha256(old_raw),
            "raw_sha_new": file_sha256(new_raw),
            "actions_sha_old": file_sha256(old_act),
            "actions_sha_new": file_sha256(new_act),
            "split_adjusted_changed_material_rows": adjusted_summary["changed_material_rows"],
            "split_adjusted_first_material_change": adjusted_summary["first_material_change"],
            **event_summary,
        })
        summaries.append(bars_summary)
        raw_changes.extend(bars_changes)
        adjusted_changes.extend(adjusted_diffs)
        action_changes.extend(event_diffs)
        if bars_summary["changed_material_rows"] or adjusted_summary["changed_material_rows"]:
            print(
                f"[audit] {number}/77 {symbol} RAW={bars_summary['changed_material_rows']} "
                f"split_adjusted={adjusted_summary['changed_material_rows']}",
                flush=True,
            )
        else:
            print(f"[audit] {number}/77 {symbol} sem diferenca material em datas comuns")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "resumo_por_ativo.csv", index=False)
    pd.DataFrame(raw_changes).to_csv(output / "divergencias_ohlcv_raw.csv", index=False)
    pd.DataFrame(adjusted_changes).to_csv(
        output / "divergencias_ohlcv_splits.csv", index=False
    )
    pd.DataFrame(action_changes).to_csv(
        output / "divergencias_corporate_actions.csv", index=False
    )
    totals = {
        "audit_version": AUDIT_VERSION,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "target_cutoff": cutoff,
        "first_observed_trade_divergence": FIRST_DIFFERENT_DECISION,
        "snapshot_original_sha256": old_manifest["snapshot_sha256"],
        "snapshot_fresh_sha256": new_manifest["snapshot_sha256"],
        "fixed_effective_asset_count": len(requested),
        "excluded_from_all_research": ["CLMT", "DOC"],
        "source_kind": "local_offline_sha256_validated_csv",
        "strategy_modified": False,
        "downloads_made": False,
        "assets_with_raw_changes_on_common_dates": sum(
            x["changed_material_rows"] > 0 for x in summaries
        ),
        "assets_with_only_appended_raw_bars": sum(
            x["changed_material_rows"] == 0 and x["new_rows_after_cutoff"] > 0
            and x["old_only_rows"] == 0 and x["new_only_rows"] == 0
            for x in summaries
        ),
        "assets_with_split_adjusted_changes": sum(
            x["split_adjusted_changed_material_rows"] > 0 for x in summaries
        ),
        "raw_divergent_rows": len(raw_changes),
        "split_adjusted_divergent_rows": len(adjusted_changes),
        "corporate_action_divergent_rows": len(action_changes),
        "raw_divergences_through_2020_07_28": sum(
            x["material_change_through_first_divergence"] for x in summaries
        ),
        "warning": (
            "Descritivo: alteracao de insumos nao demonstra causalidade "
            "entre uma linha revista e a diferenca de decisao/retorno."
        ),
    }
    (output / "resumo_auditoria.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    archive_path = output / "pacote_auditoria_drift_historico.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != archive_path:
                archive.write(path, arcname=path.name)
    print(f"[audit] concluido: {archive_path}", flush=True)
    return totals


if __name__ == "__main__":
    audit()
