"""Validacao, exclusoes estruturais e normalizacao local de splits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reproducao.dados import SnapshotPaths
from tcc_engine.config import ASSETS


REFERENCE_FILE = Path(__file__).with_name(
    "reference_10_8_74_raw_snapshot_diagnostics.json"
)


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            result[key] = None
        elif isinstance(value, float) and np.isnan(value):
            result[key] = None
        else:
            result[key] = value
    return result


def load_raw_bar_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{path.name}: colunas RAW ausentes: {', '.join(missing)}"
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame[["open", "high", "low", "close", "volume"]].copy()


def load_actions_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Corporate Actions ausente: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return [_clean_record(item) for item in frame.to_dict(orient="records")]


def structural_identity_issue(
    symbol: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized = symbol.strip().upper()
    for action in actions:
        action_type = str(action.get("action_type") or "")
        if action_type not in {
            "stock_merger",
            "stock_and_cash_merger",
            "cash_merger",
        }:
            continue
        acquiree = str(action.get("acquiree_symbol") or "").strip().upper()
        acquirer = str(action.get("acquirer_symbol") or "").strip().upper()
        if acquiree == normalized and acquirer and acquirer != normalized:
            return {
                "symbol": normalized,
                "reason": "structural_identity_change",
                "action_type": action_type,
                "process_date": action.get("process_date"),
                "effective_date": action.get("effective_date"),
                "acquiree_symbol": acquiree,
                "acquirer_symbol": acquirer,
            }
    return None


def split_normalize(
    raw: pd.DataFrame,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Replica a normalizacao usada na linha 10.8.74/10.8.84."""
    result = raw.copy()
    session_dates = pd.DatetimeIndex(result.index).tz_convert("UTC").normalize()
    applied: list[dict[str, Any]] = []
    splits = [
        action
        for action in actions
        if action.get("action_type") in {"forward_split", "reverse_split"}
        and action.get("ex_date")
        and action.get("old_rate") is not None
        and action.get("new_rate") is not None
    ]
    splits.sort(key=lambda item: str(item.get("ex_date")))

    for action in splits:
        ex_date = pd.Timestamp(action["ex_date"])
        ex_date = (
            ex_date.tz_localize("UTC")
            if ex_date.tzinfo is None
            else ex_date.tz_convert("UTC")
        ).normalize()
        old_rate = float(action["old_rate"])
        new_rate = float(action["new_rate"])
        if old_rate <= 0.0 or new_rate <= 0.0:
            continue

        price_factor = old_rate / new_rate
        volume_factor = new_rate / old_rate
        mask = session_dates < ex_date
        if not mask.any():
            continue

        for column in ("open", "high", "low", "close"):
            result.loc[mask, column] = (
                result.loc[mask, column].astype(float) * price_factor
            )
        result.loc[mask, "volume"] = (
            result.loc[mask, "volume"].astype(float) * volume_factor
        )
        applied.append(
            {
                "action_type": action.get("action_type"),
                "ex_date": str(action.get("ex_date")),
                "process_date": str(action.get("process_date")),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "price_factor": price_factor,
                "volume_factor": volume_factor,
            }
        )
    return result, applied


def prepare_model_frames(
    paths: SnapshotPaths,
    *,
    assets: tuple[str, ...] = ASSETS,
    write_normalized_csv: bool = True,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Carrega o snapshot, exclui identidades quebradas e normaliza splits."""
    paths.ensure()
    frames: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for position, symbol in enumerate(assets, start=1):
        raw_path = paths.raw_bars / f"{symbol}.csv"
        action_path = paths.corporate_actions / f"{symbol}.csv"
        raw = load_raw_bar_file(raw_path)
        actions = load_actions_file(action_path)
        issue = structural_identity_issue(symbol, actions)

        if issue is not None:
            exclusions.append(issue)
            print(
                f"[data] {position}/{len(assets)} {symbol} excluded "
                f"reason={issue['reason']} action={issue['action_type']} "
                f"acquirer={issue['acquirer_symbol']}",
                flush=True,
            )
            diagnostics.append(
                {
                    "symbol": symbol,
                    "raw_rows": len(raw),
                    "corporate_actions": len(actions),
                    "splits_applied": 0,
                    "excluded": True,
                    "exclusion_reason": issue["reason"],
                }
            )
            continue

        normalized, applied = split_normalize(raw, actions)
        frames[symbol] = normalized
        if write_normalized_csv:
            target = paths.normalized_bars / f"{symbol}.csv"
            normalized.reset_index().to_csv(
                target,
                index=False,
                float_format="%.17g",
            )
        diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": len(raw),
                "corporate_actions": len(actions),
                "splits_applied": len(applied),
                "excluded": False,
                "exclusion_reason": None,
            }
        )
        print(
            f"[data] {position}/{len(assets)} {symbol} "
            f"raw_rows={len(raw)} ca={len(actions)} splits={len(applied)}",
            flush=True,
        )

    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    expected = {
        row["symbol"]: row
        for row in reference.get("diagnostics", [])
    }
    raw_mismatches: list[str] = []
    action_mismatches: list[str] = []
    split_mismatches: list[str] = []

    for row in diagnostics:
        symbol = row["symbol"]
        if row["excluded"]:
            continue
        ref = expected.get(symbol)
        if ref is None:
            raw_mismatches.append(symbol)
            action_mismatches.append(symbol)
            split_mismatches.append(symbol)
            continue
        if int(row["raw_rows"]) != int(ref["raw_rows"]):
            raw_mismatches.append(symbol)
        if int(row["corporate_actions"]) != int(ref["corporate_actions"]):
            action_mismatches.append(symbol)
        if int(row["splits_applied"]) != int(ref["splits_applied"]):
            split_mismatches.append(symbol)

    actual_rows = sum(int(row["raw_rows"]) for row in diagnostics if not row["excluded"])
    reference_rows = int(reference["eligible_total_raw_rows"])
    audit = {
        "reference_api_version": reference.get("source_api_version"),
        "eligible_assets": len(frames),
        "reference_eligible_assets": int(reference["eligible_assets"]),
        "actual_total_eligible_raw_rows": actual_rows,
        "reference_total_eligible_raw_rows": reference_rows,
        "actual_total_splits_applied": sum(
            int(row["splits_applied"])
            for row in diagnostics
            if not row["excluded"]
        ),
        "reference_total_splits_applied": int(
            reference["eligible_total_splits_applied"]
        ),
        "raw_row_mismatch_symbols": raw_mismatches,
        "corporate_action_mismatch_symbols": action_mismatches,
        "split_mismatch_symbols": split_mismatches,
        "excluded_assets": exclusions,
    }
    print(
        "[data-audit] "
        f"eligible_rows={actual_rows} reference_rows={reference_rows} "
        f"row_mismatches={len(raw_mismatches)} "
        f"ca_mismatches={len(action_mismatches)} "
        f"split_mismatches={len(split_mismatches)}",
        flush=True,
    )
    if raw_mismatches:
        print(
            "[data-audit] raw row mismatch symbols=" + ",".join(raw_mismatches),
            flush=True,
        )
    if action_mismatches:
        print(
            "[data-audit] corporate-action mismatch symbols="
            + ",".join(action_mismatches),
            flush=True,
        )
    if split_mismatches:
        print(
            "[data-audit] split mismatch symbols=" + ",".join(split_mismatches),
            flush=True,
        )
    return frames, exclusions, diagnostics, audit
