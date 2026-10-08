"""Validacao, exclusoes estruturais e normalizacao local de splits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reproducao.dados import SnapshotPaths
from engine.configuracao import ASSETS


# Exclusoes estruturais conhecidas e deliberadas. O snapshot permanece
# congelado; a exclusao acontece apenas no pipeline de modelagem.
KNOWN_STRUCTURAL_EXCLUSIONS: dict[str, dict[str, Any]] = {
    "CLMT": {
        "symbol": "CLMT",
        "reason": "structural_identity_change",
        "action_type": "name_change",
        "process_date": "2024-07-11",
        "effective_date": None,
        "old_cusip": "131476103",
        "new_cusip": "131428104",
        "acquiree_symbol": "CLMT",
        "acquirer_symbol": "CLMT",
        "note": (
            "same ticker with CUSIP/legal-identity transition; "
            "series excluded instead of bridged"
        ),
    },
}


def _normalize_identifier(value: Any) -> str:
    """Normaliza identificadores CSV que o pandas pode ler como float."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text.endswith(".0"):
        candidate = text[:-2]
        if candidate.isdigit():
            return candidate
    return text


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


def load_raw_bar_file(
    path: Path,
    *,
    float_precision: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        float_precision=float_precision,
    )
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

    known = KNOWN_STRUCTURAL_EXCLUSIONS.get(normalized)
    if known is not None:
        matching = next(
            (
                action
                for action in actions
                if str(action.get("action_type") or "") == "name_change"
                and str(action.get("old_symbol") or "").strip().upper()
                == normalized
                and str(action.get("new_symbol") or "").strip().upper()
                == normalized
                and _normalize_identifier(action.get("old_cusip"))
                == _normalize_identifier(known.get("old_cusip"))
                and _normalize_identifier(action.get("new_cusip"))
                == _normalize_identifier(known.get("new_cusip"))
            ),
            None,
        )
        if matching is not None:
            return dict(known)

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
    allow_structural_assets: frozenset[str] = frozenset(),
    csv_float_precision: str | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Carrega o snapshot, exclui identidades quebradas e normaliza splits.

    allow_structural_assets existe apenas para campanhas de sensibilidade:
    o evento estrutural continua auditado, mas o ativo pode ser mantido no
    replay de forma explícita e registrada. O comportamento padrao continua
    sendo excluir.
    """
    paths.ensure()
    allowed_structural = {
        str(symbol).strip().upper()
        for symbol in allow_structural_assets
    }
    frames: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for position, symbol in enumerate(assets, start=1):
        raw_path = paths.raw_bars / f"{symbol}.csv"
        action_path = paths.corporate_actions / f"{symbol}.csv"
        raw = load_raw_bar_file(
            raw_path,
            float_precision=csv_float_precision,
        )
        actions = load_actions_file(action_path)
        issue = structural_identity_issue(symbol, actions)
        structural_override = bool(
            issue is not None
            and symbol.strip().upper() in allowed_structural
        )

        if issue is not None and not structural_override:
            exclusions.append(issue)
            detail = (
                issue.get("acquirer_symbol")
                or issue.get("new_cusip")
                or "n/a"
            )
            print(
                f"[data] {position}/{len(assets)} {symbol} excluded "
                f"reason={issue['reason']} action={issue['action_type']} "
                f"detail={detail}",
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

        if structural_override:
            print(
                f"[data] {position}/{len(assets)} {symbol} "
                f"structural_override=True reason={issue['reason']} "
                f"action={issue['action_type']}",
                flush=True,
            )

        normalized, applied = split_normalize(raw, actions)
        frames[symbol] = normalized
        diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": len(raw),
                "corporate_actions": len(actions),
                "splits_applied": len(applied),
                "excluded": False,
                "exclusion_reason": None,
                "structural_override": bool(structural_override),
                "structural_issue": (
                    dict(issue) if structural_override and issue is not None
                    else None
                ),
            }
        )
        print(
            f"[data] {position}/{len(assets)} {symbol} "
            f"raw_rows={len(raw)} ca={len(actions)} splits={len(applied)}",
            flush=True,
        )

    actual_rows = sum(
        int(row["raw_rows"])
        for row in diagnostics
        if not row["excluded"]
    )
    actual_splits = sum(
        int(row["splits_applied"])
        for row in diagnostics
        if not row["excluded"]
    )
    audit = {
        "eligible_assets": len(frames),
        "actual_total_eligible_raw_rows": actual_rows,
        "actual_total_splits_applied": actual_splits,
        "excluded_assets": exclusions,
    }
    print(
        "[data-audit] "
        f"eligible_assets={len(frames)} "
        f"eligible_rows={actual_rows} "
        f"splits_applied={actual_splits} "
        f"excluded_assets={len(exclusions)}",
        flush=True,
    )
    return frames, exclusions, diagnostics, audit
