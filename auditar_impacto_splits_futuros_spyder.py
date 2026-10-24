"""TCC v1.24.2 — quantifica o efeito de splits futuros sobre atributos reais.

Usa somente os 75 CSVs RAW e os eventos JA preservados no snapshot U75.
Compara a antiga normalizacao aplicada a TODOS os eventos de 2026 com
normalizacao apenas dos eventos EFETIVADOS ate a decisao em 28/07/2020.

Isto NAO prova quando o anuncio ficou disponivel. ex_date nao e first_seen.
As duas normalizacoes sao calculadas APENAS EM MEMORIA para auditar o legado,
nunca gravadas como serie de mercado, modelo novo ou lucro financeiro.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.rotacao import ROTATION_FEATURES, construir_quadro_rotacao
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import (
    load_actions_file, load_raw_bar_file, split_normalize,
)
from reproducao.universo_fixo75 import universo_75


ROOT = Path(__file__).resolve().parent
VERSION = "1.24.2"
SNAPSHOT_END = "2026-10-22"
CUTOFF = "2020-07-28"
OUTPUT_ROOT = ROOT / "output" / "impacto_splits_futuros"
FIELDS = ("open", "high", "low", "close", "volume")
RTOL = 1e-10
ATOL = 1e-12


def cutoff_timestamp(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def _effective_date(action: dict) -> pd.Timestamp | None:
    if action.get("action_type") not in ("forward_split", "reverse_split"):
        return None
    value = action.get("ex_date")
    if value is None or not str(value).strip():
        raise RuntimeError("Split real sem ex_date auditavel.")
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise RuntimeError("Split com ex_date invalida.")
    return (date.tz_localize("UTC") if date.tzinfo is None
            else date.tz_convert("UTC")).normalize()


def partition_by_effective_date(
    actions: list[dict], *, decision_day: str = CUTOFF,
) -> tuple[list[dict], list[dict]]:
    """Separar pelo evento EFETIVADO, nao presumir data do anuncio."""
    decision = cutoff_timestamp(decision_day)
    past, future = [], []
    for event in actions:
        date = _effective_date(event)
        if date is None:
            continue
        (past if date <= decision else future).append(event)
    return past, future


def _changed_count(
    a: pd.DataFrame, b: pd.DataFrame, fields: tuple[str, ...],
) -> tuple[int, dict[str, int], dict[str, float]]:
    if not a.index.equals(b.index):
        raise RuntimeError("Calendarios desalinhados: nao comparar por posicao.")
    masks = []
    per_field = {}
    max_difference = {}
    for field in fields:
        x = a[field].to_numpy(dtype=float)
        y = b[field].to_numpy(dtype=float)
        mismatch = ~np.isclose(
            x, y, atol=ATOL, rtol=RTOL, equal_nan=True,
        )
        masks.append(mismatch)
        per_field[field] = int(mismatch.sum())
        finite = np.isfinite(x) & np.isfinite(y)
        max_difference[field] = float(
            np.max(np.abs(x[finite] - y[finite])) if finite.any() else 0.0
        )
    return int(np.any(np.column_stack(masks), axis=1).sum()), per_field, max_difference


def measure_legacy_future_split_impact(
    symbol: str,
    raw: pd.DataFrame,
    actions: list[dict],
    *,
    decision_day: str = CUTOFF,
) -> tuple[dict, list[dict]]:
    """Compara entradas legadas como estavam versus excluindo splits futuros."""
    original = raw.copy()
    start = cutoff_timestamp(decision_day) + pd.Timedelta(days=1)
    prefix = raw.loc[raw.index < start, list(FIELDS)].copy()
    if len(prefix) < 250:
        raise RuntimeError(f"{symbol}: menos de 250 barras antes da decisao.")
    past, future = partition_by_effective_date(actions, decision_day=decision_day)
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(prefix, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()
    summary = {
        "symbol": symbol,
        "decision_day": decision_day,
        "raw_prefix_rows": len(prefix),
        "raw_prefix_sha256": source_hash,
        "splits_already_effective": len(past),
        "splits_effective_after_decision": len(future),
        "future_ex_dates": ";".join(sorted(
            str(item["ex_date"])[:10] for item in future
        )),
        "future_actions_with_verified_first_seen": sum(
            bool(item.get("first_seen_at_utc")) for item in future
        ),
        "raw_modified": False,
        "legacy_ohlcv_changed_rows": 0,
        "legacy_price_changed_rows": 0,
        "legacy_volume_changed_rows": 0,
        "features_materially_changed_rows": 0,
        "features_materially_changed_cells": 0,
        "features_compared": len(ROTATION_FEATURES),
        "feature_rows_common": 0,
        "feature_rows_only_one_path": 0,
    }
    if not future:
        return summary, []
    full, _ = split_normalize(raw, actions)
    asof_effective, _ = split_normalize(raw, past)
    full_prefix = full.loc[full.index < start, list(FIELDS)]
    effective_prefix = asof_effective.loc[
        asof_effective.index < start, list(FIELDS)
    ]
    total, fields, max_diffs = _changed_count(
        full_prefix, effective_prefix, FIELDS,
    )
    price_fields = ("open", "high", "low", "close")
    price_rows, _, _ = _changed_count(
        full_prefix, effective_prefix, price_fields,
    )
    summary.update({
        "legacy_ohlcv_changed_rows": total,
        "legacy_price_changed_rows": price_rows,
        "legacy_volume_changed_rows": fields["volume"],
        "max_abs_close_difference": max_diffs["close"],
        "max_abs_volume_difference": max_diffs["volume"],
    })

    # Compare 52 features do motor TCC, nao targets/labels que usam futuro.
    base_features = construir_quadro_rotacao(effective_prefix, CONFIG)
    leak_features = construir_quadro_rotacao(full_prefix, CONFIG)
    common = base_features.index.intersection(leak_features.index)
    summary["feature_rows_common"] = len(common)
    summary["feature_rows_only_one_path"] = (
        len(base_features.index.difference(leak_features.index))
        + len(leak_features.index.difference(base_features.index))
    )
    feature_differences = []
    total_cells = 0
    changed_any = np.zeros(len(common), dtype=bool)
    for feature in ROTATION_FEATURES:
        old = base_features.loc[common, feature].to_numpy(dtype=float)
        new = leak_features.loc[common, feature].to_numpy(dtype=float)
        changed = ~np.isclose(
            old, new, rtol=RTOL, atol=ATOL, equal_nan=True,
        )
        count = int(changed.sum())
        total_cells += count
        changed_any |= changed
        if count:
            valid = np.isfinite(old) & np.isfinite(new)
            maximum = float(
                np.max(np.abs(old[valid] - new[valid]))
                if valid.any() else 0.0
            )
            feature_differences.append({
                "symbol": symbol,
                "feature": feature,
                "material_changed_rows": count,
                "max_abs_difference": maximum,
            })
    summary["features_materially_changed_rows"] = int(changed_any.sum())
    summary["features_materially_changed_cells"] = total_cells
    if not original.equals(raw):
        raise RuntimeError(f"{symbol}: valores originais RAW foram alterados.")
    return summary, feature_differences


def audit(*, root: Path = ROOT, decision_day: str = CUTOFF) -> dict:
    root = Path(root).resolve()
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(historical, ())
    base = root / "dados" / ".u75_fresh_runs"
    candidates = sorted(
        base.glob(f"asof_{SNAPSHOT_END}_*/universe_75"), reverse=True,
    )
    source = None
    manifest = None
    for candidate in candidates:
        paths = SnapshotPaths.from_root(candidate)
        if paths.manifest.is_file():
            trial = validate_snapshot(paths)
            if tuple(trial.get("assets") or ()) == assets:
                source = paths
                manifest = trial
                break
    if source is None or manifest is None:
        raise FileNotFoundError(
            "Snapshot U75 completo nao encontrado; nenhum download permitido."
        )
    records, differences = [], []
    for index, symbol in enumerate(assets, 1):
        raw = load_raw_bar_file(
            source.raw_bars / f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        )
        actions = load_actions_file(
            source.corporate_actions / f"{symbol}.csv",
        )
        info, changed = measure_legacy_future_split_impact(
            symbol, raw, actions, decision_day=decision_day,
        )
        records.append(info)
        differences.extend(changed)
        if info["splits_effective_after_decision"]:
            print(
                f"[{index}/75] {symbol}: splits_futuros="
                f"{info['splits_effective_after_decision']} "
                f"ohlcv_alteradas={info['legacy_ohlcv_changed_rows']} "
                f"atributos_alterados={info['features_materially_changed_cells']}",
                flush=True,
            )

    folder = (root / "output" / "impacto_splits_futuros"
              / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(
        folder / "impacto_por_ativo.csv", index=False,
    )
    pd.DataFrame(
        differences, columns=[
            "symbol", "feature", "material_changed_rows", "max_abs_difference",
        ],
    ).to_csv(folder / "impacto_por_atributo.csv", index=False)
    report = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": manifest["snapshot_sha256"],
        "assets_checked": len(records),
        "date_of_hypothetical_decision": decision_day,
        "splits_effective_after_decision": sum(
            row["splits_effective_after_decision"] for row in records
        ),
        "assets_with_future_effective_splits": sum(
            row["splits_effective_after_decision"] > 0 for row in records
        ),
        "assets_with_changed_historical_ohlcv": sum(
            row["legacy_ohlcv_changed_rows"] > 0 for row in records
        ),
        "assets_with_material_feature_changes": sum(
            row["features_materially_changed_cells"] > 0 for row in records
        ),
        "total_material_changed_feature_cells": sum(
            row["features_materially_changed_cells"] for row in records
        ),
        "unknown_historical_announcement_availability": True,
        "original_csv_modified": False,
        "backtest_or_model_fit_performed": False,
        "interpretation": (
            "Compara normalizacao LEGADA usando todos os eventos atuais "
            "contra somente os eventos efetivados ate a decisao. Nao e "
            "reconstrucao da visibilidade historica, nem prova de "
            "vazamento de scores/rotacoes ou de impacto financeiro."
        ),
    }
    (folder / "resumo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output = folder / "pacote_impacto_splits_futuros.zip"
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("impacto_por_ativo.csv", "impacto_por_atributo.csv", "resumo.json"):
            archive.write(folder / name, arcname=name)
    print(f"[split-vintage] pacote: {output}", flush=True)
    return report


if __name__ == "__main__":
    audit()
