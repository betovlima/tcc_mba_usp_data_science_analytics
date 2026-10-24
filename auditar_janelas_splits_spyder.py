"""TCC v1.24.3 — auditar atributos/labels que atravessam splits reais.

Mantem 75 ativos, OHLCV RAW intocados e usa split_normalize APENAS EM MEMORIA
para comparar o comportamento historico legada vs RAW sem reexpressao.

Não treina nem simula patrimonio. F5 no Spyder. Saida de pesquisa:
output/auditoria_janelas_splits/<ID>/pacote_janelas_splits.zip

Importante: ainda faltam custos, custodia de fracoes e first_seen_at_utc para
um backtest economico point-in-time; NAO tratar esta auditoria como replay.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
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
VERSION = "1.24.3"
SNAPSHOT_END = "2026-10-22"
LOOKBACK_SESSIONS = 60
FORWARD_SESSIONS = 60
RTOL = 1e-10
ATOL = 1e-12
TARGETS = (
    "forward_horizon_utility_5",
    "forward_horizon_utility_10",
    "forward_horizon_utility_20",
    "forward_horizon_utility_40",
    "forward_horizon_utility_60",
    "forward_risk_adjusted_utility",
    "forward_net_log_return",
)


def _utc_day(value: str) -> pd.Timestamp:
    dt = pd.Timestamp(value)
    if pd.isna(dt):
        raise ValueError("Split sem data valida.")
    return (dt.tz_localize("UTC") if dt.tzinfo is None
            else dt.tz_convert("UTC")).normalize()


def _mask_change(x: pd.Series, y: pd.Series) -> np.ndarray:
    return ~np.isclose(
        x.to_numpy(dtype=float), y.to_numpy(dtype=float),
        equal_nan=True, rtol=RTOL, atol=ATOL,
    )


def _compare_window(
    raw_features: pd.DataFrame,
    adjusted_features: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    columns: tuple[str, ...] | list[str],
) -> tuple[int, int, int]:
    dates = timestamps.intersection(raw_features.index).intersection(
        adjusted_features.index
    )
    if dates.empty:
        return 0, 0, 0
    changed_rows = np.zeros(len(dates), dtype=bool)
    changed_cells = 0
    for col in columns:
        mismatch = _mask_change(
            raw_features.loc[dates, col],
            adjusted_features.loc[dates, col],
        )
        changed_rows |= mismatch
        changed_cells += int(mismatch.sum())
    return len(dates), int(changed_rows.sum()), int(changed_cells)


def measure_event_boundary(
    symbol: str,
    raw: pd.DataFrame,
    event: dict,
    *,
    raw_features: pd.DataFrame | None = None,
) -> dict:
    """Audita isoladamente um evento efetivado durante as barras observadas.

    O fator é aplicado exclusivamente a uma COPIA em memoria, para estudar a
    divergencia da implementacao antiga. Nao publica valores ajustados.
    """
    required = ("open", "high", "low", "close", "volume")
    if not set(required).issubset(raw.columns):
        raise ValueError(f"{symbol}: sem OHLCV obrigatorios.")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise RuntimeError(f"{symbol}: calendario RAW inconsistente.")
    ex_date = _utc_day(event["ex_date"])
    sessions = pd.DatetimeIndex(raw.index)
    pos = int(sessions.normalize().searchsorted(ex_date, side="left"))
    result = {
        "symbol": symbol,
        "event_type": str(event["action_type"]),
        "ex_date": ex_date.strftime("%Y-%m-%d"),
        "old_rate": str(event.get("old_rate", "")),
        "new_rate": str(event.get("new_rate", "")),
        "event_id": str(event.get("id") or ""),
        "source_has_first_seen": bool(event.get("first_seen_at_utc")),
        "split_within_observed_bars": False,
        "prior_session": None,
        "following_session": None,
        "previous_close_raw": None,
        "following_open_raw": None,
        "following_close_raw": None,
        "raw_open_gap_pct": None,
        "raw_close_gap_pct": None,
        "custody_neutral_open_gap_pct": None,
        "pre_event_feature_rows": 0,
        "pre_event_feature_changed_rows": 0,
        "pre_event_feature_changed_cells": 0,
        "post_event_feature_rows": 0,
        "post_event_feature_changed_rows": 0,
        "post_event_feature_changed_cells": 0,
        "pre_event_label_rows": 0,
        "pre_event_label_changed_rows": 0,
        "pre_event_label_changed_cells": 0,
        "post_event_label_rows": 0,
        "post_event_label_changed_rows": 0,
        "post_event_label_changed_cells": 0,
        "raw_modified": False,
        "model_trained": False,
    }
    # Nao inferir eventos que ocorreram antes do primeiro pregao conhecido,
    # ou depois da ultima barra disponivel.
    if pos <= 0 or pos >= len(raw):
        return result

    previous = raw.iloc[pos - 1]
    following = raw.iloc[pos]
    factor = float(event["new_rate"]) / float(event["old_rate"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Split sem fator valido.")
    prior_close = float(previous["close"])
    after_open = float(following["open"])
    after_close = float(following["close"])
    if min(prior_close, after_open, after_close) <= 0:
        raise ValueError("RAW nao positivo.")
    result.update({
        "split_within_observed_bars": True,
        "prior_session": sessions[pos - 1].date().isoformat(),
        "following_session": sessions[pos].date().isoformat(),
        "previous_close_raw": prior_close,
        "following_open_raw": after_open,
        "following_close_raw": after_close,
        "raw_open_gap_pct": 100 * (after_open / prior_close - 1),
        "raw_close_gap_pct": 100 * (after_close / prior_close - 1),
        "custody_neutral_open_gap_pct": 100 * (
            after_open * factor / prior_close - 1
        ),
    })
    raw_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()
    adjusted, applied = split_normalize(raw, [event])
    if len(applied) != 1:
        raise RuntimeError(f"{symbol}: split nao aplicado no controle em memoria.")
    baseline = (
        construir_quadro_rotacao(raw, CONFIG)
        if raw_features is None else raw_features
    )
    adjusted_frame = construir_quadro_rotacao(adjusted, CONFIG)
    before = sessions[max(0, pos - LOOKBACK_SESSIONS):pos]
    after = sessions[pos:min(len(sessions), pos + FORWARD_SESSIONS)]
    for slot, days in (("pre_event", before), ("post_event", after)):
        n, changed_rows, changed_cells = _compare_window(
            baseline, adjusted_frame, days, ROTATION_FEATURES,
        )
        result[f"{slot}_feature_rows"] = n
        result[f"{slot}_feature_changed_rows"] = changed_rows
        result[f"{slot}_feature_changed_cells"] = changed_cells
        n, changed_rows, changed_cells = _compare_window(
            baseline, adjusted_frame, days, TARGETS,
        )
        result[f"{slot}_label_rows"] = n
        result[f"{slot}_label_changed_rows"] = changed_rows
        result[f"{slot}_label_changed_cells"] = changed_cells
    after_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()
    if raw_fingerprint != after_fingerprint:
        raise RuntimeError(f"{symbol}: OHLCV RAW modificado na auditoria.")
    return result


def audit(*, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(historical, ())
    source = None
    snapshot = None
    for candidate in sorted(
        (root / "dados" / ".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_END}_*/universe_75"
        ), reverse=True,
    ):
        paths = SnapshotPaths.from_root(candidate)
        if paths.manifest.is_file():
            manifest = validate_snapshot(paths)
            if tuple(manifest.get("assets") or ()) == assets:
                source, snapshot = paths, manifest
                break
    if source is None or snapshot is None:
        raise FileNotFoundError(
            "Snapshot validado de 75 ativos nao encontrado; nao consultar Alpaca."
        )
    rows = []
    for i, symbol in enumerate(assets, 1):
        raw = load_raw_bar_file(
            source.raw_bars / f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        )
        events = load_actions_file(
            source.corporate_actions / f"{symbol}.csv"
        )
        splits = [
            event for event in events
            if event.get("action_type") in ("forward_split", "reverse_split")
        ]
        if not splits:
            continue
        features = construir_quadro_rotacao(raw, CONFIG)
        for event in splits:
            line = measure_event_boundary(
                symbol, raw, event, raw_features=features,
            )
            rows.append(line)
            if line["split_within_observed_bars"]:
                print(
                    f"[{i}/75] {symbol} {line['ex_date']}: "
                    f"pre_features={line['pre_event_feature_changed_cells']} "
                    f"pre_labels={line['pre_event_label_changed_cells']} "
                    f"post_features={line['post_event_feature_changed_cells']}",
                    flush=True,
                )
    result = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / "output" / "auditoria_janelas_splits" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    result.to_csv(folder / "eventos_e_janelas.csv", index=False)
    included = result.loc[result["split_within_observed_bars"]]
    summary = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "requested_assets": len(assets),
        "split_events_from_snapshot": len(rows),
        "splits_inside_price_coverage": len(included),
        "split_events_outside_price_coverage": len(rows) - len(included),
        "events_changing_pre_split_features": int((
            included["pre_event_feature_changed_cells"] > 0
        ).sum()),
        "events_changing_pre_split_labels": int((
            included["pre_event_label_changed_cells"] > 0
        ).sum()),
        "events_changing_post_split_features": int((
            included["post_event_feature_changed_cells"] > 0
        ).sum()),
        "events_changing_post_split_labels": int((
            included["post_event_label_changed_cells"] > 0
        ).sum()),
        "raw_csv_modified": False,
        "model_retrained": False,
        "capital_calculated": False,
        "limitation": (
            "Compara RAW com normalizacao antiga aplicada em memoria, "
            "isolando cada evento real; resultados nao provam conhecimento "
            "historico de anuncios, qualidade de previsao ou patrimonio. "
            "Ausencia de split accounting completo ainda impede replay RAW."
        ),
    }
    (folder / "resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    archive_path = folder / "pacote_janelas_splits.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(folder / "resumo.json", "resumo.json")
        zipf.write(folder / "eventos_e_janelas.csv", "eventos_e_janelas.csv")
    print(f"[split-window] resultado: {archive_path}", flush=True)
    return summary


if __name__ == "__main__":
    audit()
