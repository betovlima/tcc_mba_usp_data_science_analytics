"""Persistencia dos artefatos finais da reproducao."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.config import EXPERIMENT_VERSION


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_results(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    exclusions: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    audit: dict[str, Any],
    folds: list[dict[str, Any]],
    control_result: Any,
    control_metrics: dict[str, Any],
    soft_result: Any,
    soft_metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_dir / "summary.json",
        "comparison": output_dir / "comparison.csv",
        "control_predictions": output_dir / "control_predictions.csv",
        "control_trades": output_dir / "control_trades.csv",
        "soft_predictions": output_dir / "soft_horizon_consensus_predictions.csv",
        "soft_trades": output_dir / "soft_horizon_consensus_trades.csv",
        "folds": output_dir / "folds.csv",
        "data_diagnostics": output_dir / "data_diagnostics.csv",
        "data_audit": output_dir / "data_audit.json",
        "exclusions": output_dir / "structural_exclusions.csv",
        "summary_text": output_dir / "summary.txt",
    }

    control_result.predictions.to_csv(paths["control_predictions"], index=True)
    control_result.trades.to_csv(paths["control_trades"], index=False)
    soft_result.predictions.to_csv(paths["soft_predictions"], index=True)
    soft_result.trades.to_csv(paths["soft_trades"], index=False)

    pd.DataFrame(diagnostics).to_csv(paths["data_diagnostics"], index=False)
    pd.DataFrame(exclusions).to_csv(paths["exclusions"], index=False)
    pd.DataFrame([comparison]).to_csv(paths["comparison"], index=False)

    fold_rows: list[dict[str, Any]] = []
    for label, metrics in (("CONTROL", control_metrics), ("SOFT", soft_metrics)):
        for row in metrics.get("folds") or []:
            item = dict(row)
            item["variant"] = label
            fold_rows.append(item)
    pd.DataFrame(fold_rows).to_csv(paths["folds"], index=False)

    paths["data_audit"].write_text(
        json.dumps(audit, indent=2, default=_json_default),
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "tcc-cpu-control-vs-soft-horizon-consensus",
        "database_access": False,
        "data_transport": "alpaca_raw_sip_to_csv_per_asset",
        "snapshot_sha256": manifest.get("snapshot_sha256"),
        "control": control_metrics,
        "soft_horizon_consensus": soft_metrics,
        "comparison": comparison,
        "excluded_assets": exclusions,
        "data_audit": audit,
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    ratio = comparison.get("soft_vs_control_ratio")
    ratio_text = f"{float(ratio):+.2%}" if ratio is not None else "n/a"
    text = "\n".join(
        [
            f"TCC MBA USP - reproducao v{EXPERIMENT_VERSION}",
            "Backend: CPU",
            f"Snapshot: {manifest.get('snapshot_sha256')}",
            "",
            f"Control: US$ {float(control_metrics['ending_capital']):,.2f}",
            f"Soft:    US$ {float(soft_metrics['ending_capital']):,.2f}",
            (
                "Soft - Control: "
                f"US$ {float(comparison['soft_minus_control_capital']):,.2f} "
                f"({ratio_text})"
            ),
            (
                "Soft changed base actions: "
                f"{int(comparison['soft_changed_base_actions'])}"
            ),
            "",
            f"Requested device: {comparison.get('requested_compute_device')}",
            f"Effective device: {comparison.get('effective_compute_device')}",
        ]
    )
    paths["summary_text"].write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"[output] dir={output_dir}", flush=True)
    return paths
