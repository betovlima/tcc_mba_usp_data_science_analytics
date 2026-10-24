"""TCC v1.24.9 — identifica 17 dígitos vs menor decimal round-trip do RAW.

Somente leitura dos 2 CSVs históricos CAPR/YANG e Corporate Actions,
validados no contrato do único universo U75. Não baixar nem alterar dados.
Audita precisamente três pares da v1.24.8. Não retreina nem simula capital.

No Spyder (F5): output/auditoria_serializacao_float64/<ID>/
    pacote_auditoria_serializacao_float64.zip
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import zipfile

from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.universo_fixo75 import universo_75
from auditar_precisao_decimal_splits_spyder import (
    CASES, ROOT, SNAPSHOT_END, compare_exact_decimal_raw, file_sha256,
    read_csv_literal, raw_quote_literal,
)

VERSION = "1.24.9"


def probe_float_serialization(
    entry_literal: str, close_literal: str, quantity_factor: str,
) -> dict:
    """Usa decimal/Fraction apenas como diagnostico, nunca limpa/ajusta CSV."""
    entry = float(entry_literal)
    close = float(close_literal)
    factor = Fraction(quantity_factor)
    if not (entry > 0 and close > 0 and factor > 0):
        raise ValueError("Precos e multiplicadores devem ser positivos.")
    entry_text_17g = format(entry, ".17g")
    close_text_17g = format(close, ".17g")
    entry_shortest = repr(entry)
    close_shortest = repr(close)

    source_difference = (
        Fraction(Decimal(close_literal)) * factor
        - Fraction(Decimal(entry_literal))
    )
    binary_difference = Fraction(close) * factor - Fraction(entry)
    shortest_difference = (
        Fraction(Decimal(close_shortest)) * factor
        - Fraction(Decimal(entry_shortest))
    )

    return {
        "entry_literal": entry_literal,
        "close_literal": close_literal,
        "factor_fraction": f"{factor.numerator}/{factor.denominator}",
        "entry_reencoded_17g": entry_text_17g,
        "close_reencoded_17g": close_text_17g,
        "literal_matches_17g": (
            entry_literal == entry_text_17g
            and close_literal == close_text_17g
        ),
        "entry_shortest_round_trip_repr": entry_shortest,
        "close_shortest_round_trip_repr": close_shortest,
        "entry_float64_hex": entry.hex(),
        "close_float64_hex": close.hex(),
        "shortest_repr_same_float64": (
            float(entry_shortest) == entry and float(close_shortest) == close
        ),
        "stored_literal_delta_fraction": (
            f"{source_difference.numerator}/{source_difference.denominator}"
        ),
        "binary_float_delta_fraction": (
            f"{binary_difference.numerator}/{binary_difference.denominator}"
        ),
        "shortest_repr_delta_fraction": (
            f"{shortest_difference.numerator}/{shortest_difference.denominator}"
        ),
        "stored_literal_greater": source_difference > 0,
        "exact_binary_greater": binary_difference > 0,
        "shortest_repr_equal": shortest_difference == 0,
        "warning": (
            "repr(float) e a menor string decimal que recupera os mesmos "
            "bits. Isso nao prova a representacao textual original da API."
        ),
    }


def audit(*, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    contract = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(contract, ())
    paths = None
    snapshot = None
    for path in sorted(
        (root / "dados" / ".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_END}_*/universe_75"
        ), reverse=True,
    ):
        trial = SnapshotPaths.from_root(path)
        if trial.manifest.is_file():
            manifest = validate_snapshot(trial)
            if tuple(manifest.get("assets") or ()) == assets:
                paths, snapshot = trial, manifest
                break
    if paths is None or snapshot is None:
        raise FileNotFoundError("Snapshot U75 nao encontrado. Sem rede.")

    rows = []
    hashes = {}
    for symbol in sorted({case[0] for case in CASES}):
        raw_path = paths.raw_bars / f"{symbol}.csv"
        ca_path = paths.corporate_actions / f"{symbol}.csv"
        before = {
            "raw_sha256": file_sha256(raw_path),
            "events_sha256": file_sha256(ca_path),
        }
        raw = read_csv_literal(raw_path)
        events = read_csv_literal(ca_path)
        for sym, day, enter, exit_day in CASES:
            if sym != symbol:
                continue
            entry_text = raw_quote_literal(raw, enter, "open")
            close_text = raw_quote_literal(raw, exit_day, "close")
            exact = compare_exact_decimal_raw(
                entry_open_literal=entry_text,
                future_close_literal=close_text,
                entry_day=enter, future_day=exit_day,
                corporate_actions=events,
            )
            line = probe_float_serialization(
                entry_text, close_text,
                exact["quantity_multiplier_fraction"],
            )
            line.update({
                "symbol": sym, "decision_day": day,
                "entry_day": enter, "close_day": exit_day,
                "crossed_split_events": exact["events_effectively_crossed"],
            })
            if line["stored_literal_greater"] != (
                exact["exact_decimal_close_gt_entry"]
            ):
                raise RuntimeError("Conflito entre dois auditores racionais.")
            rows.append(line)
        after = {
            "raw_sha256": file_sha256(raw_path),
            "events_sha256": file_sha256(ca_path),
        }
        if before != after:
            raise RuntimeError(f"{symbol}: snapshot modificado!")
        hashes[symbol] = before
    if len(rows) != len(CASES):
        raise RuntimeError("A auditoria nao examinou as tres decisoes.")
    summary = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "universe_contract_verified": len(assets),
        "assets_literal_quotes_inspected": len(hashes),
        "pairs_compared": len(rows),
        "pairs_matching_17g_serialization": sum(
            item["literal_matches_17g"] for item in rows
        ),
        "pairs_with_shortest_repr_economic_tie": sum(
            item["shortest_repr_equal"] for item in rows
        ),
        "pairs_with_positive_literal_fraction": sum(
            item["stored_literal_greater"] for item in rows
        ),
        "raw_and_actions_hashes": hashes,
        "raw_modified": False,
        "model_retrained": False,
        "capital_calculated": False,
        "api_original_json_bytes_preserved": False,
        "limitations": (
            "Confirma a origem representacional das strings armazenadas "
            "mas nao estabelece qual decimal o payload original da Alpaca "
            "continha antes da conversao em float64. Nao reescrever CSVs."
        ),
    }
    output = (
        root / "output" / "auditoria_serializacao_float64"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "resultado.json").write_text(
        json.dumps({"summary": summary, "cases": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8",
    )
    archive = output / "pacote_auditoria_serializacao_float64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(output / "resultado.json", arcname="resultado.json")
    print(f"[float64-provenance] 17g={summary['pairs_matching_17g_serialization']}"
          f"/{len(rows)}; repr_ties={summary['pairs_with_shortest_repr_economic_tie']}"
          f"/{len(rows)}; output={archive}", flush=True)
    return summary


if __name__ == "__main__":
    audit()
