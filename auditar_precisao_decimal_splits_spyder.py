"""TCC v1.24.8 — origem decimal literal de tres empates float64.

Somente leitura de CSVs RAW, usando Decimal dos TEXTOS originais de precos
e taxas de split. Nao altera RAW, nao corrige empiricamente a regra `>`,
nao treina nem calcula capital. F5 no Spyder.

Investiga:
  YANG, decisao 2024-10-16, entrada 2024-10-17, fechamento 2024-11-19
  CAPR, decisao 2019-05-20, entrada 2019-05-21, fechamento 2019-06-07
  CAPR, decisao 2019-05-28, entrada 2019-05-29, fechamento 2019-08-12
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.universo_fixo75 import universo_75

ROOT = Path(__file__).resolve().parent
VERSION = "1.24.8"
SNAPSHOT_END = "2026-10-22"
CASES = (
    ("YANG", "2024-10-16", "2024-10-17", "2024-11-19"),
    ("CAPR", "2019-05-20", "2019-05-21", "2019-06-07"),
    ("CAPR", "2019-05-28", "2019-05-29", "2019-08-12"),
)


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc_day(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Data invalida em CSV RAW.")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").date().isoformat()


def read_csv_literal(path: Path) -> list[dict[str, str]]:
    """CSV sem convertê-lo para números: não limpar, deduplicar ou ordenar."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV ausente/vazio: {path}")
    return rows


def raw_quote_literal(
    rows: list[dict[str, str]], date: str, field: str,
) -> str:
    if field not in ("open", "high", "low", "close", "volume"):
        raise ValueError("Campo desconhecido.")
    matching = [
        row for row in rows
        if utc_day(row["timestamp"]) == date
    ]
    if len(matching) != 1:
        raise RuntimeError(f"{date}: barra ausente ou duplicada no CSV RAW.")
    value = matching[0][field]
    if value is None or not str(value).strip():
        raise ValueError(f"{date}/{field}: valor RAW ausente.")
    return value


def decimal_rate(value: str, *, field: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{field}: decimal invalido.") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field}: esperado decimal positivo.")
    return result


def compare_exact_decimal_raw(
    *,
    entry_open_literal: str,
    future_close_literal: str,
    entry_day: str,
    future_day: str,
    corporate_actions: list[dict[str, str]],
) -> dict:
    """Comparacao racional de cotacoes DECIMAIS sem criar precos substitutos.

    O valor de fechamento × fator de quantidade vive APENAS no relatorio.
    O texto RAW permanece intacto. Não supõe tick mínimo ou arredondamento.
    """
    if entry_day > future_day:
        raise ValueError("A entrada deve anteced    # Fraction(Decimal(texto)) evita inclusive erro por divisao decimal
    # periodica em splits como 1 para 3. O sinal é racional EXATO.
    entry = Fraction(decimal_rate(entry_open_literal, field="open"))
    close = Fraction(decimal_rate(future_close_literal, field="close"))
    factor = Fraction(1)
    effects = []
    for event in corporate_actions:
        kind = str(event.get("action_type") or "")
        if kind not in ("forward_split", "reverse_split"):
            continue
        ex_date = utc_day(event["ex_date"])
        if not (entry_day < ex_date <= future_day):
            continue
        numerator = Fraction(decimal_rate(event["new_rate"], field="new_rate"))
        denominator = Fraction(decimal_rate(event["old_rate"], field="old_rate"))
        ratio = numerator / denominator
        factor *= ratio
        effects.append({
            "ex_date": ex_date, "action_type": kind,
            "old_rate_literal": event["old_rate"],
            "new_rate_literal": event["new_rate"],
            "quantity_ratio_fraction": f"{ratio.numerator}/{ratio.denominator}",
        })

    quote_per_original_share = close * factor
    difference = quote_per_original_share - entry
    with localcontext() as ctx:
        ctx.prec = 80
        approx_factor = Decimal(factor.numerator) / Decimal(factor.denominator)
        approx_quote = (
            Decimal(quote_per_original_share.numerator)
            / Decimal(quote_per_original_share.denominator)
        )
        approx_delta = (
            Decimal(difference.numerator) / Decimal(difference.denominator)
        )
    return {
        "entry_open_literal": entry_open_literal,
        "future_close_literal": future_close_literal,
        "entry_decimal": str(Decimal(entry_open_literal)),
        "future_close_decimal": str(Decimal(future_close_literal)),
        "quantity_multiplier_fraction": f"{factor.numerator}/{factor.denominator}",
        "quantity_multiplier_decimal_approx": str(approx_factor),
        "future_value_decimal_approx": str(approx_quote),
        "difference_decimal_approx": str(approx_delta),
        "exact_difference_numerator": str(difference.numerator),
        "exact_difference_denominator": str(difference.denominator),
        "exact_decimal_equal": bool(difference == 0),
        "exact_decimal_close_gt_entry": bool(difference > 0),
        "events_effectively_crossed": len(effects),
        "event_details": effects,
        "source_precision": (
            "Texto do CSV preservado e proporções de split como "
            "frações racionais exatas; decimais de saída são apenas "
            "representações aproximadas, nunca critério de decisão."
        ),
    }rio."
            ),
        }


def audit(*, root: Path=ROOT) -> dict:
    root = Path(root).resolve()
    manifest_contract = json.loads(
        (root/"evidencias"/"manifest_universo_congelado.json").read_text(
            encoding="utf-8",
        )
    )
    universe = universo_75(manifest_contract, ())
    snapshot = None
    paths = None
    for candidate in sorted(
        (root/"dados"/".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_END}_*/universe_75"
        ), reverse=True,
    ):
        trial = SnapshotPaths.from_root(candidate)
        if trial.manifest.is_file():
            info = validate_snapshot(trial)
            if tuple(info.get("assets") or ()) == universe:
                snapshot, paths = info, trial
                break
    if paths is None or snapshot is None:
        raise FileNotFoundError("Snapshot validado U75 nao encontrado, sem downloads.")

    # Verificar o universo do snapshot mas ler preços apenas dos 2 ativos
    # envolvidos: não alegar que 75 arquivos foram reprocessados.
    findings = []
    evidence_sha = {}
    for symbol in sorted(set(item[0] for item in CASES)):
        raw_file = paths.raw_bars/f"{symbol}.csv"
        actions_file = paths.corporate_actions/f"{symbol}.csv"
        before = {
            "bars_sha256": file_sha256(raw_file),
            "actions_sha256": file_sha256(actions_file),
        }
        raw = read_csv_literal(raw_file)
        actions = read_csv_literal(actions_file)
        for sym, decision, entry_day, exit_day in CASES:
            if sym != symbol:
                continue
            result = compare_exact_decimal_raw(
                entry_open_literal=raw_quote_literal(raw, entry_day, "open"),
                future_close_literal=raw_quote_literal(raw, exit_day, "close"),
                entry_day=entry_day, future_day=exit_day,
                corporate_actions=actions,
            )
            result.update({
                "symbol": symbol, "decision_day": decision,
                "entry_day": entry_day, "divergent_close_day": exit_day,
            })
            findings.append(result)
        after = {
            "bars_sha256": file_sha256(raw_file),
            "actions_sha256": file_sha256(actions_file),
        }
        if before != after:
            raise RuntimeError(f"{symbol}: arquivos RAW/Corporate Actions modificados.")
        evidence_sha[symbol] = before

    if len(findings) != len(CASES):
        raise RuntimeError("Diagnostico nao cobriu exatamente as tres decisoes.")
    report = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "universe_contract_verified": len(universe),
        "symbols_with_quotes_inspected": 2,
        "decisions": len(findings),
        "exact_decimal_equal": sum(x["exact_decimal_equal"] for x in findings),
        "exact_decimal_gt": sum(x["exact_decimal_close_gt_entry"] for x in findings),
        "float64_prior_audit_found_one_ulp_mismatch_in_each_case": True,
        "quotes_source_file_sha256": evidence_sha,
        "raw_files_modified": False,
        "model_retrained": False,
        "decision_logic_changed": False,
        "capital_calculated": False,
        "caveat": (
            "Decimal examina o texto RAW armazenado no snapshot, nao "
            "uma copia historica da resposta original da API. O resultado "
            "pode distinguir empate literal de valor decimal ligeiramente "
            "diferente, mas nao valida politica de arredondamento de tick."
        ),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root/"output"/"decimais_raw_splits"/stamp
    output.mkdir(parents=True, exist_ok=False)
    (output/"resultado.json").write_text(
        json.dumps({"summary":report,"decisions":findings},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    archive = output/"pacote_decimais_raw_splits.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(output/"resultado.json", arcname="resultado.json")
    print(f"[decimal-raw] 3 decisoes, igualdade={report['exact_decimal_equal']} "
          f"maior={report['exact_decimal_gt']} arquivo={archive}", flush=True)
    return report


if __name__ == "__main__":
    audit()
