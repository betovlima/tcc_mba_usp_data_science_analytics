"""Consolida ate 24 replays U75+1 ja realizados, sem refazer backtest.

Executar no Spyder com F5 apos salvar os pacotes ZIP em:
output/contribuicao_marginal_externa_u75/
Nenhum modelo e treinado, nenhum dado de mercado e baixado.
O fechamento e somente da campanha exploratoria de 24 ativos.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from pesquisa.control_operacional_strategy30 import ROOT

VERSION = "u75-external-marginal-consolidation-v1.0.0"
SOURCE_DIR = ROOT / "output" / "contribuicao_marginal_externa_u75"
TARGET_DIR = ROOT / "output" / "consolidacao_externa_u75"
PATTERN = "pacote_contribuicao_marginal_externa_u75_*.zip"
SHARED = (
    "auditoria_paridade.json", "auditoria_paridade_diaria.csv",
    "control_predictions.csv", "baseline_metricas.json",
    "comparacao_capital.csv", "comparacao_por_fold.csv",
    "contrato_e_lote.json", "resultado.json", "resumo.md",
    "candidatos_reprovados.json",
)


def _read_package(path: Path):
    """Confere TODAS as entradas com o manifesto SHA-256 antes de aceitar dados."""
    raw = path.read_bytes()
    with ZipFile(BytesIO(raw)) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"{path.name}: ZIP corrompido")
        names = z.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"{path.name}: nomes duplicados")
        manifest = json.loads(z.read("manifesto_pacote.json"))
        if manifest.get("experiment") != "u75-external-marginal-capital":
            raise RuntimeError(f"{path.name}: experimento diferente")
        entries = manifest.get("files") or []
        if {row["name"] for row in entries} != set(names) - {"manifesto_pacote.json"}:
            raise RuntimeError(f"{path.name}: manifesto nao cobre os arquivos do ZIP")
        for item in entries:
            data = z.read(item["name"])
            if (sha256(data).hexdigest() != item["sha256"]
                    or len(data) != int(item["size"])):
                raise RuntimeError(f"{path.name}: hash/tamanho divergente: {item['name']}")
        result = json.loads(z.read("resultado.json"))
        if result.get("status") not in {"completed", "completed_with_exclusions"}:
            raise RuntimeError(f"{path.name}: execucao nao foi concluida")
        if not result.get("parity", {}).get("fully_reproduced"):
            raise RuntimeError(f"{path.name}: Control sem paridade comprovada no JSON")
        return result, set(names), sha256(raw).hexdigest()


def _evidence_gaps(files: set[str], symbol: str) -> list[str]:
    required = [*SHARED,
                f"decisoes_{symbol}.csv",
                f"trades_{symbol}.csv",
                f"identidade_{symbol}.json"]
    return [n for n in required if n not in files]


def _assert_equal_repeat(a: dict, b: dict) -> None:
    """Reexecucoes nao podem ser escolhidas pelo melhor patrimonio."""
    symbol = a["symbol"]
    keys = (
        "sessions", "changed_positions", "direct_candidate_days",
        "indirect_changed_days", "candidate_raw_sha256",
        "candidate_actions_sha256", "calendar_source",
    )
    if any(a.get(k) != b.get(k) for k in keys):
        raise RuntimeError(f"{symbol}: execucoes duplicadas com dados/decisoes diferentes")
    numeric = (
        "control_ending_capital", "challenger_ending_capital",
        "delta_ending_capital", "delta_compound_log_growth",
        "strategy_sharpe", "strategy_maximum_drawdown",
        "capital_rotations", "total_transaction_fees",
    )
    for key in numeric:
        if abs(float(a[key]) - float(b[key])) > (0.01 if "capital" in key else 1e-8):
            raise RuntimeError(f"{symbol}: resultados duplicados divergentes em {key}")


def consolidar(pacotes: list[Path]) -> dict:
    if not pacotes:
        raise RuntimeError("Nenhum ZIP encontrado na pasta de resultados")
    expected = None
    source_identity = None
    control_hash = None
    ending_capital = None
    records: dict[str, dict] = {}
    failures: dict[str, list[str]] = {}
    sources = []
    for path in sorted(pacotes):
        result, files, digest = _read_package(path)
        source = result["source"]
        assets = tuple(sorted(source["external_candidates"]))
        identity = (source["candidate_snapshot"], source["reference_snapshot"])
        if expected is None:
            expected = assets
            source_identity = identity
            control_hash = source["control_zip_sha256"]
            ending_capital = float(result["baseline_metrics"]["strategy_ending_capital"])
            if len(expected) != 24 or len(set(expected)) != 24:
                raise RuntimeError("Contrato de shortlist nao possui 24 candidatos unicos")
        elif (expected != assets or source_identity != identity
              or control_hash != source["control_zip_sha256"]
              or abs(ending_capital - float(result["baseline_metrics"]["strategy_ending_capital"])) > 0.01):
            raise RuntimeError(f"{path.name}: snapshots ou Control nao coincidem")
        listed = tuple(sorted(result["batch"]["symbols"]))
        delivered = tuple(sorted(r["symbol"] for r in result["candidate_results"]))
        errors = tuple(sorted(r["symbol"] for r in result.get("candidate_errors", [])))
        if len(listed) != len(set(listed)) or set(listed) != set(delivered) | set(errors):
            raise RuntimeError(f"{path.name}: lote e resultados nao correspondem")
        if not set(listed).issubset(set(expected)):
            raise RuntimeError(f"{path.name}: candidato fora da shortlist")
        for err in result.get("candidate_errors", []):
            failures.setdefault(err["symbol"], []).append(str(err["error"]))
        for row in result["candidate_results"]:
            sym = row["symbol"]
            missing = _evidence_gaps(files, sym)
            item = {
                "result": dict(row), "missing_evidence": missing,
                "package": path.name, "package_sha256": digest,
            }
            if sym in records:
                prior = records[sym]
                _assert_equal_repeat(prior["result"], item["result"])
                if len(missing) < len(prior["missing_evidence"]):
                    records[sym] = item
            else:
                records[sym] = item
        sources.append({
            "file": path.name, "sha256": digest,
            "symbols": list(listed), "result_count": len(delivered),
            "errors": list(errors),
        })
    assert expected is not None
    missing_symbols = sorted(set(expected) - set(records))
    incomplete = {k: v["missing_evidence"] for k, v in records.items() if v["missing_evidence"]}
    unresolved_failures = {k: v for k, v in failures.items() if k not in records}
    rows = []
    for sym, value in records.items():
        r = dict(value["result"])
        r["package"] = value["package"]
        r["evidence_complete"] = not bool(value["missing_evidence"])
        r["missing_evidence"] = "; ".join(value["missing_evidence"])
        rows.append(r)
    rows.sort(key=lambda r: (-float(r["delta_compound_log_growth"]), r["symbol"]))
    status = ("completed_exploratory" if (not missing_symbols and not incomplete
              and not unresolved_failures) else "incomplete")
    return {
        "schema_version": 1,
        "version": VERSION,
        "status": status,
        "scope": "24_candidatos_retrospetivos_u75_plus_1",
        "prospective_discovery_validated": False,
        "baseline_ending_capital": ending_capital,
        "control_zip_sha256": control_hash,
        "snapshots": {"reference": source_identity[1], "external": source_identity[0]},
        "expected_candidates": list(expected),
        "measured_candidates": len(records),
        "missing_candidates": missing_symbols,
        "incomplete_evidence": incomplete,
        "failed_candidates": unresolved_failures,
        "historical_failed_attempts": failures,
        "packages": sources,
        "results": rows,
    }


def main() -> dict:
    files = sorted(SOURCE_DIR.glob(PATTERN))
    result = consolidar(files)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = TARGET_DIR / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "resultado_consolidado.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    df = pd.DataFrame(result["results"])
    if not df.empty:
        df.to_csv(out / "ranking_capital_exploratorio.csv", index=False, float_format="%.17g")
    lines = [
        "# Fechamento da campanha exploratoria U75 + 1",
        "",
        f"Estado: {result['status']}",
        f"Patrimonio Control: US$ {result['baseline_ending_capital']:,.2f}",
        f"Candidatos concluidos: {result['measured_candidates']}/24",
        f"Candidatos ausentes: {result['missing_candidates']}",
        f"Candidatos com erros: {list(result['failed_candidates'])}",
        f"Resultados com evidencias incompletas: {list(result['incomplete_evidence'])}",
        "",
        "| Ativo | Patrimonio (USD) | Diferenca (USD) | Dias alterados | Comprado | Evidencias |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result["results"]:
        lines.append(
            f"| {row['symbol']} | {row['challenger_ending_capital']:,.2f} | "
            f"{row['delta_ending_capital']:+,.2f} | {row['changed_positions']} | "
            f"{row['direct_candidate_days']} | "
            f"{'completas' if row['evidence_complete'] else 'incompletas'} |"
        )
    lines.extend([
        "", "## Limites cientificos", "",
        "Esta pesquisa adicionou um candidato por vez ao U75 e refez a trajetoria completa.",
        "A shortlist foi escolhida retrospectivamente, logo nao comprova "
        "descoberta de ativos em tempo real nem superioridade futura.",
        "O Control possui limitacao herdada de elegibilidade t+1. "
        "Nenhuma regra CASH foi alterada. Nenhum modelo foi promovido ao MCT.",
        "Se alguma evidencia estiver incompleta, a campanha nao pode ser "
        "declarada auditada integralmente. Reexecucoes divergentes sao reprovadas.",
        "",
    ])
    (out / "resumo_final.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[consolidacao] {result['status']} {result['measured_candidates']}/24", flush=True)
    print(f"[relatorio] {out}", flush=True)
    return result


if __name__ == "__main__":
    main()
