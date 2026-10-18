"""Congela a shortlist da busca por agrupamento após auditoria de continuidade.

Pré-condições
-------------
- pesquisar_candidatos_por_agrupamento_spyder.py executado;
- auditar_continuidade_shortlist_cluster_spyder.py executado;
- 24 candidatos;
- review_required = 0;
- excluded = 0;
- fingerprints da busca e da auditoria coincidentes.

Este script NÃO:
- executa backtest;
- consulta capital;
- retreina modelo;
- altera a ordem da shortlist.

Execução no Spyder
------------------
Abra o arquivo e execute com F5.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output" / "candidate_cluster_search"
SEARCH_JSON = SOURCE / "candidate_cluster_search.json"
SHORTLIST_CSV = SOURCE / "candidatos_cluster_shortlist.csv"
AUDIT_DIR = SOURCE / "continuity_audit"
AUDIT_JSON = AUDIT_DIR / "shortlist_continuity_audit.json"
AUDIT_CSV = AUDIT_DIR / "shortlist_continuity_audit.csv"

DEST = ROOT / "dados" / "cluster_candidate_search"
FROZEN_CSV = DEST / "cluster_shortlist_frozen.csv"
FROZEN_AUDIT_CSV = DEST / "cluster_shortlist_continuity_frozen.csv"
FROZEN_JSON = DEST / "cluster_shortlist_freeze.json"

SCRIPT_VERSION = "cluster-shortlist-freeze-v1.0.0"
EXPECTED_SEARCH_VERSION = "candidate-cluster-search-v1.0.6"
EXPECTED_AUDIT_VERSION = "candidate-cluster-continuity-audit-v1.0.5"
EXPECTED_COUNT = 24


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Arquivo ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    search = _read_json(SEARCH_JSON)
    audit = _read_json(AUDIT_JSON)

    if search.get("script_version") != EXPECTED_SEARCH_VERSION:
        raise RuntimeError(
            "Versão da busca inesperada: "
            f"{search.get('script_version')} "
            f"esperado={EXPECTED_SEARCH_VERSION}"
        )
    if audit.get("script_version") != EXPECTED_AUDIT_VERSION:
        raise RuntimeError(
            "Versão da auditoria inesperada: "
            f"{audit.get('script_version')} "
            f"esperado={EXPECTED_AUDIT_VERSION}"
        )

    if int(audit.get("shortlist_count") or 0) != EXPECTED_COUNT:
        raise RuntimeError("A shortlist auditada não contém 24 candidatos.")
    if int(audit.get("review_required") or 0) != 0:
        raise RuntimeError("Há candidatos pendentes de revisão.")
    if int(audit.get("excluded") or 0) != 0:
        raise RuntimeError("Há candidatos excluídos na shortlist auditada.")
    if int(audit.get("eligible_for_next_stage") or 0) != EXPECTED_COUNT:
        raise RuntimeError("Nem todos os 24 candidatos estão elegíveis.")

    search_fp = str(search.get("shortlist_fingerprint") or "")
    audit_fp = str(audit.get("source_shortlist_fingerprint") or "")
    if not search_fp or search_fp != audit_fp:
        raise RuntimeError(
            "Fingerprint da busca não coincide com a auditoria."
        )

    shortlist = pd.read_csv(SHORTLIST_CSV)
    audit_table = pd.read_csv(AUDIT_CSV)

    shortlist_assets = tuple(
        shortlist["asset"].astype(str).str.strip().str.upper()
    )
    audit_assets = tuple(
        str(x)
        for x in audit.get("source_shortlist_assets") or []
    )
    if shortlist_assets != audit_assets:
        raise RuntimeError(
            "Lista/ordem da shortlist diverge da auditoria."
        )

    if shortlist["asset"].duplicated().any():
        raise RuntimeError("Shortlist contém ativos duplicados.")

    statuses = audit_table.set_index("asset")["status"].to_dict()
    invalid = [
        symbol
        for symbol in shortlist_assets
        if statuses.get(symbol) != "eligible_for_next_stage"
    ]
    if invalid:
        raise RuntimeError(
            "Ativos sem elegibilidade final: " + ",".join(invalid)
        )

    DEST.mkdir(parents=True, exist_ok=True)

    shortlist.to_csv(
        FROZEN_CSV,
        index=False,
        float_format="%.17g",
    )
    audit_table.to_csv(
        FROZEN_AUDIT_CSV,
        index=False,
        float_format="%.17g",
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "status": "frozen_before_substitution_analysis",
        "search_version": search.get("script_version"),
        "search_commit": search.get("commit"),
        "audit_version": audit.get("script_version"),
        "shortlist_fingerprint": search_fp,
        "count": len(shortlist_assets),
        "assets": list(shortlist_assets),
        "similar_assets": list(
            shortlist.loc[
                shortlist["search_role"].eq("semelhante"),
                "asset",
            ].astype(str)
        ),
        "complementary_assets": list(
            shortlist.loc[
                shortlist["search_role"].eq("complementar"),
                "asset",
            ].astype(str)
        ),
        "financial_outcomes_used_for_selection": False,
        "backtest_executed_before_freeze": False,
        "selection_locked": True,
        "next_step": (
            "build substitution-event dataset using only information "
            "available at each decision date"
        ),
    }
    FROZEN_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("TCC - SHORTLIST DE CANDIDATOS CONGELADA")
    print(f"version={SCRIPT_VERSION}")
    print(f"count={len(shortlist_assets)}")
    print(f"fingerprint={search_fp}")
    print("assets=" + ",".join(shortlist_assets))
    print("financial_outcomes_used_for_selection=False")
    print("backtest_executed_before_freeze=False")
    print("=" * 78)
    print(f"Saídas: {DEST}")


if __name__ == "__main__":
    main()
