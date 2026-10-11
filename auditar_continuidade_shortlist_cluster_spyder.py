"""Auditoria de continuidade da shortlist da busca por agrupamento.

Objetivo
--------
Antes de qualquer teste financeiro ou de substituição, verificar se os ativos
selecionados pela busca comportamental apresentam eventos corporativos que
exijam exclusão ou revisão manual.

A política desta pesquisa é conservadora:
- problemas estruturais de identidade ou continuidade não são reconstruídos;
- não se faz bridge entre emissores ou séries;
- o ativo é excluído e o motivo fica registrado.

Este script NÃO:
- treina modelos;
- executa backtest;
- mede capital;
- altera a shortlist original.

Execução no Spyder
------------------
Abra este arquivo e execute com F5.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from reproducao.dados import (
    SnapshotPaths,
    download_corporate_actions,
    load_alpaca_credentials,
)
from reproducao.preparacao import (
    load_actions_file,
    structural_identity_issue,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output" / "candidate_cluster_search"
SHORTLIST_FILE = SOURCE / "candidatos_cluster_shortlist.csv"
OUT = SOURCE / "continuity_audit"
ACTIONS = SnapshotPaths.from_root(OUT)

SCRIPT_VERSION = "candidate-cluster-continuity-audit-v1.0.0"
EXPECTED_BRANCH = "research/universe-clustering-visualization-v1"

KNOWN_RESEARCH_STRUCTURAL_EXCLUSIONS = {
    "FLG": "historical_identity_transition",
    "LBTYA": "historical_identity_transition",
    "DCOY": "historical_identity_transition",
    "BLOX": "historical_series_discontinuity",
    "VISN": "historical_identity_transition",
    "COR": "historical_identity_transition",
}

IDENTITY_SENSITIVE_ACTIONS = {
    "stock_merger",
    "stock_and_cash_merger",
    "cash_merger",
    "spin_off",
    "redemption",
    "name_change",
    "worthless_removal",
}


def _git(args: list[str]) -> str:
    import subprocess

    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return process.stdout.strip()


def _clean(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main() -> None:
    try:
        branch = _git(["branch", "--show-current"])
        commit = _git(["rev-parse", "--short", "HEAD"])
    except Exception:
        branch = "sem-git"
        commit = "sem-git"

    print("=" * 78)
    print("TCC - AUDITORIA DE CONTINUIDADE DA SHORTLIST")
    print(f"version={SCRIPT_VERSION}")
    print(f"branch={branch}")
    print(f"commit={commit}")
    print("=" * 78)

    if branch not in (EXPECTED_BRANCH, "sem-git"):
        raise RuntimeError(
            f"Branch incorreta: esperado={EXPECTED_BRANCH} observado={branch}"
        )

    if not SHORTLIST_FILE.exists():
        raise RuntimeError(
            "Shortlist ausente. Execute primeiro "
            "pesquisar_candidatos_por_agrupamento_spyder.py."
        )

    shortlist = pd.read_csv(SHORTLIST_FILE)
    shortlist["asset"] = (
        shortlist["asset"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    assets = tuple(shortlist["asset"].drop_duplicates())

    OUT.mkdir(parents=True, exist_ok=True)
    ACTIONS.ensure()

    credentials = load_alpaca_credentials(ROOT)
    download_corporate_actions(
        credentials,
        ACTIONS,
        assets=assets,
        query_end="2026-10-06",
    )

    rows = []
    event_rows = []

    for symbol in assets:
        actions = load_actions_file(
            ACTIONS.corporate_actions / f"{symbol}.csv"
        )

        for action in actions:
            event_rows.append(
                {
                    "asset": symbol,
                    **{key: _clean(value) for key, value in action.items()},
                }
            )

        known_reason = KNOWN_RESEARCH_STRUCTURAL_EXCLUSIONS.get(symbol)
        detected = structural_identity_issue(symbol, actions)

        sensitive = [
            action
            for action in actions
            if str(action.get("action_type") or "")
            in IDENTITY_SENSITIVE_ACTIONS
        ]

        if known_reason:
            status = "excluded"
            reason = known_reason
        elif detected is not None:
            status = "excluded"
            reason = str(
                detected.get("reason")
                or "structural_identity_change"
            )
        elif sensitive:
            status = "review_required"
            reason = "identity_sensitive_corporate_action"
        else:
            status = "eligible_for_next_stage"
            reason = ""

        rows.append(
            {
                "asset": symbol,
                "search_role": shortlist.loc[
                    shortlist["asset"].eq(symbol),
                    "search_role",
                ].iloc[0],
                "nearest_u67_asset": shortlist.loc[
                    shortlist["asset"].eq(symbol),
                    "nearest_u67_asset",
                ].iloc[0],
                "novelty_ratio": float(
                    shortlist.loc[
                        shortlist["asset"].eq(symbol),
                        "novelty_ratio",
                    ].iloc[0]
                ),
                "corporate_actions": len(actions),
                "identity_sensitive_actions": len(sensitive),
                "status": status,
                "reason": reason,
                "known_historical_exclusion": bool(known_reason),
                "detected_structural_issue": bool(
                    detected is not None
                ),
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["status", "search_role", "asset"]
    )
    events = pd.DataFrame(event_rows)

    audit.to_csv(
        OUT / "shortlist_continuity_audit.csv",
        index=False,
        float_format="%.17g",
    )
    events.to_csv(
        OUT / "shortlist_corporate_actions.csv",
        index=False,
    )

    eligible = audit.loc[
        audit["status"].eq("eligible_for_next_stage")
    ].copy()
    eligible.to_csv(
        OUT / "shortlist_eligible_after_continuity.csv",
        index=False,
        float_format="%.17g",
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "shortlist_count": int(len(audit)),
        "eligible_for_next_stage": int(
            audit["status"].eq("eligible_for_next_stage").sum()
        ),
        "review_required": int(
            audit["status"].eq("review_required").sum()
        ),
        "excluded": int(
            audit["status"].eq("excluded").sum()
        ),
        "financial_outcomes_used": False,
        "backtest_executed": False,
        "next_step": (
            "manual review of review_required rows; only then build the "
            "substitution-event dataset"
        ),
    }
    (OUT / "shortlist_continuity_audit.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nResumo:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nAuditoria:")
    print(
        audit[
            [
                "asset",
                "search_role",
                "nearest_u67_asset",
                "novelty_ratio",
                "identity_sensitive_actions",
                "status",
                "reason",
            ]
        ].to_string(index=False)
    )

    if (audit["status"] == "review_required").any():
        print(
            "\nHá ativos em review_required. Não avance para substituição "
            "até revisar esses eventos corporativos."
        )
    else:
        print(
            "\nNenhum ativo ficou em revisão manual. A lista elegível pode "
            "avançar para a construção do dataset de substituições."
        )

    print(f"\nSaídas: {OUT}")


if __name__ == "__main__":
    main()
