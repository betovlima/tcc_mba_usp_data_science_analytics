"""Congela os dados de mercado dos 24 candidatos já selecionados.

Objetivo
--------
Depois que a shortlist foi escolhida e auditada sem consultar resultados
financeiros, esta etapa baixa e congela os dados necessários para a análise
temporal de substituição.

O snapshot dos candidatos usa o mesmo contrato de dados do U67:
- Alpaca SIP;
- barras diárias;
- Adjustment.RAW;
- Corporate Actions;
- cutoff igual ao snapshot oficial U67.

O script NÃO:
- altera a shortlist;
- escolhe candidatos;
- executa backtest;
- consulta resultados de capital.

Execução no Spyder
------------------
Abra o arquivo e execute com F5.
Na primeira execução os 24 ativos são baixados da Alpaca.
Se o snapshot já existir e for válido, ele é apenas reutilizado.
"""

from __future__ import annotations

from pathlib import Path
import json

from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.preparacao import prepare_model_frames


ROOT = Path(__file__).resolve().parent
FROZEN_DIR = ROOT / "dados" / "cluster_candidate_search"
FREEZE_JSON = FROZEN_DIR / "cluster_shortlist_freeze.json"
FREEZE_CSV = FROZEN_DIR / "cluster_shortlist_frozen.csv"

U67 = SnapshotPaths.u67(ROOT)
CANDIDATES = SnapshotPaths.from_root(
    FROZEN_DIR / "snapshot"
)

SCRIPT_VERSION = "cluster-candidate-snapshot-v1.0.0"
EXPECTED_COUNT = 24
EXPECTED_FREEZE_VERSION = "cluster-shortlist-freeze-v1.0.0"

# Segurança: não sobrescreve um snapshot já válido.
FORCAR_RECRIACAO = False


def _load_frozen_contract() -> tuple[dict, tuple[str, ...]]:
    if not FREEZE_JSON.exists() or not FREEZE_CSV.exists():
        raise RuntimeError(
            "Shortlist congelada ausente. Execute primeiro "
            "congelar_shortlist_cluster_spyder.py."
        )

    payload = json.loads(
        FREEZE_JSON.read_text(encoding="utf-8")
    )
    if payload.get("script_version") != EXPECTED_FREEZE_VERSION:
        raise RuntimeError(
            "Versão inesperada do congelamento: "
            f"{payload.get('script_version')}"
        )
    if payload.get("status") != "frozen_before_substitution_analysis":
        raise RuntimeError(
            "A shortlist não está marcada como congelada."
        )
    if payload.get("selection_locked") is not True:
        raise RuntimeError(
            "selection_locked precisa ser True."
        )
    if payload.get("financial_outcomes_used_for_selection") is not False:
        raise RuntimeError(
            "A seleção não pode ter usado resultados financeiros."
        )
    if payload.get("backtest_executed_before_freeze") is not False:
        raise RuntimeError(
            "O congelamento deve ocorrer antes do backtest."
        )

    assets = tuple(
        str(asset).strip().upper()
        for asset in (payload.get("assets") or [])
    )
    if len(assets) != EXPECTED_COUNT or len(set(assets)) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Shortlist congelada inválida: {len(assets)} ativos."
        )

    return payload, assets


def _validate_existing(
    assets: tuple[str, ...],
    analysis_end: str,
) -> dict | None:
    if not CANDIDATES.manifest.exists():
        return None

    try:
        manifest = validate_snapshot(CANDIDATES)
    except Exception:
        if FORCAR_RECRIACAO:
            return None
        raise

    manifest_assets = tuple(
        str(asset).strip().upper()
        for asset in (manifest.get("assets") or [])
    )
    manifest_end = str(
        (manifest.get("bars") or {}).get(
            "bar_snapshot_as_of_end"
        )
        or ""
    )

    if manifest_assets != assets:
        if FORCAR_RECRIACAO:
            return None
        raise RuntimeError(
            "Snapshot existente pertence a outra shortlist."
        )
    if manifest_end != analysis_end:
        if FORCAR_RECRIACAO:
            return None
        raise RuntimeError(
            "Snapshot existente possui cutoff diferente do U67."
        )

    return manifest


def main() -> None:
    print("=" * 78)
    print("TCC - SNAPSHOT CONGELADO DOS 24 CANDIDATOS")
    print(f"version={SCRIPT_VERSION}")
    print("financial_analysis=NO")
    print("candidate_selection_change=NO")
    print("=" * 78)

    freeze, assets = _load_frozen_contract()

    u67_manifest = validate_snapshot(U67)
    analysis_end = str(
        (u67_manifest.get("bars") or {}).get(
            "bar_snapshot_as_of_end"
        )
        or ""
    )
    if not analysis_end:
        raise RuntimeError(
            "Cutoff do snapshot U67 não encontrado."
        )

    existing = _validate_existing(
        assets,
        analysis_end,
    )

    if existing is not None and not FORCAR_RECRIACAO:
        print(
            "[snapshot] existente e válido; download não será repetido."
        )
        manifest = existing
    else:
        CANDIDATES.clear_generated()
        credentials = load_alpaca_credentials(ROOT)

        raw_files = download_raw_bars(
            credentials,
            CANDIDATES,
            assets=assets,
            bar_snapshot_as_of_end=analysis_end,
            analysis_end_date=analysis_end,
        )
        action_files = download_corporate_actions(
            credentials,
            CANDIDATES,
            assets=assets,
            query_end=analysis_end,
        )
        manifest = build_snapshot_manifest(
            CANDIDATES,
            raw_files,
            action_files,
            bar_snapshot_as_of_end=analysis_end,
            analysis_end_date=analysis_end,
            assets=assets,
            snapshot_name="cluster_candidate_shortlist_24",
        )

    prepared, exclusions, diagnostics, audit = prepare_model_frames(
        CANDIDATES,
        assets=assets,
        csv_float_precision="round_trip",
    )

    prepared_assets = tuple(
        asset
        for asset in assets
        if asset in prepared
    )

    if exclusions:
        raise RuntimeError(
            "O snapshot congelado produziu exclusões estruturais. "
            "A shortlist precisa ser reavaliada antes da análise temporal. "
            + json.dumps(
                exclusions,
                ensure_ascii=False,
                default=str,
            )
        )

    if prepared_assets != assets:
        raise RuntimeError(
            "Nem todos os 24 candidatos sobreviveram à preparação. "
            f"observado={len(prepared_assets)} esperado={len(assets)}"
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "candidate_snapshot_ready",
        "selection_freeze_version": freeze.get("script_version"),
        "selection_count": len(assets),
        "assets": list(assets),
        "analysis_end": analysis_end,
        "snapshot_name": manifest.get("snapshot_name"),
        "snapshot_id": manifest.get("snapshot_sha256"),
        "prepared_count": len(prepared_assets),
        "structural_exclusions": exclusions,
        "data_audit": audit,
        "financial_analysis_performed": False,
        "backtest_executed": False,
        "next_step": (
            "build candidate-incumbent-decision-date substitution events"
        ),
    }

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    (
        FROZEN_DIR / "cluster_candidate_snapshot.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"[done] assets={len(prepared_assets)} "
        f"analysis_end={analysis_end}"
    )
    print(
        "[done] financial_analysis_performed=False "
        "backtest_executed=False"
    )
    print(f"[done] snapshot={CANDIDATES.root}")


if __name__ == "__main__":
    main()
