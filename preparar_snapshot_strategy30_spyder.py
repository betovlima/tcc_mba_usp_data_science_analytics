"""Coleta reprodutivel dos 75 CSVs Alpaca RAW/SIP para paridade Strategy #30.

Nao usa banco, nao atualiza arquivos do snapshot antigo. Nao sobrescreve
snapshot existente: um novo download deve ter identidade independente.
O novo download pode divergir do snapshot de 29/10/2026 por revisoes
retroativas da fonte; o teste de paridade verificara essa possibilidade.
"""
from __future__ import annotations

from pathlib import Path

from pesquisa.control_operacional_strategy30 import ROOT, carregar_contrato
from reproducao.dados import (
    SnapshotPaths, load_alpaca_credentials,
    download_raw_bars, download_corporate_actions,
    build_snapshot_manifest, validate_snapshot,
)

SNAPSHOT = SnapshotPaths.from_root(ROOT / "dados" / "strategy30_snapshot")
SOURCE_CUTOFF = "2026-10-28"


def main() -> dict:
    contract = carregar_contrato()
    if contract["last_session"] != SOURCE_CUTOFF:
        raise RuntimeError("O cutoff informado diverge da referencia congelada.")
    assets = tuple(contract["assets"])
    if len(assets) != 75:
        raise RuntimeError("Universo oficial nao tem 75 ativos.")
    if SNAPSHOT.manifest.is_file():
        print("[snapshot] reutilizando snapshot local versionado; sem download", flush=True)
        return validate_snapshot(SNAPSHOT)
    if any(SNAPSHOT.raw_bars.glob("*.csv")) or any(
        SNAPSHOT.corporate_actions.glob("*.csv")
    ):
        raise RuntimeError(
            "Pasta de snapshot parcial encontrada. Para evitar mistura "
            "de execucoes, mova ou preserve o diretorio atual e crie "
            "um novo diretorio de snapshot vazio."
        )
    credentials = load_alpaca_credentials(ROOT)
    SNAPSHOT.ensure()
    raw = download_raw_bars(
        credentials, SNAPSHOT, assets=assets,
        bar_snapshot_as_of_end=SOURCE_CUTOFF,
        analysis_end_date=SOURCE_CUTOFF,
    )
    actions = download_corporate_actions(
        credentials, SNAPSHOT, assets=assets,
        query_end=SOURCE_CUTOFF,
    )
    manifest = build_snapshot_manifest(
        SNAPSHOT, raw, actions,
        bar_snapshot_as_of_end=SOURCE_CUTOFF,
        analysis_end_date=SOURCE_CUTOFF,
        assets=assets,
        snapshot_name="strategy30_75_raw_sip_20261028_research",
    )
    checked = validate_snapshot(SNAPSHOT)
    if checked["snapshot_sha256"] != manifest["snapshot_sha256"]:
        raise RuntimeError("Integridade do snapshot falhou apos coleta.")
    print(
        f"[snapshot] 75 CSVs RAW/SIP, corporate actions e manifesto: {SNAPSHOT.root}",
        flush=True,
    )
    return checked


if __name__ == "__main__":
    main()
