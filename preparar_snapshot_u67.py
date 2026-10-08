"""Prepara o snapshot congelado U67 usado pela reproducao oficial.

Este arquivo e executado somente quando for necessario criar ou substituir o
snapshot de pesquisa. A reproducao oficial nao acessa a Alpaca.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from engine.configuracao import (
    U67_EXPECTED_REQUESTED_COUNT,
    U67_REQUESTED_ASSETS,
)
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)


ROOT = Path(__file__).resolve().parent
BUILD = SnapshotPaths.from_root(ROOT / "dados" / ".u67_build")
TARGET = SnapshotPaths.u67(ROOT)

SNAPSHOT_END_DATE = "2026-10-06"
SNAPSHOT_NAME = "tcc-u67-frozen-20261006"


if len(U67_REQUESTED_ASSETS) != U67_EXPECTED_REQUESTED_COUNT:
    raise RuntimeError(
        "Contrato U67 invalido antes da coleta: "
        f"esperado={U67_EXPECTED_REQUESTED_COUNT} "
        f"observado={len(U67_REQUESTED_ASSETS)}"
    )


# %% 1 - Coleta completa em area temporaria
credentials = load_alpaca_credentials(ROOT)
BUILD.clear_generated()

raw_files = download_raw_bars(
    credentials,
    BUILD,
    assets=U67_REQUESTED_ASSETS,
    bar_snapshot_as_of_end=SNAPSHOT_END_DATE,
    analysis_end_date=SNAPSHOT_END_DATE,
)
action_files = download_corporate_actions(
    credentials,
    BUILD,
    assets=U67_REQUESTED_ASSETS,
    query_end=SNAPSHOT_END_DATE,
)

manifest = build_snapshot_manifest(
    BUILD,
    raw_files,
    action_files,
    bar_snapshot_as_of_end=SNAPSHOT_END_DATE,
    analysis_end_date=SNAPSHOT_END_DATE,
    assets=U67_REQUESTED_ASSETS,
    snapshot_name=SNAPSHOT_NAME,
)
validate_snapshot(BUILD)


# %% 2 - Publicacao local do snapshot validado
if TARGET.root.exists():
    shutil.rmtree(TARGET.root)
TARGET.root.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(BUILD.root), str(TARGET.root))

validated = validate_snapshot(TARGET)
print(
    "[snapshot-u67] ready "
    f"assets={len(validated.get('assets') or [])} "
    f"sha256={validated.get('snapshot_sha256')} "
    f"path={TARGET.root}",
    flush=True,
)
print(
    "[snapshot-u67] revise os arquivos e versione dados/u67 no Git "
    "antes de usar a reproducao oficial.",
    flush=True,
)
