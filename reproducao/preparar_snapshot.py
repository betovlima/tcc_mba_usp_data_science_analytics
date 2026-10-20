"""Prepara o snapshot congelado referência usado pela reproducao oficial.

Modulo interno opcional executado com python -m reproducao.preparar_snapshot.\nEste arquivo e executado somente quando for necessario criar ou substituir o
snapshot de pesquisa. A reproducao oficial nao acessa a Alpaca.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from engine.configuracao import (
    ASSETS,
)
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    read_snapshot_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
BUILD = SnapshotPaths.from_root(ROOT / "dados" / ".reference_build")
TARGET = SnapshotPaths.referencia(ROOT)

SNAPSHOT_END_DATE = "2026-10-20"
SNAPSHOT_NAME = "tcc-reference-frozen-20261020"


# %% 1 - Coleta completa em area temporaria
credentials = load_alpaca_credentials(ROOT)
BUILD.clear_generated()

raw_files = download_raw_bars(
    credentials,
    BUILD,
    assets=ASSETS,
    bar_snapshot_as_of_end=SNAPSHOT_END_DATE,
    analysis_end_date=SNAPSHOT_END_DATE,
)
action_files = download_corporate_actions(
    credentials,
    BUILD,
    assets=ASSETS,
    query_end=SNAPSHOT_END_DATE,
)

manifest = build_snapshot_manifest(
    BUILD,
    raw_files,
    action_files,
    bar_snapshot_as_of_end=SNAPSHOT_END_DATE,
    analysis_end_date=SNAPSHOT_END_DATE,
    assets=ASSETS,
    snapshot_name=SNAPSHOT_NAME,
)
read_snapshot_manifest(BUILD)


# %% 2 - Publicacao local do snapshot validado
if TARGET.root.exists():
    shutil.rmtree(TARGET.root)
TARGET.root.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(BUILD.root), str(TARGET.root))

validated = read_snapshot_manifest(TARGET)
print(
    "[snapshot-referencia] ready "
    f"assets={len(validated.get('assets') or [])} "
    f"files={len(validated.get('row_counts') or {})} "
    f"path={TARGET.root}",
    flush=True,
)
print(
    "[snapshot-referencia] revise os arquivos e versione dados/referencia no Git "
    "antes de usar a reproducao oficial.",
    flush=True,
)
