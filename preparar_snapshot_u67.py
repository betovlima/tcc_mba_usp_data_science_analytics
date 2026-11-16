"""Prepara um NOVO snapshot para os ativos configurados sem sobrescrever arquivos congelados.

Este arquivo e executado somente quando for necessario criar ou substituir o
snapshot de pesquisa. A reproducao oficial nao acessa a Alpaca.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from engine.configuracao import ASSETS
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)


ROOT = Path(__file__).resolve().parent
BUILD = SnapshotPaths.from_root(ROOT / "dados" / ".universo_77_build")
TARGET = SnapshotPaths.from_root(ROOT / "dados" / "universo_77")

SNAPSHOT_END_DATE = "2026-10-06"
SNAPSHOT_NAME = "tcc-universo-configurado-20261006"


if not ASSETS or len(ASSETS) != len(set(ASSETS)):
    raise ValueError("ASSETS deve conter simbolos unicos e nao pode ser vazio.")


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
validate_snapshot(BUILD)


# %% 2 - Publicacao local do snapshot validado
if TARGET.root.exists():
    shutil.rmtree(TARGET.root)
TARGET.root.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(BUILD.root), str(TARGET.root))

validated = validate_snapshot(TARGET)
print(
    "[snapshot] ready "
    f"assets={len(validated.get('assets') or [])} "
    f"sha256={validated.get('snapshot_sha256')} "
    f"path={TARGET.root}",
    flush=True,
)
print(
    "[snapshot] revise os arquivos e versione o novo snapshot de 77 no Git "
    "antes de usar a reproducao oficial.",
    flush=True,
)
