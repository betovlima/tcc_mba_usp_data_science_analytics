"""Migra o snapshot local legado para o snapshot oficial versionado."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from engine.configuracao import ASSETS
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    validate_snapshot,
)


ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copia dados/reproducao para dados/pesquisa, "
            "reconstroi o manifesto e valida os hashes."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir um snapshot de pesquisa ja existente.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    origem = SnapshotPaths.legacy(ROOT)
    destino = SnapshotPaths.research(ROOT)

    if not origem.manifest.exists():
        raise RuntimeError(
            "Snapshot legado nao encontrado em dados/reproducao."
        )

    validate_snapshot(origem)

    if destino.manifest.exists() and not args.overwrite:
        raise RuntimeError(
            "dados/pesquisa ja possui snapshot. "
            "Use --overwrite somente se a substituicao for intencional."
        )

    destino.clear_generated()

    raw_files = {}
    action_files = {}
    for symbol in ASSETS:
        raw_source = origem.raw_bars / f"{symbol}.csv"
        action_source = origem.corporate_actions / f"{symbol}.csv"
        if not raw_source.exists():
            raise RuntimeError(f"RAW ausente: {raw_source}")
        if not action_source.exists():
            raise RuntimeError(f"Corporate Actions ausente: {action_source}")

        raw_target = destino.raw_bars / raw_source.name
        action_target = destino.corporate_actions / action_source.name
        shutil.copy2(raw_source, raw_target)
        shutil.copy2(action_source, action_target)
        raw_files[symbol] = raw_target
        action_files[symbol] = action_target

    manifest = build_snapshot_manifest(
        destino,
        raw_files,
        action_files,
    )
    validate_snapshot(destino)

    print("[migration] snapshot oficial criado", flush=True)
    print(f"[migration] root={destino.root}", flush=True)
    print(
        f"[migration] snapshot_sha256={manifest['snapshot_sha256']}",
        flush=True,
    )
    print("[migration] versione dados/pesquisa no Git.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
