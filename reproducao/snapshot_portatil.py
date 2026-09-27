"""Verifica snapshots do Git sem modificar os CSVs versionados.

Alguns snapshots foram gerados no Windows com CRLF, mas o Git armazena
os CSVs em LF. Uma copia efemera e permitida somente se os bytes restaurados
forem exatamente os declarados no SHA-256 original do manifesto.
Nao altera precos, datas, eventos, registros nem o manifesto.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reproducao.dados import SnapshotPaths, validate_snapshot


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def preparar_snapshot_verificado(
    origem: SnapshotPaths, destino: Path,
) -> tuple[SnapshotPaths, dict[str, int]]:
    manifest = json.loads(origem.manifest.read_text(encoding="utf-8"))
    hashes = manifest.get("file_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Manifesto sem file_hashes valido")
    escolhas: dict[str, tuple[Path, bytes, bool]] = {}
    mismatch = []
    restored = 0
    for rel, expected in sorted(hashes.items()):
        relative = Path(rel)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".csv"
            or relative.parts[0] not in {"raw_bars", "corporate_actions"}
        ):
            raise RuntimeError(f"Caminho inesperado no manifesto: {rel}")
        source = origem.root / relative
        if not source.is_file():
            mismatch.append((rel, "ausente"))
            continue
        actual = source.read_bytes()
        if _hash(actual) == expected:
            escolhas[rel] = (source, actual, False)
            continue
        # A transformacao e permitida somente se reproduzir o SHA ORIGINAL.
        logical = actual.replace(b"\r\n", b"\n")
        variants = (logical, logical.replace(b"\n", b"\r\n"))
        matching = next((v for v in variants if _hash(v) == expected), None)
        if matching is None:
            mismatch.append((rel, f"original_sha256={_hash(actual)} expected={expected}"))
        else:
            escolhas[rel] = (source, matching, True)
            restored += 1

    if mismatch:
        report = "; ".join(f"{path}: {reason}" for path, reason in mismatch[:10])
        raise RuntimeError(
            f"Snapshot nao equivale ao manifesto nem por LF/CRLF: {len(mismatch)} arquivo(s). "
            f"Nenhum arquivo foi alterado. {report}"
        )
    if not restored:
        validate_snapshot(origem)
        return origem, {"files": len(escolhas), "line_endings_rehydrated": 0}

    # Copia isolada em output: os arquivos Git de dados/pesquisa nao mudam.
    if destino.resolve() == origem.root.resolve():
        raise ValueError("Destino temporario nao pode ser a pasta versionada")
    for rel, (_, content, _) in escolhas.items():
        target = destino / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "manifest.json").write_bytes(origem.manifest.read_bytes())
    verified = SnapshotPaths.from_root(destino)
    validate_snapshot(verified)
    return verified, {
        "files": len(escolhas),
        "line_endings_rehydrated": restored,
    }
