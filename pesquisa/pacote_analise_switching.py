"""Pacote de auditoria deterministico de cada execucao da pesquisa.

Inclui somente os resultados da execucao, nao inclui dados RAW nem credenciais.
Arquivos sao checados por SHA-256 e o ZIP e validado apos a gravacao.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, BadZipFile


def gerar_zip_resultados(pasta: Path, *, versao: str, status: str) -> Path:
    pasta = Path(pasta).resolve()
    if not pasta.is_dir():
        raise FileNotFoundError(pasta)
    if status not in {"completed", "failed"}:
        raise ValueError("Status de execucao invalido.")
    if not (pasta / ("resultado.json" if status == "completed" else "erro_execucao.json")).is_file():
        raise RuntimeError("Arquivo principal da execucao nao encontrado.")
    destino = pasta.parent / f"pacote_optimal_switching_cash_{pasta.name}.zip"
    if destino.exists():
        raise FileExistsError(
            f"Pacote ja existe, nenhuma execucao sera sobrescrita: {destino}"
        )
    files = sorted(
        p for p in pasta.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}
    )
    if not files:
        raise RuntimeError("Nenhuma evidencia disponivel para compactacao.")
    entries = []
    for path in files:
        rel = path.relative_to(pasta).as_posix()
        entries.append({
            "name": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "schema_version": 1,
        "experiment": "optimal-switching-cash-reentry",
        "script_version": str(versao),
        "status": status,
        "run_id": pasta.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": entries,
        "data_raw_included": False,
        "database_used": False,
    }
    temp_path = destino.with_suffix(".zip.tmp")
    try:
        with ZipFile(temp_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(
                "manifesto_pacote.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            )
            for path in files:
                archive.write(path, arcname=path.relative_to(pasta).as_posix())
        with ZipFile(temp_path, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP gerado com arquivo corrompido.")
            payload = json.loads(archive.read("manifesto_pacote.json"))
            for info in payload["files"]:
                content = archive.read(info["name"])
                if len(content) != info["bytes"] or sha256(content).hexdigest() != info["sha256"]:
                    raise RuntimeError(f"Integridade comprometida: {info['name']}")
        temp_path.replace(destino)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return destino
