"""Pacote ZIP de diagnosticos financeiros, CSVs e graficos autocontidos.

Mantem o manifesto SHA-256 para todos os arquivos; nao inclui RAW ou
credenciais. Esta versao e especifica para relatorios com PNG e HTML.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def gerar_zip_visual(pasta: Path, *, versao: str, status: str) -> Path:
    folder = Path(pasta).resolve()
    if status not in {"completed", "failed"}:
        raise ValueError("Status invalido.")
    expected = "resultado.json" if status == "completed" else "erro_execucao.json"
    if not (folder / expected).is_file():
        raise RuntimeError(f"Arquivo principal ausente: {expected}")
    target = folder.parent / f"pacote_diagnostico_u75_{folder.name}.zip"
    if target.exists():
        raise FileExistsError(target)
    allowed = {".csv", ".json", ".md", ".txt", ".html", ".png"}
    paths = sorted(
        x for x in folder.rglob("*")
        if x.is_file() and x.suffix.lower() in allowed
    )
    entries = [
        {
            "name": x.relative_to(folder).as_posix(),
            "bytes": x.stat().st_size,
            "sha256": sha256(x.read_bytes()).hexdigest(),
        }
        for x in paths
    ]
    manifest = {
        "schema_version": 1,
        "experiment": "u75-stress-cash-diagnostics",
        "version": str(versao), "status": status,
        "run_id": folder.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries), "files": entries,
        "data_raw_included": False, "database_used": False,
    }
    temp = target.with_suffix(".zip.tmp")
    try:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("manifesto_pacote.json", json.dumps(
                manifest, ensure_ascii=False, indent=2,
            ))
            for x in paths:
                z.write(x, x.relative_to(folder).as_posix())
        with ZipFile(temp) as z:
            if z.testzip() is not None:
                raise RuntimeError("ZIP invalido.")
            check = json.loads(z.read("manifesto_pacote.json"))
            for row in check["files"]:
                b = z.read(row["name"])
                if len(b) != row["bytes"] or sha256(b).hexdigest() != row["sha256"]:
                    raise RuntimeError(f"Hash divergente: {row['name']}")
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    return target
