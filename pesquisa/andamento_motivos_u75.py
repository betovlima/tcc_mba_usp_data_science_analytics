"""Marcos persistentes de progresso do experimento U75.

Não modifica modelos, dados, sinais, custos ou políticas.
Uma interrupção abrupta do kernel deixa o último marco 'running' visível;
não afirma conclusão ou sucesso após ausência de confirmação.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from pesquisa.saida_resiliente_motivos_u75 import write_with_parent


def record_progress(
    output_dir: Path,
    stage: str,
    *,
    status: str = "running",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Grava estado mais recente e histórico incremental com flush/sync."""
    if not stage or status not in ("running", "completed", "failed"):
        raise ValueError("Invalid research progress marker")
    folder = Path(output_dir)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "details": details or {},
    }
    payload = json.dumps(event, ensure_ascii=False, indent=2, default=str)
    line = json.dumps(event, ensure_ascii=False, default=str) + "\n"

    def _append(path: Path):
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            # Explicit persistence of small milestones helps detect abnormal
            # interpreter termination. This is not a crash recovery cache.
            import os
            os.fsync(fh.fileno())

    write_with_parent(folder / "andamento.jsonl", _append)
    write_with_parent(
        folder / "andamento_atual.json",
        lambda path: path.write_text(payload + "\n", encoding="utf-8"),
    )
    print(f"[etapa] {event['timestamp_utc']} | {stage} | {status}", flush=True)
    return event


def last_progress(output_dir: Path) -> dict[str, Any] | None:
    """Retorna o último marco válido, mesmo sem resultado financeiro."""
    log = Path(output_dir) / "andamento.jsonl"
    if not log.is_file():
        return None
    last = None
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last
