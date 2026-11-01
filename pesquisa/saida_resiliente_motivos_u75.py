"""Saidas persistentes para a pesquisa de motivos historicos do U75.

Cada escrita revalida o diretorio pai para lidar com limpeza externa de
pastas temporarias durante execucoes longas. Integridade do pacote deve ser
verificada depois; recriar pasta nao significa recuperar arquivos apagados.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def write_with_parent(path: Path, writer: Callable[[Path], T]) -> T:
    """Create parents at write time; retry once if the directory disappears."""
    destination = Path(path)
    for attempt in range(2):
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            return writer(destination)
        except FileNotFoundError:
            if attempt:
                raise
    raise RuntimeError("Unreachable output writer state")


def check_required_files(directory: Path, filenames: tuple[str, ...]) -> None:
    """Never report completed after some earlier output files disappeared."""
    folder = Path(directory)
    missing = [name for name in filenames if not (folder / name).is_file()]
    if missing:
        raise RuntimeError(
            "Arquivo(s) de evidencia desapareceram durante a execucao: "
            + ", ".join(missing)
            + ". Verifique limpeza automatica do diretorio output. "
            + "Voce pode definir TCC_MOTIFS_OUTPUT_DIR para uma pasta estavel."
        )
