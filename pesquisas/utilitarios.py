"""Utilitarios pequenos para pesquisas executadas no Spyder."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import zipfile


def criar_pacote_analise(
    diretorio_resultados: Path,
    *,
    versao: str,
) -> Path:
    """Compacta somente os artefatos necessarios para analise posterior."""
    diretorio = Path(diretorio_resultados)
    if not diretorio.exists():
        raise FileNotFoundError(diretorio)

    destino = diretorio.parent / (
        "pacote_analise_" + versao.replace(".", "_").replace("-", "_") + ".zip"
    )
    if destino.exists():
        destino.unlink()

    with zipfile.ZipFile(
        destino,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as arquivo:
        for path in sorted(diretorio.rglob("*")):
            if not path.is_file():
                continue
            arquivo.write(
                path,
                arcname=str(
                    Path(diretorio.name) / path.relative_to(diretorio)
                ),
            )
    return destino


def sinal_sonoro_conclusao() -> None:
    """Emite dois tons no Windows; usa bell do terminal como fallback."""
    try:
        import winsound

        winsound.Beep(880, 220)
        time.sleep(0.08)
        winsound.Beep(1175, 420)
        return
    except (ImportError, RuntimeError, OSError):
        pass

    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        return
