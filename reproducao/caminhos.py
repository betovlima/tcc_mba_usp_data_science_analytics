"""Caminhos persistentes e migracao segura do layout local."""

from __future__ import annotations

from pathlib import Path
import shutil


def _mesclar_sem_sobrescrever(origem: Path, destino: Path) -> None:
    """Move itens ausentes da origem para o destino sem substituir arquivos."""
    if not origem.exists():
        return

    destino.mkdir(parents=True, exist_ok=True)
    for item in list(origem.iterdir()):
        alvo = destino / item.name
        if item.is_dir():
            _mesclar_sem_sobrescrever(item, alvo)
            try:
                item.rmdir()
            except OSError:
                pass
            continue

        if alvo.exists():
            continue
        shutil.move(str(item), str(alvo))

    try:
        origem.rmdir()
    except OSError:
        pass


def migrar_diretorios_legados(raiz_projeto: Path) -> list[tuple[Path, Path]]:
    """Remove o sufixo _v1 dos layouts locais antigos sem perder arquivos.

    A migracao e idempotente e nunca sobrescreve um arquivo que ja exista no
    novo destino. Diretorios antigos que contenham conflitos sao preservados.
    """
    pares = (
        (
            raiz_projeto / "dados" / "pesquisa_v1",
            raiz_projeto / "dados" / "pesquisa",
        ),
        (
            raiz_projeto / "dados" / "temporario" / "reproducao_v1",
            raiz_projeto / "dados" / "temporario" / "reproducao",
        ),
        (
            raiz_projeto / "dados" / "reproducao_v1",
            raiz_projeto / "dados" / "reproducao",
        ),
        (
            raiz_projeto / "output" / "reproducao_v1",
            raiz_projeto / "output" / "reproducao",
        ),
    )

    migrados: list[tuple[Path, Path]] = []
    for origem, destino in pares:
        if not origem.exists():
            continue
        _mesclar_sem_sobrescrever(origem, destino)
        migrados.append((origem, destino))
        print(
            f"[layout] migracao segura {origem} -> {destino}",
            flush=True,
        )
    return migrados
