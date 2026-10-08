"""Artefatos finais da reproducao oficial do TCC."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import zipfile


def criar_pacote_analise(
    diretorio_resultados: Path,
    *,
    comparison_file: str,
    execution_schema: str,
    archive_name: str,
) -> Path:
    """Valida o artefato principal e compacta a pasta de resultados."""
    diretorio = Path(diretorio_resultados)
    if not diretorio.exists():
        raise FileNotFoundError(diretorio)

    comparison_path = diretorio / str(comparison_file)
    if not comparison_path.exists():
        raise RuntimeError(
            "Pacote recusado: arquivo principal ausente: "
            f"{comparison_path.name}."
        )

    try:
        comparison_payload = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Pacote recusado: nao foi possivel validar "
            f"{comparison_path.name}."
        ) from exc

    observed_schema = comparison_payload.get("execution_schema")
    if observed_schema != str(execution_schema):
        raise RuntimeError(
            "Pacote recusado por execution_schema incompativel: "
            f"esperado={execution_schema!r} "
            f"observado={observed_schema!r}."
        )

    destino = diretorio / str(archive_name)
    if destino.exists():
        destino.unlink()

    with zipfile.ZipFile(
        destino,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as arquivo:
        for path in sorted(diretorio.rglob("*")):
            if not path.is_file() or path.resolve() == destino.resolve():
                continue
            arquivo.write(path, arcname=path.relative_to(diretorio))

    return destino


def sinal_sonoro_conclusao() -> None:
    """Toca um aviso de conclusao sem interferir no resultado da execucao."""
    mechanisms: list[str] = []

    try:
        import winsound

        try:
            alias_flag = getattr(winsound, "SND_ALIAS", None)
            if alias_flag is not None:
                winsound.PlaySound("SystemExclamation", int(alias_flag))
                mechanisms.append("PlaySound:SystemExclamation")
        except Exception:
            pass

        try:
            message_type = getattr(winsound, "MB_ICONASTERISK", -1)
            winsound.MessageBeep(int(message_type))
            mechanisms.append("MessageBeep")
        except Exception:
            pass

        try:
            winsound.Beep(880, 220)
            time.sleep(0.08)
            winsound.Beep(1175, 420)
            mechanisms.append("Beep")
        except Exception:
            pass
    except Exception:
        pass

    if not mechanisms:
        try:
            sys.stdout.write("\\a")
            sys.stdout.flush()
            mechanisms.append("terminal-bell")
        except Exception:
            mechanisms.append("none")

    try:
        print(
            "[sound] completion mechanisms=" + ",".join(mechanisms),
            flush=True,
        )
    except Exception:
        pass
