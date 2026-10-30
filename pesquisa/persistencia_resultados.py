"""Escrita resiliente de evidencias das pesquisas do TCC.

As execucoes podem demorar muito. Cada gravacao garante que a pasta da
execucao ainda existe, inclusive se tiver sido removida depois do inicio.
Nao cria dados artificiais nem recupera arquivos que foram apagados.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def garantir_pasta_resultados(pasta: Path) -> Path:
    """Recria a pasta quando ausente e preserva arquivos ja existentes."""
    pasta = Path(pasta)
    anterior = pasta.is_dir()
    pasta.mkdir(parents=True, exist_ok=True)
    if not pasta.is_dir():
        raise OSError(f"Diretorio de resultados indisponivel: {pasta}")
    if not anterior:
        print(f"[output] diretorio preparado/restaurado: {pasta}", flush=True)
    return pasta


def salvar_csv(frame: Any, arquivo: Path, **kwargs: Any) -> Path:
    """Garante a pasta imediatamente antes de cada escrita CSV."""
    arquivo = Path(arquivo)
    garantir_pasta_resultados(arquivo.parent)
    frame.to_csv(arquivo, **kwargs)
    return arquivo


def salvar_json(
    arquivo: Path,
    payload: Any,
    writer: Callable[[Path, Any], None],
) -> Path:
    arquivo = Path(arquivo)
    garantir_pasta_resultados(arquivo.parent)
    writer(arquivo, payload)
    return arquivo


def salvar_texto(arquivo: Path, conteudo: str, *, encoding: str = "utf-8") -> Path:
    arquivo = Path(arquivo)
    garantir_pasta_resultados(arquivo.parent)
    arquivo.write_text(conteudo, encoding=encoding)
    return arquivo
