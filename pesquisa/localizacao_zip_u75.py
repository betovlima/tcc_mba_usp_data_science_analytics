"""Seleção determinística do ZIP MCT congelado no output do TCC.

Nenhuma busca em Downloads/evidencias, nenhuma escolha por data de modificação.
Um override explícito é permitido para uma nova campanha de pesquisa.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


REFERENCE_ZIP = "market_cycle_trader_20261101T021336-83e49685.zip"
REFERENCE_SHA256 = "6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe"


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_mct_zip(
    project_root: Path,
    *,
    explicit: str = "",
    reference_name: str = REFERENCE_ZIP,
    reference_sha256: str = REFERENCE_SHA256,
) -> Path:
    """Encontra o backtest validado em output recursivamente ou usa caminho explícito.

    A seleção implícita exige nome e SHA-256 da execução de referência,
    mesmo que existam outros ZIPs no diretório. Não adota o mais recente.
    """
    project_root = Path(project_root).expanduser().resolve()
    if explicit and str(explicit).strip():
        selected = Path(str(explicit).strip()).expanduser()
        if not selected.is_absolute():
            selected = project_root / selected
        selected = selected.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"MCT_BACKTEST_ZIP não encontrado: {selected}")
        return selected

    output = project_root / "output"
    if not output.is_dir():
        raise FileNotFoundError(
            f"Pasta output do TCC não encontrada: {output}. "
            "Informe MCT_BACKTEST_ZIP somente se usar outra origem."
        )

    candidates = sorted(p for p in output.rglob(reference_name) if p.is_file())
    if not candidates:
        all_zips = sum(1 for _ in output.rglob("market_cycle_trader_*.zip"))
        raise FileNotFoundError(
            f"ZIP MCT de referência '{reference_name}' não encontrado em "
            f"{output} (outros ZIPs MCT encontrados: {all_zips}). "
            "Informe MCT_BACKTEST_ZIP para usar conscientemente outra execução."
        )

    valid = [p for p in candidates if _sha256_file(p) == reference_sha256]
    if not valid:
        hashes = [f"{str(p.relative_to(output))}: {_sha256_file(p)}" for p in candidates]
        raise RuntimeError(
            "ZIP MCT com nome esperado, mas SHA-256 divergente. "
            f"Esperado={reference_sha256}; encontrados={hashes}. "
            "Não substituir a referência silenciosamente."
        )
    return valid[0]
