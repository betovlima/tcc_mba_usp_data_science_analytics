"""Localiza e arquiva dentro do output a evidencia MCT congelada do TCC.

Primeiro procura os ZIPs conhecidos em output/**; se estiverem na raiz,
evidencias/ ou Downloads, copia SEM alterar o arquivo original e valida SHA-256.
A referencia atual tem prioridade sobre a antiga de CSVs economicamente
identicos. Nunca escolhe o ZIP apenas por ser mais recente.
"""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import shutil
import tempfile


REFERENCE_ZIP = "market_cycle_trader_20261101T021336-83e49685.zip"
REFERENCE_SHA256 = "6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe"
PREVIOUS_ZIP = "market_cycle_trader_20261101T005056-0acb4c89.zip"
PREVIOUS_SHA256 = "6eeafa90937b501f66356c3d1e6019bbde342430d92905570d32b4bcffa2228e"


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sources(root: Path, output: Path, name: str) -> list[Path]:
    # A busca antiga localizava ZIPs na raiz do TCC, evidencias e Downloads.
    # Output e percorrido recursivamente; nao vasculhar toda a pasta pessoal.
    result = list(output.rglob(name)) if output.is_dir() else []
    for directory in (root, root / "evidencias", Path.home() / "Downloads"):
        candidate = directory / name
        if candidate.is_file():
            result.append(candidate)
    return sorted(set(p.resolve() for p in result if p.is_file()))


def _import_to_output(source: Path, root: Path, output: Path, expected_sha: str) -> Path:
    if source == output or output in source.parents:
        return source
    destination = output / "busca_externa_lider_u75" / "referencia_mct" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"Destino do ZIP congelado ja existe com SHA-256 divergente: {destination}"
            )
        return destination

    # Copia atomica para que uma interrupcao nao deixe ZIP parcial.
    with tempfile.NamedTemporaryFile(
        prefix=".mct_zip_", suffix=".tmp", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        if _sha256_file(temporary_path) != expected_sha:
            raise RuntimeError(
                f"SHA-256 mudou durante a copia do ZIP MCT: {source}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"[fonte] Evidencia MCT copiada de {source} para {destination}", flush=True)
    return destination


def locate_mct_zip(
    project_root: Path,
    *,
    explicit: str = "",
    reference_name: str = REFERENCE_ZIP,
    reference_sha256: str = REFERENCE_SHA256,
) -> Path:
    """Seleciona evidencia conhecida e garante copia em output.

    Em testes, reference_name/hash podem ser especificados. Em producao,
    o ZIP atual tem prioridade e a reproducao anterior so e aceita pois
    seus CSVs financeiros e de ranking foram comparados byte a byte.
    """
    project_root = Path(project_root).expanduser().resolve()
    output = project_root / "output"
    if explicit and str(explicit).strip():
        path = Path(str(explicit).strip()).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCT_BACKTEST_ZIP nao encontrado: {path}")
        return path

    references = [(reference_name, reference_sha256)]
    if (reference_name, reference_sha256) == (REFERENCE_ZIP, REFERENCE_SHA256):
        references.append((PREVIOUS_ZIP, PREVIOUS_SHA256))

    mismatches = []
    for index, (name, expected_sha) in enumerate(references):
        for candidate in _sources(project_root, output, name):
            actual_sha = _sha256_file(candidate)
            if actual_sha != expected_sha:
                mismatches.append(f"{candidate}: {actual_sha}")
                continue
            selected = _import_to_output(candidate, project_root, output, expected_sha)
            if index > 0:
                print(
                    "[fonte] Usando ZIP anterior auditado: os CSVs de ranking e "
                    "operacoes sao identicos aos da nova execucao.",
                    flush=True,
                )
            print(f"[fonte] MCT congelado selecionado: {selected}", flush=True)
            return selected

    if mismatches:
        raise RuntimeError(
            "Foram localizados ZIPs com nomes esperados, mas nenhum possui "
            "SHA-256 da evidencia congelada. Divergencias: "
            + "; ".join(mismatches)
        )
    raise FileNotFoundError(
        f"ZIP MCT de referencia nao encontrado em {output} (inclusive "
        "subpastas), na raiz do TCC, em evidencias/ ou em Downloads. "
        "Coloque o ZIP fornecido no output/ do projeto ou informe "
        "MCT_BACKTEST_ZIP com o caminho exato."
    )
