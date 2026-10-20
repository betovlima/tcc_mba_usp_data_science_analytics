"""Contrato estrutural dos dois scripts executáveis do TCC."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_raiz_tem_apenas_os_dois_scripts_pactuados() -> None:
    scripts = {path.name for path in ROOT.glob("*.py")}
    assert scripts == {"buscar_ativos.py", "reproduzir_experimento.py"}


def test_pesquisa_e_reproducao_permanecem_separadas() -> None:
    source = (ROOT / "buscar_ativos.py").read_text(encoding="utf-8")
    assert "reproducao.busca_agrupamentos" in source
    assert "reproducao.expansao_u67" in source
    assert "reproduzir_experimento.py" not in source
    official = (ROOT / "reproduzir_experimento.py").read_text(encoding="utf-8")
    assert "SnapshotPaths.u67(ROOT)" in official
    assert "validate_snapshot(DATA)" in official
