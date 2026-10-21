"""Reproducao oficial do universo de ativos selecionado.

O script nao descobre ativos nem atualiza series. Utiliza o mesmo treino por
ativo, calibracao por fold e replay financeiro da pesquisa historica ja
validada. Requer o snapshot base e os arquivos locais congelados dos
candidatos, com verificacao SHA-256 antes de qualquer treinamento.

Spyder: reinicie o kernel e execute este arquivo com F5.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from reproducao.experimento import reproduzir_universo_congelado

_PANDAS4_WARNING = getattr(pd.errors, "Pandas4Warning", None)
if _PANDAS4_WARNING is not None:
    warnings.filterwarnings("ignore", category=_PANDAS4_WARNING)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "reproducao_universo"
REPRODUCTION_VERSION = "1.23.0"
EXECUTION_SCHEMA = "universe-reproduction-v3"


def main() -> dict:
    """Reproduz o universo vencedor historico sem busca ou novos downloads."""
    return reproduzir_universo_congelado(
        root=ROOT,
        output=OUT,
        version=REPRODUCTION_VERSION,
        execution_schema=EXECUTION_SCHEMA,
    )


if __name__ == "__main__":
    main()
