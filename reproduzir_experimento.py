"""Reproducao do universo configurado na unica lista ASSETS.

O script nao descobre ativos nem baixa series. Treina, calibra e executa o
backtest conjunto com todos os simbolos da configuracao, independentemente da
quantidade. Cada ativo precisa ter dados historicos locais validados.
Os arquivos RAW preservados sao verificados por SHA-256.

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
REPRODUCTION_VERSION = "1.23.0+dynamic-universe.test"
EXECUTION_SCHEMA = "universe-reproduction-v4"


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
