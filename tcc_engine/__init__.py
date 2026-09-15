"""Motor acadêmico do TCC MBA USP Data Science & Analytics.

A pasta contém somente a implementação necessária ao experimento atual.
"""

from .configuracao import ATIVOS, CONFIGURACAO, DATA_FIM, DATA_INICIO, ConfiguracaoBacktest
from .rotacao_capital import executar_modelos_rotacao

__all__ = [
    "ATIVOS",
    "CONFIGURACAO",
    "DATA_FIM",
    "DATA_INICIO",
    "ConfiguracaoBacktest",
    "executar_modelos_rotacao",
]
