from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MODO_FILTRO_CAIXA_UTILIDADE_ABSOLUTA = "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE"


@dataclass(frozen=True)
class AvaliacaoFiltroCaixaUtilidadeAbsoluta:
    melhor_pontuacao: float
    limite_ativo: float
    aceito: bool
    histerese_manutencao_mercado: bool
    histerese_bloqueio_caixa: bool


def filtro_caixa_utilidade_absoluta_ativado(configuracao: Any) -> bool:
    return (
        str(getattr(configuracao, "strategy_mode", ""))
        == MODO_FILTRO_CAIXA_UTILIDADE_ABSOLUTA
    )


def avaliar_filtro_caixa_utilidade_absoluta(
    configuracao: Any,
    *,
    melhor_pontuacao: float,
    posicao_atual: int,
) -> AvaliacaoFiltroCaixaUtilidadeAbsoluta:
    """Decide entre MERCADO e CAIXA pela utilidade Top-1 da estratégia.

    Este filtro não treina um segundo modelo preditivo. A utilidade absoluta
    Top-1 é tratada como sinal de oportunidade e submetida a uma regra de
    histerese com dois limites:

    - em CAIXA, entra apenas ao alcançar ou superar o limite de entrada;
    - investido, permanece no mercado até a utilidade cair abaixo do limite de saída.

    Os limites são parâmetros da estratégia e podem ser explorados sem alterar
    o modelo LightGBM utilizado pelo experimento.
    """
    limite_entrada = float(
        getattr(configuracao, "opportunity_utility_entry_threshold")
    )
    limite_saida = float(
        getattr(configuracao, "opportunity_utility_exit_threshold")
    )
    limite_ativo = limite_saida if int(posicao_atual) > 0 else limite_entrada
    aceito = float(melhor_pontuacao) >= limite_ativo
    dentro_faixa = limite_saida <= float(melhor_pontuacao) < limite_entrada

    return AvaliacaoFiltroCaixaUtilidadeAbsoluta(
        melhor_pontuacao=float(melhor_pontuacao),
        limite_ativo=float(limite_ativo),
        aceito=bool(aceito),
        histerese_manutencao_mercado=bool(
            int(posicao_atual) > 0 and dentro_faixa
        ),
        histerese_bloqueio_caixa=bool(
            int(posicao_atual) <= 0 and dentro_faixa
        ),
    )
