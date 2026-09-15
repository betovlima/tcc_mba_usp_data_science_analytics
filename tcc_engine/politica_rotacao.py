"""Política de seleção, permanência e rotação entre ativos."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .modelo import prever_utilidades


def criar_politica_rotacao(
    modelos: dict[str, Any],
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    configuracao: Any,
    margem_calibrada: float,
    *,
    diagnosticos: dict[pd.Timestamp, dict[str, Any]] | None = None,
    janela_id: int | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Cria a função de decisão usada durante a simulação fora da amostra."""
    margem_efetiva = max(float(configuracao.margem_troca), float(margem_calibrada))

    def politica(data: pd.Timestamp, posicao_atual: int, dias_posicao: int) -> tuple[int, float]:
        utilidades = prever_utilidades(modelos, quadros, ativos, data)
        posicoes_validas = [
            p for p in range(1, len(utilidades)) if np.isfinite(utilidades[p])
        ]
        if not posicoes_validas:
            return 0, 0.0

        ordenadas = sorted(
            posicoes_validas,
            key=lambda p: (-float(utilidades[p]), ativos[p - 1]),
        )
        melhor = ordenadas[0]
        melhor_valor = float(utilidades[melhor])
        valor_atual = float(utilidades[posicao_atual]) if posicao_atual < len(utilidades) else float("-inf")
        segundo = ordenadas[1] if len(ordenadas) > 1 else None

        motivo = ""
        destino = posicao_atual
        valor_destino = valor_atual if np.isfinite(valor_atual) else 0.0

        if posicao_atual > 0 and dias_posicao < int(configuracao.dias_minimos_posicao) and melhor != posicao_atual:
            motivo = "MANTER_PERIODO_MINIMO"
        elif melhor_valor <= float(configuracao.limiar_caixa):
            destino = 0
            valor_destino = 0.0
            motivo = "MOVER_PARA_CAIXA"
        elif posicao_atual == 0:
            if melhor_valor >= float(configuracao.limiar_caixa) + float(configuracao.vantagem_minima_entrada):
                destino = melhor
                valor_destino = melhor_valor
                motivo = "ENTRAR_MELHOR_ATIVO"
            else:
                destino = 0
                valor_destino = 0.0
                motivo = "SEM_VANTAGEM_MINIMA"
        elif melhor == posicao_atual:
            motivo = "MANTER_MELHOR_ATIVO"
        elif melhor_valor >= valor_atual + margem_efetiva:
            destino = melhor
            valor_destino = melhor_valor
            motivo = "ROTACIONAR_MELHOR_ATIVO"
        else:
            motivo = "MANTER_MARGEM_TROCA"

        if diagnosticos is not None:
            diagnosticos[pd.Timestamp(data)] = {
                "janela_id": janela_id,
                "ativo_atual": "CASH" if posicao_atual == 0 else ativos[posicao_atual - 1],
                "utilidade_atual": float(valor_atual) if np.isfinite(valor_atual) else None,
                "melhor_ativo": ativos[melhor - 1],
                "melhor_utilidade": melhor_valor,
                "segundo_ativo": ativos[segundo - 1] if segundo is not None else None,
                "segunda_utilidade": float(utilidades[segundo]) if segundo is not None else None,
                "margem_troca_efetiva": margem_efetiva,
                "dias_posicao": int(dias_posicao),
                "acao_final": "CASH" if destino == 0 else ativos[destino - 1],
                "motivo_decisao": motivo,
            }
        return int(destino), float(valor_destino)

    return politica


def combinar_politicas_por_janela(
    politicas: dict[int, Callable[[pd.Timestamp, int, int], tuple[int, float]]],
    mapa_data_janela: dict[pd.Timestamp, int],
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Seleciona a política correta para cada data fora da amostra."""
    def politica(data: pd.Timestamp, posicao: int, dias_posicao: int) -> tuple[int, float]:
        identificador = mapa_data_janela.get(pd.Timestamp(data))
        if identificador is None:
            raise KeyError(f"Nenhuma janela atribuída à data {pd.Timestamp(data)}.")
        return politicas[int(identificador)](data, posicao, dias_posicao)
    return politica
