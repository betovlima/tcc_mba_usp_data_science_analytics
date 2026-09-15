"""Funções de custos e preço de execução usadas pelo motor do TCC."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def arredondar_taxa_para_centavo(valor: float) -> float:
    """Arredonda uma taxa positiva para cima no centavo mais próximo."""
    if not np.isfinite(valor) or valor <= 0:
        return 0.0
    return math.ceil((valor - 1e-12) * 100.0) / 100.0


def calcular_taxas_referencia(
    lado: str,
    quantidade: float,
    preco: float,
    configuracao: Any,
) -> dict[str, float]:
    """Calcula comissão e taxas regulatórias de uma ordem simulada."""
    if quantidade <= 0 or preco <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }
    lado_normalizado = lado.upper()
    valor_operacao = quantidade * preco
    comissao = arredondar_taxa_para_centavo(
        valor_operacao * configuracao.commission_rate
    )
    taxa_cat = arredondar_taxa_para_centavo(
        quantidade * configuracao.cat_fee_per_share
    )
    taxa_sec = 0.0
    taxa_taf = 0.0
    if lado_normalizado == "SELL":
        taxa_sec = arredondar_taxa_para_centavo(
            valor_operacao * configuracao.sec_fee_rate
        )
        taxa_taf = arredondar_taxa_para_centavo(
            min(
                quantidade * configuracao.taf_fee_per_share,
                configuracao.taf_fee_cap,
            )
        )
    elif lado_normalizado != "BUY":
        raise ValueError(f"Lado de ordem não suportado: {lado}")
    return {
        "commission_fee": comissao,
        "sec_fee": taxa_sec,
        "taf_fee": taxa_taf,
        "cat_fee": taxa_cat,
        "total_fee": comissao + taxa_sec + taxa_taf + taxa_cat,
    }


def aplicar_deslizamento(
    preco: float,
    lado: str,
    configuracao: Any,
) -> float:
    """Aplica o deslizamento configurado ao preço da ordem simulada."""
    ajuste = configuracao.slippage_bps / 10_000
    return preco * (1 + ajuste if lado == "BUY" else 1 - ajuste)
