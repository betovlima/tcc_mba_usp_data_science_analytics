from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .optimized_allocation import (
    DecisaoAlocacao,
    _cenarios_retorno_historico,
    _pesos_atuais_seguros,
)

MODO_SOBREPOSICAO_RISCO_COMPOSTO = "COMPOUND_ROTATION_SWING_COMPOUND_RISK_OVERLAY"


def sobreposicao_risco_composto_ativada(configuracao: Any) -> bool:
    return (
        str(getattr(configuracao, "strategy_mode", ""))
        == MODO_SOBREPOSICAO_RISCO_COMPOSTO
    )


def execucao_alocacao_ativada(configuracao: Any) -> bool:
    from .concentrated_allocation import alocacao_carteira_ativada

    return (
        alocacao_carteira_ativada(configuracao)
        or sobreposicao_risco_composto_ativada(configuracao)
    )


def _cvar_cenario(valores: np.ndarray, nivel_confianca: float) -> float:
    observacoes = np.asarray(valores, dtype=float)
    observacoes = observacoes[np.isfinite(observacoes)]
    if len(observacoes) == 0:
        return float("nan")
    perdas = -observacoes
    nivel = float(np.quantile(perdas, nivel_confianca, method="higher"))
    cauda = perdas[perdas >= nivel - 1e-15]
    return max(0.0, float(np.mean(cauda)) if len(cauda) else nivel)


def _alvo_base(
    simbolos: list[str],
    atual: np.ndarray,
    posicao_alvo: int,
    pontuacao_alvo: float,
    *,
    estado: str,
    fallback_tecnico: bool,
    cvar_atual: float | None = None,
    cvar_referencia: float | None = None,
) -> DecisaoAlocacao:
    pesos = {simbolo: 0.0 for simbolo in simbolos}
    if posicao_alvo <= 0:
        alvo = np.zeros(len(simbolos) + 1, dtype=float)
        alvo[-1] = 1.0
        return DecisaoAlocacao(
            weights=pesos,
            cash_weight=1.0,
            expected_utility=(
                float(pontuacao_alvo) if np.isfinite(pontuacao_alvo) else 0.0
            ),
            expected_relative_alpha=0.0,
            confidence_adjusted_relative_alpha=0.0,
            allocation_reward=0.0,
            confidence_adjusted_allocation_reward=0.0,
            normalized_cvar=0.0,
            risk_reference=cvar_referencia,
            estimated_cvar=0.0,
            turnover=float(np.abs(alvo[:-1] - atual[:-1]).sum()),
            objective_value=0.0,
            eligible_assets=(),
            optimizer_status=estado,
        )

    indice = int(posicao_alvo) - 1
    pesos[simbolos[indice]] = 1.0
    alvo_risco = np.zeros(len(simbolos), dtype=float)
    alvo_risco[indice] = 1.0
    normalizado = (
        float(cvar_atual) / float(cvar_referencia)
        if cvar_atual is not None
        and cvar_referencia is not None
        and np.isfinite(cvar_atual)
        and np.isfinite(cvar_referencia)
        and cvar_referencia > 1e-12
        else None
    )
    return DecisaoAlocacao(
        weights=pesos,
        cash_weight=0.0,
        expected_utility=(
            float(pontuacao_alvo) if np.isfinite(pontuacao_alvo) else 0.0
        ),
        expected_relative_alpha=0.0,
        confidence_adjusted_relative_alpha=0.0,
        allocation_reward=1.0,
        confidence_adjusted_allocation_reward=1.0,
        normalized_cvar=normalizado,
        risk_reference=cvar_referencia,
        estimated_cvar=cvar_atual,
        turnover=float(np.abs(alvo_risco - atual[:-1]).sum()),
        objective_value=None if fallback_tecnico else 1.0,
        eligible_assets=(simbolos[indice],),
        optimizer_status=estado,
    )


def _taxas_estimadas_custo_operacao(
    quadro: pd.DataFrame,
    instante: pd.Timestamp,
    configuracao: Any,
) -> tuple[float, float]:
    if instante not in quadro.index:
        return (0.0, 0.0)
    preco = float(quadro.loc[instante].get("close", float("nan")))
    if not np.isfinite(preco) or preco <= 0:
        return (0.0, 0.0)
    deslizamento = (
        max(0.0, float(getattr(configuracao, "slippage_bps", 0.0))) / 10000.0
    )
    comissao = max(0.0, float(getattr(configuracao, "commission_rate", 0.0)))
    cat = max(0.0, float(getattr(configuracao, "cat_fee_per_share", 0.0))) / preco
    sec = max(0.0, float(getattr(configuracao, "sec_fee_rate", 0.0)))
    taf = max(0.0, float(getattr(configuracao, "taf_fee_per_share", 0.0))) / preco
    return (
        deslizamento + comissao + cat,
        deslizamento + comissao + cat + sec + taf,
    )


def otimizar_sobreposicao_risco_composto(
    posicao_alvo: int,
    pontuacao_alvo: float,
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    instante: pd.Timestamp,
    pesos_atuais: dict[str, float] | None,
    configuracao: Any,
) -> DecisaoAlocacao:
    atual = _pesos_atuais_seguros(simbolos, pesos_atuais)
    if posicao_alvo <= 0:
        return _alvo_base(
            simbolos,
            atual,
            0,
            pontuacao_alvo,
            estado="base_policy_cash",
            fallback_tecnico=False,
        )

    indice_alvo = int(posicao_alvo) - 1
    if indice_alvo < 0 or indice_alvo >= len(simbolos):
        return _alvo_base(
            simbolos,
            atual,
            posicao_alvo,
            pontuacao_alvo,
            estado="technical_fallback_base_policy:invalid_target_position",
            fallback_tecnico=True,
        )

    simbolo_alvo = simbolos[indice_alvo]
    quadro = quadros.get(simbolo_alvo)
    if quadro is None or instante not in quadro.index:
        return _alvo_base(
            simbolos,
            atual,
            posicao_alvo,
            pontuacao_alvo,
            estado="technical_fallback_base_policy:missing_target_market_data",
            fallback_tecnico=True,
        )

    janela_configurada = max(
        20,
        int(getattr(configuracao, "allocation_lookback_days", 126)),
    )
    janela_atual = max(252, janela_configurada)
    janela_referencia = max(756, janela_atual * 3)
    cenarios_atuais = _cenarios_retorno_historico(
        {simbolo_alvo: quadro},
        [simbolo_alvo],
        instante,
        janela_atual,
        configuracao,
    )
    cenarios_referencia = _cenarios_retorno_historico(
        {simbolo_alvo: quadro},
        [simbolo_alvo],
        instante,
        janela_referencia,
        configuracao,
    )
    minimo_atual = max(60, min(126, janela_atual // 2))
    minimo_referencia = max(126, min(252, janela_referencia // 3))
    if (
        cenarios_atuais.shape[0] < minimo_atual
        or cenarios_referencia.shape[0] < minimo_referencia
    ):
        return _alvo_base(
            simbolos,
            atual,
            posicao_alvo,
            pontuacao_alvo,
            estado=(
                "technical_fallback_base_policy:insufficient_asset_risk_history"
                f":current={cenarios_atuais.shape[0]}"
                f":reference={cenarios_referencia.shape[0]}"
            ),
            fallback_tecnico=True,
        )

    nivel_confianca = float(
        getattr(configuracao, "allocation_cvar_confidence", 0.95)
    )
    cvar_atual = _cvar_cenario(cenarios_atuais[:, 0], nivel_confianca)
    cvar_referencia = _cvar_cenario(cenarios_referencia[:, 0], nivel_confianca)
    if (
        not np.isfinite(cvar_atual)
        or not np.isfinite(cvar_referencia)
        or cvar_referencia <= 1e-12
    ):
        return _alvo_base(
            simbolos,
            atual,
            posicao_alvo,
            pontuacao_alvo,
            estado="technical_fallback_base_policy:invalid_asset_risk_estimate",
            fallback_tecnico=True,
            cvar_atual=cvar_atual if np.isfinite(cvar_atual) else None,
            cvar_referencia=(
                cvar_referencia if np.isfinite(cvar_referencia) else None
            ),
        )

    risco_normalizado = max(0.0, float(cvar_atual) / float(cvar_referencia))
    aversao_risco = max(
        0.0,
        float(getattr(configuracao, "allocation_cvar_penalty", 1.0)),
    )
    penalidade_giro = max(
        0.0,
        float(getattr(configuracao, "allocation_turnover_penalty", 0.0025)),
    )
    recompensa = max(
        1e-12,
        float(getattr(configuracao, "allocation_signal_scale", 1.0)),
    )
    peso_maximo = min(
        1.0,
        max(0.0, float(getattr(configuracao, "allocation_max_asset_weight", 1.0))),
    )
    peso_atual_alvo = float(atual[indice_alvo])
    peso_outros_riscos = float(np.delete(atual[:-1], indice_alvo).sum())
    taxa_compra, taxa_venda = _taxas_estimadas_custo_operacao(
        quadro,
        instante,
        configuracao,
    )

    def objetivo(peso: float) -> float:
        valor = min(peso_maximo, max(0.0, float(peso)))
        custo_risco = 0.5 * aversao_risco * (risco_normalizado * valor) ** 2
        giro_risco = peso_outros_riscos + abs(valor - peso_atual_alvo)
        custo_operacao = (
            taxa_compra * max(0.0, valor - peso_atual_alvo)
            + taxa_venda * max(0.0, peso_atual_alvo - valor)
        )
        return (
            recompensa * valor
            - custo_risco
            - penalidade_giro * giro_risco
            - custo_operacao
        )

    candidatos = {
        0.0,
        peso_maximo,
        min(peso_maximo, max(0.0, peso_atual_alvo)),
    }
    curvatura = aversao_risco * risco_normalizado * risco_normalizado
    if curvatura > 1e-12:
        estacionario_inferior = (
            recompensa + penalidade_giro + taxa_venda
        ) / curvatura
        estacionario_superior = (
            recompensa - penalidade_giro - taxa_compra
        ) / curvatura
        if 0.0 <= estacionario_inferior <= min(peso_maximo, peso_atual_alvo):
            candidatos.add(float(estacionario_inferior))
        if (
            max(0.0, peso_atual_alvo)
            <= estacionario_superior
            <= peso_maximo
        ):
            candidatos.add(float(estacionario_superior))

    melhor_peso = max(candidatos, key=lambda valor: (objetivo(valor), valor))
    melhor_peso = min(peso_maximo, max(0.0, float(melhor_peso)))

    risco = np.zeros(len(simbolos), dtype=float)
    risco[indice_alvo] = melhor_peso
    peso_caixa = max(0.0, 1.0 - melhor_peso)
    giro_risco = float(np.abs(risco - atual[:-1]).sum())
    cvar_carteira = float(cvar_atual) * melhor_peso
    estado = "optimal_compound_risk_overlay"
    if melhor_peso >= peso_maximo - 1e-9:
        estado = "optimal_compound_risk_overlay_full_exposure"
    elif melhor_peso <= 1e-9:
        estado = "optimal_compound_risk_overlay_cash"

    return DecisaoAlocacao(
        weights={
            simbolo: float(risco[indice])
            for indice, simbolo in enumerate(simbolos)
        },
        cash_weight=float(peso_caixa),
        expected_utility=(
            float(pontuacao_alvo) if np.isfinite(pontuacao_alvo) else 0.0
        ),
        expected_relative_alpha=0.0,
        confidence_adjusted_relative_alpha=0.0,
        allocation_reward=float(melhor_peso),
        confidence_adjusted_allocation_reward=float(melhor_peso),
        normalized_cvar=float(risco_normalizado),
        risk_reference=float(cvar_referencia),
        estimated_cvar=float(cvar_carteira),
        turnover=giro_risco,
        objective_value=float(objetivo(melhor_peso)),
        eligible_assets=(simbolo_alvo,),
        optimizer_status=estado,
    )
