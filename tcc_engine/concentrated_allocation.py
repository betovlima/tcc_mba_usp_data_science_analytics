from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .optimized_allocation import (
    MODO_ALOCACAO_OTIMIZADA,
    DecisaoAlocacao,
    ErroTecnicoAlocacao,
    CalibradorRetornoEsperado,
    _tudo_caixa,
    _cenarios_retorno_historico,
    _pesos_atuais_seguros,
    sinal_relativo_transversal,
)

MODO_ALOCACAO_CONCENTRADA = "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION"


def alocacao_concentrada_ativada(configuracao: Any) -> bool:
    return (
        str(getattr(configuracao, "strategy_mode", ""))
        == MODO_ALOCACAO_CONCENTRADA
    )


def alocacao_carteira_ativada(configuracao: Any) -> bool:
    return str(getattr(configuracao, "strategy_mode", "")) in {
        MODO_ALOCACAO_OTIMIZADA,
        MODO_ALOCACAO_CONCENTRADA,
    }


def forca_candidatos_concentrados(
    utilidade: np.ndarray,
    *,
    limite_candidatos: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    valores = np.asarray(utilidade, dtype=float)
    forcas = np.zeros(valores.shape, dtype=float)
    posicoes_finitas = np.flatnonzero(np.isfinite(valores))
    if len(posicoes_finitas) == 0:
        return np.asarray([], dtype=int), forcas

    ordenadas = sorted(
        posicoes_finitas.tolist(),
        key=lambda indice: (-float(valores[indice]), int(indice)),
    )
    selecionadas = np.asarray(
        ordenadas[: max(1, int(limite_candidatos))],
        dtype=int,
    )
    valor_topo = float(valores[selecionadas[0]])
    valores_finitos = valores[posicoes_finitas]
    centro = float(np.median(valores_finitos))
    escala = float(np.median(np.abs(valores_finitos - centro))) * 1.4826
    if not np.isfinite(escala) or escala <= 1e-12:
        escala = float(np.std(valores_finitos, ddof=0))
    if not np.isfinite(escala) or escala <= 1e-12:
        escala = 1.0

    for posicao in selecionadas:
        diferenca = max(0.0, (valor_topo - float(valores[posicao])) / escala)
        forcas[posicao] = float(np.exp(-0.5 * diferenca * diferenca))
    forcas[selecionadas[0]] = 1.0
    return selecionadas, forcas


def otimizar_alocacao_concentrada(
    utilidades: np.ndarray,
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    instante: pd.Timestamp,
    pesos_atuais: dict[str, float] | None,
    configuracao: Any,
    *,
    calibrador_retorno_esperado: CalibradorRetornoEsperado | None = None,
    oportunidade: Any | None = None,
    limite_oportunidade: float | None = None,
) -> DecisaoAlocacao:
    atual = _pesos_atuais_seguros(simbolos, pesos_atuais)
    utilidade = np.asarray(utilidades[1 : len(simbolos) + 1], dtype=float)
    finitos = np.isfinite(utilidade)
    if not finitos.any():
        return _tudo_caixa(
            simbolos,
            atual,
            status="no_finite_ranking_signal",
            oportunidade=oportunidade,
            limite_oportunidade=limite_oportunidade,
        )

    sinal_relativo = sinal_relativo_transversal(utilidade)
    selecionadas, proximidade = forca_candidatos_concentrados(
        utilidade,
        limite_candidatos=3,
    )
    if len(selecionadas) == 0:
        return _tudo_caixa(
            simbolos,
            atual,
            status="no_ranked_candidate",
            oportunidade=oportunidade,
            limite_oportunidade=limite_oportunidade,
        )

    sinal_relativo_minimo = float(
        getattr(configuracao, "allocation_minimum_utility", 0.0)
    )
    indice_primario = int(selecionadas[0])
    elegiveis = np.zeros(len(simbolos), dtype=bool)
    elegiveis[indice_primario] = True
    for indice in selecionadas[1:]:
        if (
            np.isfinite(sinal_relativo[indice])
            and float(sinal_relativo[indice]) > sinal_relativo_minimo
            and float(proximidade[indice]) > 1e-6
        ):
            elegiveis[indice] = True

    calibrado = (
        calibrador_retorno_esperado.prever(utilidade)
        if calibrador_retorno_esperado is not None
        else np.full(utilidade.shape, np.nan, dtype=float)
    )
    confianca = (
        min(1.0, max(0.0, float(oportunidade.confidence)))
        if oportunidade is not None
        and getattr(oportunidade, "confidence", None) is not None
        else 1.0
    )
    ajustado_confianca = calibrado * confianca
    vetor_recompensa = np.where(elegiveis, proximidade, 0.0)
    vetor_recompensa[indice_primario] = 1.0
    recompensa_ajustada = vetor_recompensa * confianca

    janela = int(getattr(configuracao, "allocation_lookback_days", 126))
    cenarios = _cenarios_retorno_historico(
        quadros,
        simbolos,
        instante,
        janela,
        configuracao,
    )
    minimo_cenarios = max(20, min(60, janela // 2))
    if cenarios.shape[0] < minimo_cenarios:
        raise ErroTecnicoAlocacao(
            "A alocação concentrada possui histórico de risco sincronizado "
            f"insuficiente em {pd.Timestamp(instante)}: "
            f"{cenarios.shape[0]} cenários."
        )

    quantidade_ativos = len(simbolos)
    quantidade_cenarios = int(cenarios.shape[0])
    indice_caixa = quantidade_ativos
    indice_alpha = quantidade_ativos + 1
    inicio_folga = indice_alpha + 1
    inicio_giro = inicio_folga + quantidade_cenarios
    fim_giro = inicio_giro + quantidade_ativos + 1
    indice_cvar = fim_giro
    indice_penalidade_risco = indice_cvar + 1
    quantidade_variaveis = indice_penalidade_risco + 1

    nivel_confianca = float(
        getattr(configuracao, "allocation_cvar_confidence", 0.95)
    )
    aversao_risco = float(
        getattr(configuracao, "allocation_cvar_penalty", 1.0)
    )
    penalidade_giro = float(
        getattr(configuracao, "allocation_turnover_penalty", 0.0025)
    )
    peso_maximo_ativo = float(
        getattr(configuracao, "allocation_max_asset_weight", 1.0)
    )
    escala_sinal = float(
        getattr(configuracao, "allocation_signal_scale", 1.0)
    )
    custo_estimado = (
        max(0.0, float(getattr(configuracao, "slippage_bps", 0.0))) / 10000.0
    )
    custo_estimado += max(
        0.0,
        float(getattr(configuracao, "commission_rate", 0.0)),
    )

    def cvar_cenario(valores: np.ndarray) -> float:
        perdas = -np.asarray(valores, dtype=float)
        nivel = float(np.quantile(perdas, nivel_confianca, method="higher"))
        cauda = perdas[perdas >= nivel - 1e-15]
        return max(0.0, float(np.mean(cauda)) if len(cauda) else nivel)

    cvars_individuais = np.asarray(
        [
            cvar_cenario(cenarios[:, indice])
            for indice in range(quantidade_ativos)
        ],
        dtype=float,
    )
    cvars_positivos = cvars_individuais[
        np.isfinite(cvars_individuais) & (cvars_individuais > 1e-8)
    ]
    referencia_risco = (
        float(np.median(cvars_positivos)) if len(cvars_positivos) else 0.01
    )
    referencia_risco = max(referencia_risco, 1e-6)
    teto_risco = max(
        referencia_risco * 3.0,
        (
            float(np.max(cvars_positivos)) * 1.5
            if len(cvars_positivos)
            else 0.03
        ),
    )

    c = np.zeros(quantidade_variaveis, dtype=float)
    c[:quantidade_ativos] = -escala_sinal * recompensa_ajustada
    c[inicio_giro : inicio_giro + quantidade_ativos] = penalidade_giro
    custo_estimado_normalizado = custo_estimado / referencia_risco
    c[inicio_giro : inicio_giro + quantidade_ativos] += custo_estimado_normalizado
    c[indice_penalidade_risco] = aversao_risco

    a_eq = np.zeros((1, quantidade_variaveis), dtype=float)
    a_eq[0, : quantidade_ativos + 1] = 1.0
    b_eq = np.asarray([1.0], dtype=float)

    linhas: list[np.ndarray] = []
    lados_direitos: list[float] = []
    for indice_cenario, retornos in enumerate(cenarios):
        linha = np.zeros(quantidade_variaveis, dtype=float)
        linha[:quantidade_ativos] = -retornos
        linha[indice_alpha] = -1.0
        linha[inicio_folga + indice_cenario] = -1.0
        linhas.append(linha)
        lados_direitos.append(0.0)

    linha_cvar = np.zeros(quantidade_variaveis, dtype=float)
    linha_cvar[indice_alpha] = 1.0
    linha_cvar[inicio_folga:inicio_giro] = 1.0 / max(
        1e-12,
        (1.0 - nivel_confianca) * quantidade_cenarios,
    )
    linha_cvar[indice_cvar] = -1.0
    linhas.append(linha_cvar)
    lados_direitos.append(0.0)

    for ponto_risco in np.linspace(0.0, teto_risco, 13):
        inclinacao = float(ponto_risco) / (referencia_risco**2)
        intercepto = -0.5 * (
            (float(ponto_risco) / referencia_risco) ** 2
        )
        linha = np.zeros(quantidade_variaveis, dtype=float)
        linha[indice_cvar] = inclinacao
        linha[indice_penalidade_risco] = -1.0
        linhas.append(linha)
        lados_direitos.append(float(-intercepto))

    for indice in selecionadas[1:]:
        if not bool(elegiveis[indice]):
            continue
        linha = np.zeros(quantidade_variaveis, dtype=float)
        linha[int(indice)] = 1.0
        linha[indice_primario] = -float(proximidade[indice])
        linhas.append(linha)
        lados_direitos.append(0.0)

    for indice in range(quantidade_ativos + 1):
        indice_z = inicio_giro + indice
        linha_positiva = np.zeros(quantidade_variaveis, dtype=float)
        linha_positiva[indice] = 1.0
        linha_positiva[indice_z] = -1.0
        linhas.append(linha_positiva)
        lados_direitos.append(float(atual[indice]))

        linha_negativa = np.zeros(quantidade_variaveis, dtype=float)
        linha_negativa[indice] = -1.0
        linha_negativa[indice_z] = -1.0
        linhas.append(linha_negativa)
        lados_direitos.append(float(-atual[indice]))

    limites: list[tuple[float | None, float | None]] = []
    for indice in range(quantidade_ativos):
        limites.append(
            (
                0.0,
                peso_maximo_ativo if bool(elegiveis[indice]) else 0.0,
            )
        )
    limites.append((0.0, 1.0))
    limites.append((None, None))
    limites.extend([(0.0, None)] * quantidade_cenarios)
    limites.extend([(0.0, None)] * (quantidade_ativos + 1))
    limites.append((0.0, None))
    limites.append((0.0, None))

    resultado = linprog(
        c,
        A_ub=np.asarray(linhas, dtype=float),
        b_ub=np.asarray(lados_direitos, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=limites,
        method="highs",
    )
    if not bool(resultado.success):
        raise ErroTecnicoAlocacao(
            "O solucionador da alocação concentrada falhou em "
            f"{pd.Timestamp(instante)}: "
            f"{str(resultado.message or 'desconhecido')[:160]}"
        )

    solucao = np.asarray(resultado.x, dtype=float)
    risco = np.clip(solucao[:quantidade_ativos], 0.0, 1.0)
    peso_caixa = float(np.clip(solucao[indice_caixa], 0.0, 1.0))
    total = float(risco.sum() + peso_caixa)
    if total <= 0 or not np.isfinite(total):
        raise ErroTecnicoAlocacao(
            "A alocação concentrada retornou solução inválida em "
            f"{pd.Timestamp(instante)}."
        )
    risco /= total
    peso_caixa /= total

    retornos_carteira = cenarios @ risco
    perdas = -retornos_carteira
    nivel_var = float(np.quantile(perdas, nivel_confianca, method="higher"))
    cauda = perdas[perdas >= nivel_var - 1e-15]
    cvar_estimado = float(np.mean(cauda)) if len(cauda) else nivel_var
    cvar_normalizado = float(max(0.0, cvar_estimado) / referencia_risco)
    alvo = np.concatenate([risco, np.asarray([peso_caixa])])
    giro = float(np.abs(alvo[:-1] - atual[:-1]).sum())
    utilidade_esperada = float(
        np.dot(np.where(np.isfinite(utilidade), utilidade, 0.0), risco)
    )
    alpha_relativo_esperado = float(
        np.dot(np.where(np.isfinite(calibrado), calibrado, 0.0), risco)
    )
    alpha_relativo_ajustado = float(
        np.dot(
            np.where(np.isfinite(ajustado_confianca), ajustado_confianca, 0.0),
            risco,
        )
    )
    recompensa_alocacao = float(np.dot(vetor_recompensa, risco))
    recompensa_alocacao_ajustada = float(
        np.dot(recompensa_ajustada, risco)
    )
    ativos_elegiveis = tuple(
        simbolos[indice]
        for indice in range(quantidade_ativos)
        if elegiveis[indice]
    )

    return DecisaoAlocacao(
        weights={
            simbolo: float(risco[indice])
            for indice, simbolo in enumerate(simbolos)
        },
        cash_weight=float(peso_caixa),
        expected_utility=utilidade_esperada,
        expected_relative_alpha=alpha_relativo_esperado,
        confidence_adjusted_relative_alpha=alpha_relativo_ajustado,
        allocation_reward=recompensa_alocacao,
        confidence_adjusted_allocation_reward=recompensa_alocacao_ajustada,
        normalized_cvar=cvar_normalizado,
        risk_reference=float(referencia_risco),
        estimated_cvar=cvar_estimado,
        turnover=giro,
        objective_value=float(-resultado.fun),
        eligible_assets=ativos_elegiveis,
        optimizer_status="optimal_concentrated",
        opportunity_probability=(
            float(oportunidade.probability)
            if oportunidade is not None
            else None
        ),
        opportunity_confidence=(
            float(oportunidade.confidence)
            if oportunidade is not None
            else None
        ),
        opportunity_threshold=(
            float(limite_oportunidade)
            if limite_oportunidade is not None
            else None
        ),
        opportunity_accepted=(
            bool(oportunidade.accepted)
            if oportunidade is not None
            else None
        ),
    )
