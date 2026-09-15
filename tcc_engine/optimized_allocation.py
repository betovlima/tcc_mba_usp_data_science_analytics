from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

MODO_ALOCACAO_OTIMIZADA = "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION"


class ErroTecnicoAlocacao(RuntimeError):
    pass


def sinal_relativo_transversal(utilidade: np.ndarray) -> np.ndarray:
    valores = np.asarray(utilidade, dtype=float)
    resultado = np.full(valores.shape, np.nan, dtype=float)
    finitos = np.isfinite(valores)
    quantidade = int(finitos.sum())
    if quantidade == 0:
        return resultado
    if quantidade == 1:
        resultado[finitos] = 1.0
        return resultado
    valores_finitos = valores[finitos]
    postos = pd.Series(valores_finitos).rank(method="average").to_numpy(dtype=float)
    resultado[finitos] = (2.0 * (postos - 1.0) / float(quantidade - 1)) - 1.0
    return resultado


def forca_separacao_transversal(utilidade: np.ndarray) -> np.ndarray:
    valores = np.asarray(utilidade, dtype=float)
    resultado = np.full(valores.shape, np.nan, dtype=float)
    finitos = np.isfinite(valores)
    if not finitos.any():
        return resultado
    valores_finitos = valores[finitos]
    centro = float(np.median(valores_finitos))
    desvio_absoluto = np.abs(valores_finitos - centro)
    escala = float(np.median(desvio_absoluto)) * 1.4826
    if not np.isfinite(escala) or escala <= 1e-12:
        escala = float(np.std(valores_finitos, ddof=0))
    if not np.isfinite(escala) or escala <= 1e-12:
        resultado[finitos] = 0.0
        return resultado
    pontuacao_z = (valores_finitos - centro) / escala
    resultado[finitos] = np.tanh(np.maximum(pontuacao_z, 0.0))
    return resultado


def forca_ordinal_transversal(utilidade: np.ndarray) -> np.ndarray:
    sinal_relativo = sinal_relativo_transversal(utilidade)
    separacao = forca_separacao_transversal(utilidade)
    resultado = np.full(sinal_relativo.shape, np.nan, dtype=float)
    finitos = np.isfinite(sinal_relativo) & np.isfinite(separacao)
    if not finitos.any():
        return resultado
    forca_posto = np.clip(sinal_relativo[finitos], 0.0, 1.0)
    resultado[finitos] = np.clip(0.75 * forca_posto + 0.25 * separacao[finitos], 0.0, 1.0)
    return resultado


@dataclass(frozen=True)
class CalibradorAlphaRelativo:
    model: Any | None
    constant_alpha: float | None
    sample_count: int
    signal_min: float
    signal_max: float
    realized_alpha_mean: float
    realized_alpha_std: float
    center_prediction: float
    method: str

    @property
    def utilidade_minima(self) -> float:
        return self.signal_min

    @property
    def utilidade_maxima(self) -> float:
        return self.signal_max

    @property
    def media_retorno_realizado(self) -> float:
        return self.realized_alpha_mean

    @property
    def desvio_retorno_realizado(self) -> float:
        return self.realized_alpha_std

    def sinal_relativo(self, utilidade: np.ndarray) -> np.ndarray:
        return sinal_relativo_transversal(utilidade)

    def prever(self, utilidade: np.ndarray) -> np.ndarray:
        sinal = self.sinal_relativo(utilidade)
        resultado = np.full(sinal.shape, np.nan, dtype=float)
        finitos = np.isfinite(sinal)
        if not finitos.any():
            return resultado
        if self.model is None:
            resultado[finitos] = float(self.constant_alpha or 0.0)
            return resultado
        previsto = np.asarray(self.model.predict(sinal[finitos]), dtype=float)
        resultado[finitos] = previsto - float(self.center_prediction)
        return resultado


CalibradorRetornoEsperado = CalibradorAlphaRelativo


@dataclass(frozen=True)
class DecisaoAlocacao:
    weights: dict[str, float]
    cash_weight: float
    expected_utility: float
    expected_relative_alpha: float
    confidence_adjusted_relative_alpha: float
    allocation_reward: float
    confidence_adjusted_allocation_reward: float
    normalized_cvar: float | None
    risk_reference: float | None
    estimated_cvar: float | None
    turnover: float
    objective_value: float | None
    eligible_assets: tuple[str, ...]
    optimizer_status: str
    opportunity_probability: float | None = None
    opportunity_confidence: float | None = None
    opportunity_threshold: float | None = None
    opportunity_accepted: bool | None = None

    @property
    def retorno_liquido_esperado(self) -> float:
        return self.expected_relative_alpha

    @property
    def retorno_esperado_ajustado_confianca(self) -> float:
        return self.confidence_adjusted_relative_alpha


def alocacao_otimizada_ativada(configuracao: Any) -> bool:
    return str(getattr(configuracao, "strategy_mode", "")) == MODO_ALOCACAO_OTIMIZADA


def montar_amostras_alpha_relativo(modelos, quadros, simbolos, datas, utilidades_no_instante, *, horizonte_rotulo: int) -> pd.DataFrame:
    quantidade_segura = max(0, len(datas) - max(1, int(horizonte_rotulo)))
    linhas = []
    for instante in datas[:quantidade_segura]:
        ts = pd.Timestamp(instante)
        utilidades = np.asarray(utilidades_no_instante(modelos, quadros, simbolos, ts), dtype=float)
        utilidade_bruta = np.asarray([float(utilidades[posicao]) if posicao < len(utilidades) else float("nan") for posicao in range(1, len(simbolos) + 1)], dtype=float)
        realizado = np.asarray([float(quadros[simbolo].loc[ts].get("forward_net_log_return", float("nan"))) if ts in quadros[simbolo].index else float("nan") for simbolo in simbolos], dtype=float)
        validos = np.isfinite(utilidade_bruta) & np.isfinite(realizado)
        if int(validos.sum()) < 2:
            continue
        sinal = sinal_relativo_transversal(np.where(validos, utilidade_bruta, np.nan))
        referencia = float(np.median(realizado[validos]))
        alpha_relativo = realizado - referencia
        for indice, simbolo in enumerate(simbolos):
            if not bool(validos[indice]) or not np.isfinite(sinal[indice]):
                continue
            linhas.append({"timestamp": ts, "symbol": simbolo, "utility": float(utilidade_bruta[indice]), "relative_signal": float(sinal[indice]), "realized_net_log_return": float(realizado[indice]), "realized_relative_alpha": float(alpha_relativo[indice])})
    colunas = ["timestamp", "symbol", "utility", "relative_signal", "realized_net_log_return", "realized_relative_alpha"]
    if not linhas:
        return pd.DataFrame(columns=colunas)
    return pd.DataFrame(linhas, columns=colunas).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def montar_amostras_retorno_esperado(modelos, quadros, simbolos, datas, utilidades_no_instante, *, horizonte_rotulo: int) -> pd.DataFrame:
    return montar_amostras_alpha_relativo(modelos, quadros, simbolos, datas, utilidades_no_instante, horizonte_rotulo=horizonte_rotulo)


def ajustar_calibrador_alpha_relativo(modelos, quadros, simbolos, datas_calibracao, utilidades_no_instante, *, horizonte_rotulo: int) -> CalibradorAlphaRelativo:
    amostras = montar_amostras_alpha_relativo(modelos, quadros, simbolos, datas_calibracao, utilidades_no_instante, horizonte_rotulo=horizonte_rotulo)
    minimo_amostras = max(100, len(simbolos) * 4)
    if len(amostras) < minimo_amostras:
        raise ValueError(f"A calibração de alpha relativo da alocação otimizada exige pelo menos {minimo_amostras} observações válidas fora da amostra; apenas {len(amostras)} estão disponíveis.")
    x = amostras["relative_signal"].to_numpy(dtype=float)
    y = amostras["realized_relative_alpha"].to_numpy(dtype=float)
    inferior, superior = np.quantile(y, [0.01, 0.99]) if len(y) >= 100 else (float(np.min(y)), float(np.max(y)))
    y_limitado = np.clip(y, float(inferior), float(superior))
    modelo = None
    constante = 0.0
    previsao_central = 0.0
    metodo = "zero_relative_alpha_no_rank_resolution"
    if np.unique(x).size >= 2:
        from sklearn.isotonic import IsotonicRegression
        modelo = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=None, y_max=None)
        modelo.fit(x, y_limitado)
        previsao_central = float(np.asarray(modelo.predict(np.asarray([0.0], dtype=float)), dtype=float)[0])
        metodo = "out_of_sample_isotonic_cross_sectional_relative_alpha_v2"
    return CalibradorAlphaRelativo(model=modelo, constant_alpha=constante if modelo is None else None, sample_count=int(len(amostras)), signal_min=float(np.min(x)), signal_max=float(np.max(x)), realized_alpha_mean=float(np.mean(y_limitado)), realized_alpha_std=float(np.std(y_limitado, ddof=1)) if len(y_limitado) > 1 else 0.0, center_prediction=float(previsao_central), method=metodo)


def ajustar_calibrador_retorno_esperado(modelos, quadros, simbolos, datas_calibracao, utilidades_no_instante, *, horizonte_rotulo: int) -> CalibradorAlphaRelativo:
    return ajustar_calibrador_alpha_relativo(modelos, quadros, simbolos, datas_calibracao, utilidades_no_instante, horizonte_rotulo=horizonte_rotulo)


def _pesos_atuais_seguros(simbolos: list[str], pesos_atuais: dict[str, float] | None) -> np.ndarray:
    atual = pesos_atuais or {}
    risco = np.asarray([max(0.0, float(atual.get(simbolo, 0.0) or 0.0)) for simbolo in simbolos], dtype=float)
    caixa = max(0.0, float(atual.get("CASH", 0.0) or 0.0))
    valores = np.concatenate([risco, np.asarray([caixa], dtype=float)])
    total = float(valores.sum())
    if not np.isfinite(total) or total <= 0:
        valores[:] = 0.0
        valores[-1] = 1.0
        return valores
    return valores / total


def _pesos_horizonte_normalizados(configuracao: Any) -> tuple[tuple[int, ...], np.ndarray]:
    horizontes = tuple(int(valor) for valor in list(getattr(configuracao, "rotation_target_horizons", []) or []))
    pesos_brutos = np.asarray(list(getattr(configuracao, "rotation_target_horizon_weights", []) or []), dtype=float)
    if not horizontes or len(horizontes) != len(pesos_brutos) or not np.isfinite(pesos_brutos).all() or float(pesos_brutos.sum()) <= 0:
        horizonte = max(1, int(getattr(configuracao, "rotation_horizon_days", 5)))
        return (horizonte,), np.asarray([1.0], dtype=float)
    return horizontes, pesos_brutos / float(pesos_brutos.sum())


def _cenarios_retorno_historico(quadros, simbolos, instante, dias_janela, configuracao) -> np.ndarray:
    horizontes, pesos = _pesos_horizonte_normalizados(configuracao)
    horizonte_maximo = max(horizontes)
    series = []
    for simbolo in simbolos:
        quadro = quadros[simbolo]
        if instante not in quadro.index:
            return np.empty((0, len(simbolos)), dtype=float)
        localizacao = quadro.index.get_loc(instante)
        if not isinstance(localizacao, (int, np.integer)):
            return np.empty((0, len(simbolos)), dtype=float)
        inicio = max(0, int(localizacao) - int(dias_janela) - horizonte_maximo - 2)
        fechamentos = quadro.iloc[inicio:int(localizacao) + 1]["close"].astype(float)
        ponderado = pd.Series(0.0, index=fechamentos.index, dtype=float)
        valido = pd.Series(True, index=fechamentos.index, dtype=bool)
        for horizonte, peso in zip(horizontes, pesos, strict=True):
            componente = np.log(fechamentos / fechamentos.shift(int(horizonte)))
            valido &= componente.notna() & np.isfinite(componente)
            ponderado = ponderado + float(peso) * componente.fillna(0.0)
        ponderado = ponderado.where(valido)
        ponderado.name = simbolo
        series.append(ponderado)
    if not series:
        return np.empty((0, 0), dtype=float)
    combinado = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if len(combinado) > dias_janela:
        combinado = combinado.iloc[-dias_janela:]
    return combinado.to_numpy(dtype=float)


def _tudo_caixa(simbolos, atual, *, status: str, oportunidade, limite_oportunidade: float | None = None) -> DecisaoAlocacao:
    alvo = np.zeros(len(simbolos) + 1, dtype=float)
    alvo[-1] = 1.0
    return DecisaoAlocacao(weights={simbolo: 0.0 for simbolo in simbolos}, cash_weight=1.0, expected_utility=0.0, expected_relative_alpha=0.0, confidence_adjusted_relative_alpha=0.0, allocation_reward=0.0, confidence_adjusted_allocation_reward=0.0, normalized_cvar=0.0, risk_reference=None, estimated_cvar=0.0, turnover=float(np.abs(alvo[:-1] - atual[:-1]).sum()), objective_value=0.0, eligible_assets=(), optimizer_status=status, opportunity_probability=float(oportunidade.probability) if oportunidade is not None else None, opportunity_confidence=float(oportunidade.confidence) if oportunidade is not None else None, opportunity_threshold=float(limite_oportunidade) if limite_oportunidade is not None else None, opportunity_accepted=bool(oportunidade.accepted) if oportunidade is not None else None)


def otimizar_alocacao(utilidades, quadros, simbolos, instante, pesos_atuais, configuracao, *, calibrador_retorno_esperado: CalibradorRetornoEsperado | None = None, oportunidade=None, limite_oportunidade: float | None = None) -> DecisaoAlocacao:
    atual = _pesos_atuais_seguros(simbolos, pesos_atuais)
    utilidade = np.asarray(utilidades[1:len(simbolos) + 1], dtype=float)
    finitos = np.isfinite(utilidade)
    sinal_relativo = sinal_relativo_transversal(utilidade)
    forca_ordinal = forca_ordinal_transversal(utilidade)
    sinal_relativo_minimo = float(getattr(configuracao, "allocation_minimum_utility", 0.0))
    elegiveis = finitos & np.isfinite(sinal_relativo) & np.isfinite(forca_ordinal) & (sinal_relativo > sinal_relativo_minimo) & (forca_ordinal > 0.0)
    if not elegiveis.any():
        return _tudo_caixa(simbolos, atual, status="no_eligible_relative_rank", oportunidade=oportunidade, limite_oportunidade=limite_oportunidade)
    calibrado = calibrador_retorno_esperado.prever(utilidade) if calibrador_retorno_esperado is not None else np.full(utilidade.shape, np.nan, dtype=float)
    confianca = min(1.0, max(0.0, float(oportunidade.confidence))) if oportunidade is not None and getattr(oportunidade, "confidence", None) is not None else 1.0
    ajustado_confianca = calibrado * confianca
    vetor_recompensa = np.where(elegiveis, forca_ordinal, 0.0)
    recompensa_ajustada = vetor_recompensa * confianca
    janela = int(getattr(configuracao, "allocation_lookback_days", 126))
    cenarios = _cenarios_retorno_historico(quadros, simbolos, instante, janela, configuracao)
    minimo_cenarios = max(20, min(60, janela // 2))
    if cenarios.shape[0] < minimo_cenarios:
        raise ErroTecnicoAlocacao(f"A alocação otimizada possui histórico de risco sincronizado insuficiente em {pd.Timestamp(instante)}: {cenarios.shape[0]} cenários.")
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
    nivel_confianca = float(getattr(configuracao, "allocation_cvar_confidence", 0.95))
    aversao_risco = float(getattr(configuracao, "allocation_cvar_penalty", 1.0))
    penalidade_giro = float(getattr(configuracao, "allocation_turnover_penalty", 0.0025))
    peso_maximo_ativo = float(getattr(configuracao, "allocation_max_asset_weight", 1.0))
    escala_sinal = float(getattr(configuracao, "allocation_signal_scale", 1.0))
    custo_estimado = max(0.0, float(getattr(configuracao, "slippage_bps", 0.0))) / 10000.0 + max(0.0, float(getattr(configuracao, "commission_rate", 0.0)))
    def cvar_cenario(valores: np.ndarray) -> float:
        perdas = -np.asarray(valores, dtype=float)
        nivel = float(np.quantile(perdas, nivel_confianca, method="higher"))
        cauda = perdas[perdas >= nivel - 1e-15]
        return max(0.0, float(np.mean(cauda)) if len(cauda) else nivel)
    cvars_individuais = np.asarray([cvar_cenario(cenarios[:, indice]) for indice in range(quantidade_ativos)], dtype=float)
    cvars_positivos = cvars_individuais[np.isfinite(cvars_individuais) & (cvars_individuais > 1e-8)]
    referencia_risco = max(float(np.median(cvars_positivos)) if len(cvars_positivos) else 0.01, 1e-6)
    teto_risco = max(referencia_risco * 3.0, float(np.max(cvars_positivos)) * 1.5 if len(cvars_positivos) else 0.03)
    c = np.zeros(quantidade_variaveis, dtype=float)
    c[:quantidade_ativos] = -escala_sinal * recompensa_ajustada
    c[inicio_giro:inicio_giro + quantidade_ativos] = penalidade_giro
    c[inicio_giro:inicio_giro + quantidade_ativos] += custo_estimado / referencia_risco
    c[indice_penalidade_risco] = aversao_risco
    a_eq = np.zeros((1, quantidade_variaveis), dtype=float)
    a_eq[0, :quantidade_ativos + 1] = 1.0
    b_eq = np.asarray([1.0], dtype=float)
    linhas, lados_direitos = [], []
    for indice_cenario, retornos in enumerate(cenarios):
        linha = np.zeros(quantidade_variaveis, dtype=float); linha[:quantidade_ativos] = -retornos; linha[indice_alpha] = -1.0; linha[inicio_folga + indice_cenario] = -1.0; linhas.append(linha); lados_direitos.append(0.0)
    linha_cvar = np.zeros(quantidade_variaveis, dtype=float); linha_cvar[indice_alpha] = 1.0; linha_cvar[inicio_folga:inicio_giro] = 1.0 / max(1e-12, (1.0 - nivel_confianca) * quantidade_cenarios); linha_cvar[indice_cvar] = -1.0; linhas.append(linha_cvar); lados_direitos.append(0.0)
    for ponto_risco in np.linspace(0.0, teto_risco, 13):
        linha = np.zeros(quantidade_variaveis, dtype=float); linha[indice_cvar] = 2.0 * float(ponto_risco) / (referencia_risco ** 2); linha[indice_penalidade_risco] = -1.0; linhas.append(linha); lados_direitos.append(float((float(ponto_risco) / referencia_risco) ** 2))
    for indice in range(quantidade_ativos + 1):
        indice_z = inicio_giro + indice
        linha_positiva = np.zeros(quantidade_variaveis, dtype=float); linha_positiva[indice] = 1.0; linha_positiva[indice_z] = -1.0; linhas.append(linha_positiva); lados_direitos.append(float(atual[indice]))
        linha_negativa = np.zeros(quantidade_variaveis, dtype=float); linha_negativa[indice] = -1.0; linha_negativa[indice_z] = -1.0; linhas.append(linha_negativa); lados_direitos.append(float(-atual[indice]))
    limites = [(0.0, peso_maximo_ativo if bool(elegiveis[indice]) else 0.0) for indice in range(quantidade_ativos)] + [(0.0, 1.0), (None, None)] + [(0.0, None)] * quantidade_cenarios + [(0.0, None)] * (quantidade_ativos + 1) + [(0.0, None), (0.0, None)]
    resultado = linprog(c, A_ub=np.asarray(linhas, dtype=float), b_ub=np.asarray(lados_direitos, dtype=float), A_eq=a_eq, b_eq=b_eq, bounds=limites, method="highs")
    if not bool(resultado.success):
        raise ErroTecnicoAlocacao(f"O solucionador da alocação otimizada falhou em {pd.Timestamp(instante)}: {str(resultado.message or 'desconhecido')[:160]}")
    solucao = np.asarray(resultado.x, dtype=float)
    risco = np.clip(solucao[:quantidade_ativos], 0.0, 1.0)
    peso_caixa = float(np.clip(solucao[indice_caixa], 0.0, 1.0))
    total = float(risco.sum() + peso_caixa)
    if total <= 0 or not np.isfinite(total):
        raise ErroTecnicoAlocacao(f"A alocação otimizada retornou solução inválida em {pd.Timestamp(instante)}.")
    risco /= total; peso_caixa /= total
    retornos_carteira = cenarios @ risco; perdas = -retornos_carteira; nivel_var = float(np.quantile(perdas, nivel_confianca, method="higher")); cauda = perdas[perdas >= nivel_var - 1e-15]
    cvar_estimado = float(np.mean(cauda)) if len(cauda) else nivel_var
    cvar_normalizado = float(max(0.0, cvar_estimado) / referencia_risco)
    alvo = np.concatenate([risco, np.asarray([peso_caixa])]); giro = float(np.abs(alvo[:-1] - atual[:-1]).sum())
    utilidade_esperada = float(np.dot(np.where(np.isfinite(utilidade), utilidade, 0.0), risco)); alpha_relativo_esperado = float(np.dot(np.where(np.isfinite(calibrado), calibrado, 0.0), risco)); alpha_relativo_ajustado = float(np.dot(np.where(np.isfinite(ajustado_confianca), ajustado_confianca, 0.0), risco)); recompensa_alocacao = float(np.dot(np.where(np.isfinite(vetor_recompensa), vetor_recompensa, 0.0), risco)); recompensa_ajustada_confianca = float(np.dot(np.where(np.isfinite(recompensa_ajustada), recompensa_ajustada, 0.0), risco)); ativos_elegiveis = tuple(simbolos[indice] for indice in range(quantidade_ativos) if elegiveis[indice])
    return DecisaoAlocacao(weights={simbolo: float(risco[indice]) for indice, simbolo in enumerate(simbolos)}, cash_weight=float(peso_caixa), expected_utility=utilidade_esperada, expected_relative_alpha=alpha_relativo_esperado, confidence_adjusted_relative_alpha=alpha_relativo_ajustado, allocation_reward=recompensa_alocacao, confidence_adjusted_allocation_reward=recompensa_ajustada_confianca, normalized_cvar=cvar_normalizado, risk_reference=float(referencia_risco), estimated_cvar=cvar_estimado, turnover=giro, objective_value=float(-resultado.fun), eligible_assets=ativos_elegiveis, optimizer_status="optimal", opportunity_probability=float(oportunidade.probability) if oportunidade is not None else None, opportunity_confidence=float(oportunidade.confidence) if oportunidade is not None else None, opportunity_threshold=float(limite_oportunidade) if limite_oportunidade is not None else None, opportunity_accepted=bool(oportunidade.accepted) if oportunidade is not None else None)
