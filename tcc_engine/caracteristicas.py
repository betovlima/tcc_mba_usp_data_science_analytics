"""Construção dos atributos técnicos e dos alvos do modelo."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CARACTERISTICAS_ROTACAO = [
    "retorno_1", "retorno_2", "retorno_3", "retorno_5", "retorno_10",
    "retorno_20", "retorno_40", "retorno_60", "retorno_120",
    "volatilidade_5", "volatilidade_10", "volatilidade_20", "volatilidade_40",
    "volatilidade_60", "razao_volatilidade_5_20", "razao_volatilidade_10_40",
    "razao_volatilidade_20_60", "distancia_ema_5", "distancia_ema_10",
    "distancia_ema_20", "distancia_ema_50", "distancia_ema_100",
    "ema_5_sobre_20", "ema_20_sobre_50", "ema_50_sobre_100",
    "inclinacao_ema_20_5", "inclinacao_ema_50_10", "inclinacao_ema_100_20",
    "rsi_14", "atr_pct_14", "distancia_maxima_20", "distancia_minima_20",
    "distancia_maxima_50", "distancia_minima_50", "distancia_maxima_100",
    "distancia_minima_100", "distancia_maxima_200", "distancia_minima_200",
    "posicao_canal_20", "posicao_canal_50", "posicao_canal_100",
    "posicao_canal_200", "eficiencia_tendencia_10", "eficiencia_tendencia_20",
    "eficiencia_tendencia_40", "eficiencia_tendencia_60",
    "aceleracao_momento_5_20", "aceleracao_momento_20_60",
    "expansao_amplitude_5_20", "zscore_volume_20", "zscore_volume_60",
    "razao_volume_5_20",
]


def _dividir_seguro(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    return numerador / denominador.replace(0, np.nan)


def _rsi(fechamento: pd.Series, periodo: int = 14) -> pd.Series:
    variacao = fechamento.diff()
    ganhos = variacao.clip(lower=0)
    perdas = -variacao.clip(upper=0)
    media_ganhos = ganhos.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    media_perdas = perdas.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    razao = _dividir_seguro(media_ganhos, media_perdas)
    return 100 - 100 / (1 + razao)


def _amplitude_verdadeira(quadro: pd.DataFrame) -> pd.Series:
    fechamento_anterior = quadro["close"].shift(1)
    componentes = [
        quadro["high"] - quadro["low"],
        (quadro["high"] - fechamento_anterior).abs(),
        (quadro["low"] - fechamento_anterior).abs(),
    ]
    return pd.concat(componentes, axis=1).max(axis=1)


def construir_quadro_rotacao(barras: pd.DataFrame, configuracao: Any) -> pd.DataFrame:
    """Calcula atributos observáveis e o alvo prospectivo de utilidade."""
    horizontes = [int(valor) for valor in configuracao.horizontes_alvo]
    pesos = np.asarray(configuracao.pesos_horizontes, dtype=float)
    pesos = pesos / pesos.sum()
    maior_horizonte = max(horizontes)

    dados = barras.copy().sort_index()
    dados.index = pd.to_datetime(dados.index, utc=True)
    fechamento = dados["close"].astype(float)
    abertura = dados["open"].astype(float)
    maxima = dados["high"].astype(float)
    minima = dados["low"].astype(float)
    volume = dados["volume"].astype(float)
    retorno_diario = fechamento.pct_change()
    amplitude_diaria = _dividir_seguro(maxima - minima, fechamento)

    for periodo in [1, 2, 3, 5, 10, 20, 40, 60, 120]:
        dados[f"retorno_{periodo}"] = fechamento.pct_change(periodo)
    for periodo in [5, 10, 20, 40, 60]:
        dados[f"volatilidade_{periodo}"] = retorno_diario.rolling(periodo).std()
    dados["razao_volatilidade_5_20"] = _dividir_seguro(dados["volatilidade_5"], dados["volatilidade_20"])
    dados["razao_volatilidade_10_40"] = _dividir_seguro(dados["volatilidade_10"], dados["volatilidade_40"])
    dados["razao_volatilidade_20_60"] = _dividir_seguro(dados["volatilidade_20"], dados["volatilidade_60"])

    medias_exponenciais: dict[int, pd.Series] = {}
    for periodo in [5, 10, 20, 50, 100]:
        medias_exponenciais[periodo] = fechamento.ewm(span=periodo, adjust=False).mean()
        dados[f"distancia_ema_{periodo}"] = _dividir_seguro(fechamento, medias_exponenciais[periodo]) - 1
    dados["ema_5_sobre_20"] = _dividir_seguro(medias_exponenciais[5], medias_exponenciais[20]) - 1
    dados["ema_20_sobre_50"] = _dividir_seguro(medias_exponenciais[20], medias_exponenciais[50]) - 1
    dados["ema_50_sobre_100"] = _dividir_seguro(medias_exponenciais[50], medias_exponenciais[100]) - 1
    dados["inclinacao_ema_20_5"] = medias_exponenciais[20].pct_change(5)
    dados["inclinacao_ema_50_10"] = medias_exponenciais[50].pct_change(10)
    dados["inclinacao_ema_100_20"] = medias_exponenciais[100].pct_change(20)

    dados["rsi_14"] = _rsi(fechamento) / 100.0
    atr = _amplitude_verdadeira(dados).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dados["atr_pct_14"] = _dividir_seguro(atr, fechamento)

    for periodo in [20, 50, 100, 200]:
        maxima_movel = maxima.rolling(periodo).max()
        minima_movel = minima.rolling(periodo).min()
        dados[f"distancia_maxima_{periodo}"] = _dividir_seguro(fechamento, maxima_movel) - 1
        dados[f"distancia_minima_{periodo}"] = _dividir_seguro(fechamento, minima_movel) - 1
        dados[f"posicao_canal_{periodo}"] = _dividir_seguro(fechamento - minima_movel, maxima_movel - minima_movel)

    mudanca_absoluta = fechamento.pct_change().abs()
    for periodo in [10, 20, 40, 60]:
        movimento_liquido = fechamento.pct_change(periodo).abs()
        caminho = mudanca_absoluta.rolling(periodo).sum()
        dados[f"eficiencia_tendencia_{periodo}"] = _dividir_seguro(movimento_liquido, caminho)

    dados["aceleracao_momento_5_20"] = dados["retorno_5"] - dados["retorno_20"] * (5.0 / 20.0)
    dados["aceleracao_momento_20_60"] = dados["retorno_20"] - dados["retorno_60"] * (20.0 / 60.0)
    dados["expansao_amplitude_5_20"] = _dividir_seguro(
        amplitude_diaria.rolling(5).mean(),
        amplitude_diaria.rolling(20).mean(),
    )

    for periodo in [20, 60]:
        media_volume = volume.rolling(periodo).mean()
        desvio_volume = volume.rolling(periodo).std()
        dados[f"zscore_volume_{periodo}"] = _dividir_seguro(volume - media_volume, desvio_volume)
    dados["razao_volume_5_20"] = _dividir_seguro(volume.rolling(5).mean(), volume.rolling(20).mean())

    custo_ida_volta = min(
        0.25,
        2.0 * (
            max(0.0, float(configuracao.deslizamento_bps)) / 10_000.0
            + max(0.0, float(configuracao.comissao))
        ),
    )
    log_custo = math.log(max(1e-12, 1.0 - custo_ida_volta))

    minimas = minima.to_numpy(dtype=float)
    maximas = maxima.to_numpy(dtype=float)
    fechamentos = fechamento.to_numpy(dtype=float)
    aberturas = abertura.to_numpy(dtype=float)
    componentes_utilidade = np.full((len(dados), len(horizontes)), np.nan, dtype=float)
    componentes_retorno = np.full((len(dados), len(horizontes)), np.nan, dtype=float)
    captura_movimento = np.full(len(dados), np.nan, dtype=float)
    persistencia_tendencia = np.full(len(dados), np.nan, dtype=float)

    for indice in range(max(0, len(dados) - maior_horizonte)):
        entrada = aberturas[indice + 1]
        if not np.isfinite(entrada) or entrada <= 0:
            continue
        fechamentos_completos = fechamentos[indice + 1: indice + maior_horizonte + 1]
        maximas_completas = maximas[indice + 1: indice + maior_horizonte + 1]
        if len(fechamentos_completos) != maior_horizonte:
            continue

        alta_maxima = max(0.0, float(np.nanmax(maximas_completas) / entrada - 1.0))
        proporcao_positiva = float(np.mean(fechamentos_completos > entrada))
        variacoes = np.diff(fechamentos_completos)
        passos_positivos = float(np.mean(variacoes > 0)) if len(variacoes) else 0.0
        captura_movimento[indice] = math.log1p(alta_maxima)
        persistencia_tendencia[indice] = 0.5 * proporcao_positiva + 0.5 * passos_positivos

        for indice_componente, horizonte in enumerate(horizontes):
            minimas_futuras = minimas[indice + 1: indice + horizonte + 1]
            fechamentos_futuros = fechamentos[indice + 1: indice + horizonte + 1]
            if len(fechamentos_futuros) != horizonte:
                continue
            retorno_log_bruto = math.log(max(float(fechamentos_futuros[-1]) / entrada, 1e-12))
            menor_minima = float(np.nanmin(minimas_futuras))
            queda = max(0.0, 1.0 - menor_minima / entrada)
            caminho_precos = np.concatenate(([entrada], fechamentos_futuros))
            picos = np.maximum.accumulate(caminho_precos)
            rebaixamentos = 1.0 - np.divide(
                caminho_precos,
                picos,
                out=np.ones_like(caminho_precos),
                where=picos > 0,
            )
            drawdown_caminho = max(0.0, float(np.nanmax(rebaixamentos)))
            retorno_log_liquido = retorno_log_bruto + log_custo
            componentes_retorno[indice, indice_componente] = retorno_log_liquido
            componentes_utilidade[indice, indice_componente] = (
                retorno_log_liquido
                - float(configuracao.penalidade_baixa) * queda
                - float(configuracao.penalidade_drawdown) * drawdown_caminho
            )

    utilidade_ponderada = np.nansum(componentes_utilidade * pesos.reshape(1, -1), axis=1)
    retorno_ponderado = np.nansum(componentes_retorno * pesos.reshape(1, -1), axis=1)
    linhas_invalidas = np.isnan(componentes_utilidade).any(axis=1)
    utilidade_ponderada[linhas_invalidas] = np.nan
    retorno_ponderado[linhas_invalidas] = np.nan

    dados["retorno_log_liquido_futuro"] = retorno_ponderado
    dados["utilidade_base_futura"] = utilidade_ponderada
    dados["captura_movimento_futura"] = captura_movimento
    dados["persistencia_tendencia_futura"] = persistencia_tendencia
    dados["utilidade_ajustada_risco_futura"] = (
        utilidade_ponderada
        + float(configuracao.peso_captura_movimento) * dados["captura_movimento_futura"]
        + float(configuracao.peso_persistencia_tendencia) * dados["persistencia_tendencia_futura"]
    )

    obrigatorias = CARACTERISTICAS_ROTACAO + ["open", "high", "low", "close", "volume"]
    dados = dados.replace([np.inf, -np.inf], np.nan)
    return dados.dropna(subset=obrigatorias)


def preparar_painel_rotacao(
    barras_por_ativo: dict[str, pd.DataFrame],
    configuracao: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Alinha os ativos em um calendário comum usado pelo experimento."""
    quadros = {
        ativo: construir_quadro_rotacao(quadro, configuracao)
        for ativo, quadro in barras_por_ativo.items()
        if quadro is not None and not quadro.empty
    }
    if len(quadros) < 2:
        raise ValueError("A rotação precisa de pelo menos dois ativos com dados válidos.")

    ancoras = [ativo for ativo in configuracao.ativos_ancora_calendario if ativo in quadros]
    if len(ancoras) < 2:
        ancoras = sorted(quadros)

    datas_comuns: pd.DatetimeIndex | None = None
    for ativo in ancoras:
        indice = pd.DatetimeIndex(quadros[ativo].index)
        datas_comuns = indice if datas_comuns is None else datas_comuns.intersection(indice)
    if datas_comuns is None or len(datas_comuns) < configuracao.linhas_minimas_treinamento:
        raise ValueError("O histórico alinhado é insuficiente para treinamento e validação temporal.")

    datas_comuns = datas_comuns.sort_values()
    alinhados = {ativo: quadro.reindex(datas_comuns).copy() for ativo, quadro in quadros.items()}
    return alinhados, datas_comuns
