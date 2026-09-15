from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _instante(valor: Any) -> pd.Timestamp | None:
    if valor is None:
        return None
    try:
        convertido = pd.Timestamp(valor)
    except (TypeError, ValueError):
        return None
    if convertido.tzinfo is None:
        return convertido.tz_localize("UTC")
    return convertido.tz_convert("UTC")


def _numero(valor: Any) -> float | None:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if np.isfinite(numero) else None


def _quadro_do_ativo(
    quadros: dict[str, pd.DataFrame],
    simbolo: str,
) -> pd.DataFrame | None:
    quadro = quadros.get(str(simbolo or ""))
    if quadro is None or quadro.empty:
        return None
    if not isinstance(quadro.index, pd.DatetimeIndex):
        return None
    if quadro.index.tz is None:
        quadro = quadro.copy()
        quadro.index = quadro.index.tz_localize("UTC")
    return quadro.sort_index()


def _retorno_abertura_abertura(
    quadros: dict[str, pd.DataFrame],
    simbolo: str,
    inicio_em: Any,
    fim_em: Any,
) -> float | None:
    quadro = _quadro_do_ativo(quadros, simbolo)
    inicio = _instante(inicio_em)
    fim = _instante(fim_em)
    if quadro is None or inicio is None or fim is None or fim <= inicio:
        return None
    try:
        abertura_inicial = _numero(quadro.loc[inicio, "open"])
        abertura_final = _numero(quadro.loc[fim, "open"])
    except (KeyError, TypeError):
        return None
    if abertura_inicial in {None, 0.0} or abertura_final is None:
        return None
    return float(abertura_final / abertura_inicial - 1.0)


def _excursoes_posicao(
    quadros: dict[str, pd.DataFrame],
    simbolo: str,
    entrada_em: Any,
    saida_em: Any,
    preco_entrada: Any,
) -> dict[str, float | None]:
    quadro = _quadro_do_ativo(quadros, simbolo)
    inicio = _instante(entrada_em)
    fim = _instante(saida_em)
    base = _numero(preco_entrada)
    if quadro is None or inicio is None or fim is None or base in {None, 0.0} or fim < inicio:
        return {
            "maximum_favorable_excursion": None,
            "maximum_adverse_excursion": None,
        }

    periodo = quadro.loc[(quadro.index >= inicio) & (quadro.index < fim)]
    maximas = (
        pd.to_numeric(periodo.get("high"), errors="coerce").dropna()
        if not periodo.empty
        else pd.Series(dtype=float)
    )
    minimas = (
        pd.to_numeric(periodo.get("low"), errors="coerce").dropna()
        if not periodo.empty
        else pd.Series(dtype=float)
    )

    abertura_saida = None
    try:
        abertura_saida = _numero(quadro.loc[fim, "open"])
    except (KeyError, TypeError):
        pass

    valores_maximos = maximas.tolist()
    valores_minimos = minimas.tolist()
    if abertura_saida is not None:
        valores_maximos.append(abertura_saida)
        valores_minimos.append(abertura_saida)
    if not valores_maximos or not valores_minimos:
        return {
            "maximum_favorable_excursion": None,
            "maximum_adverse_excursion": None,
        }

    return {
        "maximum_favorable_excursion": max(0.0, float(max(valores_maximos) / base - 1.0)),
        "maximum_adverse_excursion": min(0.0, float(min(valores_minimos) / base - 1.0)),
    }


def _proxima_saida(
    linhas: list[dict[str, Any]],
    indice_compra: int,
    simbolo: str,
) -> dict[str, Any] | None:
    for linha in linhas[indice_compra + 1 :]:
        if str(linha.get("asset") or "") != simbolo:
            continue
        if str(linha.get("action") or "").upper() in {"SELL", "FINAL_SELL"}:
            return linha
    return None


def enriquecer_diagnosticos_operacoes(
    registros: Iterable[dict[str, Any]],
    quadros: dict[str, pd.DataFrame],
    simbolos: Iterable[str],
) -> list[dict[str, Any]]:
    linhas = [dict(linha) for linha in registros]
    universo = [str(simbolo) for simbolo in simbolos]

    for linha in linhas:
        if str(linha.get("action") or "").upper() not in {"SELL", "FINAL_SELL"}:
            continue
        excursoes = _excursoes_posicao(
            quadros,
            str(linha.get("asset") or ""),
            linha.get("entry_timestamp"),
            linha.get("timestamp"),
            linha.get("entry_price"),
        )
        linha.update(excursoes)
        retorno_realizado = _numero(linha.get("position_return"))
        mfe = _numero(excursoes.get("maximum_favorable_excursion"))
        linha["profit_capture_ratio"] = (
            max(0.0, retorno_realizado) / mfe
            if retorno_realizado is not None and mfe is not None and mfe > 0
            else None
        )

    for indice, compra in enumerate(linhas):
        if str(compra.get("action") or "").upper() != "BUY":
            continue
        identificador_rotacao = str(compra.get("rotation_id") or "").strip()
        ativo_origem = str(compra.get("rotation_from_asset") or "").strip()
        ativo_destino = str(compra.get("rotation_to_asset") or compra.get("asset") or "").strip()
        if not identificador_rotacao or not ativo_origem or not ativo_destino:
            continue

        linha_saida = _proxima_saida(linhas, indice, ativo_destino)
        if linha_saida is None:
            continue
        inicio_em = compra.get("timestamp")
        fim_em = linha_saida.get("timestamp")
        retorno_escolhido = _retorno_abertura_abertura(quadros, ativo_destino, inicio_em, fim_em)
        retorno_anterior = _retorno_abertura_abertura(quadros, ativo_origem, inicio_em, fim_em)

        retornos_alternativos: list[tuple[str, float]] = []
        for simbolo in universo:
            valor = _retorno_abertura_abertura(quadros, simbolo, inicio_em, fim_em)
            if valor is not None:
                retornos_alternativos.append((simbolo, valor))
        melhor_ativo = None
        melhor_retorno = None
        if retornos_alternativos:
            melhor_ativo, melhor_retorno = max(retornos_alternativos, key=lambda item: item[1])

        valor_adicionado = (
            retorno_escolhido - retorno_anterior
            if retorno_escolhido is not None and retorno_anterior is not None
            else None
        )
        custo_oportunidade = (
            max(0.0, melhor_retorno - retorno_escolhido)
            if melhor_retorno is not None and retorno_escolhido is not None
            else None
        )

        compra.update(
            {
                "subsequent_position_return": _numero(linha_saida.get("position_return")),
                "chosen_market_return": retorno_escolhido,
                "counterfactual_previous_asset_return": retorno_anterior,
                "rotation_value_added": valor_adicionado,
                "rotation_regret": max(0.0, -valor_adicionado) if valor_adicionado is not None else None,
                "best_alternative_asset": melhor_ativo,
                "best_alternative_return": melhor_retorno,
                "opportunity_cost": custo_oportunidade,
                "maximum_favorable_excursion": _numero(linha_saida.get("maximum_favorable_excursion")),
                "maximum_adverse_excursion": _numero(linha_saida.get("maximum_adverse_excursion")),
                "profit_capture_ratio": _numero(linha_saida.get("profit_capture_ratio")),
                "subsequent_holding_days": _numero(linha_saida.get("holding_bars")),
            }
        )

    return linhas
