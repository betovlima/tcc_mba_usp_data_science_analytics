"""Visão econômica causal (em memória) sem reescrever barras Alpaca RAW.

Preços da visão são cotados em valor econômico de UMA ação-base no início
do snapshot: aplica retrospectivamente NADA, e acumula exclusivamente
eventos já EFETIVADOS, para t>=ex_date. É equivalente em razões a uma série
ajustada para observações pós-split, sem depender de eventos futuros para
features em t. Volume espelha a mudança no número de ações.

Esta visão alimenta SÓ features/rótulos em memória. Compras/vendas
executam a cotação RAW original e a custódia muda sua QUANTIDADE ao split.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def fingerprint_frame(raw: pd.DataFrame) -> str:
    """Fingerprint dos valores, índice e dtype pandas em memória."""
    from hashlib import sha256
    data = pd.util.hash_pandas_object(raw, index=True).to_numpy(
        dtype=np.uint64,
    )
    return sha256(data.tobytes()).hexdigest()


def extrair_fatores_efetivos(
    raw: pd.DataFrame,
    actions: list[dict[str, Any]],
) -> tuple[tuple[pd.Timestamp, float], ...]:
    """Falha se um evento está mal formado: não excluir splits silenciosamente."""
    if raw.empty or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise RuntimeError("Índice RAW vazio, repetido ou não ordenado.")
    days = pd.DatetimeIndex(raw.index)
    if days.tz is None:
        raise RuntimeError("O índice RAW deve ter timezone explícito.")
    start = days[0].tz_convert("UTC").normalize()
    end = days[-1].tz_convert("UTC").normalize()
    events: list[tuple[pd.Timestamp, float]] = []
    for action in actions:
        if action.get("action_type") not in ("forward_split", "reverse_split"):
            continue
        try:
            date = pd.Timestamp(action["ex_date"])
            if pd.isna(date):
                raise ValueError("ex_date ausente")
            date = (
                date.tz_localize("UTC") if date.tzinfo is None
                else date.tz_convert("UTC")
            ).normalize()
            numerator = float(action["new_rate"])
            denominator = float(action["old_rate"])
            multiplier = numerator / denominator
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            raise RuntimeError("Corporate Action split sem dados auditáveis.") from exc
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise RuntimeError("Corporate Action split com fator inválido.")
        # Evento anterior à primeira observação já pertence à unidade do RAW.
        # Evento posterior à última barra não pode afetar nenhum instante atual.
        if start < date <= end:
            events.append((date, float(multiplier)))
    return tuple(sorted(events, key=lambda x: x[0]))


def construir_visao_economica(
    raw: pd.DataFrame,
    events: tuple[tuple[pd.Timestamp, float], ...],
) -> pd.DataFrame:
    """Constrói vista split-neutral prospectiva; input RAW não se modifica."""
    columns = ("open", "high", "low", "close", "volume")
    if not set(columns).issubset(raw.columns):
        raise RuntimeError("OHLCV completo obrigatório.")
    before = fingerprint_frame(raw)
    dates = pd.DatetimeIndex(raw.index)
    if dates.tz is None or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise RuntimeError("Calendário RAW inválido.")
    day = dates.tz_convert("UTC").normalize()
    weight = np.ones(len(raw), dtype=float)
    cumulative = 1.
    start = 0
    for date, factor in events:
        boundary = int(day.searchsorted(date, side="left"))
        if boundary < start:
            raise RuntimeError("Eventos não ordenados.")
        weight[start:boundary] = cumulative
        cumulative *= factor
        if not np.isfinite(cumulative) or cumulative <= 0:
            raise RuntimeError("Fator acumulado inválido.")
        start = boundary
    weight[start:] = cumulative
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise RuntimeError("Fator de quantidade inválido.")
    view = raw.copy(deep=True)
    for column in ("open", "high", "low", "close"):
        view[column] = raw[column].to_numpy(dtype=float) * weight
    view["volume"] = raw["volume"].to_numpy(dtype=float) / weight
    if not np.isfinite(view[list(columns)].to_numpy(dtype=float)).all():
        raise RuntimeError("Vista econômica contém valores não finitos.")
    if fingerprint_frame(raw) != before:
        raise RuntimeError("Violação: RAW alterado.")
    return view


def restaurar_execucao_raw(
    feature_frame: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Retém indicadores/labels econômicos, mas negocia OHLCV RAW literal."""
    fields = ["open", "high", "low", "close", "volume"]
    if not feature_frame.index.isin(raw.index).all():
        raise RuntimeError("Sem barras RAW para todo o calendário de decisão.")
    output = feature_frame.copy(deep=True)
    output[fields] = raw.loc[output.index, fields]
    return output
