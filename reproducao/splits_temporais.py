"""Politica temporal de splits: metadados pontuais, RAW imutavel, custodia.

NAO reconstroi anuncios historicos ausentes. `process_date` da Alpaca nao
substitui comprovante `first_seen_at_utc` de disponibilidade da informacao.
O motor oficial permanece intocado; a custodia sintetica NAO e um replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd


SPLIT_TYPES = frozenset({"forward_split", "reverse_split"})
OBSERVATION_FIELD = "first_seen_at_utc"


@dataclass(frozen=True)
class SplitEvent:
    symbol: str
    action_type: str
    ex_date: pd.Timestamp
    old_rate: Decimal
    new_rate: Decimal
    first_seen_at_utc: pd.Timestamp | None
    event_key: str

    @property
    def quantity_factor(self) -> Decimal:
        return self.new_rate / self.old_rate


def _utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}: timestamp invalido") from error
    if pd.isna(stamp):
        raise ValueError(f"{field}: data ausente")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def split_from_action(action: dict[str, Any]) -> SplitEvent | None:
    """Lê fatos da API sem inventar quando o dado ficou disponivel."""
    kind = str(action.get("action_type") or "")
    if kind not in SPLIT_TYPES:
        return None
    symbol = str(action.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Split sem simbolo; nao inferir identidade.")
    ex_date = _utc(action.get("ex_date"), field="ex_date").normalize()
    try:
        old_rate = Decimal(str(action["old_rate"]))
        new_rate = Decimal(str(action["new_rate"]))
    except (KeyError, ArithmeticError, ValueError) as error:
        raise ValueError("Split sem proporcao valida.") from error
    if not (old_rate.is_finite() and new_rate.is_finite() and
            old_rate > 0 and new_rate > 0):
        raise ValueError("Split requer proporcoes positivas e finitas.")
    observed = action.get(OBSERVATION_FIELD)
    first_seen = None if observed is None or str(observed).strip() == "" else _utc(
        observed, field=OBSERVATION_FIELD,
    )
    key = str(action.get("id") or (
        f"{symbol}|{kind}|{ex_date.date()}|{old_rate}:{new_rate}"
    ))
    return SplitEvent(
        symbol, kind, ex_date, old_rate, new_rate, first_seen, key,
    )


def known_as_of(
    events: Iterable[SplitEvent], decision_time: Any,
) -> tuple[SplitEvent, ...]:
    """Somente anuncio observado e datado ate o instante da decisao."""
    instant = _utc(decision_time, field="decision_time")
    return tuple(
        event for event in events
        if event.first_seen_at_utc is not None
        and event.first_seen_at_utc <= instant
    )


def recorded_effective_by(
    events: Iterable[SplitEvent], valuation_time: Any,
) -> tuple[SplitEvent, ...]:
    """Ocorrencia economica efetivada; nao atesta disponibilidade historica."""
    instant = _utc(valuation_time, field="valuation_time")
    return tuple(event for event in events if event.ex_date <= instant)


def apply_custody_split(
    *,
    quantity: Decimal,
    average_unit_cost: Decimal,
    event: SplitEvent,
    valuation_time: Any,
    applied_event_keys: frozenset[str] = frozenset(),
) -> tuple[Decimal, Decimal, frozenset[str]]:
    """Teste da transformacao patrimonial sem editar cotacoes RAW.

    A custodia real precisa lidar com fracoes/cash-in-lieu, eventos
    anunciados/revistos, market timing, contas e corretagem; nao e simulador.
    """
    instant = _utc(valuation_time, field="valuation_time")
    if instant < event.ex_date:
        raise ValueError("Nao se pode efetivar split antes da ex_date.")
    if event.event_key in applied_event_keys:
        raise ValueError("Split aplicado mais de uma vez.")
    if quantity <= 0 or average_unit_cost <= 0:
        raise ValueError("Posicao deve ter quantidade e custo positivos.")
    factor = event.quantity_factor
    result_quantity = quantity * factor
    result_unit_cost = average_unit_cost / factor
    if quantity * average_unit_cost != result_quantity * result_unit_cost:
        raise AssertionError("Split alterou a base economica da posicao.")
    return result_quantity, result_unit_cost, applied_event_keys | {event.event_key}


def inspect_raw_invariance(
    historical_bars: pd.DataFrame,
    future_actions: Iterable[dict[str, Any]],
    *,
    as_of: Any,
) -> dict[str, Any]:
    """Audita prefixo RAW e visibilidade de novos eventos, sem modificar barras.

    Este teste nao equivale a executar e comparar o LightGBM inteiro.
    """
    cutoff = _utc(as_of, field="as_of")
    original = historical_bars.loc[historical_bars.index <= cutoff].copy()
    if original.empty:
        raise ValueError("Nao ha barras anteriores ao cutoff da auditoria.")
    raw_hash_before = pd.util.hash_pandas_object(original, index=True).to_numpy()
    parsed = [event for value in future_actions if (event := split_from_action(value))]
    # A politica estrita apenas registra os eventos, nao os aplica as barras.
    untouched = original.copy()
    if not np.array_equal(
        raw_hash_before,
        pd.util.hash_pandas_object(untouched, index=True).to_numpy(),
    ):
        raise AssertionError("OHLCV alterado por metadados de splits.")
    known = known_as_of(parsed, cutoff)
    return {
        "cutoff": cutoff.isoformat(),
        "bars_tested": len(original),
        "future_events_supplied": len(parsed),
        "known_announcements_as_of_cutoff": len(known),
        "events_without_verified_first_seen": sum(
            event.first_seen_at_utc is None for event in parsed
        ),
        "events_already_effective": len(recorded_effective_by(parsed, cutoff)),
        "raw_values_preserved": True,
        "causal_limit": (
            "Teste de RAW/metadados, nao e comparacao de previsoes LightGBM. "
            "Sem log historico first_seen_at_utc nao afirmar quando a API "
            "disponibilizou o anuncio."
        ),
    }
