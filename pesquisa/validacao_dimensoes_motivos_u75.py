"""Dimensions of frozen reference replays, driven by the reference contract.

Do not confuse reference parity with a forward extension. This validates the
*declared* asset/calendar/fold counts, not literals copied from another run.
"""
from __future__ import annotations

from typing import Any


def validar_dimensoes_reproducao(
    symbols: list[str],
    folds: list[dict[str, Any]],
    decisions: object,
    decision_to_fold: dict,
    contract: dict[str, Any],
) -> dict[str, int]:
    """Allow any declared size, but never silently mismatch the frozen source."""
    assets = list(contract["assets"])
    expected_assets = int(contract["effective_asset_count"])
    if expected_assets < 1 or len(assets) != expected_assets:
        raise RuntimeError("Contrato sem universo coerente.")
    if assets != sorted(set(assets)):
        raise RuntimeError("Universo do contrato com duplicatas ou fora de ordem.")
    if sorted(symbols) != assets or len(symbols) != len(set(symbols)):
        raise RuntimeError("Universo preparado difere do contrato da referencia.")

    expected_counts = dict(contract["expected_fold_counts"])
    if not expected_counts or not folds:
        raise RuntimeError("Referencia sem periodos cronologicos declarados.")
    observed_ids = [str(int(fold["fold_id"])) for fold in folds]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected_counts):
        raise RuntimeError("Folds preparados diferem dos identificadores do contrato.")

    expected_rows = int(contract["expected_rows"])
    number_of_decisions = len(decisions)
    if expected_rows < 1 or number_of_decisions != expected_rows + 1:
        raise RuntimeError(
            "Calendario nao corresponde ao replay historico congelado. "
            "Para adicionar pregoes, executar um estudo de extensao temporal "
            "separado, sem fingir que e o mesmo Control."
        )
    if len(decision_to_fold) != expected_rows:
        raise RuntimeError("Datas das politicas por fold incompletas ou duplicadas.")
    if not set(decisions[:-1]) == set(decision_to_fold):
        raise RuntimeError("Datas de decisao das politicas divergem do calendario.")
    return {
        "ativos":expected_assets,
        "folds":len(expected_counts),
        "sessoes_avaliadas":expected_rows,
        "datas_de_decisao":number_of_decisions,
    }
