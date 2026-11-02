"""Nenhuma contagem de universo, folds ou calendario fica fixa no executor."""
from __future__ import annotations

import pandas as pd
import pytest

from pesquisa.validacao_dimensoes_motivos_u75 import validar_dimensoes_reproducao


@pytest.mark.parametrize(("universe_size", "fold_count"), [
    (65, 2), (75, 3), (82, 4),
])
def test_dimensions_are_read_from_reference_contract(universe_size, fold_count):
    names = [f"S{i:03d}" for i in range(universe_size)]
    dates = pd.date_range("2020-01-02", periods=13, freq="B", tz="UTC")
    folds = [{"fold_id":i} for i in range(1,fold_count+1)]
    mapping = {d:(i % fold_count)+1 for i,d in enumerate(dates[:-1])}
    contract = {
        "assets":names,
        "effective_asset_count":universe_size,
        "expected_rows":len(dates)-1,
        "expected_fold_counts":{
            str(i):1 for i in range(1,fold_count+1)
        },
    }
    outcome=validar_dimensoes_reproducao(
        names,folds,dates,mapping,contract,
    )
    assert outcome["ativos"] == universe_size
    assert outcome["folds"] == fold_count
    assert outcome["datas_de_decisao"] == len(dates)


def test_reference_rejects_extra_dates_without_modifying_frozen_result():
    names = ["AAA","BBB"]
    dates = pd.date_range("2020-01-02", periods=8, freq="B", tz="UTC")
    mapping = {d:1 for d in dates[:-1]}
    contract = {
        "assets":names, "effective_asset_count":2,
        "expected_rows":len(dates)-2,
        "expected_fold_counts":{"1":6},
    }
    with pytest.raises(RuntimeError, match="extensao temporal"):
        validar_dimensoes_reproducao(
            names,[{"fold_id":1}],dates,mapping,contract,
        )


def test_reference_rejects_missing_assets_even_with_correct_size():
    dates = pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC")
    contract = {
        "assets":["AAA","BBB"], "effective_asset_count":2,
        "expected_rows":2, "expected_fold_counts":{"1":2},
    }
    with pytest.raises(RuntimeError, match="Universo preparado"):
        validar_dimensoes_reproducao(
            ["AAA","CCC"],[{"fold_id":1}],dates,
            {d:1 for d in dates[:-1]},contract,
        )


def test_reference_rejects_duplicate_fold_ids():
    names=["AAA"]
    dates = pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC")
    contract = {
        "assets":names, "effective_asset_count":1,
        "expected_rows":2, "expected_fold_counts":{"1":1,"2":1},
    }
    with pytest.raises(RuntimeError, match="Folds preparados"):
        validar_dimensoes_reproducao(
            names,[{"fold_id":1},{"fold_id":1}],dates,
            {d:1 for d in dates[:-1]},contract,
        )
