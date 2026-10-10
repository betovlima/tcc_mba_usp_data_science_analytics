"""A decisão não pode depender da existência ou dos preços da sessão seguinte."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from engine.rotacao import (
    ROTATION_FEATURES,
    _executar_compra,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
    _utilidades_modelo,
)


class ModeloConstante:
    def __init__(self, value=0.8):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def _quadros():
    dates = pd.date_range("2026-01-05", periods=3, freq="B", tz="UTC")
    frame = pd.DataFrame(1.0, index=dates, columns=ROTATION_FEATURES)
    for field in ("open", "high", "low", "close"):
        frame[field] = 10.0
    frame["volume"] = 1000.0
    return {"AAA": frame.copy(), "BBB": frame.copy()}, dates


def _scores(models, frames, date, mode):
    symbols = list(frames)
    if mode == "individual":
        return _utilidades_modelo(models, frames, symbols, date, CONFIG)
    cache, _ = _precalcular_utilidades_modelo(
        models, frames, symbols, pd.DatetimeIndex([date]), CONFIG
    )
    return cache[date]


@pytest.mark.parametrize("mode", ["individual", "lote"])
@pytest.mark.parametrize("field", ["open", "close"])
@pytest.mark.parametrize("future_value", [np.nan, 0.0, -1.0, np.inf])
def test_preco_futuro_nao_altera_escore_ou_ativo_escolhido(mode, field, future_value):
    frames, dates = _quadros()
    models = {"AAA": ModeloConstante(0.8), "BBB": ModeloConstante(0.4)}
    before = _scores(models, frames, dates[0], mode)
    frames["AAA"].loc[dates[1], field] = future_value
    after = _scores(models, frames, dates[0], mode)
    np.testing.assert_array_equal(after, before)
    assert np.argmax(after) == 1


@pytest.mark.parametrize("mode", ["individual", "lote"])
def test_ultima_observacao_disponivel_pode_gerar_previsao(mode):
    frames, dates = _quadros()
    models = {"AAA": ModeloConstante(), "BBB": ModeloConstante(0.4)}
    full = _scores(models, frames, dates[0], mode)
    prefix = {symbol: frame.iloc[:1].copy() for symbol, frame in frames.items()}
    np.testing.assert_array_equal(_scores(models, prefix, dates[0], mode), full)


@pytest.mark.parametrize("mode", ["individual", "lote"])
@pytest.mark.parametrize("condition", ["sem_modelo", "sem_sessao", "atributo_ausente", "atributo_infinito"])
def test_informacao_presente_insuficiente_mantem_ativo_inelegivel(mode, condition):
    frames, dates = _quadros()
    models = {"AAA": ModeloConstante(), "BBB": ModeloConstante(0.4)}
    if condition == "sem_modelo":
        models.pop("AAA")
    elif condition == "sem_sessao":
        frames["AAA"] = frames["AAA"].iloc[1:]
    else:
        frames["AAA"].loc[dates[0], ROTATION_FEATURES[0]] = (
            np.nan if condition == "atributo_ausente" else np.inf
        )
    scores = _scores(models, frames, dates[0], mode)
    assert scores[0] == 0.0
    assert scores[1] == -np.inf
    assert scores[2] == pytest.approx(0.4)


def test_previsoes_individuais_e_em_lote_concordam_com_futuro_incompleto():
    frames, dates = _quadros()
    models = {"AAA": ModeloConstante(), "BBB": ModeloConstante(0.4)}
    frames["AAA"].loc[dates[1], "close"] = np.nan
    frames["BBB"].loc[dates[1], ROTATION_FEATURES[1]] = np.nan
    cache, _ = _precalcular_utilidades_modelo(models, frames, list(frames), dates, CONFIG)
    for date in dates:
        np.testing.assert_array_equal(
            cache[date], _utilidades_modelo(models, frames, list(frames), date, CONFIG)
        )


def test_politica_escolhe_mesmo_ativo_quando_so_preco_futuro_muda():
    frames, dates = _quadros()
    models = {"AAA": ModeloConstante(), "BBB": ModeloConstante(0.4)}
    policy = _politica_utilidade(models, frames, list(frames), CONFIG, 0.0005)
    before = policy(dates[0], 0, 0)
    frames["AAA"].loc[dates[1], "close"] = np.nan
    assert policy(dates[0], 0, 0) == before
    assert before[0] == 1


def _sem_taxas(side, quantity, price, config):
    return {"total_fee": 0.0}


def _sem_deslizamento(price, side, config):
    return price


@pytest.mark.parametrize("price", [np.nan, np.inf, 0.0, -1.0])
def test_compra_exige_preco_valido_na_execucao(price):
    with pytest.raises(ValueError, match="execution price"):
        _executar_compra(1000.0, price, CONFIG, _sem_taxas, _sem_deslizamento)


def test_abertura_invalida_interrompe_execucao_sem_reescolher_ativo():
    frames, dates = _quadros()
    frames["AAA"].loc[dates[2], "open"] = np.nan
    calls = []

    def policy(date, position, holding):
        calls.append(date)
        return (1 if date == dates[0] else 2), 0.8

    with pytest.raises(ValueError, match="AAA.*open.*execution"):
        _simular_exato(
            "test", policy, frames, list(frames), dates,
            CONFIG, _sem_taxas, _sem_deslizamento,
        )
    assert calls == [dates[0], dates[1]]
