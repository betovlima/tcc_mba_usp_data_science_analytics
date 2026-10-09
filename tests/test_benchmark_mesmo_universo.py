"""Contrato financeiro da comparacao no mesmo universo fixo."""
import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from engine.rotacao import _benchmark_pesos_iguais, _simular_exato
from reproducao.experimento import summarize_metrics


def _precos():
    dates = pd.date_range('2020-01-01', periods=3, tz='UTC')
    frames = {
        'AAA': pd.DataFrame({'open': [10.0, 10.0, 20.0], 'close': [10.0, 10.0, 20.0]}, index=dates),
        'BBB': pd.DataFrame({'open': 20.0, 'close': 20.0}, index=dates),
    }
    for frame in frames.values():
        frame['high'] = frame['close']
        frame['low'] = frame['close']
    return frames, dates


def _taxa_fixa(side, quantity, price, config):
    return {'commission_fee': 1.0, 'sec_fee': 0.0, 'taf_fee': 0.0, 'cat_fee': 0.0, 'total_fee': 1.0}


def _sem_deslizamento(price, side, config):
    return price


def test_compra_manutencao_divide_capital_igualmente_e_liquida_com_taxas():
    frames, dates = _precos()
    curve = _benchmark_pesos_iguais(frames, ['AAA', 'BBB'], dates[1:], 1000.0, CONFIG, _taxa_fixa, _sem_deslizamento)
    # US$ 500 por ativo, menos US$ 1 por compra. AAA dobra e BBB permanece
    # constante. A liquidacao final desconta mais US$ 1 de cada ativo.
    assert curve.tolist() == pytest.approx([998.0, 1495.0])


@pytest.mark.parametrize('invalid_price', [np.nan, 0.0, -1.0, np.inf])
def test_preco_invalido_nao_reduz_silenciosamente_universo(invalid_price):
    frames, dates = _precos()
    frames['BBB'].loc[dates[-1], 'close'] = invalid_price
    with pytest.raises(ValueError, match='BBB: incomplete or invalid prices'):
        _benchmark_pesos_iguais(frames, ['AAA', 'BBB'], dates[1:], 1000.0, CONFIG, _taxa_fixa, _sem_deslizamento)


def test_ativo_ausente_nao_reduz_silenciosamente_universo():
    frames, dates = _precos()
    del frames['BBB']
    with pytest.raises(ValueError, match='BBB: missing prices'):
        _benchmark_pesos_iguais(frames, ['AAA', 'BBB'], dates[1:], 1000.0, CONFIG, _taxa_fixa, _sem_deslizamento)


def test_sessao_ausente_nao_reduz_silenciosamente_universo():
    frames, dates = _precos()
    frames['BBB'] = frames['BBB'].drop(index=dates[-1])
    with pytest.raises(ValueError, match='BBB: incomplete or invalid prices'):
        _benchmark_pesos_iguais(frames, ['AAA', 'BBB'], dates[1:], 1000.0, CONFIG, _taxa_fixa, _sem_deslizamento)


@pytest.mark.parametrize('symbols', [[], ['AAA', 'AAA']])
def test_universo_vazio_ou_duplicado_e_rejeitado(symbols):
    frames, dates = _precos()
    with pytest.raises(ValueError, match='unique assets'):
        _benchmark_pesos_iguais(frames, symbols, dates[1:], 1000.0, CONFIG, _taxa_fixa, _sem_deslizamento)


def test_replay_compara_universo_inteiro_mesmo_quando_rotacao_so_compra_um_ativo():
    frames, dates = _precos()
    config = CONFIG.copiar_modelo(update={'assets': ('AAA', 'BBB'), 'initial_capital': 1000.0})
    result = _simular_exato('test', lambda date, position, holding: (1, 1.0), frames, ['AAA', 'BBB'], dates, config, _taxa_fixa, _sem_deslizamento)
    metrics = summarize_metrics(result, [], config.initial_capital)
    assert set(result.trades['asset']) == {'AAA'}
    assert metrics['benchmark_assets'] == ['AAA', 'BBB']
    assert metrics['benchmark_asset_count'] == 2
    assert metrics['benchmark_same_universe'] is True
    assert result.predictions['buy_hold_equity'].tolist() == pytest.approx([998.0, 1495.0])
    assert metrics['ending_capital'] == pytest.approx(1997.0)
    assert metrics['buy_hold_ending_capital'] == pytest.approx(1495.0)
