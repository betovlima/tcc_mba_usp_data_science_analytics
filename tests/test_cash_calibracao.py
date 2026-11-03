"""Guardrails da pesquisa de CASH com o score de calibracao por fold."""
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pesquisar_cash_calibracao_spyder import analisar


def _fixture(tmp_path: Path, *, scores=(-0.4, -0.2), leakage=False, corrupt=False):
    df = pd.DataFrame([{
        'scenario': 'TCC_SELECTED_UNIVERSE',
        'fold_id': i,
        'candidate_margin': 0.0,
        'calibration_score': score,
        'selected': True,
    } for i, score in enumerate(scores, start=1)])
    test_dates = ['2021-01-04', '2022-01-04']
    manifest_folds = [{
        'fold_id': i,
        'calibration_end': ('2022-01-04' if leakage and i == 2 else f'{2019+i}-06-30'),
        'test_start': day,
        'test_end': day,
        'strategy_starting_capital': (10000 if i == 1 else 15000),
        'strategy_ending_capital': (15000 if i == 1 else 20000),
        'strategy_return': 0.5 if i == 1 else 1/3,
    } for i, day in enumerate(test_dates, start=1)]
    history = pd.DataFrame({
        'timestamp': ['2021-01-04T00:00:00+00:00', '2022-01-04T00:00:00+00:00'],
        'strategy_equity': [15000, 21000 if corrupt else 20000],
        'walk_forward_fold': [1, 2]
    })
    manifest = {
        'status': 'completed', 'reproduced_exact_to_cent': True,
        'reproduction_version': 'fixture',
        'metrics': {'ending_capital': 20000, 'folds': manifest_folds},
    }
    file = tmp_path / 'fake.zip'
    with zipfile.ZipFile(file, 'w') as z:
        z.writestr('fold_calibration.csv', df.to_csv(index=False))
        z.writestr('predictions.csv', history.to_csv(index=False))
        z.writestr('reproducao_universo.json', json.dumps(manifest))
    return file


def test_always_negative_gate_is_exact_all_cash(tmp_path):
    payload = analisar(_fixture(tmp_path), tmp_path / 'out')
    assert payload['folds_score_negativo'] == 2
    assert payload['cash_gate_capital_final'] == 10000
    assert payload['control_capital_final'] == 20000
    assert payload['cash_gate_sharpe'] is None
    history = pd.read_csv(tmp_path / 'out' / 'patrimonio_control_vs_cash.csv')
    assert list(history.cash_gate_equity) == [10000, 10000]
    assert (tmp_path / 'out' / 'control_vs_cash_calibracao.png').is_file()


def test_mixed_gate_requires_policy_replay_not_naive_mask(tmp_path):
    file = _fixture(tmp_path, scores=(-0.2, 0.3))
    with pytest.raises(RuntimeError, match='replay da politica'):
        analisar(file, tmp_path / 'out')


def test_reject_future_information(tmp_path):
    file = _fixture(tmp_path, leakage=True)
    with pytest.raises(ValueError, match='Vazamento temporal'):
        analisar(file, tmp_path / 'out')


def test_reject_wrong_reference_capital(tmp_path):
    file = _fixture(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match='divergiu'):
        analisar(file, tmp_path / 'out')
