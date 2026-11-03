"""Pesquisa offline: CASH quando o score de calibracao selecionado e negativo.

Executar no Spyder (F5) ou terminal. Nao faz downloads, treina modelos,
altera o Control nem reconstitui scores diarios inexistentes.

A inferencia e EXATA somente se todos os folds ficarem em CASH. Quando houver
folds mistos, o experimento interrompe: o replay contrafactual teria de
executar novamente a politica com o estado de posicao efetivamente modificado.
"""
# %% Imports e entradas
from __future__ import annotations
import argparse
import io
import json
import math
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent
PACOTE_PADRAO = Path('output/reproducao_universo/pacote_reproducao_universo.zip')
SAIDA_PADRAO = Path('output/pesquisa_cash_calibracao')


def _ler(pacote: Path):
    with zipfile.ZipFile(pacote) as z:
        nomes = set(z.namelist())
        esperados = {'fold_calibration.csv', 'predictions.csv', 'reproducao_universo.json'}
        if not esperados.issubset(nomes):
            raise ValueError('Pacote nao contem as evidencias de calibracao e replay exigidas.')
        folds = pd.read_csv(io.BytesIO(z.read('fold_calibration.csv')))
        previsoes = pd.read_csv(io.BytesIO(z.read('predictions.csv')))
        manifesto = json.loads(z.read('reproducao_universo.json'))
    return folds, previsoes, manifesto


def analisar(pacote: Path, saida: Path) -> dict:
    # %% Contrato temporal e identidade da reproducao
    folds, previsoes, manifesto = _ler(pacote)
    if manifesto.get('status') != 'completed' or not manifesto.get('reproduced_exact_to_cent'):
        raise ValueError('Somente reproducao oficial concluida e conferida e aceita.')
    inicial = float(manifesto['metrics']['folds'][0]['strategy_starting_capital'])
    esperado = float(manifesto['metrics']['ending_capital'])
    if not (math.isfinite(inicial) and inicial > 0 and math.isfinite(esperado)):
        raise ValueError('Capital da referencia invalido.')
    if len(previsoes) == 0:
        raise ValueError('O replay original nao contem sessoes.')
    ultimo = float(previsoes['strategy_equity'].iloc[-1])
    if abs(ultimo - esperado) > 0.01:
        raise ValueError('Equity historica divergiu do manifesto. Nao usar este pacote.')
    vencedores = folds.loc[folds['selected'].astype(str).str.lower().isin(('true', '1'))].copy()
    vencedores['fold_id'] = vencedores['fold_id'].astype(int)
    vencedores['calibration_score'] = pd.to_numeric(vencedores['calibration_score'], errors='raise')
    if vencedores['fold_id'].duplicated().any():
        raise ValueError('Mais de uma margem selecionada por fold.')
    manifest_folds = manifesto['metrics']['folds']
    if set(vencedores['fold_id']) != {int(f['fold_id']) for f in manifest_folds}:
        raise ValueError('Fold selecionado nao corresponde ao conjunto temporal do manifesto.')
    registros = []
    for fold in manifest_folds:
        ident = int(fold['fold_id'])
        selecao = vencedores.set_index('fold_id').loc[ident]
        fim_calibracao = pd.to_datetime(fold['calibration_end'], utc=True)
        inicio_teste = pd.to_datetime(fold['test_start'], utc=True)
        if fim_calibracao >= inicio_teste:
            raise ValueError(f'Vazamento temporal no fold {ident}.')
        score = float(selecao['calibration_score'])
        if not math.isfinite(score):
            raise ValueError('Pontuacao de calibracao nao finita.')
        registros.append({
            'fold_id': ident,
            'calibration_end': fim_calibracao.isoformat(),
            'test_start': inicio_teste.isoformat(),
            'test_end': str(fold['test_end']),
            'calibration_score': score,
            'candidate_margin': float(selecao['candidate_margin']),
            'gate_cash': bool(score < 0.0),
            'control_fold_return': float(fold['strategy_return']),
            'control_fold_end_capital': float(fold['strategy_ending_capital']),
        })
    tabela = pd.DataFrame(registros)
    # Para folds positivos, usar retornos da trajetoria original seria
    # incorreto: a posicao herdada pode ter mudado durante o CASH anterior.
    if not tabela['gate_cash'].all():
        raise RuntimeError(
            'Este primeiro teste so e exato no caso ALL-CASH; para folds '+
            'mistos precisamos de replay da politica com estado da carteira.'
        )

    # %% Contrafactual exato: com todos os folds vetados nao ha ordens
    datas = pd.to_datetime(previsoes['timestamp'], utc=True)
    if not datas.is_monotonic_increasing or datas.duplicated().any():
        raise ValueError('Calendario do replay original invalido.')
    serie = pd.DataFrame({
        'timestamp': datas.dt.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'control_equity': previsoes['strategy_equity'].astype(float),
        'cash_gate_equity': inicial,
        'fold_id': previsoes['walk_forward_fold'].astype(int),
    })
    if set(serie['fold_id']) != set(tabela['fold_id']):
        raise ValueError('Folds do replay e do manifesto nao correspondem.')
    out = {
        'hipotese': 'entrar_em_cash_se_calibration_score_do_fold_menor_que_zero',
        'tipo': 'contrafactual_exato_somente_com_gate_ativo_em_todos_os_folds',
        'fonte': pacote.name,
        'reproducao_original': manifesto.get('reproduction_version'),
        'folds_avaliados': len(tabela),
        'folds_score_negativo': int(tabela['gate_cash'].sum()),
        'sessoes_fora_amostra': len(serie),
        'control_capital_inicial': inicial,
        'control_capital_final': esperado,
        'cash_gate_capital_final': inicial,
        'cash_gate_exposicao': 0.0,
        'cash_gate_drawdown_maximo': 0.0,
        'cash_gate_sharpe': None,
        'cash_gate_rendimento_juros': 'zero_por_hipotese',
        'diferenca_capital_final_cash_menos_control': inicial - esperado,
        'regra_predefinida': 'score < 0',
        'limite': 'avaliacao por fold; nao equivale a recalibracao diaria do sistema operacional',
    }

    # %% Persistencia do relatorio, sem mutar os dados originais
    saida.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(saida / 'folds_calibracao_e_decisao_cash.csv', index=False)
    serie.to_csv(saida / 'patrimonio_control_vs_cash.csv', index=False)
    (saida / 'resultado.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(datas, serie['control_equity'], label='Control original', lw=2.2)
    ax.plot(datas, serie['cash_gate_equity'], label='CASH: calibração < 0', lw=2.0)
    ax.set_yscale('log')
    ax.set_ylabel('Patrimônio em dólares (escala logarítmica)')
    ax.set_xlabel('Sessões fora da amostra')
    ax.set_title('TCC: regra de CASH pela calibração, experimento histórico')
    ax.grid(alpha=0.23)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(saida / 'control_vs_cash_calibracao.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pacote', type=Path, default=PACOTE_PADRAO)
    parser.add_argument('--saida', type=Path, default=SAIDA_PADRAO)
    args, _ = parser.parse_known_args()
    resultado = analisar(args.pacote, args.saida)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
