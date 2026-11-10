"""TCC | Triagem historica de CASH sobre a carteira Control da Winner.

IMPORTANTE: universo configurado 77, mas o backtest historico tem 75 series
elegiveis. Este estudo NAO e um teste com os 77 ativos completos e nao
substitui a pesquisa causal em pesquisar_protecao_reversao_spyder.py.

Spyder: F5. Procura o backtest historico congelado em dados/ e grava
resultados isolados em out/reversao_cash_overlay_historico_u75_20261110/.

PROTOCOLO: treino=fold 1; calibracao de limiar=fold 2; teste=fold 3.
O replay e um overlay APROXIMADO sobre retornos do Control; nao reproduz
fills, impacto, retreinamento, rotacoes alternativas nem estado da corretora.
CASH no pregao t e decidido pelas variaveis do fechamento anterior.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ / 'dados'
PASTA_SAIDA = RAIZ / 'out' / 'reversao_cash_overlay_historico_u75_20261110'
ZIP_SAIDA = PASTA_SAIDA / 'pacote_protecao_reversao_overlay_historico_u75.zip'
BACKTEST_SHA256 = '1cdeda67bafccf424a75b675b5f16084b14c282ec1ba1b9599c6e234312dc0f2'
SEED = 42
PERDA_SEVERA = -0.03
CUSTO_IDA_VOLTA = 0.001  # 10 pontos-base, custo extra de overlay, nao custo de broker
MAX_CASH_VALIDACAO = 0.25
MIN_DIAS_CASH_VALIDACAO = 6
LIMIARES = np.round(np.arange(.20, .901, .025), 4)

# Variaveis do estado do mercado e do candidato, registradas antes da execucao.
# Campos de estado da posicao sao apenas diagnosticos: usalos na reentrada de
# um replay stateful exigiria reconstruir a posicao da carteira alternativa.
ATRIBUTOS = [
    'raw_best_score', 'second_score', 'best_vs_second_gap',
    'universe_score_mean', 'universe_score_std', 'best_score_zscore',
    'positive_score_count', 'spy_return_5', 'spy_return_20',
    'spy_realized_volatility_20', 'universe_breadth_5', 'universe_breadth_20',
]

# So para veto de uma sessao: estado do ativo atual no Control.
# Estes atributos NAO autorizam replay stateful de reentrada em CASH.
ATRIBUTOS_POSICAO = ATRIBUTOS + [
    'current_score', 'position_return_since_entry', 'position_drawdown_from_peak',
    'position_mae_so_far', 'score_change_from_entry',
    'current_asset_rank', 'consecutive_days_current_not_top1',
]

def carregar():
    candidatos = list(PASTA_DADOS.glob('market_cycle_trader_*.zip'))
    candidatos += list(RAIZ.glob('market_cycle_trader_*.zip'))
    if not candidatos:
        raise FileNotFoundError('Backtest nao encontrado em dados/. Coloque o ZIP original nessa pasta.')
    pacotes_certos = [p for p in candidatos if hashlib.sha256(p.read_bytes()).hexdigest() == BACKTEST_SHA256]
    if not pacotes_certos:
        raise ValueError(
            'E preciso o ZIP historico exato (SHA-256 ' + BACKTEST_SHA256 + '). '
            'Nenhum dado U67, U75 ou de outros backtests sera substituido automaticamente.'
        )
    pacote = max(pacotes_certos, key=lambda p: p.stat().st_mtime)
    with zipfile.ZipFile(pacote) as z:
        prefix = [x for x in z.namelist() if x.startswith('PORTFOLIO_') and x.endswith('_predictions.csv')]
        met = [x for x in z.namelist() if x.startswith('PORTFOLIO_') and x.endswith('_metrics.json')]
        if len(prefix) != 1 or len(met) != 1:
            raise ValueError('Esperado um arquivo de predictions e um metrics em PORTFOLIO_.')
        df = pd.read_csv(io.BytesIO(z.read(prefix[0])), low_memory=False)
        metrics = json.loads(z.read(met[0]))
        manifest = json.loads(z.read('experiment_manifest.json'))
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['decision_date'] = pd.to_datetime(df['decision_date'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.timestamp.duplicated().any() or df.decision_date.isna().any():
        raise ValueError('Calendario invalido ou timestamps duplicados.')
    if not (df.decision_date < df.timestamp).all():
        raise ValueError('Risco de vazamento temporal: decision_date nao anterior ao timestamp.')
    for nome in ATRIBUTOS + ['strategy_equity']:
        if nome not in df:
            raise ValueError('Coluna obrigatoria ausente: ' + nome)
    if df['walk_forward_fold'].nunique() != 3:
        raise ValueError('Este estudo requer tres folds numerados 1/2/3.')
    if int(manifest['configured_asset_count']) != 77 or int(manifest['eligible_asset_count']) != 75:
        raise ValueError('Universo diferente do registro oficial de 77 configurados / 75 elegiveis.')
    eq = df.strategy_equity.to_numpy(float)
    if not np.all(np.isfinite(eq)) or np.any(eq <= 0):
        raise ValueError('Patrimonio inconsistente.')
    if abs(eq[-1] - float(metrics['strategy_ending_capital'])) > 0.02:
        raise ValueError('Patrimonio diverge entre serie e metricas.')
    inicial = float(metrics['initial_capital'])
    df['retorno_controle'] = np.diff(np.r_[inicial, eq]) / np.r_[inicial, eq[:-1]]
    if (df['retorno_controle'] <= -1).any():
        raise ValueError('Retorno invalido para capitalizacao.')
    df['perda_severa'] = (df['retorno_controle'] <= PERDA_SEVERA).astype(int)
    if not np.isclose(np.prod(1.0 + df['retorno_controle'].to_numpy(float)), eq[-1]/inicial, rtol=1e-9):
        raise ValueError('Retornos do controle nao reconstroem o patrimonio final.')
    sha = hashlib.sha256(pacote.read_bytes()).hexdigest()
    return pacote, df, metrics, manifest, sha


def simular(sub, sinal_cash, custo=CUSTO_IDA_VOLTA, capital_inicial=1.0):
    """Substitui o fator realizado do controle por 1 em dias de CASH.

    E um teste de triagem de sinais, NAO um contrafactual de execucao:
    custo extra aproximado por mudanca market/cash e reentrada no final.
    """
    m = np.asarray(sinal_cash, dtype=bool)
    r = sub['retorno_controle'].to_numpy(float)
    if len(m) != len(r):
        raise ValueError('Mascara e retornos com extensoes distintas.')
    transicoes = np.zeros(len(m), dtype=int)
    anterior = False
    for i, agora in enumerate(m):
        transicoes[i] = int(agora != anterior)
        anterior = agora
    # Custo total adicional equivale a custo/2 por transicao.
    fator = np.where(m, 1.0, 1.0 + r) * (1.0 - custo / 2.0) ** transicoes
    # Cobra custo de reentrada no final, quando a posicao terminar em CASH.
    custo_final = (1.0 - custo / 2.0) if len(m) and m[-1] else 1.0
    caminho = capital_inicial * np.cumprod(fator)
    if len(caminho) and m[-1]:
        caminho[-1] *= custo_final
    capital_final = float(caminho[-1]) if len(caminho) else float(capital_inicial)
    maxdd = float(np.min(caminho / np.maximum.accumulate(np.r_[capital_inicial, caminho])[1:] - 1.0)) if len(m) else 0.0
    daily = fator - 1
    q = float(np.quantile(daily, .05)) if len(daily) else None
    es = float(np.mean(daily[daily <= q])) if len(daily) else None
    episodios = int(np.sum(m & ~np.r_[False, m[:-1]])) if len(m) else 0
    return {
        'final': capital_final, 'drawdown': maxdd, 'cash_dias': int(m.sum()),
        'cash_fracao': float(np.mean(m)) if len(m) else 0.0,
        'cash_episodios': episodios, 'custo_aproximado': float(np.sum(transicoes)*custo/2 + (int(m[-1])*custo/2 if len(m) else 0)),
        'severa_perda_coberta': int((m & sub['perda_severa'].to_numpy(bool)).sum()),
        'perda_severa_total': int(sub['perda_severa'].sum()),
        'dias_alta_perdidos': int((m & (r >= 0.03)).sum()),
        'expected_shortfall_5pct': es,
        'caminho': caminho, 'sinal': m,
    }


def usar_probabilidade(proba, limiar):
    # CASH num pregao se a probabilidade prevista superar limiar;
    # em caso contrario, reentra no alvo do proprio Control nesse pregao.
    return proba >= limiar


def selecionar_limiar(cal, proba):
    controle = simular(cal, np.zeros(len(cal), dtype=bool))
    resultados = []
    for t in LIMIARES:
        r = simular(cal, usar_probabilidade(proba, t))
        vantagem = np.log(r['final'] / controle['final'])
        melhora_dd = max(0., -controle['drawdown'] + r['drawdown'])
        r['limiar'] = float(t)
        r['vantagem_log'] = float(vantagem)
        r['score_validacao'] = float(vantagem + .35 * melhora_dd)
        if (r['cash_dias'] >= MIN_DIAS_CASH_VALIDACAO and r['cash_fracao'] <= MAX_CASH_VALIDACAO
                and r['final'] >= controle['final'] and r['drawdown'] >= controle['drawdown'] - .02):
            resultados.append(r)
    if not resultados:
        return None, []
    melhor = max(resultados, key=lambda x: (x['score_validacao'], -x['cash_dias']))
    return melhor['limiar'], resultados


def sem_caminho(obj):
    return {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in obj.items() if k not in ('caminho', 'sinal')}


def intervalo_bootstrap_blocos(relativos_controle, caminho_shadow, blocos=5, replicas=2500):
    """CI exploratorio da vantagem relativa com reamostragem temporal em blocos.

    Reamostra DIFERENCAS logaritmicas por sessao, e nao o patrimonio bruto.
    Nao constitui inferencia de performance de trading live.
    """
    fatores_shadow = caminho_shadow / np.r_[1.0, caminho_shadow[:-1]]
    delta = np.log(fatores_shadow) - np.log1p(np.asarray(relativos_controle, dtype=float))
    n = len(delta)
    rng = np.random.default_rng(SEED)
    pontos = []
    for _ in range(replicas):
        blocos_idx = rng.integers(0, max(1, n - blocos + 1), size=int(np.ceil(n / blocos)))
        amostra = np.concatenate([delta[i:i+blocos] for i in blocos_idx])[:n]
        pontos.append(np.expm1(float(amostra.sum())))
    return [float(x) for x in np.quantile(pontos, [.025, .975])]


def diagnostico_auc(y, p):
    if len(np.unique(y)) < 2:
        return {'auc': None, 'average_precision': None, 'brier': None}
    return {'auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p))}


def desenhar(dados, linha_base, linhas, saida):
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 130})
    fig, ax = plt.subplots(figsize=(12.5, 5))
    x = dados['timestamp'].dt.tz_convert(None)
    ax.plot(x, linha_base, label='Controle (Winner sem CASH)', lw=1.8)
    for titulo, valores in linhas:
        ax.plot(x, valores, label=titulo, lw=1.4)
    ax.set_yscale('log')
    ax.set_title('Holdout: patrimonio relativo em escala logaritmica (shadow replay aproximado)')
    ax.set_ylabel('Multiplo do capital no inicio do teste')
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(saida / '01_patrimonio_holdout.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 4))
    eq = dados['strategy_equity'].to_numpy(float)
    base_dd = eq / np.maximum.accumulate(np.r_[eq[0], eq])[1:] - 1.
    ax.plot(x, base_dd * 100., label='Controle', lw=1.7)
    for titulo, valores in linhas:
        d = valores / np.maximum.accumulate(np.r_[1., valores])[1:] - 1
        ax.plot(x, d * 100., label=titulo, lw=1.2)
    ax.set_title('Drawdown no periodo de teste (shadow aproximado)')
    ax.set_ylabel('Queda desde o topo (%)')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida / '02_drawdown_holdout.png', dpi=150)
    plt.close(fig)


def executar():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    pacote, df, metrics, manifest, sha = carregar()
    treino = df[df['walk_forward_fold'].eq(1)].copy()
    calibracao = df[df['walk_forward_fold'].eq(2)].copy()
    teste = df[df['walk_forward_fold'].eq(3)].copy()
    print('Backtest:', pacote.name, '| Treino/calibracao/teste:', len(treino), len(calibracao), len(teste))
    ytr = treino['perda_severa'].to_numpy(dtype=int)
    ycal = calibracao['perda_severa'].to_numpy(dtype=int)
    yte = teste['perda_severa'].to_numpy(dtype=int)
    modelos = {
      'Regressao_logistica': Pipeline([
          ('imputar', SimpleImputer(strategy='median', keep_empty_features=True)),
          ('escalar', StandardScaler()),
          ('modelo', LogisticRegression(C=0.25, max_iter=3000, class_weight='balanced', random_state=SEED)),
      ]),
      'LightGBM_raso': Pipeline([
          ('imputar', SimpleImputer(strategy='median', keep_empty_features=True)),
          ('modelo', LGBMClassifier(n_estimators=70, learning_rate=.035, max_depth=2, num_leaves=3,
                                   min_child_samples=35, reg_lambda=5.0, verbosity=-1,
                                   n_jobs=1, random_state=SEED)),
      ]),
    }
    modelos['Regressao_logistica_posicao'] = Pipeline([
        ('imputar', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('escalar', StandardScaler()),
        ('modelo', LogisticRegression(C=.25, max_iter=3000, class_weight='balanced', random_state=SEED)),
    ])
    modelos['LightGBM_posicao'] = Pipeline([
        ('imputar', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('modelo', LGBMClassifier(n_estimators=70, learning_rate=.035, max_depth=2, num_leaves=3,
                                 min_child_samples=35, reg_lambda=5.0, verbosity=-1,
                                 n_jobs=1, random_state=SEED)),
    ])
    baseline_test = simular(teste, np.zeros(len(teste), dtype=bool))
    base_eq = np.cumprod(1 + teste['retorno_controle'].to_numpy(float))
    relatorio = {
      'status': 'triagem_historica_u75_sem_backtest_de_execucao',
      'alerta_universo': '77 configurados, 75 elegiveis no backtest historico. Nao e pesquisa com 77 series completas.',
      'arquivo_original': pacote.name, 'arquivo_sha256': sha,
      'configurados': manifest['configured_asset_count'],
      'elegiveis': manifest['eligible_asset_count'],
      'exclusoes': manifest['structural_exclusions'],
      'data_inicio': str(df.iloc[0]['timestamp'].date()),
      'data_fim': str(df.iloc[-1]['timestamp'].date()),
      'modelo_original': manifest.get('winner_engine_compatibility'),
      'controle_capital_inicial': metrics['initial_capital'],
      'controle_capital_final': metrics['strategy_ending_capital'],
      'controle_mdd': metrics['strategy_maximum_drawdown'],
      'replay': 'Fatores diarios do Controle sao zerados em CASH, com custo extra por transicao; nao simula fills nem carteira alternativa completa.',
      'retorno_alvo': 'perda diaria da carteira de -3% ou pior na sessao seguinte ao sinal',
      'treino': {'fold':1,'linhas':len(treino),'perdas_severas':int(ytr.sum())},
      'calibracao': {'fold':2,'linhas':len(calibracao),'perdas_severas':int(ycal.sum())},
      'teste': {'fold':3,'linhas':len(teste),'perdas_severas':int(yte.sum()),'controle':sem_caminho(baseline_test)},
      'modelos': {},
      'advertencias': [
        'Avaliacao do sinal e do overlay shadow; sem precos OHLC por ativo e custos reais de fill, nao e evidencia causal de retorno realizavel.',
        'Variaveis de estado da carteira foram isoladas em modelos adicionais para teste de veto de uma sessao; seriam invalidas para reentrada stateful sem reconstruir o estado.',
        'Limiar calibrado somente no fold 2; fold 3 reservado para o teste independente, sem ajuste retroativo.',
        'Manifest informa split_normalization_uses_future_events=true; investigar possivel informacao futura no ajuste de splits antes de alegar causalidade.',
        'Live Winner 2026-11 nao e identica ao replay de treinamento nem aos retornos da conta real.',
      ],
    }
    linhas=[]
    diario = teste[['timestamp','decision_date','walk_forward_fold','selected_asset','retorno_controle','perda_severa','strategy_equity','universe_breadth_20','raw_best_score']].copy()
    diario['controle_relativo'] = base_eq
    tabela_validacao=[]
    for nome, mod in modelos.items():
        colunas = ATRIBUTOS_POSICAO if nome.endswith('_posicao') else ATRIBUTOS
        Xtr = treino[colunas].replace([np.inf, -np.inf], np.nan)
        Xcal = calibracao[colunas].replace([np.inf, -np.inf], np.nan)
        Xte = teste[colunas].replace([np.inf, -np.inf], np.nan)
        mod.fit(Xtr, ytr)
        pcal = mod.predict_proba(Xcal)[:,1]
        pte = mod.predict_proba(Xte)[:,1]
        limiar, opcoes = selecionar_limiar(calibracao,pcal)
        for o in opcoes:
            tabela_validacao.append({'modelo':nome, **sem_caminho(o)})
        if limiar is None:
            s = np.zeros(len(teste), dtype=bool)
            sc = np.zeros(len(calibracao), dtype=bool)
            status = 'ABSTENCAO: nenhum limiar melhorou o controle na calibracao sob restricoes'
        else:
            s = usar_probabilidade(pte, limiar)
            sc = usar_probabilidade(pcal, limiar)
            status = 'LIMIAR_SELECIONADO_NO_FOLD_2'
        res = simular(teste, s)
        rescal = simular(calibracao, sc)
        ganho = res['final']/baseline_test['final']-1
        relatorio['modelos'][nome] = {
          'status': status, 'limiar_probabilidade':limiar, 'atributos':colunas,
          'somente_veto_uma_sessao':bool(nome.endswith('_posicao')),
          'calibracao_metricas': diagnostico_auc(ycal,pcal),
          'teste_metricas': diagnostico_auc(yte,pte),
          'teste': sem_caminho(res),
          'calibracao':sem_caminho(rescal),
          'vantagem_terminal_teste':float(ganho),
          'intervalo_vantagem_bootstrap_5_sessoes_95pct': intervalo_bootstrap_blocos(
             teste['retorno_controle'].to_numpy(float), res['caminho']),
          'reducao_mdd_teste_pontos_percentuais':100*(res['drawdown']-baseline_test['drawdown']),
          'captura_CAPR_2026_07_30':bool(teste.loc[teste.timestamp.dt.strftime('%Y-%m-%d').eq('2026-07-30')].index.size and
              s[np.where(teste.timestamp.dt.strftime('%Y-%m-%d').eq('2026-07-30'))[0][0]]),
        }
        diario['probabilidade_'+nome] = pte
        diario['cash_'+nome] = s
        diario['capital_relativo_'+nome] = res['caminho']
        linhas.append((nome, res['caminho']))

    # Auditoria Live opcional: agrega contagens sem exportar linhas privadas.
    livelist = list(PASTA_DADOS.glob('mct_live_transaction_audit_*.zip'))
    if livelist:
        with zipfile.ZipFile(max(livelist,key=lambda p:p.stat().st_mtime)) as livezip:
            decisoes = pd.read_csv(io.BytesIO(livezip.read('live_decisions_period.csv')))
            relatorio['live_auditoria'] = {
               'decisoes_salvas':len(decisoes),
               'decisoes_CAPR_hold':int(((decisoes['current_asset']=='CAPR') & (decisoes['target_asset']=='CAPR')).sum()),
               'calibracao_negativa_e_hold':int(((decisoes['calibration_score']<0) & (decisoes['action']=='hold')).sum()),
               'alerta':'Dados Live apenas descritivos; 4 dias nao validam a regra de reversao.'
            }
    (PASTA_SAIDA/'resultado_pesquisa.json').write_text(json.dumps(relatorio,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    diario.to_csv(PASTA_SAIDA/'sessoes_teste_fold3.csv', index=False)
    pd.DataFrame(tabela_validacao).to_csv(PASTA_SAIDA/'calibracao_fold2.csv', index=False)
    desenhar(teste,base_eq,linhas,PASTA_SAIDA)
    capr = diario[diario['selected_asset'].eq('CAPR')].copy()
    capr.to_csv(PASTA_SAIDA/'capr_fold3.csv',index=False)
    texto=[
      '# Triagem historica: reversao da Winner para CASH (U75 observado)',
      '', f'Pacote de referencia: `{pacote.name}`; SHA-256: `{sha}`.',
      '', '**Escopo:** execucao historica com 77 configurados e 75 series elegiveis; nao comprova desempenho para 77 ativos completos. Nao altera a pesquisa causal ja existente.',
      '', '## Referencia e protocolo',
      f"- Universo: {manifest['configured_asset_count']} configurados, {manifest['eligible_asset_count']} elegiveis; DOC e CLMT tem exclusoes estruturais documentadas.",
      f"- Capital final do controle: US$ {metrics['strategy_ending_capital']:,.2f}; MaxDD: {metrics['strategy_maximum_drawdown']:.2%}; CASH: {metrics['cash_days']} dias.",
      f'- Janela: {len(treino)} sessoes treino (fold 1), {len(calibracao)} calibracao (fold 2), {len(teste)} teste intocado (fold 3).',
      '- Alvo: perda de pelo menos 3% na sessao seguinte, prevista com indicadores disponiveis na data da decisao.',
      '- Custo extra ilustrativo de 0,1% por round trip em CASH, aplicado por mudanca de exposicao.',
      '', '## Resultados independentes: fold 3',
      f"Controle: indice final {baseline_test['final']:.5f}, MDD {baseline_test['drawdown']:.2%}, Expected Shortfall dos piores 5% {baseline_test['expected_shortfall_5pct']:.2%}.",
      '', '| Modelo | AUC fold 3 | Limiar calibrado | CASH fold 3 | Protecao das perdas <= -3% | Capital vs controle | MDD fold 3 |',
      '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for nome,a in relatorio['modelos'].items():
        te=a['teste']; au=a['teste_metricas']['auc']; li=a['limiar_probabilidade'];
        texto.append(f"| {nome} | {au:.3f} | {li if li is not None else 'Abstencao'} | {te['cash_dias']} | {te['severa_perda_coberta']}/{te['perda_severa_total']} | {a['vantagem_terminal_teste']:+.2%} | {te['drawdown']:.2%} |")
    texto += ['', '## Incerteza da vantagem terminal',
      'O intervalo de 95% por bootstrap temporal de blocos de 5 sessoes aplica-se somente ao teste de triagem, nao a ganhos reais.']
    for nome,a in relatorio['modelos'].items():
        lo,hi=a['intervalo_vantagem_bootstrap_5_sessoes_95pct']
        texto.append(f'- {nome}: vantagem simulada {a["vantagem_terminal_teste"]:+.2%}; intervalo exploratorio [{lo:+.2%}, {hi:+.2%}].')
    texto += ['', '## Limites da evidencia',
      'O replay substitui somente o fator de retorno do controle nos dias marcados como CASH. O modelo original continua decidindo os ativos nos outros dias.',
      'Isso nao reproduz execucoes Alpaca, precos do ativo para voltar ao mercado, custos e slippage dependentes da liquidez, nem treinamento da politica apos a intervencao.',
      'Os modelos *_posicao usam o estado anterior da posicao do controle, que divergiria apos entrar em CASH; esses resultados sao apenas de triagem de veto pontual.',
      'Nenhum resultado pode ser promovido ao MCT Live sem simulacao financeira fiel e nova validacao OOS.',
      'Existe no manifesto um sinal de uso de eventos futuros na normalizacao de splits; a auditoria causal precisa verificar esse ponto.',
      'Sinais sao do fechamento anterior; perdas overnight ou gaps podem ocorrer antes da abertura e nao sao preveniveis com garantia.',
      'Intervalo bootstrap temporal e exploratorio: nao corrige vieses de simulacao, amostragem de ativos ou instabilidade estrutural.',
      '', '## Arquivos', 'sessoes_teste_fold3.csv, capr_fold3.csv, calibracao_fold2.csv, resultado_pesquisa.json e graficos PNG (gerados localmente, nao versionados).',
    ]
    (PASTA_SAIDA/'relatorio.md').write_text('\n'.join(texto),encoding='utf-8')
    with zipfile.ZipFile(ZIP_SAIDA,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PASTA_SAIDA.iterdir()):
            if f.is_file() and f != ZIP_SAIDA:
                z.write(f,arcname=f.name)
    for nome,a in relatorio['modelos'].items():
        print(nome,a['status'],'limiar=',a['limiar_probabilidade'],
              'AUC=',a['teste_metricas']['auc'],
              'dias_cash=',a['teste']['cash_dias'],'vantagem=',round(a['vantagem_terminal_teste']*100,2),'%')
    print('Resultados em:',ZIP_SAIDA)
    return relatorio

if __name__=='__main__':
    executar()