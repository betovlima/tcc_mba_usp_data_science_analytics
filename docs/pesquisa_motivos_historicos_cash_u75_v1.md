# Pesquisa: movimentos históricos do U75 e permanência em CASH

**Versão:** \`u75-historical-motifs-cash-v1.0.0\`  
**Branch:** \`research/u75-historical-motifs-cash-v1\`  
**Executor Spyder:** \`pesquisar_motivos_historicos_cash_u75_spyder.py\`  
**Escopo:** TCC somente; nenhuma alteração no ambiente operacional, no Control ou nos parâmetros atuais.

## Hipótese

Se o movimento recente do U75 ou do ativo que a política pretende manter for parecido com episódios anteriores que **continuaram caindo**, seria possível substituir temporariamente a posição por **CASH**, decidindo ao fechamento e executando na abertura seguinte, sem fixar uma duração obrigatória em dinheiro? 

Este é um novo teste matemático e estatístico. **Não** busca chegar a US$ 587 milhões; os checkpoints financeiros da Strategy #30 servem apenas para auditar a reprodução exata do Control antes de comparar políticas. A prioridade científica é perda evitada, ES95, drawdown e recuperação, não maximizar patrimônio terminal.

## Contrato dos dados

Reaproveitar o snapshot imutável Strategy #30:
- \`dados/strategy30_snapshot/raw_bars/*.csv\` e \`corporate_actions/*.csv\`;
- ZIP original da Strategy #30 identificado por \`pesquisa/control_oficial_strategy30.json\`;
- 75 ativos exatos, três folds originais, calendário e custos iguais ao Control;
- sem download automático, banco, preenchimento ou reconstrução de histórico quebrado.

A execução primeiro **treina os 75 modelos originais por fold, recalibra a margem pelo procedimento oficial e compara cada patrimônio e posição com o ZIP congelado**. Se a paridade falhar, nenhum cenário alternativo pode ser aceito.

## Método matemático

Em cada fechamento \`t\`, para cada consulta, construir a trajetória de log-preços nos \`w\` pregões anteriores, separadamente normalizada por canal (forma temporal, não nível absoluto de preço):

- **canal coletivo**: índice diário equal-weight diagnóstico dos 75 ativos, sem pressupor que essa seja a posição real da carteira;
- **canal individual**: histórico do ativo candidato escolhido pela política base no estado **efetivo** da simulação.

A busca é um **distance profile causal**: compara cada consulta com todas as subsequências históricas disponíveis, pré-seleciona as mais próximas por distância euclidiana normalizada e reordena por **DTW multivariado com banda de Sakoe-Chiba**.

**Terminologia precisa:** não é o algoritmo integral STOMP/STUMPY de Matrix Profile, mas uma consulta causal de perfil de distâncias entre subsequências, seguida de DTW. Foi escolhida para não exigir biblioteca externa além das dependências já presentes no TCC e para permitir auditoria de causalidade.

Cada episódio anterior possui fim \`j\` e um retorno subsequente observado somente até \`j+h\`. Ele é elegível se, e só se:

\[
j+h\leq t.
\]

Não consultar retorno de \`t+1\` na escolha, não misturar resultados posteriores ao corte, não converter dado ausente em retorno zero. Episódios selecionados exigem pelo menos \`h\` pregões de distância entre suas datas finais, para reduzir contagem duplicada do mesmo resultado futuro.

A partir dos retornos \`r_k=\log(P_{j_k+h}/P_{j_k})\`, o risco é avaliado por estimativa de média \`m\` e limite unilateral aproximado:

\[
U = \overline r + 1{,}645 \frac{s(r)}{\sqrt{k}}.
\]

Um sinal exploratório de redução de exposição ocorre se **\`U < 0\`**, isto é, mesmo o limite superior aproximado da média dos análogos permanece negativo. A estimativa da fração de perdas e o ES95 amostral dos análogos também são registrados. Com poucas observações e episódios economicamente dependentes, **o valor 1,645 não garante cobertura de 95%**. A hipótese é estatística exploratória, não certificação de risco.

Os padrões e retornos de análogos históricos são recalculados por data, sempre respeitando maturação. Não há treinamento com dados posteriores à decisão nem hiperparametrização pelo capital final.

## Comparações pré-declaradas

| Cenário | Janela / horizonte | Causa potencial de CASH |
|---|---|---|
| \`Control\` | Motor original | Sem nova proteção |
| \`market_w20_h5\` | 20 / 5 pregões | Padrão adverso no U75 agregado |
| \`market_or_asset_w20_h5\` | 20 / 5 pregões | Padrão adverso no U75 **ou** no ativo que a política escolheu manter |
| \`market_or_asset_w40_h10\` | 40 / 10 pregões | Sensibilidade a trajetória e horizonte maiores |

O protocolo usa até 48 candidatos na distância euclidiana, 16 análogos após DTW, banda 4, histórico mínimo 252 e z=1,645. Nenhum desses valores deve ser ajustado ao observar o resultado financeiro deste replay; são escolhas piloto que exigem validação de sensibilidade em cortes futuros. A configuração é exposta explicitamente no script e registrada no JSON.

**Uma avaliação de rotação por pregão**. Quando uma condição adversa é identificada, o alvo do dia é CASH; na próxima decisão, o Control original é consultado com o estado **real da política protegida**, inclusive a posição CASH e seus dias de permanência, podendo reentrar imediatamente se não houver mais sinal. Não existem travas artificiais de permanência ou rotações no mesmo dia.

O Control original **não é renomeado nem corrigido silenciosamente**. O mecanismo herdado de elegibilidade relacionado à sessão seguinte continua uma limitação de causalidade conhecida e será explicitado no resultado. Os sinais novos são construídos com informação temporal causal; isso **não transforma o Control original em prova de comportamento operacional livre de lookahead**.

## Validação e interpretação

As perdas em AMS (maio-junho), CAPR (julho) e CAPR (setembro-outubro) **já foram inspecionadas pelo autor**. Podemos verificar como o piloto teria se comportado nessas datas, mas não chamá-las de holdout virgem nem ajustar o método olhando seus resultados.

Em \`output/motivos_historicos_cash_u75/<timestamp>/\`, analisar:

- \`control_paridade.json\` e \`control_paridade_diaria.csv\`: identidade real do Control;
- \`indice_u75_cobertura.csv\`: quantidade de ativos válidos e retornos equal-weight;
- \`*_decisoes_posicoes.csv\`: cada decisão, sinal coletivo/individual, motivo da escolha, posição real e patrimônio;
- \`*_operacoes.csv\`: custos e negociações do simulador oficial;
- \`patrimonio_cenarios.csv\` e \`patrimonio_comparado.png\`: trajetórias, incluindo CASH;
- \`metricas_pareadas.csv\`: patrimônio final **secundário**, drawdown máximo, ES95 empírico dos retornos diários da carteira e dias de perda;
- \`resultado.json\`, \`resumo.md\`: limites e dados da execução.

O ES95 da carteira é a média das piores 5% sessões, baseada em retornos diários **do replay** e não deve ser confundido com a previsão de ES95 entre análogos. Caso o script falhe, ele gera \`ultimo_erro.txt\` e um pacote ZIP de falha, sem alegar \`completed\`.

### Testes

\`python -m unittest discover -s tests -p test_motivos_historicos_cash_u75.py -v\`

Os testes sintéticos cobrem: distância nula em sequências iguais, invariância do sinal perante mudanças em preços posteriores, maturação dos resultados dos análogos, dados ausentes, preservação do universo U75, passagem da posição real para o CASH e reentrada sem duração fixa. Estes testes não substituem a execução com os 75 CSVs reais.

## Critério de evolução

Se os sinais reduzirem drawdowns e ES95 sem excesso de falsos alertas nem prejuízo desproporcional nas recuperações, o próximo trabalho será (a) medir intervalos de confiança por episódios não sobrepostos e (b) validar em dados de corte posterior que não foram usados para formular essa hipótese.

Não promover ao ambiente operacional por esta execução exploratória. A pesquisa externa ao U75 fica para uma etapa separada; primeiro verificar se manter CASH teria servido como proteção.
