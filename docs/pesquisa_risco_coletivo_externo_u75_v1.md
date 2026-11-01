# Risco coletivo U75 e oportunidades externas: piloto P1 shadow

**Versão:** \`u75-risk-gated-external-v1.0.0\`  
**Executor Spyder:** \`pesquisar_risco_coletivo_externo_u75_spyder.py\`.  
**Objetivo científico:** verificar se medidas matemáticas e estatísticas do universo U75 antecipam perdas coletivas e se, nos alertas, há candidatos externos disponíveis com condições técnicas mais favoráveis que o Top-1.

O objetivo **não é maximizar US$ 587 milhões** nem repetir a hiperparametrização LightGBM. O patrimônio histórico é somente uma possível conferência técnica de paridade em protocolos próprios, e não é usado como rótulo ou função-objetivo deste experimento.

## Dados e execução

Este ramo parte de \`research/u75-external-leader-oos-results-v1\` e reutiliza os dados existentes, sem downloads:

- \`dados/strategy30_snapshot/\` e o contrato congelado Strategy #30;
- \`dados/cluster_candidate_search/\` com shortlist congelada de 24 ativos, snapshot e auditoria estrutural;
- ZIP MCT auditado contendo o \`raw_best_asset\`, localizado pelo módulo \`pesquisa/localizacao_zip_u75.py\`.

Execute F5 no Spyder, em \`pesquisar_risco_coletivo_externo_u75_spyder.py\`. O programa valida as fontes antes de começar. Os CSVs de candidatos e demais snapshots privados já precisam existir na máquina, conforme pesquisa externa anterior. Não fazer bridge, reconstruir históricos ou baixar dados de outra fonte para preencher lacunas.

As saídas são escritas em \`output/risco_coletivo_externo_u75/\` e empacotadas em \`pacote_risco_coletivo_externo_u75_*.zip\`. O stack trace, em caso de erro, fica em \`ultimo_erro.txt\`.

## Medidas matemáticas

Em cada fechamento t, no universo U75 congelado:

1. Matriz de covariância Ledoit-Wolf calculada somente com os **126 pregões anteriores**, com pelo menos 68 ativos completos, sem imputação.
2. Turbulência de Mahalanobis: \`(r_t - mu)^T inv(Sigma) (r_t - mu)\`. É uma anomalia estatística, não um sinal direcional isolado.
3. Proporção da variância explicada pelo maior autovalor da matriz de correlação robusta (PCA).
4. Fração dos 75 ativos com retorno de 20 pregões negativo, mais retorno coletivo nos horizontes de um e 20 pregões.

O índice equal-weight diário é **diagnóstico**, não representa a carteira real do Control.

## Previsão por análogos cronológicos

O evento ex post é uma perda do índice coletivo nos **cinco pregões seguintes** maior que o quantil inferior de 5% **estimado somente no treino anterior ao teste**.

Para cada ano externo, treina-se apenas com datas cujos rótulos de cinco pregões maturaram antes da primeira decisão desse ano. \`StandardScaler\` também é reajustado exclusivamente no treino. A probabilidade condicional de perda de cauda é estimada pelos \`ceil(sqrt(N))\` análogos históricos mais próximos, com suavização Beta(1,1). O ES95 dos análogos e o ES95 do treino são registrados.

O sinal exploratório é disparado quando a probabilidade condicional supera a incondicional em pelo menos **1,645 erros-padrão binomiais aproximados** e o retorno médio dos análogos é negativo. Trata-se de critério fixo, não otimizado pelo patrimônio. Como os retornos temporais dependem uns dos outros, essa estatística **não garante controle real de erro de 5%**. Na próxima análise, são necessários intervalos em blocos e conferência de calibração. Se não houver alertas, não reduzir o critério depois de olhar o teste.

## Busca externa condicionada ao risco

O gatilho não depende do Top-1; a busca externa somente é avaliada nas datas de alerta do U75. O Top-1 é o \`raw_best_asset\` do ZIP MCT histórico; **não equivale à posição efetivamente mantida**.

Como benchmark estatístico básico e pré-fixado, o escore de cada candidato é \`return_20 / (sqrt(20) * vol_20)\`. O melhor candidato externo é comparado com \`max(0, escore do Top-1)\`. As saídas de observação são \`EXTERNAL\`, \`TOP1\` ou \`CASH\`. Na falta de candidato externo com atributos válidos, o dia recebe \`no_external_snapshot\`; a ausência de dados **não é tratada como sinal de entrar em CASH**.

A escolha não consulta preços da sessão seguinte. Retornos de abertura para fechamento de t+1 são anexados depois, exclusivamente para auditoria. Não converter em patrimônio composto: faltam permanência overnight, execução completa, custos e estado de carteira.

**Limitações conhecidas:** a shortlist de 24 ativos foi congelada em pesquisa histórica e não representa descoberta prospectiva independente de todo o mercado; o \`raw_best_asset\` do motor histórico pode herdar elegibilidade que consulta t+1; análises externas condicionadas ainda não demonstram a capacidade de evitar perdas da posição efetiva. Nenhum resultado financeiro novo é afirmado antes da execução.

## Arquivos

- \`indice_coletivo_u75.csv\`: atributos em t e retornos futuros apenas como rótulos.
- \`previsoes_risco_oos.csv\`: probabilidades, ES95, limiar de cauda treinado, alertas e outcomes.
- \`folds_cronologicos.csv\`: evidência das fronteiras temporais.
- \`decisoes_externas_shadow.csv\`: Top-1, candidato, CASH e outcomes intradiários auditados.
- \`metricas_risco_por_ano.csv\`: precisão, sensibilidade, alarmes falsos e Brier score frente ao benchmark.
- \`probabilidade_queda_u75.png\`, \`resultado.json\`, \`resumo.md\`.

## Testes

\`python -m unittest discover -s tests -p test_risco_coletivo_externo_u75.py -v\`

Cinco testes cobrem alterações de preços futuros, cobertura incompleta, rótulos maturados, preços futuros de candidatos e falta de snapshot externo. Passaram localmente com dados sintéticos, **não representam uma execução do U75 real**.

## Próxima porta de validação

Somente se as probabilidades mostrarem antecipação de risco fora da amostra com falsos alertas toleráveis, fazer replay P2/P3 completo com posição efetiva, custos, CASH e alternativas externas; avaliar perdas evitadas, ES95, drawdown por episódio e oportunidades sacrificadas, com capital final como medida secundária. O MCT operacional permanece intocado.
