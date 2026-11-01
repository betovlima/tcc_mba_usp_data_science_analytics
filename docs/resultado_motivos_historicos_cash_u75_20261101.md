# Auditoria do experimento U75: motivos históricos, DTW e CASH

**Relatório:** `u75-historical-motifs-cash-audit-v1.0.0`  
**Pacote analisado:** `pacote_motivos_historicos_cash_u75_20261101T205201014162Z.zip`  
**SHA-256:** `ca844c3f8c32207d4778236264732ff306197900346dadb01fd94b333ed5f084`  
**Executor:** `u75-historical-motifs-cash-v1.0.0`  
**Estado:** `completed`. Relatório documental, sem alterações no Control, motor, parâmetros, snapshots ou ambiente operacional.

## 1. Paridade e objetivo

O arquivo `control_paridade.json` comprova 1.576 sessões históricas (22/07/2020 a 28/10/2026), divergência de **zero dias** e capital final exatamente igual à referência Strategy #30, **US$ 80.685.454,78647688**. O snapshot possui os mesmos 75 ativos, folds, posições e curvas com tolerância de US$ 0,01 por sessão.

**US$ 80,69 milhões não é alvo de otimização:** checkpoint de paridade. A pergunta é se padrões históricos reconhecidos exclusivamente no passado permitem reduzir quedas e ES95, entrando em CASH sem tempo de permanência imposto.

## 2. Resultado financeiro pareado

| Métrica | Control | Apenas coletivo 20/5 | Coletivo + ativo 20/5 | Coletivo + ativo 40/10 |
|---|---:|---:|---:|---:|
| Capital final (US$ mi) | 80,6855 | 79,4440 | 37,9921 | 43,9616 |
| Variação capital final vs Control | 0% | −1,5386% | −52,9134% | −45,5149% |
| Máximo drawdown | −41,6828% | **−38,8062%** | −38,8062% | −41,6828% |
| ES95 empírico de retorno diário | −7,7878% | −7,7421% | **−7,6390%** | −7,6426% |
| Pior pregão | −36,2253% | −36,2253% | −36,2253% | −36,2253% |
| Dias em CASH | 0 | **14** | 75 | 71 |
| Períodos de CASH | 0 | 11 | 57 | 51 |
| Reentradas em ativos | 0 | 11 | 57 | 51 |
| Sharpe reportado | 2,3177 | 2,3208 | 2,1821 | 2,3144 |

A proteção coletiva 20/5 diminuiu o máximo drawdown em **2,8767 pontos percentuais**, mas o ES95 melhorou somente **0,0456 ponto percentual**. As versões condicionadas ao ativo elevaram a frequência de CASH e reduziram pouco o ES95, sacrificando grandes oportunidades positivas. Todas mantiveram exatamente a pior queda diária, de **−36,2253%**. Não foi comprovada antecipação do maior choque.

A mudança em capital final serve para quantificar o **custo econômico dos alarmes**; não define o objetivo da pesquisa.

## 3. Os três episódios de 2026 examinados pelo autor

As perdas abaixo são calculadas como `equity(fundo)/equity(pico)-1` usando as **mesmas datas** para cada política, não os novos picos individuais de cada carteira.

| Episódio | Control | Coletivo 20/5 | Coletivo + ativo 20/5 | Coletivo + ativo 40/10 |
|---|---:|---:|---:|---:|
| AMS 28/05→08/06 | −20,7317% | −20,7317% | −20,7317% | −20,7317% |
| CAPR 27/07→31/07 | −41,6828% | **−38,8062%** | **−38,8062%** | −41,6828% |
| CAPR 30/09→26/10 | −32,3589% | −32,3589% | −32,3589% | −32,1625% |

### O sinal foi tempestivo?

- Maio–junho, AMS: **nenhuma sessão em CASH dentro da janela de 28/05 a 08/06** em qualquer um dos três cenários. O cenário com ativo 20/5 chegou a sair de AMS em 07/05, mas isso foi **antes da janela marcada**, sem prevenir a queda mais tarde.
- Julho, CAPR: em **30/07** o Control perdeu **36,2253% no dia** e nenhum cenário estava em CASH. O sinal **coletivo** foi calculado no fechamento desse dia, e a entrada em CASH ocorreu na abertura **31/07**, reduzindo a queda de 27 a 31/07 de 41,68% para 38,81%. Trata-se de **reação após a perda principal**, não de antecipação dela.
- Setembro–outubro, CAPR: coletivo 20/5 e coletivo+ativo 20/5 não fizeram CASH na janela de 30/09 a 26/10. A variante 40/10 ficou em CASH em **09/10** por sinal no fechamento de 08/10 e reduziu a queda do episódio apenas de 32,3589% para 32,1625%; houve também um CASH em 25/09, **antes** do pico, que não constitui proteção contínua do declínio subsequente.

CASH e simulações são contabilizados no **pregão de execução**, e a tabela `*_decisoes_posicoes.csv` relaciona `decision_date` (sinal) a `timestamp` (execução). Uma decisão só é útil para proteção se estiver disponível **antes** da exposição a risco que pretende evitar.

## 4. A perda de oportunidades com alarmes falsos importa

- **Coletivo 20/5**: 14 sessões em CASH distribuídas em 11 episódios; diferenças de posições para o Control em 17 sessões, sendo 3 divergências adicionais após os dias de CASH devido ao estado de carteira diferente.
- **Coletivo + ativo 20/5**: 75 sessões em CASH em 57 episódios; 109 dias com posição diferente da referência, incluindo 34 divergências em que a carteira também estava investida em outro ativo.
- **Coletivo + ativo 40/10**: 71 sessões em CASH em 51 episódios; 106 dias com posição diferente, dos quais 35 em ativos distintos do Control, sem CASH naquele dia.

No cenário 40/10, o sistema permaneceu em CASH nas execuções **26, 27 e 28/12/2023**, enquanto o Control permaneceu exposto a **VNCE**. De 22 a 28/12, o patrimônio do Control cresceu aproximadamente **130,29%** (US$ 1,079 mi → US$ 2,485 mi), enquanto o patrimônio protegido permaneceu constante (US$ 1,270 mi). Esse episódio exemplifica o grande custo potencial de decidir CASH a partir de analogias que não se generalizam. Os valores iniciais das políticas já diferiam antes do episódio; comparar **retorno percentual local**, não diferença absoluta de patrimônios como causalidade.

Nas 14 sessões CASH da versão coletiva 20/5, o retorno diário do Control somou aproximadamente **+2,91 pontos percentuais** (soma aritmética de 14 retornos de datas distintas), com 7 pregões positivos e 7 negativos. Atenção: a rentabilidade da política em CASH **não é necessariamente zero no dia da saída**, pois a execução ocorre na abertura e há exposição até esse instante. Não interpretar `selected_asset=CASH` como ausência de qualquer retorno intradiário/overnight daquele dia.

## 5. Limitações do método

1. A distância é computada sobre **trajetórias normalizadas individualmente** (cada janela pelo seu próprio desvio padrão). Isso compara a **forma** temporal, mas perde parte da **magnitude absoluta e escala de volatilidade**. Movimentos parecidos na forma e diferentes na severidade podem ser tratados como análogos. É uma hipótese plausível para a instabilidade do sinal, não uma causa comprovada.
2. A busca de episódios históricos usa **prefiltro de distância euclidiana e DTW causal**, não uma implementação exata de STUMP/STOMP do Matrix Profile.
3. São até 16 análogos, por protocolo; exclusões por sobreposição podem reduzir a amostra. O ES95 da própria distribuição dos análogos é muito sensível a uma ou poucas observações extremas. A média e o limite superior Gaussiano `1,645*s/sqrt(n)` não oferecem cobertura nominal garantida com padrões temporais dependentes.
4. A política escolhe CASH quando o limite superior da média de retornos logarítmicos futuros dos análogos fica abaixo de zero. **Essa métrica não mede diretamente o benefício econômico esperado de sair e reentrar**, nem os custos dos falsos alarmes na mesma função de decisão.
5. A camada de risco usa dados observáveis no fechamento e rótulos dos análogos maturados, mas o **Control original possui elegibilidade histórica dependente do pregão seguinte**, já conhecida. A paridade perfeita é verificabilidade técnica, não comprovação de causalidade integral.
6. Os episódios foram examinados anteriormente. Este é **replay retrospectivo**, não novo teste independente fora de todas as escolhas metodológicas. NÃO ajustar janelas ou z pelo resultado deste ZIP e depois chamar o mesmo período de OOS novo.

## 6. Decisão científica e próxima hipótese

**Não promover nenhum dos três overlays como proteção validada.** A versão coletiva 20/5 é o melhor compromisso **entre as três variantes deste piloto** em termos de redução de drawdown e número de alertas, mas sua proteção chegou tarde no episódio CAPR e falhou em AMS e na queda de outubro. A versão 40/10 perdeu uma valorização excepcional em VNCE; isso não deve ser corrigido com exclusões ex post.

Próxima pesquisa **sem otimização do patrimônio histórico**: comparação condicional de episódios usando **forma + amplitude + volatilidade + breadth U75**, com identificação do ativo efetivo, previsão de probabilidade de **continuidade de queda futura** e distribuição de retorno da **alternativa CASH vs posição**, não só sinal de média negativa do análogo. Pré-especificar método de calibração, episódios independentes, custos e limites de incerteza antes de avaliar datas adicionais, idealmente em período novo e intacto ou com avaliação aninhada robusta.

**Não alterar Control nem ambiente operacional.** Apenas documentação desta execução auditada.
