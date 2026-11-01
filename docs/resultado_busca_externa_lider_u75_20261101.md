# Resultado da busca externa ao líder U75, execução de 01/11/2026

**Estado:** auditoria dos resultados efetivamente obtidos, sem homologação e sem alteração do MCT.  
**Versão da rotina:** `u75-external-leader-search-v1.0.2`.  
**Pacote analisado:** `pacote_busca_externa_lider_u75_20261101T033836Z.zip`.  
**SHA-256 do pacote:** `3f35f4009bf0f88e8bcc4532226d47fdee2987fe4ace1f857f0eb0c3ae5285c7`.  
**Origem do ranking U75:** `market_cycle_trader_20261101T021336-83e49685.zip`, SHA-256 `6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe`.  
**Contrato científico:** 75 ativos internos; 24 candidatos externos congelados, todos fora do U75; dados RAW/SIP e Corporate Actions locais; sem download; sem reexecução de modelos do Control neste executor.

## 1. Execução e cobertura

O pacote declara `status=completed`. Há 1.578 decisões recebidas do ZIP do MCT; 1.561 têm painel de comparação válido (37.464 pares de ativo/dia, 24 candidatos por decisão). As últimas 16 decisões de 07 a 28/10/2026 não têm atributos completos dos candidatos no snapshot original, e 29/10/2026 não tem quadro U75 completo para a decisão. **Nenhuma dessas datas deve ser completada com cotações futuras ou preenchimento artificial.**

São efetivamente avaliadas 1.194 sessões fora do treino, de **03/01/2022 a 06/10/2026**, em cinco folds anuais de janela expansiva. Em um dia o candidato mais semelhante não tinha retorno de execução mensurável, portanto são 1.193 observações para esse critério.

O eixo da pesquisa é a vantagem de *log-retorno abertura-fechamento* da próxima sessão de um candidato externo frente a `raw_best_asset` do U75, não o P&L efetivo da política com custos e manutenção de posição.

## 2. Resultado agregado por sessão

| Indicador | Vizinho externo mais próximo | Ridge contextual |
|---|---:|---:|
| Sessões medidas | 1.193 | 1.194 |
| Diferença média de log-retorno por pregão | −0,5107% | −0,3798% |
| Mediana da diferença de log-retorno | −0,4068% | 0,0000% |
| Sessões em que superou o líder (sobre todas as avaliadas) | 527 (44,17%) | 399 (33,42%) |
| IC95% exploratório por bootstrap de meses, log-retorno médio | [−0,7477%, −0,2987%] | [−0,6687%, −0,0931%] |

A decisão do Ridge possui uma opção explícita de **não intervir**. Em 340 dos 1.194 pregões (28,48%) manteve o líder U75, produzindo delta exatamente zero. A mediana de zero não representa habilidade preditiva nem superioridade.

Nas **854 substituições** efetivamente sugeridas pelo Ridge, a diferença média realizada foi **−0,5310% por sessão**, com 399 resultados positivos, 454 negativos e uma igualdade. Nessas mesmas 854 recomendações, o modelo previa diferença média de **+0,3422%**, mas a correlação entre o escore previsto e a vantagem realizada foi aproximadamente **−0,0068**. A regressão não apresentou calibração útil da vantagem do candidato no teste realizado.

## 3. Consistência temporal

| Ano OOS | Sessões | Vizinho: diferença média diária | Ridge: diferença média diária | Substituições pelo Ridge |
|---|---:|---:|---:|---:|
| 2022 | 251 | −0,7332% | −0,3828% | 193 |
| 2023 | 250 | −0,6334% | −0,9458% | 187 |
| 2024 | 252 | −0,2811% | +0,1185% | 159 |
| 2025 | 250 | −0,4483% | −0,3988% | 188 |
| 2026 | 191 | −0,4416% | −0,2677% | 127 |

A similaridade ficou negativa em **todos os cinco anos**. A regressão Ridge só superou a referência em média no ano de 2024, sem compensar os demais anos.

O **oráculo** que escolheria retrospectivamente o candidato de maior resultado atingiu vantagem média positiva de **+6,2705%** em log-retorno diário, mas essa comparação depende integralmente do preço da própria sessão futura. É apenas uma referência ex post, não uma política executável nem prova de capacidade de previsão.

## 4. Importante desalinhamento do alvo com o objetivo de preservação de capital

No backtest MCT identificado acima, o capital entre 29 e 30/07/2026 caiu de aproximadamente US$ 64,66 milhões para US$ 41,23 milhões, uma mudança de **−36,23% entre fechamentos**. No painel da pesquisa, a sessão de execução de **30/07/2026** registra para o líder CAPR `leader_log_open_close = +0,31102317`, equivalente a cerca de **+36,48% da abertura ao fechamento**.

Isso é compatível com forte queda anterior à abertura seguida de recuperação intradiária, mas os arquivos fornecidos não permitem atribuir a diferença exclusivamente ao gap sem uma conciliação de preços, posições e custos. **O essencial é que os dois indicadores medem horizontes distintos**. Na decisão de 29/07, o Ridge manteve CAPR e o melhor resultado retrospectivo entre os 24 candidatos foi inferior ao retorno abertura-fechamento de CAPR. Esse episódio evidencia que a métrica escolhida não serve, isoladamente, para avaliar preservação do capital da carteira.

Como a regra executa mudanças na **abertura seguinte**, uma saída decidida no fechamento anterior pode ainda sofrer o gap antes da execução. Qualquer contrafactual deve modelar quando o ativo foi realmente mantido e quando a transição ocorreu, incluindo custos de venda/compra, não apenas comparar retornos internos do dia.

## 5. Limitações e decisão científica

1. O conjunto externo de 24 ativos foi obtido em campanha histórica e não reproduz descoberta temporal do mercado inteiro. A validação aqui é condicional à shortlist já escolhida.
2. A referência `raw_best_asset` foi importada do ZIP MCT e não é igual, em todas as datas, à posição efetiva após regras de holding e troca. O ranking do motor de referência tem dependência futura de elegibilidade ainda sob investigação.
3. Os snapshots científicos e a execução operacional têm datas-limite diferentes. A interseção é auditada, mas o trabalho não reproduz a trajetória inteira do MCT a partir do mesmo dataset congelado.
4. A avaliação usa somente abertura-fechamento em t+1, sem custos, slippage, posição anterior, efeitos de caminhos de rotação nem drawdown do patrimônio.
5. O intervalo de confiança por reamostragem de meses é exploratório; não corrige a seleção retrospectiva da shortlist ou tentativas científicas sucessivas.
6. Os anos 2022–2026 **já foram examinados**. Ajustar agora features, hiperparâmetros ou filtros para melhorar esse período e reportar o mesmo resultado como validação independente seria sobreajuste de pesquisa.

**Conclusão:** não há base para promover nenhum dos dois seletores externos ou executar substituições reais. A hipótese específica "escolher o candidato mais semelhante ao líder produz vantagem na sessão seguinte" foi contrariada nesse conjunto. O Ridge contextual também fracassou na vantagem média e na calibração das decisões de troca.

**Próximo estudo, somente no TCC:** definir o alvo alinhado à política financeira (trajetória de patrimônio entre decisões, com entrada na abertura seguinte, permanência, custos e drawdown), preservar Control e políticas já validadas, e testar descoberta externa cronológica em pool independente, com baseline sem interferência. Só executar se houver dados congelados suficientes e protocolo prospectivo não contaminado; não ampliar o universo a posteriori para maximizar o resultado de 2022–2026.
