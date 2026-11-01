# Pesquisa v1.0.1: candidato externo semelhante ao líder do U75

**Branch:** `research/u75-external-leader-search-v1`  
**Motor operacional:** nenhum arquivo do MCT modificado.  
**Estado:** implementação científica inicial, testes automatizados sintéticos aprovados; **a avaliação financeira real depende de executar o script sobre snapshots locais do TCC**. Não confundir teste de software com resultado OOS de mercado.

## Pergunta

Dado o líder bruto do ranking LightGBM do U75 no fechamento `t`, um candidato externo que tenha atributos semelhantes, ou que um modelo contextual considere superior, apresenta retorno intradiário maior na sessão de execução `t+1`?

Esta primeira etapa examina uma única sessão de execução, não o valor total do episódio, o patrimônio de uma política de rotação ou um universo dinâmico plenamente validado.

## Fontes disponíveis, sem novo download

1. O ZIP de uma execução U75 do MCT contém `decision_date`, `timestamp` da execução seguinte e `raw_best_asset`. A origem do líder é **raw_best_asset**, não `selected_asset`, pois holding, margem ou CASH podem fazer a posição realizada diferir do maior score. O universo deve corresponder aos 75 símbolos do contrato congelado do TCC.
2. O conjunto RAW/SIP e Corporate Actions de `dados/strategy30_snapshot/` é conferido pelo contrato e ZIP Control Strategy #30, reutilizando `pesquisa/paridade_strategy30.py`.
3. O snapshot dos 24 candidatos externos da pesquisa por agrupamento fica em `dados/cluster_candidate_search/`. A seleção da shortlist e seus hashes são auditados pelo mesmo contrato de `pesquisar_substituicoes_cluster_spyder.py`. Candidatos que agora pertencem ao U75 são excluídos explicitamente da busca externa.
4. Os 12 atributos selecionados são nomes existentes em `ROTATION_FEATURES` e calculados pelo próprio `construir_quadro_rotacao` do motor de referência. Nenhum novo ajuste de preços ou mudança de fonte.

**Observação de compatibilidade:** o ZIP MCT pode ter cutoff posterior ao do snapshot TCC dos 75 ativos e ao dos candidatos externos. O experimento usa somente a interseção exata das datas de decisão e execução, informa cobertura e **não interpola nem estende preços**. A identidade de cada snapshot é relatada; sem prova de igualdade entre os snapshots MCT e TCC, o resultado deve ser interpretado como diagnóstico sobre um líder importado, não reprodução exata da política MCT com os preços originais desse job.

## Experimentos pré-definidos

**A. Proximidade do líder:** no fechamento, padronizar os 12 atributos de todos os ativos pelo IQR e mediana do próprio U75 naquela data. A distância euclidiana do candidato externo ao líder é calculada **sem acessar preço da sessão seguinte**. Selecionar o candidato externo de menor distância. Não escolher manualmente datas nem ativos vencedores.

**B. Regressão contextual:** para cada par (líder U75, candidato externo, dia), criar atributos relativos e distância. Rótulo de treinamento, conhecido só depois da sessão de execução, é `log(close_externo / open_externo) - log(close_líder / open_líder)`. A regressão Ridge é ajustada por janela anual expansiva, usando apenas rótulos cuja sessão de execução ocorreu antes do primeiro dia de teste. A cada dia de teste, selecionar o candidato de maior vantagem prevista **somente quando essa vantagem for positiva**; caso contrário manter o líder. `reference_cost_bps=0` no estudo bruto; esse valor não representa custos executáveis do MCT e pode ser elevado em análises de sensibilidade, sem calibrá-lo olhando o teste.

**C. Limite superior ex post:** a maior vantagem realizada entre os candidatos de um dia aparece em `oracle_delta` **somente para diagnóstico**, nunca como critério de seleção.

A avaliação não escolhe o candidato usando retorno de `t+1` ou dados de execução. Se o ativo escolhido não tiver preço de execução válido, a sessão permanece marcada como não mensurável, em vez de trocar para o segundo colocado conhecido retrospectivamente.

## Saídas do Spyder

Execute `pesquisar_busca_externa_lider_u75_spyder.py` com F5, no diretório raiz do projeto:

- `output/busca_externa_lider_u75/candidate_pairs.csv`: todas as distâncias e vantagens realizadas, separadas por data/candidato.
- `decision_coverage.csv`: datas sem cobertura de atributos, falta de candidatos e líder inválido.
- `oos_decisions.csv`: escolhas da proximidade, escolha da regressão, escolha de permanecer no U75, e labels registrados.
- `expanding_folds.csv`: fronteiras de treino/teste para auditar vazamentos de rótulo.
- `excesso_medio_anual.png`: excesso médio *de retorno abertura-fechamento*, não crescimento real de patrimônio.
- `resultado.json` e `resumo.md`: origem, contratos, hashes e métricas.
- `pacote_busca_externa_lider_u75_<UTC>.zip`: pacote de auditoria da execução.

Não faz chamadas Alpaca, não acessa MongoDB, não mexe na pesquisa Control ou no MCT. O ZIP MCT é lido em modo somente leitura.

## Pré-requisitos locais

```bash
# Git Bash, na raiz do repositório TCC
git fetch origin
git switch research/u75-external-leader-search-v1
git pull --ff-only origin research/u75-external-leader-search-v1
python -m unittest discover -s tests -p 'test_busca_externa_lider_u75.py' -v
```

Se a saída do experimento anterior `dados/cluster_candidate_search/` não existir, **não refaça automaticamente a busca e não escolha candidatos ex post**. Reconstitua primeiro o snapshot original e verifique os hashes do manifesto congelado; se isso for impossível, interrompa esta linha e prepare um novo experimento com outro contrato de seleção de candidatos.

O ZIP de referência do MCT deve estar em **`output/` do próprio repositório TCC**, inclusive em uma subpasta. A versão v1.0.1 procura recursivamente o arquivo exato `market_cycle_trader_20261101T021336-83e49685.zip` e verifica seu SHA-256 `6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe`. A presença de outros ZIPs não é erro, e o script nunca escolhe o último arquivo por acaso. Não copie para `evidencias/` ou `Downloads`. Para mudar conscientemente a execução de referência, defina `MCT_BACKTEST_ZIP` com o caminho exato do novo ZIP (absoluto ou relativo à raiz do TCC). O ZIP Control Strategy #30 permanece localizado pelo método `localizar_zip_original()` do projeto. Execute F5 no Spyder.

## Limitações científicas que impedem promoção

- A shortlist de 24 foi escolhida usando uma campanha histórica com janela completa, não redescoberta a cada ano. Mesmo com regressão temporal, isto é **validação condicional em um conjunto retrospectivamente pré-selecionado**, não validação prospectiva de descoberta de ativos em todo o mercado.
- O líder vem de um ranking já produzido pelo motor Control/MCT, que possui uma dependência de disponibilidade da próxima sessão identificada em auditorias anteriores. Esta implementação não remove nem mascara essa dependência herdada. Os testes de invariância dos candidatos não certificam causalidade do baseline.
- Retorno de abertura até fechamento de uma única sessão não é a trajetória da carteira que entra numa ação, eventualmente permanece por vários dias e paga custos reais.
- Proximidade estatística pode encontrar candidatos similares e **não** melhores. O teste pode corretamente rejeitar a hipótese.
- Se houver melhora exploratória, ainda serão obrigatórios: amostra de candidatos externos redescoberta somente com informação disponível em t, eliminação ou sensibilidade do vazamento do motor de referência, validação com calendário/execução/custos exatos, avaliação de múltiplos testes, robustez por blocos e comparação de drawdown/Expected Shortfall da política completa.

**Critério de avanço:** só propor replay de patrimônio se a estratégia contextual superar o líder em períodos futuros não usados para treino, com cobertura adequada e confiança estatística; ainda assim, os resultados desta shortlist não homologam a alteração no MCT.

## Correção v1.0.1: busca automática em output

- Novo módulo `pesquisa/localizacao_zip_u75.py`: seleção determinística por nome e SHA-256 da evidência; pesquisa `output/**`.
- Testes `tests/test_localizacao_zip_u75.py`: 4 ZIPs simultâneos, ausência da referência, hash incorreto, override explícito e caminho inválido.
- Não mexe em decisões, regressão Ridge, snapshots, U75 ou backtest.
