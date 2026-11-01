# Reotimização cronológica dos hiperparâmetros do U75

**Branch:** `research/u75-nested-hyperparameters-v1`  
**Versão:** `u75-nested-hyperparameter-research-v1.0.0`  
**Estado:** código de experimento preparado e testes unitários aprovados. **Resultados financeiros do experimento novo não disponíveis** até a execução local dos 75 históricos congelados.

## Pergunta

Os hiperparâmetros LightGBM selecionados na campanha histórica de agosto de 2026 continuam adequados ao universo U75, ou uma reotimização temporal sobre o mesmo U75 melhora **fora da janela de otimização** a política de rotação? Os 52 atributos do motor são mantidos constantes; esta pesquisa **não é seleção de features**. A importância por ganho será exportada apenas como diagnóstico; não demonstra contribuição causal ou qualidade preditiva isolada.

## Identidade dos dados e referência

Executor oficial: `reproduzir_experimento.py`. O conjunto congelado é o mesmo universo selecionado na campanha U67+10, composto por 77 ativos solicitados, 75 efetivos, até 06/10/2026. O manifesto `evidencias/manifest_universo_congelado.json` fixa o snapshot-base `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`, os dez símbolos adicionais e vinte SHA-256 de CSVs (RAW e Corporate Actions). Os arquivos de candidatos ficam no cache local `dados/.u67_expansion_cache/full_raw_20261006`. Não reconstruir, baixar, imputar nem substituir um arquivo que não passe no contrato.

**Trava de paridade:** a execução nova reproduz primeiro o U75 com as margens e os hiperparâmetros originais. Se o capital final diferir mais de US$ 0,01 do checkpoint histórico **US$ 587.165.941,1330212**, interrompe o estudo antes de gerar conclusão econômica.

## Comparação pareada

| Cenário | Hiperparâmetros LightGBM | Calibração da margem | Finalidade |
|---|---|---|---|
| `frozen_original` | parâmetros fixos de agosto | todas as 126 sessões anteriores ao teste | conferir checkpoint, **não é a comparação pareada primária** |
| `frozen_matched` | mesmos parâmetros congelados | últimas sessões da calibração, sem usar a busca interna | controle causal pareado |
| `optimized_matched` | escolhidos por LHS + processo gaussiano em cada fold | a mesma janela do `frozen_matched` | challenger pareado |

### Fronteiras temporais em cada fold externo

1. **Treino:** período existente na configuração congelada, terminado antes do embargo de 60 pregões.
2. **Busca de hiperparâmetros:** primeiras 60 sessões da janela de calibração existente.
3. **Separação:** dois pregões, para não sobrepor o último retorno de calibração à janela seguinte.
4. **Calibração da margem de rotação:** 64 sessões finais da janela de 126, sem modificar os 52 atributos.
5. **Refit:** somente informações históricas permitidas pelo `final_fit_end_index` do fold.
6. **Avaliação:** intervalo OOS desse fold, nunca consultado para escolher hiperparâmetros ou margens.

Há três folds externos. Para a comparação primária, ambas as políticas usam os mesmos snapshots, datas de decisão, custos, margens elegíveis e motor de simulação. As diferenças são apenas os hiperparâmetros escolhidos na janela interna.

### Otimização bayesiana

Implementação com `sklearn.gaussian_process.GaussianProcessRegressor` (kernel Matérn) e *expected improvement*. É uma **reimplementação da ideia LHS + aquisição bayesiana**, não o CARO histórico exato de 13/08/2026, cujo snapshot original e limites exatos de folds não foram recuperados no manifesto. Não chamamos uma reprodução histórica do CARO de algo que não é.

Configuração piloto pré-fixada: **8 propostas por fold** (uma âncora com os parâmetros históricos, três amostras por hipercubo latino e quatro propostas adaptativas), total de 24 avaliações de busca e três folds. Os oito domínios coincidem com os documentados no `MANIFEST.md`, seção 22:

| Parâmetro | Limite inferior | Limite superior |
|---|---:|---:|
| Árvores | 220 | 380 |
| Taxa de aprendizado | 0,020 | 0,050 |
| Profundidade | 2 | 4 |
| Folhas | 4 | 12, limitado também por `2^profundidade` |
| Mínimo de exemplos por folha | 15 | 30 |
| Fração de atributos | 0,75 | 0,95 |
| Regularização L1 | 0 | 0,50 |
| Regularização L2 | 1 | 4 |

As demais opções LightGBM, as 52 features, a função de utilidade, os horizontes, as taxas e o algoritmo de decisão ficam inalterados. A função de aquisição usa exclusivamente a **utilidade ajustada ao risco na janela interna de busca**, calculada pelo mesmo simulador simplificado de calibração existente no TCC; não utiliza capital/Sharpe/MaxDD do fold externo na escolha. A margem efetiva é escolhida posteriormente em outra janela, e o replay econômico externo é realizado com o simulador exato e custos originais.

## Execução

No Git Bash:

```bash
git fetch origin
git switch --track origin/research/u75-nested-hyperparameters-v1
python -m unittest discover -s tests -p 'test_otimizacao_hiperparametros_u75.py' -v
```

No Spyder, reinicie o kernel, defina o diretório de trabalho como a raiz do TCC e execute com F5:

`pesquisar_hiperparametrizacao_u75_spyder.py`

Não execute `preparar_snapshot_u67.py` nem `pesquisar_expansao_u67_spyder.py` para atualizar os dados. O executor acessa o módulo legado apenas para reutilizar as rotinas de preparação existentes, sem chamar o `main()` da campanha.

**Custo computacional:** no piloto, 8 combinações × 3 folds × 75 modelos só na busca, além dos ajustes de margem e finais. A execução pode ser pesada em CPU, pois o Control congela `device_type="cpu"`. Não é uma falha se demorar. O teste não faz atualização incremental nem impõe limite artificial de tempo. O arquivo `progresso_busca.csv` é exportado durante a execução; se falhar, `output/hiperparametrizacao_u75/ultimo_erro.txt` informa o stack trace. Um rerun reinicia uma campanha, não reutiliza resultados parciais de outra assinatura.

## Saídas

Todos os arquivos ficam em `output/hiperparametrizacao_u75/<UTC>/`:

- `hiperparametros_testados.csv`, `hiperparametros_escolhidos_por_fold.csv` e `calibracao_margens.csv` para auditoria da seleção;
- `importancia_52_atributos.csv` com os ganhos de importância LightGBM por ativo, fold e política, sem selecionar atributos;
- `predictions_frozen_original.csv`, `predictions_frozen_matched.csv`, `predictions_optimized_matched.csv`;
- `trades_*.csv` correspondentes, `comparacao_patrimonio_u75.png`, `resultado.json`;
- `pacote_hiperparametrizacao_u75_<UTC>.zip` para avaliação independente.

**Comparação primária:** `optimized_matched` versus `frozen_matched` nos mesmos 1.560 pregões fora do treino, observando capital final, Sharpe, drawdown máximo e o pior fold. A execução original é apenas a prova de paridade. Não anunciar ganho antes de receber e auditar o ZIP.

## Limitações científicas obrigatórias

O universo U75 foi descoberto retrospectivamente com acesso a toda a janela. Sua composição fixa e os hiperparâmetros originais podem trazer viés de seleção. A pesquisa já estimou PBO de 53,57% para retorno e 59,52% para Sharpe na seleção de subconjuntos. Mesmo que a nova reotimização melhore o fold externo de uma configuração fixa, isso **não demonstra descoberta prospectiva do universo**.

A auditoria anterior também apontou no motor congelado uma elegibilidade dependente da presença do preço da sessão seguinte; esta versão **não altera silenciosamente o Control original**. Não é apropriado promover o challenger para o MCT sem corrigir/medir a dependência temporal e executar uma validação futura independente. Também não há garantia de que um aumento de capital no backtest se repita no mercado.

**Decisão a partir do piloto:** analisar os resultados da busca e da paridade antes de aumentar o orçamento de otimização ou de acrescentar ativos externos. O teste dos hiperparâmetros é separado do teste de contribuição marginal de cada novo ativo.
