# Resultado observado da reotimização U75: comparação temporal pareada

**Versão do relatório:** u75-hyperparameter-oos-audit-v1.0.0  
**Pacote:** `pacote_hiperparametrizacao_u75_20261101T182817Z.zip`  
**SHA-256:** `dd560fb67e2f2feccd2ac22900f4c3ad15d0bb0e428764b99f7f356ef7eaf6c5`  
**Motor:** `u75-nested-hyperparameter-research-v1.0.0`.  
**Estado:** `completed`. Sem mudança no MCT, no Control ou nos dados congelados.

## Resultado central

O arquivo `resultado.json` confirma a paridade do checkpoint original em **US$ 587.165.941,1330212**, com 75 ativos, 52 atributos e 1.560 pregões OOS até 06/10/2026.

| Cenário | Capital final (US$) | Sharpe | MaxDD | Registros de trades |
|---|---:|---:|---:|---:|
| `frozen_original` | 587.165.941,13 | 2,7285 | −41,68% | 710 |
| `frozen_matched` | 308.904.391,62 | 2,6522 | −41,68% | 730 |
| `optimized_matched` | 15.732.700,18 | 1,9551 | **−84,29%** | 746 |

**Comparação primária correta:** `optimized_matched` contra `frozen_matched`, ambos com a mesma janela de seleção de margem. A nova otimização **perdeu US$ 293.171.691,44 (−94,9069%)** de capital final relativamente à referência pareada.

A diferença adicional entre `frozen_original` e `frozen_matched` é **−US$ 278.261.549,51 (−47,39%)**, porque a calibração da margem passou de todas as 126 sessões para o segmento final da janela. **Não atribuir esta diferença à mudança de hiperparâmetros.** No fold 2, a margem original escolhida foi 0,0100; na calibração pareada, 0,0025, gerando 30 decisões de posição diferentes em 504 pregões.

## Escolha de hiperparâmetros por fold

| Fold | Intervalo de teste | Trial vencedor | Mudanças de posição vs `frozen_matched` |
|---|---|---:|---:|
| 1 | 2020-07-22 a 2022-07-21 | 0, parâmetros originais | 0 / 504 |
| 2 | 2022-07-22 a 2024-07-24 | 0, parâmetros originais | 0 / 504 |
| 3 | 2024-07-25 a 2026-10-06 | 7, `gaussian_process_ei` | **219 / 552** |

No fold 3, o trial 7 superou a configuração congelada **apenas na janela interna de seleção** (objetivo **0,394472** contra **0,323419**). A configuração escolhida alterou n_estimators 329→279; learning_rate 0,020731→0,047140; num_leaves 6→7; min_child_samples 18→17; colsample_bytree 0,88067→0,793770; reg_alpha 0,050837→0,312086; reg_lambda 3,596305→1,918163.

No teste do fold 3, ambas as políticas partiram de **US$ 5.770.808,56**. O cenário com parâmetros originais e janela pareada atingiu **US$ 308.904.391,62**; o otimizado, **US$ 15.732.700,18**. A vantagem do trial 7 na janela interna não se generalizou.

### Episódio crítico de CAPR em julho de 2026

Em **27/07/2026**, a referência pareada mantinha **WDAY** e registrou **+9,014%**; a política reotimizada mantinha **CAPR** e registrou **−64,492%** no dia. A desvantagem de log-retorno nesse dia foi **−1,121732**.

Em **30/07/2026**, ambas estavam expostas a CAPR e caíram **−36,225%**. A reotimização ampliou o problema de exposição, mas a própria referência também sofreu perda relevante em CAPR.

No otimizado, a curva atingiu **US$ 36.236.916,03** em **28/05/2026**, depois caiu para **US$ 5.693.277,03** em **31/07/2026**: **−84,29%** de drawdown. O choque de 27/07 explica aproximadamente **37,7% da diferença negativa acumulada em log-retorno** no fold 3; o restante permanece negativo.

### Diagnóstico pareado suplementar

No fold 3, a média da diferença de log-retorno diário `optimized - frozen_matched` foi **−0,5394 ponto percentual**. Bootstrap descritivo de **28 meses inteiros**, 2.000 sorteios com seed 42, forneceu intervalo percentil 95% de **[−1,1177; −0,1048] ponto percentual**. Isso não é uma validação prospectiva independente; serve apenas para avaliar a concentração temporal do resultado observado.

## O que não foi demonstrado

1. **Não foi provado que os 52 atributos são os melhores**. Todos foram mantidos iguais. A importância `gain` por árvore não mede contribuição causal nem garante estabilidade preditiva.
2. **Não foi reproduzido o CARO histórico**. O experimento utilizou LHS + `GaussianProcessRegressor` com expected improvement, 8 trials por fold.
3. **Não foi demonstrado que a configuração antiga é globalmente ótima**. Apenas que essa rotina de reotimização, nesse orçamento e nessas janelas, não conseguiu melhorá-la.
4. O conjunto de ativos e os hiperparâmetros históricos tiveram seleção retrospectiva com sobreposição a períodos avaliados. A elegibilidade do motor original ainda possui dependência da sessão seguinte. Isso limita a interpretação OOS e impede afirmações de lucratividade prospectiva.

## Decisão e próximos passos

**Não promover a configuração selecionada pelo fold 3. Não modificar o Control nem o MCT.** A hipótese de reotimização automática a cada inserção de ativo não foi sustentada por este teste.

Antes de qualquer campanha maior, separar **(a) estabilidade da calibração da margem** e **(b) estabilidade da seleção dos hiperparâmetros** em uma janela interna mais representativa, usando somente histórico disponível antes da nova validação. Não ajustar configurações olhando os 552 pregões do fold 3 já avaliados. Priorizar o diagnóstico da dependência futura de elegibilidade e da concentração em ativos como CAPR.

O próximo experimento deve comparar novo universo e baseline com protocolo temporal verdadeiramente independente; resultados de importância dos atributos continuam descritivos até avaliação individual e conjunta apropriada.
