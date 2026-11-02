# Correção de regressão: dimensões fixas no experimento de motivos históricos

**Executor:** `u75-historical-motifs-cash-v1.0.3`  
**Núcleo matemático:** `u75-historical-motifs-cash-v1.0.1`  
**Branch:** `fix/u75-motifs-manifest-driven-dimensions-v1`.

## Origem da regressão

Ao derivar o novo executor do mecanismo de reprodução do Control Strategy #30, foram reaproveitados *três números literais* que pertenciam àquela execução histórica: **75 ativos**, **3 folds** e **1.577 datas de decisão**. O erro reapareceu apesar de o projeto já ter adotado construção de calendário e quantidade de folds a partir de dados e configuração no motor principal.

Outros locais no mesmo percurso também repetiam 75 (normalização de séries, auditoria de pacote, configuração de referência). O retorno desses testes rígidos **foi uma regressão introduzida na pesquisa nova**; não significa uma alteração intencional no algoritmo de rotação ou no protocolo dinâmico geral.

## Correção implementada

- `pesquisar_motivos_historicos_cash_u75_spyder.py` não contém mais comparação com contagens literais de 75, 3 e 1.577: a validação confere ativos, folds e total de sessões declarados em `control_oficial_strategy30.json`.
- `pesquisa/validacao_dimensoes_motivos_u75.py` valida qualquer quantidade positiva de ativos, folds e sessões que esteja coerente com o contrato; testes incluem **65/2, 75/3 e 82/4**, sem forçar os números da execução existente.
- `pesquisa/motivos_historicos_cash_u75.py` usa cobertura **relativa de 90%** (arredondamento para cima), em lugar do número fixo de 68 ativos válidos. Assim, com 65 ativos são necessários 59 históricos válidos; com 75, **68**, reproduzindo a cobertura do protocolo existente; com 82, 74. A presença de ativos não declarados, falta de séries ou duplicidade continua gerando erro, sem reconstruir preços.
- `pesquisa/control_operacional_strategy30.py` e `pesquisa/paridade_strategy30.py` deixam de repetir quantidade fixa e verificam `effective_asset_count` do próprio manifesto. O Control oficial e seu universo canônico **continuam congelados**, sem editar símbolos, resultados, preços nem hashes.

## Distinção essencial: dinâmica de pesquisa versus referência histórica

**O código agora não possui aquelas contagens literais, mas o replay da Strategy #30 continua, corretamente, vinculado ao contrato de identidade de seu ZIP congelado.** Hoje o contrato contém 75 ativos, 3 folds e 1.576 sessões avaliadas, portanto 1.577 datas de decisão. O script de **reprodução histórica** deve rejeitar calendário ou ativos diferentes desse manifesto: se não rejeitar, estaria comparando um backtest atualizado com resultado financeiro de outro universo/período e afirmando paridade falsa.

Para incorporar novas sessões e alterar o universo num **novo experimento temporal**, é necessário criar/recalcular uma **nova referência para esse período** e rodar a validação correspondente, ou executar outro workflow de extensão que não exija igualdade com o arquivo histórico. Não se deve interpretar a remoção dos números fixos como implementação completa dessa extensão; esta correção cuida da regressão de parametrização, sem criar novo modo de operação.

## Como executar

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch --track origin/fix/u75-motifs-manifest-driven-dimensions-v1
```

Reiniciar o kernel Spyder e executar `pesquisar_motivos_historicos_cash_u75_spyder.py` com F5. Os arquivos `andamento_atual.json`, `andamento.jsonl` e o pacote de resultados continuam disponíveis como na versão anterior.

Nenhuma alteração na matemática DTW, no ranqueamento LightGBM, na regra de entrar em CASH, nos custos, na seleção de Top-1 ou nos períodos de treino. A única generalização estatística é manter cobertura percentual constante quando o tamanho do universo muda; para o Control atual de 75 ativos, a exigência continua de 68.

**Não afirmar nova execução financeira ou novo resultado OOS sem rodar o programa com o snapshot compatível.**
