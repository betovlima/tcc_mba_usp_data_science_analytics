# Auditoria do resultado econômico U75 em 02/11/2026

**Fonte:** pacote_vantagem_economica_rotacao_u75_20261102T171418989313Z.zip  
**SHA-256:** `0244ee29cfcaa16f52ff2763b79b3fa1527f381c28cc9604b37740a368ffde2b`  
**Executor:** `u75-economic-rotation-recovery-v1.0.1`  
**Status:** `completed`, ZIP íntegro (25 arquivos, SHA-256 de todos conferidos).  
**Controle:** Control Strategy #30, 75 ativos, 1.576 sessões. Paridade ao centavo e posições idênticas em todos os dias.

## Resultados históricos

| Métrica | Control | Risco econômico | Risco econômico + recuperação |
|---|---:|---:|---:|
| Capital final (USD) | 80.685.454,79 | 14.970.107,25 | 19.691.495,83 |
| Sharpe | 2,3177 | 2,0359 | 2,1064 |
| MaxDD | -41,6828% | -44,1846% | -41,8218% |
| ES95 empírico diário | -7,7878% | -7,4413% | -7,3898% |
| Pior dia | -36,2253% | -36,2253% | -36,2253% |
| Rotacoes | 348 | 443 | 439 |
| Dias CASH | 0 | 42 | 46 |
| Decisões diferentes do Control | n/a | 269 | 268 |
| Pregões com posições diferentes do Control | n/a | 463 | 455 |

A variante com recuperação terminou com 31,5388% mais capital que a variante
econômica sem recuperação, mas ficou 75,5947% abaixo do Control. Gerou 9
decisões com razão explícita `RECOVERY_ECONOMIC_ROTATION`; a trajetória
de posições diferiu da política econômica simples em 66 pregões. Não
atribuir toda a diferença de capital exclusivamente a essas nove decisões,
pois há divergências consequentes de estado de carteira e compounding.

### Diagnóstico por episódios já inspecionados

Comparação entre as MESMAS datas de pico e fundo do Control:

| Episódio | Control | Econômica | Com recuperação |
|---|---:|---:|---:|
| AMS 28/05–08/06/2026 | -20,7317% | -20,7317% | -20,7317% |
| CAPR 27/07–31/07/2026 | -41,6828% | -41,6829% | -41,6829% |
| CAPR 30/09–26/10/2026 | -32,3589% | -18,4067% | -18,4067% |

A política econômica não antecipou a principal perda de CAPR em julho.
Em 27/07/2026, o próprio classificador marcou `p_downside(CAPR)=0,997982`,
mas a escolha final continuou CAPR. Não interpretar essa probabilidade
como certeza ou calibração validada para o episódio já inspecionado.

## Falha lógica independente dos retornos realizados

No fechamento de 27/07/2026, os atributos e modelos PRODUZIDOS naquela data
registraram `risk_score(CAPR)=-0,019593`, enquanto o maior `risk_score`
bruto disponível foi `ADC=+0,000646`. O custo estimado de mudança na
política é `0,003`. O custo de comprar ADC partindo de WDAY tornaria
o retorno econômico incremental líquido aproximado de ADC
`0,000646-0,003=-0,002354`; CASH, após custo de saída, teria
aproximadamente `-0,003`. A compra de CAPR (também uma transição)
teria valor aproximado `-0,019593-0,003=-0,022593`.

A implementação v1.0.1 exigia simultaneamente uma **alternativa com
retorno líquido positivo** para rotacionar e a **ausência de qualquer
alternativa com retorno bruto positivo** para considerar CASH.
Consequentemente, a existência de ADC com ganho bruto `+0,000646`
bloqueou CASH, embora sua compra não cobrisse o custo de troca.

**Isso é uma inconsistência matemática de escolha**, não uma indicação
de que ADC teria lucrado ou de que CASH teria antecipado com certeza a
queda de 30/07/2026.

### Outros eventos de oportunidade perdida

- Em 09/04/2025 o Control ganhou 24,53%, enquanto ambas variantes
  econômicas encerraram o pregão em CASH e perderam essa valorização.
- Em 01/05/2026 o Control ganhou 17,24% em AMS, e as variantes
  econômicas aproximadamente 0,07% em SCSC.
- Em 03/06/2025 as variantes econômicas em PNBK perderam 18,18%
  enquanto o Control em TSLA ganhou aproximadamente 0,46%.
- Já em 11/01/2024 o Control perdeu 18,23% em VNCE enquanto as
  variantes econômicas ganharam cerca de 1,71% em ORCL.
- Em 05/10/2026 o Control perdeu cerca de 13,07% em CAPR e as
  variantes econômicas com MKSI ganharam cerca de 0,48%.

Esses exemplos mostram ganhos e perdas de oportunidade em ambas direções.
Não escolher os mais favoráveis como justificativa para promoção.

## Correção v1.0.2, segregada em nova branch

Branch: `research/u75-economic-rotation-recovery-v1.0.2`.

A correção usa retorno ajustado **líquido do custo de transição a partir
da posição efetiva** na comparação entre Control, alternativas e CASH.
Se nenhuma alternativa oferece retorno líquido positivo e a escolha
do Control prevê perda severa, CASH pode ser escolhida **desde que
compense economicamente a alternativa do Control**. Se manter a
posição atual é mais vantajoso que CASH e que a rotação proposta pelo
Control, a política pode preservar a posição sem uma troca desnecessária.
A trava original de permanência mínima continua válida.

Nenhum limiar, horizonte, peso, LightGBM, contrato de snapshot, U75 ou
sistema MCT foi ajustado usando o resultado financeiro do ZIP.
O histórico de v1.0.1 não foi sobrescrito. A revisão constitui uma
hipótese metodológica adicional após inspeção de resultados e continua
exploratória; precisa de nova avaliação em datas posteriores sem
vazamento de informação e em novo corte prospectivo.

### Reexecução

```bash
git status --short
git fetch origin
git switch --track origin/research/u75-economic-rotation-recovery-v1.0.2
python -m py_compile pesquisa/vantagem_economica_rotacao_u75.py pesquisar_vantagem_economica_rotacao_u75_spyder.py
python -m unittest discover -s tests -p "test_economia_liquida_cash_v102.py" -v
python -m unittest discover -s tests -p "test_vantagem_economica_rotacao_u75.py" -v
```

No Spyder (F5):

```python
%run pesquisar_vantagem_economica_rotacao_u75_spyder.py
```

Sem dados novos a nova execução é um **replay de depuração exploratório**,
não avaliação prospectiva independente. Uma melhoria financeira posterior
não comprovará generalização. Não promover ao MCT apenas com esse replay.
