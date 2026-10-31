# Diagnóstico do universo U75: fraqueza coletiva e posições efetivas

**Versão de pesquisa:** `u75-universe-stress-diagnostics-v1.0.0`.
**Branch:** `research/u75-stress-positions-diagnostics-v1`.
**Origem:** Control congelado Strategy #30, US$ 80.685.454,79,
75 ativos, 1.576 sessões, 22/07/2020 a 28/10/2026.
**Implementação operacional:** nenhuma alteração.
**Status:** diagnóstico exploratório, NÃO uma nova estratégia aprovada.

## Pergunta científica

Queremos distinguir um ativo isolado em queda de uma deterioração de todo
o U75 e descobrir quando a carteira permaneceu investida numa situação
em que não havia sequer **candidatos tecnicamente favoráveis** no
universo. O conceito de candidato técnico não garante lucro futuro;
não deve ser usado como prova de que *nenhum ativo bom existe*.

O resultado histórico é calculado com o Control exato já reproduzido.
Antes da comparação, um novo ajuste LightGBM reproduz obrigatoriamente
as 1.576 sessões de posição e patrimônio até US$ 0,01. Se falhar, a
pesquisa interrompe a avaliação.

## Indicadores calculados no fechamento, usando todos os 75 ativos

| Indicador | Cálculo | Interpretação |
|---|---|---|
| Participação positiva 5 pregões | ativos com retorno de 5 sessões positivo / ativos com indicadores válidos | força recente de todo o U75 |
| Participação positiva 20 pregões | mesmo cálculo em 20 sessões | força intermediária |
| Queda típica 20 pregões | mediana dos retornos individuais de 20 sessões | evita que um extremo domine a média |
| Abaixo da média de 50 pregões | quantidade com preço abaixo da média simples de 50 fechamentos / válidos | abrangência da tendência negativa |
| Candidatos técnicos | retorno 5 e 20 positivos **e** fechamento acima das médias de 20 e 50 sessões | quantidade de papéis com tendência favorável, não projeção de retorno |
| Cobertura | quantidade de ativos com indicadores disponíveis | evita classificar dados faltantes como crise |

**Critérios fixados ANTES desta nova execução exploratória:**
cobertura de ao menos 68/75; participação positiva em 5 pregões
no máximo 35%; em 20 pregões no máximo 45%; pelo menos 55% dos
ativos abaixo da média de 50 pregões; mediana dos retornos de
20 pregões no máximo −1,5%. Todas as condições devem ocorrer
simultaneamente para indicar **estresse coletivo**.

O diagnóstico de **estresse sem candidato técnico** requer ainda
zero candidatos técnicos em todos os 75 ativos. Se houver um
candidato tecnicamente favorável, a política exploratória não
recomenda saída; isso segue a exigência de avaliar o universo inteiro.

Esses limites são hipóteses fixas para teste, não foram validados para
provar detecção preditiva. Não são parâmetros a ajustar escolhendo
o maior patrimônio no período já observado. O indicador usa pesos
iguais de presença por ativo, não valor de mercado. O U75 contém
ações e outros instrumentos, incluindo exposição inversa; os
resultados descrevem o universo selecionado, não necessariamente
o índice acionário americano inteiro.

## Simulação financeira da permanência em ativos versus CASH

O sinal usa as características do **fechamento t**. A posição efetiva
é aquela executada no pregão **t+1**. O episódio só entra na auditoria
se o Control estiver investido, houver estresse coletivo E não
existir candidato técnico no U75.

Para isolar uma decisão, partimos do mesmo patrimônio e da mesma
posição do Control: em um replay impomos CASH durante **5 decisões**
e, a seguir, devolvemos o controle à política original, que decide
usando o estado real da carteira. Comparamos o patrimônio após
**10 sessões**, com o mesmo simulador e suas taxas. Cada replay
individual recomeça com todo o histórico do Control até o episódio,
não é apenas uma subtração simplista de retornos de preços.

Episódios contíguos viram uma única ocorrência. Episódios com janelas
de comparação que se sobrepõem não entram duas vezes na contagem.
Além disso, rodamos uma carteira alternativa com todas as intervenções
não sobrepostas, apenas como cenário exploratório. Nenhuma decisão é
tomada escolhendo retrospectivamente os episódios com melhor resultado.

**Os ganhos/perdas dos replays individuais NÃO são somáveis.** A
simulação completa com intervenções consecutivas pode ter outra
trajetória e outras condições de entrada, e seus resultados não
validam causalmente uma nova política de produção.

## Mapa mensal de ativos mantidos

O mapa mostra a **posição efetiva após cada sessão**, como no conceito
visual utilizado na aplicação operacional, e não o ticker previsto nem
apenas as ordens enviadas. A posição pode continuar igual durante
vários dias, mesmo sem nova negociação.

- `mapa_mensal_ativos.png`: visão por ano e mês, com o ativo que
  teve mais pregões e a quantidade de sessões. Intensidade indica
  concentração no ativo dominante.
- `mapa_mensal_interativo.html`: página autônoma, sem internet nem
  servidor. Clique em um mês para abrir todos os ativos efetivamente
  mantidos, seus pregões, entradas, saídas, sessões em CASH, retorno
  da carteira no mês e quantos pregões houve estresse coletivo.
- `mapa_mensal_resumo.csv` e `mapa_mensal_todos_ativos.csv`:
  contagens exatas; não omitem ativos com poucas sessões.
- `forca_diaria_u75.png`: participação positiva em 5 e 20 pregões
  e intervalos de estresse.
- `patrimonio_e_estresse.png`: evolução do capital e estresse coletivo
  observado no fechamento anterior, em escala logarítmica.

Também são exportados: indicadores diários e por ativo, decisões e
estresse, episódios contíguos, contrafactual individual, posições e
operações reais do Control, replay exploratório, auditoria de margens,
auditoria da paridade e `resultado.json`.

## Atenção especial: dependência de informação futura no motor congelado

A fonte de referência de LightGBM (funções
`_precalcular_utilidades_modelo` e `_utilidades_modelo` em
`pesquisa/referencia_strategy30/rotacao.py`) consulta abertura e
fechamento do **pregão seguinte** para decidir se há dados de mercado
válidos para um ativo. Isso pode afetar elegibilidade e constitui
uma dependência temporal a investigar.

A nova medida de estresse **não consulta esses valores futuros**;
usa apenas fechamentos de t e anteriores, e por isso não usa a contagem
de pontuações positivas do motor para determinar que não há candidatos.
Pontuações do modelo continuam em coluna de auditoria separada,
marcadas como observacionais e não causais.

**Não mudamos o motor Control congelado:** fazer isso destruiria
a referência de comparação. Corrigir e medir o efeito da elegibilidade
futura deve ser um estudo diferente, com outro hash e outra versão.
Mesmo que o motor reproduza exatamente US$ 80,69 milhões, isso não
significa ausência de vazamento temporal ou sobreajuste.

## Execução no Spyder

Faça o checkout desta branch, mantenha o ZIP original da Strategy #30
em `evidencias/` ou `Downloads`, com assinatura conhecida, e
reutilize o snapshot em `dados/strategy30_snapshot`.

Execute com F5:

```text
pesquisar_diagnostico_u75_spyder.py
```

A saída será criada em `output/diagnostico_u75/`, com
`pacote_diagnostico_u75_<identificador>.zip`. O pacote inclui CSVs,
documentação, imagens PNG e HTML interativo, mais um manifesto de
SHA-256 de todos os arquivos. Falhas também produzem ZIP de diagnóstico.

Não usar dados do banco, não baixar novamente o snapshot e não
promover a política experimental para a aplicação operacional.
