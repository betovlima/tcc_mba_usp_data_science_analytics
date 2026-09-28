# TCC — Dependência sequencial das decisões e validação confirmatória

<!-- [TCC-DFL:ANALYSIS-003] -->
**Data:** 27/09/2026. **Versão:** `v1.3.0-dev.3`.
**Branch única:** `research/decision-focused-oof-v2`. **Status:** análise
concluída sobre histórico já observado; proposta metodológica, SEM novo modelo,
SEM tuning, SEM nova amostra OOS.
**Origem:** `decision_focused(4).zip` enviado pelo pesquisador,
SHA-256 `38f409435cbb7c337e527e15d56e9266db5b19f33c8aa113ed2b4c7da2a0202a`.
Foram examinados os artefatos de `decision_focused/active/`: pares diários,
curvas da bridge, operações, diagnósticos de decisão, rótulos OOF e reconciliação.
A execução do run `36349759992` já concluiu com sucesso e os resultados
foram reproduzidos no ZIP anterior. Ver
`docs/results/decision-focused-fix-004-episode-attribution.md`.

## 1. Pergunta de pesquisa

**Uma intervenção local altera somente o ativo do próximo pregão ou também
a trajetória de posições, holding e riqueza?** A auditoria identifica
o comportamento efetivamente observado, sem inferir a rentabilidade de
uma nova política ou afirmar causalidade no mercado real.

As seis intervenções locais da regressão estão em três episódios. O par
(BASELINE, EP001) adiciona o primeiro; (EP001, EP002) adiciona o segundo;
(EP002, EP003) adiciona o terceiro. Cada par compartilha toda a história
anterior e, depois do episódio, executa a mesma política-base nos estados
efetivamente visitados. Os efeitos são **marginais condicionais à ordem
cronológica**; são replays contrafactuais simulados, não experimentos
aleatórios independentes.

## 2. O que os números mostram

| Episódio | Datas de decisão | Sessões futuras com ativos distintos | Sessões futuras com mesmo ativo mas capital distinto | Primeiro ativo novamente igual | Último ativo distinto | Capital final com/sem episódio |
|---|---|---:|---:|---|---|---:|
| EP001 | 19/01/2023 | 2 | 916 | 24/01/2023 | 23/01/2023 | 99,051783% |
| EP002 | 08–11/10/2024 | 4 | 482 | 15/10/2024 | 14/10/2024 | 99,028828% |
| EP003 | 14/01/2026 | 3 | 166 | 16/01/2026 | 21/01/2026 | 77,242460% |

Para EP003, a primeira igualdade nominal do ativo em 16/01 **não é
convergência permanente**: as exposições divergem de novo em 20–21/01.
Depois da última sessão com ativo distinto, restam **165 sessões com
ativos iguais**, mas capitais diferentes. O índice de capital relativo
no primeiro dia após a última divergência é 0,772429504 e no último
dia 0,772424604 (diferença de apenas 0,049 ponto-base no fator de
capital, compatível com pequenos efeitos de custos não estritamente
proporcionais). Não tratar igualdade de símbolo como igualdade de
`(capital, quantidade, entrada, holding)`.

### Decomposição de EP003

| Marco | Capital com episódio menos capital sem episódio |
|---|---:|
| Execução de 15/01/2026, CORT vs AVGO | −US$ 120.146,17 |
| 16/01/2026, ambos em AVGO | −US$ 165.732,81 |
| 20/01/2026, CORT vs AVGO | −US$ 466.175,26 |
| 21/01/2026, CORT vs AVGO | −US$ 650.737,76 |
| 22/01/2026, ambos em AVGO após venda de CORT no cenário-base | −US$ 1.223.153,86 |
| Após 20 sessões | −US$ 1.412.175,96 |
| Capital final | −US$ 2.250.682,75 |

A trajetória sem a intervenção realizou P&L de +US$ 986.548,86 na
venda de CORT em 22/01/2026. Essa quantidade é um P&L da operação
no cenário, **não** o efeito causal isolado da intervenção: para isso,
usar diferenças de capital entre as duas trajetórias simuladas,
incluindo preços, quantidades, transições e custos.

O delta em dólares aumentou em **US$ 838.506,78** após o marco de 20
sessões, mas isso NÃO indica necessariamente novas decisões divergentes
posteriores: o fator relativo de capital permaneceu quase constante
desde 22/01. Grande parte do aumento do *gap nominal* é crescimento
composto aplicado sobre bases de riqueza distintas. A janela de
20 sessões capturou aproximadamente 62,74% da diferença terminal
**em dólares**, mas essa fração depende da janela restante e não mede
a porcentagem de decisões erradas ou de oportunidade econômica perdida
fora do horizonte.

Os três episódios têm efeitos terminais respectivamente de
−US$ 95.603,28; −US$ 96.989,26; −US$ 2.250.682,75. Reconciliam
exatamente −US$ 2.443.275,28, com diferença zero entre bridge final
e regressão original e entre BASELINE e Control oficial.

## 3. Interpretação metodológica

O target atual já calcula contrafactual **sequencial por 20 sessões**:
`log(W_20(ação, continuação_control) /
W_20(ação_control, continuação_control))`.
Não é correto dizer que ele mede apenas o retorno do ativo do primeiro
dia. O que a auditoria adiciona é que:

1. Uma decisão altera o **estado futuro da carteira**. Mesmo quando a
   política-base é idêntica, seu resultado depende de ativo incumbente,
   dias de holding, preço de entrada, caixa/quantidade e restrições.
2. Os rótulos gerados a partir de estados OOF internos podem estar
   desajustados à distribuição de estados do Control final: os modelos
   internos da pesquisa exigem 250 linhas mínimas, e o Control oficial
   exige 700; essa diferença já está documentada.
3. O horizonte de 20 sessões captura efeitos de curto/médio prazo,
   mas sua tradução para capital terminal depende das exposições
   subsequentes e do capital composto. Prolongar arbitrariamente o
   horizonte no mesmo OOS seria tuning retrospectivo, não evidência
   confirmatória.
4. Os 7.717 rótulos são linhas de ações candidatas sobre datas
   repetidas e janelas sobrepostas; **não** são 7.717 observações
   temporalmente independentes.
5. Uma métrica que só conta dias com ativos diferentes perderia o
   principal efeito observado: EP003 tem três sessões de exposição
   distinta no replay pareado, mas 166 sessões futuras com mesmo
   símbolo e valor patrimonial diferente.

**A observação não demonstra que uma nova loss, LGBM com outro target ou
um threshold produzirá ganho.** Demonstra uma limitação concreta da
interpretação local de overrides.

## 4. Próxima hipótese, explicitamente não implementada

Hipótese testável: incorporar à avaliação das decisões a distribuição
de *trajetórias completas condicionais ao estado*, e não somente o
retorno do ativo/candidato ou a confiança em uma vantagem agregada.
Uma formulação de pesquisa para estudo posterior é:

`Q_H(s_t,a_t) = E[ log(W_(t+H)(s_t,a_t,pi)/W_t) | informacao ate t ]`

comparada à ação-base no **mesmo** `s_t`, com ambas as trajetórias
continuando sob uma política `pi` congelada e custos idênticos.
O estado mínimo a auditar é: ativo/posição corrente, dias de holding,
capital, quantidade, preço/instante de entrada, score do incumbente,
scores de candidatos, custos e regras de guarda. Distinguir capital
normalizado para aprendizado de capital absoluto para auditoria.
Não misturar no mesmo teste mudança no modelo-base e mudança na
política de intervenção.

Essa é uma **proposta pré-experimental**, não uma afirmação de que
mais atributos ou um horizonte maior melhoram o resultado. Antes de
codificar/otimizar, fixar: definição de `s`, horizontes de labels,
estratégia de treinamento, critério de intervenção, orçamento de
busca e ablação com Control.

## 5. Protocolo de validação sem reciclar OOS já visto

**Etapa A — desenvolvimento histórico (já disponível, não
confirmatório):** usar os folds atuais apenas para verificar integridade
de replay, agrupamento por data, pares no mesmo estado, estabilidade,
cobertura e definição de features. Toda seleção de método nesses folds
deve ser rotulada como exploratória.

**Etapa B — pré-registro:** fixar em commit anterior ao período de teste:
método escolhido, features disponíveis no instante da decisão, horizonte,
purge, custos, parâmetros, seeds, regra de fallback, métricas primárias,
código/snapshot hash e critério para declarar ganho. A métrica primária
deverá avaliar capital líquido composto contra Control na MESMA janela;
reportar também MaxDD, Sharpe, pior segmento, rotações, fees, exposição
e distribuição de margens. Não usar avaliação de mercado futuro para
escolher variante.

**Etapa C — amostra cronologicamente nova:** o snapshot oficial congelado
termina em **17/09/2026**. Primeiro período elegível para uma avaliação
prospectiva independente inicia depois desse cutoff; os dados futuros
devem ficar em **snapshot separado e versionado**, nunca sobrescrever
`dados/pesquisa` oficial. Qualquer modelo treinado com targets
multi-horizonte de até 60 sessões deve usar apenas exemplos cujos
*desfechos já estavam disponíveis* no instante do fit; purge/embargo
adequado preservado. Se ainda não houver 20/60 sessões de desfecho,
não afirmar que um backtest longo está completo. Os dados temporários
mais novos usados no MCT não devem contaminar o TCC congelado.

**Etapa D — avaliação:** executar Control e candidato na mesma janela,
mesmos custos, sem otimizar limiares vendo o resultado. Se houver
retreinamento, sua cadência e as datas elegíveis devem ser pré-fixadas.
Não comparar capitais de janelas diferentes como se fossem mesmo OOS.
Guardar decisões, curva diária e diferenças de estado, além de IC ou
análise de incerteza por blocos cronológicos quando houver extensão
temporal suficiente.

## 6. Decisão nesta entrega

**Não alterar o LightGBM nem a política agora.** A primeira entrega
é este diagnóstico de dependência sequencial e o protocolo de prova.
A validação futura não foi executada e não existe, neste ZIP, um
resultado de política nova. Preservar `v1.3.0-dev.3`, branch e único
executável `pesquisar_decision_focused_spyder.py`. Não criar scripts
`_v2`, `_v3` etc. Próximo marcador `[TCC-DFL:FIX-005]` somente
quando ocorrer uma alteração concreta e autorizada no código.
