# Control de pesquisa adotado em 30/10/2026: Strategy #30

A partir desta decisao, o Control das **novas** pesquisas e o backtest
operacional da Strategy #30, encerrado em **28/10/2026**.

| Campo | Referencia congelada |
|---|---|
| Patrimonio inicial | US$ 10.000,00 |
| Patrimonio final | **US$ 80.685.454,79** |
| Inicio do teste | 22/07/2020 |
| Fim do teste | 28/10/2026 |
| Ativos efetivos | **75** |
| Dias executados | **1.576** |
| Walk-forward | 3 folds: 504 / 504 / 568 |
| Sharpe | 2,3177188747 |
| Drawdown maximo | -41,682843% |
| Sessoes CASH | 0 |
| Rotacoes | 348 |

A fonte de evidencia e o ZIP recebido da execucao Strategy #30,
congelado pelo SHA-256:
`8f41720fc7af78ec1a07bff3d72d637a081326601b92f1e2bce9f117ee3ccb7a`.

A identidade da politica, o hash da configuracao, a assinatura dos
dados de mercado, o calendario e as contas sao conferidos em
`pesquisa/control_oficial_strategy30.json`.

## Reproducao e auditoria

No Spyder execute `pesquisar_controle_strategy30_spyder.py`.
O arquivo original e localizado em `evidencias/`, na raiz do projeto
ou em `Downloads`, ou por `CONTROL_STRATEGY30_ZIP`. Se o pacote
nao for encontrado, o script informa o caminho necessario.

O arquivo gerado em
`output/controle_strategy30/pacote_optimal_switching_cash_*.zip`
contem as series de capital, decisoes, operacoes, o manifesto de
integridade SHA-256 e um resumo. Falhas tambem geram ZIP de diagnostico.

## Regra para proximos experimentos

O antigo checkpoint de US$ 587.165.941,13 e preservado apenas como
**referencia historica arquivada** das pesquisas anteriores.
Ele nao deve ser substituido numericamente em novos scripts sem a
implementacao correspondente da politica.

O periodo antigo de avaliacao terminava em 06/10/2026; o novo termina
em 28/10/2026. A trajetoria Control e o criterio de calibracao tambem
sao diferentes. Nenhuma comparacao direta de patrimonios ou
experimentos CASH e metodologicamente valida sem reproduzir o mesmo
periodo, o mesmo snapshot de dados, os mesmos custos e a politica de
rotacao Strategy #30.

**A importacao do ZIP atesta apenas a identidade e consistencia do
resultado recebido.** Ela nao e uma reproducao independente completa
do modelo com os dados RAW da corretora. O ZIP de resultados
nao contem todo o historico de mercado para 75 ativos.
Portanto, ate essa paridade existir, os novos estudos CASH ficam em
fase de integracao de baseline, sem afirmacao de desempenho comparativo.

A aplicacao operacional nao e alterada por esta pesquisa.
