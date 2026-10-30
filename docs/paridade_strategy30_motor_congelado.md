# Reprodução independente do Control Strategy #30

O Control de pesquisa vigente segue sendo o
resultado **US$ 80.685.454,79**, com 75 ativos e ate 28/10/2026.

## Fonte de codigo congelada

O pacote `pesquisa/referencia_strategy30` contem copias da fonte do
motor LightGBM na revisao
`f78b718ae4a459e34d91cc78a486dd6bbba197b5`.
Os arquivos `rotacao.py`, `modelo_lightgbm.py`, `execucao.py` e
`diagnosticos.py` tem os mesmos hashes Git da fonte original.
Esta copia reside **apenas no TCC**; nao altera o motor historico
`engine/`, sua referencia antiga de 587 milhoes ou qualquer sistema externo.

## Fluxo de reproducao (Spyder)

1. Copie o ZIP original de resultados da Strategy #30 para
   `evidencias/` ou `Downloads`, ou configure `CONTROL_STRATEGY30_ZIP`.
   O ZIP nao traz os dados brutos de mercado.
2. Execute `preparar_snapshot_strategy30_spyder.py`, com credenciais
   Alpaca no arquivo `.env`. O script baixa 75 CSVs RAW/SIP/1Day,
   corporate actions, sem banco, para
   `dados/strategy30_snapshot`, corte `2026-10-28`. Nunca substitui
   o snapshot antigo. Se ja existe `manifest.json` integro, reutiliza.
3. Execute `pesquisar_paridade_strategy30_spyder.py`.

O script confere identidade do ZIP, perfil, hashes de configuracao e de
resultados, carregamento de 75 ativos, datas e numero de folds
**504/504/568**, e reproduz o aprendizado LightGBM do zero com o motor
congelado. Os parametros sao aplicados **a partir do manifesto original**,
sem ajuste manual para tentar aproximar o resultado.

A auditoria compara, em **todas as 1576 sessoes**, o ativo selecionado,
o fold e a evolucao do capital, com diferenca maxima de um centavo.
Qualquer divergencia gera `erro_execucao.json`,
`comparacao_sessao_a_sessao.csv` quando disponivel e ZIP com status
`failed`. Os pacotes ficam em `output/paridade_strategy30`.

## Limites de reproducibilidade

O download novo da Alpaca nao e prova de identidade com os dados baixados
na data da execucao original. O ZIP de resultados apresenta assinaturas
dos 75 ativos, mas nao traz todos os seus precos RAW; a auditoria faz
checagem inicial de datas/linhas e revela diferencas remanescentes pela
comparacao de todas as decisoes. Uma correspondencia perfeita seria
evidencia forte de reproducao do backtest, nao uma validacao
prospectiva de lucro.

Se nao houver paridade, a pesquisa de CASH continua bloqueada.
Nao alterar as margens dos folds com base no patrimonio final e nao
usar a curva de capital importada como substituta de uma politica executavel.

O checkpoint anterior de US$ 587.165.941,13 permanece arquivado como
resultado de outra configuracao e periodo e nunca entra na aprovacao
da reproduçao atual.
