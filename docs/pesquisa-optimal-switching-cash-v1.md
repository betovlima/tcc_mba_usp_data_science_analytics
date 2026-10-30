# Optimal Switching / CASH: pesquisa com universo unico de 75 ativos

**Versao:** `optimal-switching-cash-reentry-v1.0.2`
**Branch:** `research/optimal-switching-cash-reentry-v1`
**Status:** pesquisa, sem promocao operacional.

## Contrato de universo

O experimento trabalha sempre e exclusivamente com **75 ativos**. A lista
canonica e versionada em `pesquisa/universo_control_75.json`; o carregamento
de dados ocorre numa unica passagem sobre essa lista, antes de qualquer
treinamento ou avaliacao.

Os arquivos historicos continuam armazenados nas pastas originais do
projeto, sem reorganizar ou substituir seus conteudos. O carregador procura
as fontes congeladas correspondentes aos ativos do manifesto, verifica
a identidade do snapshot base e os hashes SHA-256 de todos os arquivos
externos previstos no contrato. Se algum ativo estiver ausente ou tiver
identidade estrutural invalida, o experimento **aborta**. Nenhum ativo e
removido, reconstruido ou substituido.

O calendario e o mesmo referencia AAPL usado no experimento historico.
O LightGBM e ajustado por fold diretamente com todos os **75 ativos** e
as margens de rotacao sao calibradas no universo integral. Nao existe
replay de um universo reduzido nem logs com contagem intermediaria.

## Hipotese matematica

O modulo `pesquisa/optimal_switching_cash.py` e independente das utilidades
LightGBM. Ele estima, em cada fold, tres regimes a partir de momentum
historico padronizado por volatilidade. Para cada regime, estima retorno
realizavel, perda negativa media e transicoes de estado. Programacao
dinamica finita compara duas alternativas: continuar investido no ativo
indicado pelo Control ou ficar em CASH, incorporando custos aproximados
de troca. A simulacao final contabiliza as taxas efetivas do motor.

Todas as estimativas para o fold usam observacoes anteriores ao periodo
de teste. As decisoes sao feitas no fechamento e executadas na abertura
do pregao seguinte. O metodo nao garante encontrar o fundo ou topo.

## Reproducao e verificacao

1. Atualize a branch e abra `pesquisar_optimal_switching_cash_spyder.py`
   no Spyder, diretório de trabalho na raiz do repositorio.
2. Execute F5. O programa primeiro valida e carrega **75/75 ativos**.
3. A simulacao Control deve reproduzir **US$ 587.165.941,13**
   ate a precisao de centavos. Divergencia interrompe o experimento.
4. Depois, o Optimal Switching utiliza o mesmo universo, calendario,
   folds, custos e motor financeiro. A unica variavel alterada e a
   politica experimental de exposicao/CASH.
5. As evidencias sao geradas em
   `output/optimal_switching_cash/<timestamp>/`. Nenhuma execucao
   anterior e sobrescrita. O `resultado.json` transforma valores
   nao finitos em `null`, sem gerar JSON invalido.

Dados RAW/SIP e eventos corporativos sao consultados exclusivamente nos
CSVs locais historicos. Nenhum download Alpaca e feito. A janela de
pesquisa termina em **06/10/2026**, antes do episodio de CAPR posterior
a essa data.

## Limites cientificos

O universo foi historicamente selecionado e nao constitui validacao
prospectiva da descoberta de ativos. O modelo inicial de Switching
assume o candidato escolhido pelo Control como unico estado de risco;
portanto, sua programacao dinamica nao resolve a troca direta entre
quaisquer pares de ativos. Os custos dessa troca continuam na
simulacao financeira. A avaliacao nao comprova retornos negociaveis ou
superioridade do modulo sem resultados fora da amostra e testes de
liquidez.

**Esta pesquisa nao modifica o codigo Control oficial nem a main.**


## Pacote automatico de auditoria, versao v1.0.3

Cada execucao pelo Spyder gera **automaticamente** um ZIP:

`output/optimal_switching_cash/pacote_optimal_switching_cash_<id_da_execucao>.zip`

O pacote usa apenas os resultados daquela execucao, incluindo:
- `resultado.json`: metricas Control vs Switching, parametros, folds e verificacao de universo;
- `control_predictions.csv`, `control_trades.csv`, `control_75_fold_calibration.csv`;
- `switching_predictions.csv`, `switching_trades.csv`, `switching_decisions.csv`;
- `switching_fold_models.csv`: recompensas, perdas e matriz de transicao dos regimes;
- `resumo_para_analise.md`: resumo numerico e alerta para politica degenerada que so escolhe CASH;
- `manifesto_pacote.json`: versao, identidade da execucao, arquivos, tamanho e SHA-256.

Mesmo quando um erro interrompe o experimento, o runner cria um ZIP marcado como
`failed` contendo `erro_execucao.json` e as evidencias ja escritas. O erro
continua sendo exibido no Spyder. Um pacote `failed` nao pode ser confundido
com resultado completo.

O ZIP nao inclui RAW/SIP, eventos corporativos, arquivos `.env`, credenciais
ou o banco de dados. O arquivo e testado internamente por integridade. Uma
execucao nunca sobrescreve o ZIP de outra.

### Observacao sobre a execucao inicial de 30/10/2026

O ZIP recebido exibiu 1.560/1.560 decisoes `cash_preferred` sobre o
universo oficial de 75 ativos. O Switching terminou em US$ 10.000,
sem operacoes, enquanto o Control reproduziu US$ 587.165.941,13.
A recompensa ajustada ficou negativa em todos os 675 estados-modelo
(75 ativos x 3 folds x 3 regimes). A pesquisa nao deve ser ajustada
usando o proprio teste fora da amostra como criterio de selecao.


## v1.0.4: calendario cronologico do Control restaurado

A execucao anterior com 75 ativos falhou o checkpoint ao gerar 4 folds
(e capital US$ 667.225,51), em vez dos 3 folds originais do Control.
A causa foi utilizar as 2.705 datas de OHLCV bruto de AAPL como calendario.

O motor `construir_quadro_rotacao` exige indicadores com janelas de
ate 200 sessoes e descarta 199 datas iniciais sem atributos suficientes.
A reproducao valida possui **2.506 sessoes de calendario preparado**,
3 folds e **1.560 sessoes fora da amostra**, com primeiro replay em
22/07/2020 e ultimo em 06/10/2026.

A v1.0.4 reutiliza diretamente o indice da serie preparada de AAPL,
aplica-o aos 75 ativos e valida a estrutura cronologica **antes de qualquer
treinamento**. O contrato de calendario esta em
`pesquisa/contrato_calendario_control_75.json`, elaborado com os metadados
do checkpoint historico entregue pelo pesquisador.
A contagem de 75 ativos permanece obrigatoria. Nao ha exclusao nem
reconstrucao de ativos.

A cifra de US$ 587.165.941,13 continua como checkpoint imutavel. Os
testes automatizados validam invariantes e nao substituem o replay completo
dos dados historicos locais, que deve confirmar o valor ao centavo.
O pacote ZIP automatico permanece ativo em sucessos e falhas.


## v1.0.5: duas hipoteses matematicas sem contaminar o teste por fold

O Control continua com **75 ativos**, 3 folds, 1.560 sessoes OOS e checkpoint
US$ 587.165.941,13 imutavel. Sao testadas duas politicas de exposicao:

1. **Log-growth**: mesma programacao dinamica, peso adicional de risco 0.
   A recompensa utiliza o retorno logaritmico historico estimado para cada
   regime. O objetivo e descobrir se a penalizacao extra explica a politica
   degenerada observada em v1.0.4.
2. **Risco calibrado**: pesos candidatos **0, 0.05, 0.10, 0.20, 0.35, 0.50,
   1.00**, definidos antes da nova execucao. Em cada fold, modelos de regime
   sao ajustados somente no trecho de treinamento (antes da calibracao).
   Um Control treinado antes dessa calibracao indica os ativos candidatos.
   O overlay simula cada peso na janela de calibracao, pontuado pela mesma
   funcao de recompensa usada no procedimento historico. O peso escolhido
   maximiza essa recompensa de calibracao; empate escolhe o menor peso.
   Depois disso, os modelos estatisticos sao reajustados ate o fim do
   trecho final de ajuste do fold, antes das datas OOS.

Os dois overlays usam o mesmo ranking Control, o mesmo simulador financeiro,
custos e calendario. Saida CASH e reentrada continuam sendo decididas
na sessao de fechamento com execucao na abertura seguinte.

**Limitacoes:** calibracao por ganho historico nao equivale a impor um limite
estatistico de CVaR; as medias estimadas por regime continuam incondicionais
a selecao feita pelo LightGBM, embora a calibracao avalie as decisoes de
candidatos do Control. A margem de rotação do Control e o peso de risco usam
a mesma janela de calibracao. Isso aumenta risco de sobreajuste.

**Importante para o TCC:** esta versao foi planejada depois que os resultados
OOS da v1.0.4 foram examinados. Portanto, embora cada fold evite vazamento
futuro, o intervalo historico ja foi consultado em uma iteracao cientifica
anterior. Os resultados nao podem ser tratados como uma validacao externa
intocada para a escolha entre v1.0.4 e v1.0.5. Exige-se uma avaliacao temporal
posterior e independente antes de qualquer promocao operacional.

O ZIP automatico agora contem `control_*.csv`, `log_growth_*.csv`,
`switching_*.csv` (risco calibrado), `switching_risk_calibration.csv`,
`switching_fold_models.csv`, `resultado.json`, `resumo_para_analise.md`
e `manifesto_pacote.json` com SHA-256. Se ocorrer erro, cria pacote de falha.
