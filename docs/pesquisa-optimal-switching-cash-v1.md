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
