# Pesquisa: saída intradiária adaptativa

Mesma branch, mesmo script, mesmo ZIP. Nenhuma alteração na main ou no MCT em produção.

## Entrada e execução

Coloque o ZIP original exportado pelo MCT `market_cycle_trader_20261124T005959-1a1fa2f1.zip` na raiz do TCC. Não precisa descompactar. Alternativamente indique `--mct-zip CAMINHO`. Não há fallback para o histórico congelado antigo. A leitura valida identificação, curva oficial, seleção, trades e datas da decisão. Strategy #33 revisão 4: 75 ativos elegíveis, 1594 sessões, patrimônio final US$ 173.441.204,20; 51 ativos efetivamente selecionados.

```bash
python pesquisar_saida_intradiaria_adaptativa_spyder.py --self-test
python pesquisar_saida_intradiaria_adaptativa_spyder.py
```

No Spyder, reinicie o kernel e execute F5. Dependência adicional: exchange-calendars>=4.8,<5 (já em requirements.txt).

## Experimento

- Treino: fold 1 e primeira metade do fold 2, 755 sessões. Validação: segunda metade do fold 2, 251 sessões. Duas sessões de embargo, teste exploratório: fold 3 inteiro, 586 sessões. Como resultados desse período foram vistos em pesquisas anteriores, uma futura janela independente é necessária para confirmação. A primeira camada usa somente seleção exportada, cuja decisão antecede a sessão.
- Control principal: patrimônio contínuo de US$ 10 mil a US$ 173.441.204,20, mantendo o capital acumulado antes do fold 3. O overlay causal só é aplicado ao fold 3, sem usar políticas treinadas retrospectivamente nos dois primeiros folds. A identidade matemática com a curva oficial é verificada; não é reprodução independente do motor. O resultado do fold 3 reescalado a US$ 10 mil permanece em coluna auxiliar.
- Comparação proporcional: uma venda de proteção no primeiro open observado após o sinal, antes do open da última barra da sessão. Recompra programada do mesmo ativo já selecionado pela Winner no open da última barra (15:55 NY em dia normal). Fator adicional = preço de venda / preço de recompra × (1-custo)/(1+custo). Mantém custos originais e cobra 10 bps extras por lado. Recompra antes do fechamento preserva exposição overnight; na abertura seguinte, a decisão original da Winner continua soberana. É proxy sem limite de liquidez, não fill comprovado.
- Replay com liquidez separado: até 1% do volume anterior, novas compras na janela inicial e tentativa de recompra ao open da última barra após venda de proteção, sujeita à capacidade disponível. Liquidações de rotação residuais podem continuar ao longo do dia. Capital inicial US$ 10 mil no fold 3. Se faltar preço do ativo mantido, resultados financeiros são marcados inválidos; não comparar patrimônio artificialmente congelado.
- Série histórica SPLIT/SIP de cinco minutos (ajuste de preços e volumes). Cache próprio em dados_intraday/split_sip_5min_full_month, sem misturar ou apagar RAW. Consultas limitadas ao último fechamento do backtest, inclusive mês corrente. Busca o ticker realmente mantido em caso de liquidação residual. Erros com mensagens sanitizadas. Não envia ordens reais.
- Kalman, multirrolling, BOCPD, LightGBM, híbrido, LightGBM com MFE/MAE, CUSUM normalizado pela volatilidade, LightGBM quantílico (10/50/90%), classificação de três regimes com recuperação e sobrevivência discreta em intervalos de 30 min. Alvos futuros somente no treino e na auditoria. Frequências 5, 15, 30, 60 min e penalização de risco escolhidas na validação, inclusive opção sem saídas.
- Sem gatilho percentual fixo, sem rotação pela camada 2, no máximo uma decisão de venda. O ZIP diário não comprova sinal de utilidade independente disponível antes do fechamento. A recompra programada do ativo já selecionado é uma hipótese causal de pesquisa, não a seleção de um novo ativo pela Winner no fechamento. A implementação completa dessa avaliação requer sinal datado e executável produzido pela primeira camada.
- Vantagem da venda: preço de saída dividido pelo open programado de recompra da mesma série SPLIT, descontando os dois custos. Não retirar o gap overnight se a recompra foi realizada. Não misturar preço RAW com preço ajustado. Fechamento SPLIT comparado ao preço implícito no Control; auditoria em 0,1% e tolerância de escala de 3% (não é um stop). Divergências relevantes invalidam a avaliação financeira afetada. Diferenças de leilão de fechamento e dividendos ainda exigem reconciliação independente.

- Oracle retrospectivo: calcula o melhor fill após um checkpoint disponível, para cada frequência de 5/15/30/60 min, somente se a venda e recompra gerarem ganho líquido. Ausência de dados mantém Control; não usar o resultado como teto global certificado. Exporta cobertura, oportunidades, patrimônio acumulado em todo o histórico e no fold 3, além da fração de ganho logarítmico capturada pelas políticas. O Oracle jamais alimenta seleção de parâmetros ou previsão causal. Alvos de retorno e excursões foram alinhados ao open de recompra.

## Saída

Sempre `output/pesquisa_saida_intradiaria_adaptativa.zip` e `output/pesquisa_saida_intradiaria_adaptativa.log`. ZIP sobrescrito atomicamente após execução; falha preserva o anterior, consulte data e log.

Inclui qualidade preditiva em suporte comum contra previsão constante zero, Brier de recuperação e cobertura quantílica, comparação ancorada, comparação com liquidez, referência MCT diária, cobertura e cotações ausentes, ordens e curvas dos dois replays, previsões, seleção de validação, sensibilidade de custos, figura, manifesto, hashes, código e log. Teste sintético verifica causalidade, recompra somente no instante programado, aritmética do overlay, invariância de escala e corte do Oracle. A execução real depende do ambiente local e do acesso histórico Alpaca.

O teste sintético adicional verifica invariância de escala multiplicando toda a série intradiária por 40. A primeira execução do cache SPLIT exige nova coleta. O ZIP anterior não é alterado se houver erro.

Novos arquivos: oracle_comparacao.csv, oracle_oportunidades_diarias.csv, oracle_patrimonio.csv, patrimonio_continuo.csv. A comparação principal inclui retorno e risco de todo o período, além das métricas do fold 3. Reutilize o cache SPLIT da rodada anterior.

## Diagnóstico gratuito IEX × SIP

Execute no mesmo script:

```bash
python pesquisar_saida_intradiaria_adaptativa_spyder.py --feed-test
```

No Spyder: `%run pesquisar_saida_intradiaria_adaptativa_spyder.py --feed-test`.
O modo usa o leitor histórico e as credenciais Alpaca já configuradas na API local. Nenhuma ordem, alteração de plano, conexão Mongo ou mudança na Winner. Compara DNN, CAPR, MKSI e NVDA nos últimos dez pregões completos, baixando 1Min SPLIT de IEX e SIP (SIP com fim pelo menos 20 minutos no passado). Agrega janelas de 1, 5, 15, 30 e 60 min alinhadas à abertura. Mede cobertura, preço, extremos, participação do volume, idade da última observação e concordância da direção de retorno. Não preenche preços faltantes.

Oracle diagnóstico: mesmo fill SIP observado 1 minuto depois do checkpoint, mesma recompra SIP cinco minutos antes do fechamento, custo de 10 bps de cada lado. Compara todos os checkpoints SIP com os checkpoints também observáveis no IEX. Conhece o futuro e não é modelo de execução. O retorno negativo de janela é uma referência diagnóstica simples, não sinal da Winner. Este piloto não revalida os modelos ML em IEX.

Se executado durante o pregão regular, após os primeiros dez minutos, testa também consultas REST 1Min dos últimos dez minutos nos feeds IEX e SIP, registra a idade da última barra completa e eventuais restrições de assinatura. Fora desse período, realiza apenas a comparação histórica. REST recente não certifica latência contínua de WebSocket; histórico pode conter correções posteriores.

Mesmo ZIP de saída: acrescenta `feed_test_resumo.csv`, `feed_test_diario.csv`, `feed_test_checkpoints.csv`, `feed_test_barras_1min.csv`, `feed_test_requisicoes.csv`, `feed_test_recente.csv`, `feed_test_manifesto.json`, código e log do diagnóstico. Se o ZIP anterior existir e seus hashes estiverem válidos, preserva os demais resultados e substitui somente os arquivos feed_test_. A execução normal sem o argumento continua realizando a pesquisa original. Nenhuma credencial é gravada no ZIP.


## Monitoramento prospectivo durante um pregao

Mesmo script, branch e ZIP. Modo separado, sem backtest, sem enviar ordens, usando as credenciais da API Alpaca local:

```bash
python pesquisar_saida_intradiaria_adaptativa_spyder.py --feed-monitor-self-test
python pesquisar_saida_intradiaria_adaptativa_spyder.py --feed-monitor
```

No Spyder: `%run pesquisar_saida_intradiaria_adaptativa_spyder.py --feed-monitor`.

Inicie antes da abertura (o processo local aguarda a primeira barra fechada) ou durante o pregao regular. O calendario XNYS determina abertura, fechamento, feriados e sessoes curtas. Depois do encerramento regular, novas execucoes sao recusadas. Deixe o computador e o processo Python ligados. O monitor consulta DNN, CAPR, MKSI, NVDA e AMS, uma vez por minuto, 10 segundos apos a conclusao da barra. Usa IEX recente e SIP com 20 minutos de atraso, sem tentar acessar SIP recente repetidamente. O SIP continua ate 20 minutos apos o fechamento para parear os ultimos minutos da sessao. Em pregao normal sao 390 observacoes programadas por feed e por ativo, 3900 consultas logicas, com ate dez por minuto (retentativas internas da API podem aumentar as chamadas reais). Cinco workers, sem sobreposicao entre ciclos. Ciclos perdidos sao registrados, nao preenchidos retroativamente. Nao usar credenciais em logs.

O resultado distingue resposta HTTP bem-sucedida de disponibilidade da barra exata, idade da ultima barra, atraso adicional sobre o cutoff solicitado, tempo de resposta, erros classificados, intervalos sem barra e ciclos omitidos. O pareamento IEX/SIP usa o mesmo minuto de mercado recebido em momentos diferentes: nao e uma feature conhecida previamente. Cada consulta arquiva as barras efetivamente retornadas, inclusive repeticoes, permitindo investigar correcoes. A medicao e de polling REST e nao certifica latencia de WebSocket.

O ZIP e atualizado atomicamente apos o primeiro ciclo, a cada cinco ciclos e ao terminar ou receber Ctrl+C. Preserva todos os demais arquivos validos do ZIP e substitui os arquivos `feed_monitor_*` pela rodada corrente (nao acumula duas rodadas do monitor). Uma interrupcao abrupta preserva o ultimo checkpoint; para terminar antes do fechamento prefira Ctrl+C. Se iniciar tarde, o manifesto e os slots ausentes identificam cobertura parcial. Nao interrompa o script por fechamento forcado do Spyder se puder usar Ctrl+C.

Arquivos: `feed_monitor_manifesto.json`, `feed_monitor_consultas.csv`, `feed_monitor_barras_recebidas.csv`, `feed_monitor_slots_ausentes.csv`, `feed_monitor_resumo.csv`, `feed_monitor_minutos_pareados.csv`, `feed_monitor_comparacao.csv`, codigo e log. O resultado continua em `output/pesquisa_saida_intradiaria_adaptativa.zip`. Validar dados antes de treinar ou integrar qualquer politica de venda.

## Yahoo Finance versus Alpaca

Mesmo script, branch e ZIP. Diagnostico separado da pesquisa de modelos e do monitor continuo. A biblioteca yfinance nao e API oficial e os termos do Yahoo restringem coleta automatizada sem permissao previa. So use a coleta direta se autorizado pelo provedor; o argumento nao concede permissao. Alternativamente, importe CSV obtido e utilizado de forma autorizada, com colunas timestamp (ISO 8601 com fuso), symbol, open, high, low, close, volume.

```bash
python pesquisar_saida_intradiaria_adaptativa_spyder.py --yahoo-self-test
# Somente com permissao de coleta automatizada:
python -m pip install yfinance
python pesquisar_saida_intradiaria_adaptativa_spyder.py --yahoo-test --yahoo-authorized
# Alternativa com dados autorizados ja disponiveis:
python pesquisar_saida_intradiaria_adaptativa_spyder.py --yahoo-test --yahoo-csv cotacoes_yahoo.csv
```

No Spyder use %run com os mesmos argumentos. Usa ate cinco pregoes completos dentro dos ultimos sete dias corridos, por padrao todos os ativos elegiveis do _metrics.json da exportacao MCT indicada por --mct-zip ou pela referencia padrao. A exportacao atual possui 75 ativos. Candles de 1 minuto agregados em 1/5/15/30/60 minutos. --yahoo-pilot restringe aos cinco ativos do piloto. Alpaca RAW IEX/SIP e Yahoo auto_adjust=False; auditoria de escala (nao e stop) exclui sessoes sem conciliacao do Oracle. Mede cobertura, diferencas de preco/volume/extremos/horarios, idade da ultima observacao, concordancia direcional e erros em suporte comum dos tres feeds. Oracle retrospectivo usa fills SIP e mesmos dias comparaveis, nao constitui modelo operacional. Se mercado aberto apos 30 minutos, modo direto testa tambem snapshots recentes Yahoo e IEX e SIP atrasado 20 minutos para os cinco ativos do piloto e o ultimo ativo selecionado no backtest. --yahoo-recent-all amplia para todos, em consultas sequenciais, nao simultaneas. --yahoo-historical-only desabilita snapshots recentes. CSV nao testa disponibilidade Yahoo recente. Snapshot unico nao mede latencia continua.

Arquivos yahoo_test_* sao adicionados ao mesmo ZIP preservando resultados anteriores validos. Nao altera MCT, nao envia ordens, nao contrata planos. Nao redistribua dados de terceiros sem permissao aplicavel. A execucao normal e o monitor Alpaca nao mudam.

### Diagnostico do universo completo

```bash
# Somente se a coleta direta for autorizada pelo provedor:
python pesquisar_saida_intradiaria_adaptativa_spyder.py --yahoo-test --yahoo-authorized --yahoo-historical-only
# Ou dados Yahoo autorizados em CSV:
python pesquisar_saida_intradiaria_adaptativa_spyder.py --yahoo-test --yahoo-csv cotacoes_yahoo.csv --yahoo-historical-only
```

A referencia MCT original deve estar na raiz do TCC (ou indicar --mct-zip CAMINHO). O manifesto inclui hash, lista completa e quantidade de ativos, bem como contadores de consultas historicas. Com 75 ativos, 225 consultas logicas previstas (75 por fonte; retentativas internas podem aumentar chamadas). Se Yahoo sinalizar rate limit, as consultas Yahoo restantes sao marcadas como skipped; nao tentar contornar restricao. Nenhum resultado parcial significa cobertura zero real do mercado. 

Novos arquivos: yahoo_test_universo.csv (uma linha por ativo, status das tres consultas, cobertura por frequencia, comparacao em suporte comum, escala, Oracle, ocorrencias historicas na carteira e anomalias de volume); yahoo_test_auditoria_volume.csv (por ativo, sessao e fonte, razoes de volume no mesmo minuto, mediana, p95, maximo, contagens acima de 10x e 100x, horario do maximo). Razoes extremas sao flags de auditoria e nao prova de volume acumulado; nao corrigir ou imputar automaticamente. Nenhum ativo e excluido silenciosamente. Dados historicos nao certificam disponibilidade ao vivo. Mesmo ZIP, sem alterar a estrategia ou criar novas versoes.



## Optimal stopping: venda intradiaria, caixa ate a proxima abertura (regra vigente)

A regra anterior de recompra no fechamento esta SUPERADA. Os resultados antigos de Oracle e os modos historicos anteriores a esta correcao foram mantidos apenas para reproducibilidade, e nao representam a politica vigente. A partir de agora, use --optimal-stop.

```bash
python pesquisar_saida_intradiaria_adaptativa_spyder.py --optimal-stop-self-test
python pesquisar_saida_intradiaria_adaptativa_spyder.py --optimal-stop
```

Spyder: %run pesquisar_saida_intradiaria_adaptativa_spyder.py --optimal-stop. Mesmo ZIP, branch e exportacao MCT; --mct-zip CAMINHO se necessario. Preserva demais resultados validos e substitui apenas optimal_stop_*. Usa cache SPLIT/SIP 5Min, sem volume/VWAP como features. Sem chamadas Yahoo, sem ordens e sem alteracao Winner.

Regra: monitorar somente o ativo selecionado pela Winner na abertura; manter ou emitir uma unica venda de protecao durante a sessao. Uma vez vendida a posicao integralmente, permanecer em caixa ate a abertura seguinte, inclusive durante a noite e fins de semana. Nao ha recompra no fechamento nem rotacao intradiaria. No replay proporcional, compra do alvo oficial congelado somente na abertura seguinte. A Winner real partindo de CASH pode escolher outro ativo por regras de permanencia e margem; reexecutar o motor diario nesse estado e etapa separada, nao demonstrada neste replay.

Recompensa: ganho log incremental da venda versus manter ate a abertura seguinte. Na aproximacao de posicao integral e mesmo ativo na abertura seguinte, R=log(S/O)+log((1-c)/(1+c)), S=fill de venda intradiaria, O=abertura do ativo vendido no proximo pregao. O calculo implementado incorpora caixa residual e quantidade oficiais, usa preco e taxa de venda oficiais se Control rotacionou e evita cobrar outra compra quando a compra oficial ja esta no fator diario. O dia da venda e marcado a caixa; na proxima abertura remove-se o gap overnight da referencia. Se o horizonte terminar no fechamento, nao inventar proxima compra. Falta de abertura apos venda invalida a avaliacao; nao usar futuro para impedir sinais. Custos 10 bps por lado adicional, testes de sensibilidade 5/10/25/50 bps. Fills sao proxies. Replay de liquidez separado, com janela de abertura (primeira hora) e participacao de 1% do volume anterior, pode ter saldos residuais; nunca recompra apos venda de protecao no mesmo dia.

Metodos: optimal_stopping (regressao de vantagem de venda vs hold), LSMC (inducao retroativa Ridge), FQI (12 iteracoes aproximadas LightGBM), decision_focused (busca regularizada sobre ganho log da primeira venda), Expected Shortfall (cauda inferior condicional a 10%). Control sem intervencao e horario fixo sao comparadores. Cadencias 5/15/30/60 min; minimo de seis barras consecutivas e r6 observavel. Oracle usa mesma elegibilidade causal, mas escolhe retrospectivamente o melhor fill posterior. Exclui labels de treino cuja abertura seguinte esteja fora do treino. Validacao cronologica seleciona frequencia e politica, HOLD concorre; fold3 exploratorio ja inspecionado. Controle oficial continuo e identidade sem vendas verificadas; overlay somente fold3.

Arquivos optimal_stop_comparacao.csv, optimal_stop_validacao.csv, optimal_stop_oracle.csv, optimal_stop_oracle_curvas.csv, optimal_stop_oracle_oportunidades.csv, optimal_stop_referencia_overnight.csv, optimal_stop_previsoes.csv, optimal_stop_patrimonio.csv, optimal_stop_ordens.csv, optimal_stop_cobertura.csv, optimal_stop_liquidez*.csv, optimal_stop_custos.csv e manifesto/log/codigo. O manifesto identifica cash_policy=overnight_until_next_winner_open. Esta rodada nao demonstra transferibilidade Yahoo, nem generalizacao independente nem reranking real da Winner a partir de caixa.


### Contexto anterior e primeira venda (rodada vigente)

O mesmo comando --optimal-stop agora testa J1 (somente sessao atual, controle metodologico anterior), J2 (60 min anteriores), J3 (120 min anteriores), J4 (pregao anterior completo), J5 (dois pregoes anteriores completos), J6 (cinco pregoes anteriores completos), sempre acrescidos das barras ja encerradas da sessao atual. Usa o historico do MESMO ticker selecionado na sessao atual, mesmo que nao estivesse comprado anteriormente. Calendario XNYS; cache SIP SPLIT, pode buscar meses adicionais que ainda nao estejam em cache. Nao espera abertura do mercado e nao consulta Yahoo nesta rodada.

Contexto anterior e resumido por retornos observados dentro de cada sessao, volatilidade, retorno recente, faixa, drawdown, cobertura e gaps anteriores separados. Gap atual = log(open atual / close anterior); retorno atual e retorno da primeira barra entram como features normalizadas. Nenhuma barra futura, nenhum preenchimento de preco, nenhum volume/VWAP nas features. Para J2-J6, exige contexto anterior disponivel, close da ultima barra regular anterior, ao menos seis retornos observados e abertura atual observada. Assim a primeira barra de cinco minutos encerrada pode emitir sinal, com fill no open posterior; J1 continua a exigir r6 e seis barras recentes. Se contexto nao disponivel, a janela nao emite sinal, sem fallback silencioso. As janelas anteriores sao features agregadas, nao concatenacao ingenua de retornos de 5min com gap overnight.

Cinco modelos x seis janelas x quatro cadencias = 120 combinacoes de modelo/janela/cadencia, mais comparador de horario fixo, parametros de ES e HOLD. Oracle separado por janela e cadencia com a mesma elegibilidade de checkpoint e recompensa overnight. Escolha de janela, cadencia e parametro por validacao cronologica; empate favorece HOLD. Fold3 permanece exploratorio. Selecao congelada da Winner na abertura seguinte (nao reexecuta ranking a partir de CASH). Venda unica, sem recompra no mesmo dia; o replay considera a primeira venda, embora exporte previsoes posteriores para diagnostico contrafactual.

Novos arquivos optimal_stop_janelas.csv (resultado de cada metodo/janela escolhido na validacao), optimal_stop_contexto_cobertura.csv (primeiro minuto elegivel, contagem de checkpoints, disponibilidade de contexto), Oracle e previsoes com coluna window. Mesmos arquivos optimal_stop_* e mesmo ZIP, substituidos na rodada. Logs de cache e contexto a cada 100 pregoes, etapas de calculo, cada janela/frequencia/modelo e tempo decorrido. --optimal-stop-self-test inclui teste sintetico de separacao de gaps e elegibilidade inicial.
