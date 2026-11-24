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
- Control principal: retornos oficiais diários, reescalados para US$ 10 mil no início do teste. A identidade matemática sobre toda a curva é verificada, não constitui reprodução independente do motor. Valor final do Control reescalado no fold 3: US$ 308.899,33.
- Comparação proporcional: saída no primeiro open real posterior ao sinal, permanece cash até a abertura da sessão seguinte. Ajusta retorno restante do pregão e exposição overnight, mantém taxas originais da referência e cobra custo extra de 10 bps por lado. Não simula fill ilimitado como execução comprovada.
- Replay com liquidez separado: até 1% do volume anterior, novas compras na janela inicial, liquidações residuais podem continuar ao longo do dia. Se faltar preço do ativo mantido, resultados financeiros são marcados inválidos; não comparar patrimônio artificialmente congelado.
- Série histórica SPLIT/SIP de cinco minutos (ajuste de preços e volumes). Cache próprio em dados_intraday/split_sip_5min_full_month, sem misturar ou apagar RAW. Consultas limitadas ao último fechamento do backtest, inclusive mês corrente. Busca o ticker realmente mantido em caso de liquidação residual. Erros com mensagens sanitizadas. Não envia ordens reais.
- Kalman, multirrolling, BOCPD, LightGBM, híbrido, LightGBM com MFE/MAE, CUSUM normalizado pela volatilidade, LightGBM quantílico (10/50/90%), classificação de três regimes com recuperação e sobrevivência discreta em intervalos de 30 min. Alvos futuros somente no treino e na auditoria. Frequências 5, 15, 30, 60 min e penalização de risco escolhidas na validação, inclusive opção sem saídas.
- Sem gatilho de preço fixo, sem rotação nova durante o intraday, sem recompra no mesmo dia. Não existe sinal de fechamento independente no ZIP; o modelo usa a abertura seguinte.
- Vantagem da venda: preço de saída dividido pelo fechamento da mesma série SPLIT, descontando custo. Gap overnight na permanência: abertura e fechamento anterior da mesma série SPLIT; na rotação: venda e fechamento anterior oficiais. Não misturar preço RAW com preço ajustado. Fechamento SPLIT comparado ao preço implícito no Control; auditoria em 0,1% e tolerância de escala de 3% (não é um stop). Divergências relevantes invalidam a avaliação financeira afetada. Diferenças de leilão de fechamento e dividendos ainda exigem reconciliação independente.

## Saída

Sempre `output/pesquisa_saida_intradiaria_adaptativa.zip` e `output/pesquisa_saida_intradiaria_adaptativa.log`. ZIP sobrescrito atomicamente após execução; falha preserva o anterior, consulte data e log.

Inclui qualidade preditiva em suporte comum contra previsão constante zero, Brier de recuperação e cobertura quantílica, comparação ancorada, comparação com liquidez, referência MCT diária, cobertura e cotações ausentes, ordens e curvas dos dois replays, previsões, seleção de validação, sensibilidade de custos, figura, manifesto, hashes, código e log. Teste sintético verifica causalidade, ausência de recompra e aritmética overnight. A execução real depende do ambiente local e do acesso histórico Alpaca.

O teste sintético adicional verifica invariância de escala multiplicando toda a série intradiária por 40. A primeira execução do cache SPLIT exige nova coleta. O ZIP anterior não é alterado se houver erro.
