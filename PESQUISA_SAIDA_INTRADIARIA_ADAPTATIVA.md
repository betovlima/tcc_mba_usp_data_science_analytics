# Pesquisa: saída intradiária adaptativa

Experimento isolado, sem alteração do MCT ou da Winner. Base: research/intraday-second-layer-comparison-v1. Reutiliza o histórico congelado de posições Control (não confundir com a estratégia corrente em produção), cache local de candles RAW/SIP de 5 minutos e leitor histórico da API local. Consulta adicional do ticker anterior em dias de rotação. Nenhum endpoint de ordens.

## Execução

```bash
python -m pip install -r requirements.txt
python pesquisar_saida_intradiaria_adaptativa_spyder.py --self-test
python pesquisar_saida_intradiaria_adaptativa_spyder.py
```

No Spyder, reinicie o kernel, abra o script e execute F5. As dependências e o cache precisam estar no ambiente local. O arquivo ZIP é sobrescrito atomicamente somente após conclusão; em caso de falha, preserve o resultado anterior e leia o log (verifique a data do ZIP).

## Desenho

- Control intradiário: carteira contínua sem saída antecipada, mesmas cotações, regras e custos.
- Kalman local linear, multirrolling e BOCPD: estimadores estatísticos de estado, seguidos de regressão Ridge do retorno restante e magnitude esperada de perda.
- LightGBM com indicadores tradicionais; variante híbrida com Kalman, multirrolling e BOCPD.
- Checkpoints horários, candles concluídos. Label: retorno logarítmico do open do próximo candle ao fechamento regular da sessão. Não existe preço futuro nas features.
- Escore: retorno esperado menos penalização de risco mais custo evitado de vender. A saída é acionada quando o escore é negativo. A penalização (0, 0.25, 0.5 ou 1) ou a ausência de saída é escolhida exclusivamente na validação (log do patrimônio + 0.5 MaxDD). Teste temporal intocado, divisão 60/20/20 com embargo de uma sessão entre partições.
- A camada 2 não recompra no mesmo dia. Alocação da primeira camada apenas na abertura seguinte, em janela de 09:35 a 10:25 NY; não inventar fill ao fechamento nem comprar retrospectivamente. O snapshot não certifica timestamps das decisões ou disponibiliza um sinal independente de fechamento. A hipótese de disponibilidade pré-abertura precisa ser auditada antes de uma conclusão de produção.
- Preço proxy: open posterior ao sinal, limitado a 1% do volume anterior; custos padrão 10 bps por lado, sensibilidade 5/10/25/50 bps mantendo política congelada. Liquidação parcial pode ocorrer.
- Capital inicial US$ 10 mil no início do teste. Sem equivalência ao patrimônio original do TCC ou à Winner atual. RAW não incorpora dividendos e splits à carteira: eventos societários precisam ser reconciliados antes de qualquer validação de patrimônio. Sem confirmação de fills reais; não promover resultado parcial.
- Não existe trava de permanência mínima ou stop-loss de preço fixo na segunda camada. O alvo da Winner é mantido congelado, sem buscar novos ativos.

## Saída

`output/pesquisa_saida_intradiaria_adaptativa.zip` e `output/pesquisa_saida_intradiaria_adaptativa.log`.

ZIP: comparação de estratégias, seleção em validação, curva diária, PNG, ordens simuladas, previsões fora da amostra, sensibilidade a custos, auditoria de cobertura, manifesto, log, código e hashes. Resultados com dados incompletos ou rotações bloqueadas são explicitamente marcados. O teste sintético verifica execução após sinal, ausência de recompra no dia e invariância do passado do Kalman. Não é evidência de retorno real.

A decisão de adotar uma camada exige evidência fora da amostra de proteção relevante, custos e liquidez viáveis, e ausência de degradação material da estratégia principal. Nenhuma estratégia é escolhida com base no desempenho do teste para nova otimização no mesmo período.
