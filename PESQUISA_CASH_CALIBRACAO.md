# Pesquisa: CASH quando a pontuação de calibração é negativa

**Hipótese pré-declarada:** no começo de cada janela de teste, entrar em CASH e permanecer em CASH durante toda a janela se o `selected_calibration_score` da calibração anterior for menor que zero.

**Condição de causalidade:** `calibration_end` deve preceder `test_start` em cada fold. Não usar scores recalculados com observações futuras. O score é a recompensa cumulativa ajustada ao risco do processo de calibração da política, **não** o retorno futuro de CAPR nem uma probabilidade de queda.

## Execução isolada

O script `pesquisar_cash_calibracao_spyder.py` é uma ferramenta de **pesquisa offline**, não altera `engine/`, não refaz treinamento, não faz downloads e não usa banco de dados. Execute no Spyder ou no terminal:

```bash
python pesquisar_cash_calibracao_spyder.py \
  --pacote output/reproducao_universo/pacote_reproducao_universo.zip \
  --saida output/pesquisa_cash_calibracao
```

É necessário possuir o ZIP da reprodução oficial `v1.23.0` que inclua `fold_calibration.csv`, `predictions.csv` e `reproducao_universo.json`. O repositório **não** possui todos os 10 arquivos locais dos candidatos, de modo que o replay oficial completo não é pressuposto neste teste.

## Evidência do checkpoint congelado

Foi analisado um pacote histórico de reprodução `v1.23.0`, com SHA-256 `d0f222c275b97a9c0b332ad2ef59b3232992957829b92444b2721f846e21709b`, recebido em conversa e validado contra o capital do manifesto.

| Fold | Calibração encerrada | Teste fora da amostra | Score selecionado | Capital final Control |
|---|---|---|---:|---:|
| 1 | 24/04/2020 | 22/07/2020 a 21/07/2022 | -1,1759056753 | US$ 67.999,95 |
| 2 | 25/04/2022 | 22/07/2022 a 24/07/2024 | -0,7855353494 | US$ 10.968.842,01 |
| 3 | 26/04/2024 | 25/07/2024 a 06/10/2026 | -0,0595317803 | US$ 587.165.941,13 |

Todas as três pontuações de calibração selecionadas foram negativas antes de seus respectivos períodos de teste. O Control cresceu de US$ 10.000 para US$ 587.165.941,13, com 1.560 pregões fora da amostra.

Com a regra `score < 0`, todos os folds ficariam integralmente em CASH. Portanto, considerando juros zero e ausência de operações, o resultado contrafactual é **US$ 10.000**, exposição 0%, drawdown 0% e Sharpe indefinido. Aqui o contrafactual é exato porque o gate nunca libera entrada. A regra **falha nesse primeiro teste**: score negativo isoladamente não é um critério suficiente para saída.

## Limitações e próximos testes

- Isto **não** testa o caso Live de calibração diária. O replay oficial guarda apenas a margem e a pontuação por fold, cada um com centenas de sessões. Aplicar a mesma regra em recalibrações diárias exige reconstruir cada score usando somente dados disponíveis até aquela data, com custos e estado de carteira realistas.
- Na presença de folds mistos, o script interrompe em vez de multiplicar retornos do Control: ao ficar em CASH, a posição e o tempo de permanência mudam, de modo que um simples overlay da série original seria incorreto.
- Não ajustar o limiar a posteriori para perseguir o checkpoint. O próximo teste deve usar limiares e métricas pré-definidos com avaliação fora da amostra separada.
- Não aplicar esta hipótese ao ambiente operacional Live sem evidência adicional; o Control deve permanecer congelado.
