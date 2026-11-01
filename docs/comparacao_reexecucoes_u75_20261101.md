# Reexecução idêntica do backtest U75: 01/11/2026

**Versão deste relatório:** `u75-capital-preservation-audit-v1.1`.  
**Estado:** comparação entre duas execuções, **não** validação causal da preservação de capital.  
**Contratos:** `u75-daily-market-protection-v1`, `u75_daily_cash_protection_enabled=true`, 75 ativos efetivos.

## Entradas, identidade e escopo

| Campo | Execução anterior | Nova execução |
|---|---|---|
| Job | `20261101T005056-0acb4c89` | `20261101T021336-83e49685` |
| Arquivo | `market_cycle_trader_20261101T005056-0acb4c89.zip` | `market_cycle_trader_20261101T021336-83e49685.zip` |
| SHA-256 ZIP | `6eeafa90937b501f66356c3d1e6019bbde342430d92905570d32b4bcffa2228e` | `6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe` |

Ambos contêm 1.578 pregões do período 22/07/2020–30/10/2026. **8 dos 11 arquivos internos são idênticos byte a byte**, incluindo os históricos CSV completos de posições/previsões, trades, diagnóstico de decisões, diagnóstico agregado, resumo textual, comparação e `failures.csv`. Os três JSONs diferentes (`strategy_manifest`, `experiment_manifest` e `metrics`) possuem apenas **16 campos diferentes**: 8 campos de identidade/tempo de execução entre os manifestos; 6 campos de profiling de tempo, além de caminho/hash do snapshot temporário em `metrics.json`. Nenhuma política, preço, escolha, retorno, operação ou métrica econômica mudou.

## Métricas idênticas

| Métrica | Ambas as execuções |
|---|---:|
| Capital inicial | US$ 10.000,00 |
| Capital final | US$ 76.710.842,01 |
| Maior patrimônio | US$ 102.034.355,76 |
| Drawdown máximo | **−41,68%** |
| Drawdown ainda aberto no fim | −24,82% |
| Expected Shortfall empírico 95% (79 piores retornos em 1.577) | **−7,80%** |
| Sessões com indicador de estresse U75 | 163 |
| Sessões de CASH confirmado | 2: 17/06/2022 e 19/04/2024 |
| Sessões com `u75_cash_override=true` | 2 |
| Sessões em 2026 | 209; 0 CASH |

A regra exige `u75_market_stress=true`, cobertura válida de todos os 75 e **0** candidatos técnicos. Nas duas saídas reais, todos os 75 estavam cobertos e o número de candidatos era zero; durante as demais 161 sessões de estresse, a intervenção não ocorreu.

## Conclusão científica

A execução repetida comprova **reprodutibilidade das decisões e resultados financeiros sob o mesmo snapshot/configuração**, e a correção visual da linha verde não alterou a política. **Não comprova que a saída para CASH evitou perdas**. Para isso é necessário comparar U75 com overlay versus a mesma estratégia Control **sem overlay de CASH** no mesmo motor, calendário, dados e custos. O `buy_hold_equity` do ZIP não representa esse controle contrafactual.

O maior drawdown −41,68% ocorreu no episódio CAPR entre 27 e 31/07/2026, quando o estresse coletivo não foi acionado no fundo. Uma queda idiossincrática do ativo escolhido não implica risco sistêmico dos 75 e, portanto, não demonstra defeito da guarda coletiva. A pesquisa deve distinguir proteção coletiva do risco de concentração antes de propor alterações matemáticas. Em nenhuma hipótese ajustar parâmetros usando as perdas já conhecidas e apresentar esse mesmo período como avaliação independente.

## Artefatos locais da comparação

A auditoria comparou os arquivos internos via SHA-256 e os objetos JSON de maneira recursiva, mantendo as duas evidências originais intactas. Arquivos gerados para o pesquisador: `comparacao_metricas.csv`, `auditoria_hashes_arquivos.csv`, `diferencas_metadados.json`, `resumo_anual_nova_execucao.csv`, `sessoes_u75_nova_execucao.csv`, `relatorio_comparacao.md`. São resultados descritivos que podem ser verificados com os dois pacotes ZIP e o script `pesquisar_preservacao_capital_u75_spyder.py` entregue na conversa, sem acesso a banco de dados ou API.

**Próxima experiência:** Control idêntico versus U75 com CASH diário, sem alterar o Winner/MCT, em protocolo causal no TCC.
