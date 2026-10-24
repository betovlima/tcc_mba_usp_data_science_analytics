# Correção v1.25.9 — identidade de META versus ETF METV

**Data:** 23/10/2026. **Branch única:** `research/v1.23.1-fresh-snapshot-20261022`. `main` e MCT operacional intocados. **Executor:** `pesquisar_mitigacao_overfitting_spyder.py`.

## Incidente real

O executor fixed75 v1.25.8 interrompeu `prepare_once` antes de criar saída e antes de treinar, com:

```
RuntimeError: Ativo estruturalmente inconsistente: META:
{'symbol': 'META', 'reason': 'structural_identity_change',
 'action_type': 'name_change', 'process_date': '2022-01-31',
 'old_symbol': 'META', 'new_symbol': 'METV',
 'old_cusip': '53656F417', 'new_cusip': '53656F417'}
```

O método legado `_generic_identity_issue` classificou uma transição de ticker de **outro emissor** como quebra da identidade de ações de Meta Platforms.

## Evidências no próprio snapshot congelado

No CSV versionado `dados/u67/corporate_actions/META.csv` (não modificado) coexistem:
- **31/01/2022:** `META -> METV`, CUSIP `53656F417` em ambos os lados — **Roundhill Ball Metaverse ETF**, e não Meta Platforms.
- **09/06/2022:** `FB -> META`, CUSIP `30303M102` em ambos os lados — **Meta Platforms Inc.**
- Os dividendos contemporâneos também estão associados a `30303M102`.

A série `dados/u67/raw_bars/META.csv` está presente desde 04/01/2016, encerrando 31/01/2022 a US$ 313,26, coerente com FB/Meta Platforms. A confusão de metadados resulta da **reutilização do ticker**, não de uma fusão legítima entre ETF e ação.

Fontes independentes do mercado:
- OCC, memorando #49960: https://infomemo.theocc.com/infomemos?number=49960
- Site do ETF: https://www.roundhillinvestments.com/etf/metv/
- Nasdaq, ECA2022-125: https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-125
- Comunicado da Meta Platforms: https://investor.atmeta.com/investor-news/press-release-details/2022/Meta-Platforms-Inc.-to-Change-Ticker-Symbol-to-META-on-June-9/default.aspx

## Tratamento realizado

- Função pura `_verified_unrelated_meta_etf_action` valida somente esse evento com símbolo, data, CUSIPs e presença do evento verdadeiro `FB -> META` nos **mesmos registros de fonte**.
- Caso validado, o CSV de preços **permanece intacto**, META **permanece entre os 75 ativos** e o conflito de metadados é registrado explicitamente em `identity_audit.json` como `unrelated_prior_issuer_ticker_reuse`.
- Qualquer outra indicação de ruptura estrutural **continua bloqueando a execução**. Não foi instituído allowlist genérico nem reconstrução de histórico, troca de ticker, limpeza de CSV, download ou tratamento especial para retornos.
- Preservados o contrato de SHA-256 do snapshot e dos dez arquivos adicionais, exatamente 75 ativos treinados/calibrados conjuntamente por fold.
- **Corrigida a interpretação do checkpoint legado**: US$ 587.165.941,13 é resultado oficial de **motor histórico 65+10**, não uma meta imposta ao novo motor fixed75. Na v1.25.9, a diferença com a referência é **reportada**, não força parada nem tentativa de ajustar dados. Os cálculos da versão v1.23.0 permanecem preservados em seu próprio executor.
- Nova saída esperada: `output/mitigacao_overfitting_v1259/<run_id>/pacote_mitigacao_overfitting_v1259.zip`.

## Testes

`tests/test_mitigacao_fixed75_v1258.py` estendido para conferir o evento real dos dois CUSIPs no CSV congelado, ancorar o fechamento FB em 31/01/2022, rejeitar eventos de outros emissores/datas/CUSIPs, preservar 75 ativos unificados, registrar auditoria, e impedir a antiga exigência de paridade monetária com arquitetura diferente.

**Estado da validação:** código publicado; ainda não rodado contra o cache dos dez ativos adicionais disponível apenas no Spyder do autor. Testes unitários e inspeções de código não equivalem a replay financeiro de 75 ativos. Não afirmar ausência de todo risco de identidade em demais ativos sem auditoria separada.

## Procedimento

```powershell
git status
git fetch origin
git switch research/v1.23.1-fresh-snapshot-20261022
git pull --ff-only origin research/v1.23.1-fresh-snapshot-20261022
python -m pytest tests/test_mitigacao_fixed75_v1258.py -v
```

No Spyder, reiniciar kernel e F5 em `pesquisar_mitigacao_overfitting_spyder.py`. Se surgir outra inconsistência, **não mudar os CSVs nem excluir ativos automaticamente**; investigar qual emissor a ação corporativa descreve e se a série é contínua antes de qualquer decisão.