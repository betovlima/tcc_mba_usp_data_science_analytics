# Snapshot oficial da pesquisa v1

Este diretorio contem o snapshot congelado utilizado pelo TCC e deve ser
versionado no Git.

Conteudo esperado:

```text
raw_bars/<ATIVO>.csv
corporate_actions/<ATIVO>.csv
manifest.json
```

Os arquivos RAW usam Alpaca SIP, timeframe diario e adjustment RAW. O
`manifest.json` registra hashes SHA-256 para permitir comprovar que os dados
usados na reproducao sao exatamente os mesmos da pesquisa.
