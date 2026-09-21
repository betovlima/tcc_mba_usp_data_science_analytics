# Dados da reproducao

Esta pasta existe apenas para documentar a estrutura local usada pelo experimento.

Nenhum dado de mercado e versionado no Git.

Na primeira execucao de `reproduzir_experimento_spyder.py`, o projeto cria:

```text
dados/
└── reproducao_v1/
    ├── raw_bars/
    │   └── <ATIVO>.csv
    ├── corporate_actions/
    │   └── <ATIVO>.csv
    ├── normalized_bars/
    │   └── <ATIVO>.csv
    └── manifest.json
```

- `raw_bars/`: barras OHLCV RAW/SIP baixadas da Alpaca, um CSV por ativo.
- `corporate_actions/`: eventos corporativos, um CSV por ativo.
- `normalized_bars/`: OHLCV apos normalizacao local de splits.
- `manifest.json`: identidade do snapshot e hashes SHA-256 dos arquivos.

Os arquivos gerados permanecem locais e sao ignorados pelo Git.
