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
    └── manifest.json
```

- `raw_bars/`: barras OHLCV RAW/SIP baixadas da Alpaca, um CSV por ativo.
- `corporate_actions/`: eventos corporativos, um CSV por ativo.
- `manifest.json`: identidade do snapshot e hashes SHA-256 dos arquivos.

Os arquivos gerados permanecem locais e sao ignorados pelo Git.


A normalizacao de splits e calculada em memoria durante a reproducao e nao e
persistida em uma segunda copia dos dados.
