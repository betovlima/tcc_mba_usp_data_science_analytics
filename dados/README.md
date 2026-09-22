# Dados da pesquisa

O projeto separa o snapshot oficial do TCC dos downloads usados em execucoes
com dados novamente consultados na Alpaca.

```text
dados/
├── pesquisa_v1/                 # versionado no Git
│   ├── raw_bars/
│   ├── corporate_actions/
│   └── manifest.json
└── temporario/reproducao_v1/    # ignorado pelo Git
    ├── raw_bars/
    ├── corporate_actions/
    └── manifest.json
```

`pesquisa_v1/` e a evidencia congelada da pesquisa e deve permanecer
versionada.

`temporario/` e recriado para novas consultas a Alpaca e nao deve ser enviado
ao Git.

A normalizacao de splits continua sendo calculada em memoria.
