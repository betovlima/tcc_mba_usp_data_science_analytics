# Dados da pesquisa

O projeto separa o snapshot oficial do TCC dos downloads usados em execucoes
com dados novamente consultados na Alpaca.

```text
dados/
├── pesquisa/                 # versionado no Git
│   ├── raw_bars/
│   ├── corporate_actions/
│   └── manifest.json
└── temporario/reproducao/    # ignorado pelo Git
    ├── raw_bars/
    ├── corporate_actions/
    └── manifest.json
```

`pesquisa/` e a evidencia congelada da pesquisa e deve permanecer
versionada.

`temporario/` e recriado para novas consultas a Alpaca e nao deve ser enviado
ao Git.

A normalizacao de splits continua sendo calculada em memoria.

## Modos de execucao

`USAR_DADOS_PESQUISA_CONGELADOS=True` usa somente `pesquisa/` e nunca
baixa ou altera o snapshot.

Com `USAR_DADOS_PESQUISA_CONGELADOS=False`, toda leitura e escrita ocorre em
`temporario/reproducao/`. Se `FORCAR_DOWNLOAD=True`, essa pasta temporaria
e limpa antes do download completo. O snapshot `pesquisa/` permanece
intocado.
