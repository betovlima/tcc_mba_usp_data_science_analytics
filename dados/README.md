# Dados do TCC

A reprodução oficial usa **um único snapshot U67 congelado e versionado**.

Estrutura esperada:

```text
dados/
└── u67/
    ├── raw_bars/
    │   └── <ATIVO>.csv
    ├── corporate_actions/
    │   └── <ATIVO>.csv
    └── manifest.json
```

O snapshot contém os 67 ativos solicitados pelo contrato U67. O pipeline recebe
os 67 e aplica, em tempo de reprodução, as exclusões estruturais previstas para
CLMT e DOC. O conjunto efetivo entregue ao modelo deve permanecer com 65 ativos.

Os CSVs de barras usam Alpaca SIP, timeframe diário e adjustment RAW. Corporate
Actions são preservados separadamente e a normalização de splits ocorre em
memória durante a preparação.

## Criar ou substituir o snapshot

A coleta de dados foi separada da reprodução científica. Para criar o snapshot
U67, execute:

```bash
python preparar_snapshot_u67.py
```

Esse script baixa os dados para uma área temporária
`dados/.u67_build/`, valida o snapshot completo e somente então publica o
resultado em `dados/u67/`.

Depois da coleta, revise e versione no Git:

```bash
git add dados/u67
git commit -m "data: freeze official U67 snapshot"
```

A reprodução oficial é executada por `reproduzir_experimento.py` e **não
acessa a Alpaca**. Ela apenas valida e lê `dados/u67/`.

O `manifest.json` registra o universo, datas, contagens e hashes SHA-256 dos
arquivos. Esses hashes existem para verificar integridade e não participam do
treinamento do modelo.
