# Dados da pesquisa

A reprodução oficial do TCC usa snapshots congelados e não baixa dados novos.

## U67 oficial

O universo oficial que reproduz aproximadamente **US$ 58,56 milhões** é
formado por:

- 56 ativos-base em `pesquisa/`;
- COLB, AMS e FOXF em `pesquisa_expansao_76_b2/`;
- THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC em
  `pesquisa_smart_candidates/`.

Estrutura necessária no ambiente de reprodução:

~~~text
dados/
├── pesquisa/
│   ├── raw_bars/
│   ├── corporate_actions/
│   └── manifest.json
├── pesquisa_expansao_76_b2/
│   ├── raw_bars/
│   ├── corporate_actions/
│   └── manifest.json
└── pesquisa_smart_candidates/
    ├── raw_bars/
    ├── corporate_actions/
    └── manifest.json
~~~

A normalização de splits é calculada em memória durante a preparação.

O repositório mantém versionado o snapshot-base em `dados/pesquisa/`. Os
snapshots históricos B2 e SMART utilizados nas campanhas de expansão precisam
estar presentes localmente para executar o checkpoint U67. O runner oficial
valida os respectivos manifestos e hashes antes do backtest e não faz download
para substituir esses dados.

Os arquivos de trabalho das pesquisas de assinatura encerradas foram removidos
da árvore atual. O histórico dos experimentos continua preservado pelo Git e
pelo `CONTEXTO_MESTRE.md`.

## Dados temporários

O diretório `temporario/` continua ignorado pelo Git e pode ser usado por
rotinas auxiliares ou novas buscas. Ele não participa da reprodução oficial.

Snapshots congelados não devem ser substituídos por downloads atuais quando o
objetivo for reproduzir o checkpoint U67.
