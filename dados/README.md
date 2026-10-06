# Dados da pesquisa

A reprodução oficial do TCC usa snapshots versionados e não baixa dados novos.

## U59 reproduzível

O universo oficial que reproduz aproximadamente US$ 30 milhões é formado por:

- 56 ativos-base em pesquisa/;
- COLB, AMS e FOXF em pesquisa_expansao_76_b2/.

Estrutura principal:

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
├── pesquisa_smart_candidates/
└── assinatura_matematica/
~~~

A normalização de splits é calculada em memória durante a preparação.

O arquivo reproduzir_experimento.py usa apenas pesquisa/ e
pesquisa_expansao_76_b2/. Os demais diretórios preservam evidências das
campanhas de pesquisa anteriores e não são necessários para reproduzir o U59.

## Dados temporários

O diretório temporario/ continua ignorado pelo Git e pode ser usado por
rotinas auxiliares ou por novas buscas. Ele não participa da reprodução
oficial do checkpoint U59.

Os snapshots congelados não devem ser substituídos por downloads atuais quando
o objetivo for reproduzir o resultado oficial.
