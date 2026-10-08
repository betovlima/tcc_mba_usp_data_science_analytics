# CONTEXTO_MESTRE — TCC MBA USP Data Science & Analytics

## Estado atual

O projeto contém somente o fluxo necessário para preparar, reproduzir, auditar
e documentar o experimento final do TCC.

Versão atual: `1.22.0-dev.7`

Branch ativa de refatoração:

```text
refactor/tcc-final-cleanup-v1
```

A `main` não deve ser alterada até a conclusão da validação da branch.

## Objetivo científico

Avaliar uma estratégia de rotação de capital entre ativos financeiros usando
LightGBM, validação temporal walk-forward, calibração cronológica e replay
financeiro reproduzível.

A reprodução final deve ser independente de banco de dados e de qualquer outra
aplicação externa. Todos os parâmetros, dados, regras e checkpoints necessários
devem estar definidos dentro deste repositório.

## Universo U67

O universo de entrada contém 67 ativos.

O pipeline aplica duas exclusões estruturais documentadas:

```text
CLMT
DOC
```

Assim, o conjunto efetivo entregue ao modelo contém 65 ativos.

O contrato do universo está centralizado em:

```text
engine/configuracao.py
```

## Snapshot oficial

A reprodução usa um único snapshot congelado:

```text
dados/u67/
├── raw_bars/
├── corporate_actions/
└── manifest.json
```

O snapshot contém os 67 ativos de entrada.

A preparação dos dados é separada da reprodução:

```text
preparar_snapshot_u67.py
```

Esse script pode acessar a Alpaca para criar ou substituir o snapshot. A coleta
é feita primeiro em:

```text
dados/.u67_build/
```

Somente depois da validação o snapshot é publicado em `dados/u67/`.

A reprodução oficial:

```text
reproduzir_experimento.py
```

não acessa a Alpaca e trabalha apenas com o snapshot congelado.

## Contrato dos dados

- fonte: Alpaca;
- feed: SIP;
- timeframe: 1Day;
- adjustment: RAW;
- início histórico: 2016-01-01;
- cutoff oficial: 2026-10-06;
- Corporate Actions preservados separadamente;
- splits normalizados em memória;
- CSVs lidos com `float_precision="round_trip"`;
- manifest e hashes usados somente para integridade e auditoria.

## Modelo final

Modelo:

```text
LightGBM Control
```

A configuração científica fica em `engine/configuracao.py`.

A validação é temporal e usa folds walk-forward, janela de calibração, purge
temporal e avaliação fora da amostra. O replay financeiro usa posição única,
custos e regras de rotação congelados no código.

## Checkpoints de regressão

Os checkpoints atuais do próprio TCC são:

```text
analysis_end = 2026-10-06
capital final esperado = US$ 76,927,051.38897176
checkpoint 2026-09-17 = US$ 78,782,538.31270888
```

Tolerância para a conferência monetária:

```text
US$ 0.01
```

Esses valores existem para detectar regressões após refatorações. Eles não são
entrada do modelo nem alvo de treinamento.

## Métricas da reprodução validada

Na execução que estabeleceu os checkpoints atuais:

```text
CAGR = 322.775261%
Sharpe = 2.56904268
MaxDD = -30.358970%
Worst fold = 282.589554%
```

O capital inicial é de US$ 10.000.

Os resultados são históricos e não constituem previsão ou garantia de retorno.

## Estrutura atual

```text
.
├── preparar_snapshot_u67.py
├── reproduzir_experimento.py
├── engine/
│   ├── configuracao.py
│   ├── diagnosticos.py
│   ├── execucao.py
│   ├── modelo_lightgbm.py
│   └── rotacao.py
├── reproducao/
│   ├── artefatos.py
│   ├── dados.py
│   ├── experimento.py
│   ├── graficos.py
│   └── preparacao.py
├── dados/
│   └── u67/
├── tests/
├── requirements.txt
├── README.md
└── CONTEXTO_MESTRE.md
```

## Refatorações concluídas

Foram removidos do runtime atual:

- pasta `pesquisas/`;
- pesquisa Directional Change;
- runner antigo de busca de ativos;
- caminhos genéricos de estratégias descartadas;
- configuração de modelos não utilizados;
- arquivos de diagnóstico históricos;
- layouts de dados antigos;
- dependências de snapshots de referência removidos;
- código auxiliar sem uso no experimento final.

O histórico dessas etapas permanece disponível no Git, mas não faz parte da
árvore operacional atual do TCC.

## Artefatos oficiais

A reprodução gera em `output/reproducao/`:

```text
u67_requested_assets.csv
u67_effective_assets.csv
u67_runtime_exclusions.csv
u67_fold_margins.csv
u67_fold_calibration_candidates.csv
u67_market_data_hashes.csv
u67_predictions.csv
u67_trades.csv
u67_runtime_environment.json
reproducao_u67.json
graficos/
pacote_reproducao_u67.zip
```

## Regras de manutenção

1. A reprodução oficial não deve depender de banco de dados, serviços externos
   ou arquivos históricos fora do snapshot congelado.
2. Mudanças de organização não devem alterar parâmetros científicos ou
   resultados sem uma nova decisão metodológica explícita.
3. Problemas estruturais de identidade de ativos devem resultar em exclusão
   explícita e reproduzível, sem reconstrução manual de séries.
4. Toda alteração concreta deve ser versionada em branch, testada e registrada
   neste arquivo.
5. Antes de merge na `main`, Ruff e pytest devem passar.
6. A árvore final deve conter somente código e documentação necessários ao TCC.

## 2026-10-08 — limpeza final de referências externas

A reprodução foi tornada autônoma. Nomes, comparações, identificadores,
artefatos e documentação ligados a sistemas externos foram removidos do projeto.

O resultado passou a ser validado apenas contra checkpoints congelados do
próprio TCC.

Versão: `1.22.0-dev.7`.
