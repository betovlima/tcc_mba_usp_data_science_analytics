# TCC MBA USP — Rotação de capital com Machine Learning

**Versão de refatoração:** `v1.22.4-dev.1`

O estudo utiliza um universo de ativos **declarado pelos dados e pela seleção versionada**, sem quantidades codificadas nos métodos. Mantém o Control LightGBM, validação cronológica walk-forward, calibração por fold, purge temporal de 60 pregões e swing trade com permanência mínima de dois pregões.

## Executáveis na raiz

| Programa | Responsabilidade |
| --- | --- |
| `reproduzir_experimento.py` | Reproduzir a política Control sobre o universo declarado |
| `buscar_ativos.py` | Buscar candidatos por atributos técnicos, PCA e agrupamentos, sem promover resultados automaticamente |

Os demais códigos científicos ficam em `engine/` e `reproducao/`.

## Composição do universo

- `dados/referencia/manifest.json`: fornece a lista de ativos do snapshot de referência; os CSVs permanecem idênticos aos históricos, sem modificação de conteúdo.
- `dados/selecao_ativos.json`: registra a seleção adicional da campanha que gerou a referência histórica, sem contagem fixa no executor.
- Dados adicionais: histórico local da campanha anterior com `raw_bars/` e `corporate_actions/`. O programa procura um conjunto completo em `dados/` ou aceita o diretório indicado em `TCC_CANDIDATE_DATA_ROOT`.

A seleção adicional histórica e o valor de aproximadamente **US$ 587,17 milhões** são preservados como **referência descritiva**, e não como condição de sucesso. Mudanças na seleção exigem uma nova avaliação cronológica; não promovem automaticamente o resultado antigo ao universo novo.

Não existem bloqueios por hash, versões de branch ou checkpoints financeiros para iniciar o cálculo. Permanecem verificações de arquivos presentes, símbolos duplicados, preços utilizáveis, continuidade das séries e exclusões estruturais documentadas. Não são reconstruídas identidades de ativos nem mescladas fontes para corrigir problemas estruturais.

## Execução

Instale as dependências Python 3.12 com:

```bash
python -m pip install -r requirements.txt
```

Depois execute no Spyder com F5 ou no terminal:

```bash
python reproduzir_experimento.py
python buscar_ativos.py
```

A busca pode consultar a Alpaca, mediante credenciais locais `.env`; a reprodução financeira utiliza CSVs locais e não acessa banco de dados nem envia ordens.

## Validação temporal e evidências

A campanha em preparação tem **data de corte de 20/10/2026**. O arquivo de referência ainda precisa ser coletado e versionado para essa data, assim como o histórico dos candidatos adicionais. Os dados existentes com corte anterior continuam sendo evidência histórica e não devem ser renomeados ou tratados como dados de 20/10. A validação walk-forward, a calibração por fold e o purge de 60 pregões permanecem no método. A nova campanha ainda não foi executada ou validada.

Os resultados, versões antigas, arquivos de evidências, figuras e decisões científicas permanecem documentados no `MANIFEST.md`. Identificadores com quantidades de ativos em seções históricas são registros de procedência, não regras da implementação atual.

## Testes

```bash
python -m pytest -q
python -m ruff check engine reproducao buscar_ativos.py reproduzir_experimento.py tests --select F401,F811,F821,F841
```

O CI verifica código e contratos, mas não reproduz a campanha financeira completa quando o cache histórico adicional não está disponível no runner.
