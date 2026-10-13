# TCC MBA USP — Reprodução do Control U67+10

**Versão de consolidação:** v1.22.3-dev.1  
**Política de referência:** Control com 75 ativos, LightGBM de utilidade
ajustada ao risco e swing trade com permanência mínima de dois pregões.

## Apenas dois programas de entrada no Spyder

### 1. `reproduzir_experimento.py` — Control de US$ 587 milhões

```python
%run reproduzir_experimento.py
```

O script chama `reproducao/control_u77.py`, que utiliza o fluxo
de treinamento e composição de previsões da campanha U67+10
encerrada na v1.0.11. O experimento histórico registrou
**US$ 587.165.941,13**, com 75 ativos, três folds OOS e
margens calibradas historicamente. Esse valor é referência
**descritiva**, não é validação obrigatória nem limite de resultado.
Se o capital mudar, o programa informa a diferença em vez de
abortar pela não coincidência.

**Dados necessários para a mesma execução:**

- `dados/u67/raw_bars/` e
  `dados/u67/corporate_actions/` — universo U67 versionado;
- `dados/.u67_expansion_cache/full_raw_20261006/raw_bars/` e
  `dados/.u67_expansion_cache/full_raw_20261006/corporate_actions/`
  — dez ativos adicionais `CRT, NBR, CAPR, PNBK, PENN, JBL,
  STAG, MCHB, LAD, TOWN` usados na campanha original.

**Importante:** os dez históricos adicionais não constam no
snapshot U67 do Git. Na mesma máquina onde a campanha v1.0.11
foi executada, o cache normalmente já existe. Se faltar,
o script informa os arquivos ausentes; não finge reproduzir
os US$ 587 milhões com outro universo. Esses arquivos de
mercado são necessários ao cálculo, **não constituem uma
restrição arbitrária**.

A execução reproduz LightGBM individual por ativo,
compõe previsões U67+10, aplica decisões com dois pregões
mínimos, calibra a margem no histórico anterior a cada
fold e gera operações, curva financeira, métricas, gráficos,
planilhas e `output/reproducao/pacote_reproducao_u77.zip`.

**Não há SHA-256, controle de hashes ou bloqueio por
checkpoint, contagem antiga de ativos, branch ou metadados
de versões passadas.** Os CSVs locais são lidos normalmente.
São mantidas verificações indispensáveis de dados presentes,
preços válidos e exclusões de séries com problemas estruturais;
não se tenta reconstruir/mesclar identidades de ativos.

### 2. `buscar_ativos.py` — busca independente

```python
%run buscar_ativos.py
```

Faz triagem técnica e busca semelhança/complementaridade
com PCA, agrupamento e geometria no universo de referência.
Pode acessar Alpaca; os downloads de candidatos ficam no
cache local `output/candidate_cluster_search/cache_bars/`.
A shortlist não promove ativos automaticamente nem modifica
o universo do Control. Possui som diferente para sucesso e
erro.

Para consultas à Alpaca, configure localmente um arquivo
`.env` com `ALPACA_API_KEY` e `ALPACA_SECRET_KEY`.

## Regras aprovadas preservadas

- Swing trade com **mínimo obrigatório de dois pregões**,
  podendo manter a posição por mais tempo;
- previsão LightGBM de utilidade ajustada ao risco, com
  atributos conhecidos na data da decisão;
- execução cronológica, folds walk-forward e histórico
  de calibração separado do OOS;
- exclusão explícita de ativos com problemas estruturais
  de histórico, sem bridges ou correções manuais;
- comparação financeira com compra/manutenção no mesmo
  universo, taxas e slippage;
- dados em CSV e resultados exportados; nenhum banco de dados;
- aviso sonoro ao fim do backtest e da busca.

O resultado anterior de **US$ 76,93 milhões** corresponde à
reprodução histórica de apenas 65 ativos e não é o Control
principal desta versão. Os oito rankings e as ablações
v1.0.9–v1.0.11 foram experiências diagnósticas, sem
promoção à estratégia atual.

## Estrutura interna e testes

As pastas `engine/` e `reproducao/` contêm as funções
internas; `dados/` contém séries de mercado;
`MANIFEST.md` guarda a cronologia das decisões.
Só `buscar_ativos.py` e `reproduzir_experimento.py`
são arquivos de execução na raiz.

```bash
python -m pytest -q
python -m ruff check engine reproducao buscar_ativos.py reproduzir_experimento.py tests --select F401,F811,F821,F841
```

O CI verifica sintaxe, contratos e testes unitários.
**Não executa** os backtests completos com os dez
históricos adicionais, ausentes do ambiente CI.
Confirmação do capital de US$ 587 milhões exige execução
local no Spyder com o mesmo cache de dados.
