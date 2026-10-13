# TCC MBA USP — Rotação de Capital com Machine Learning

**Estrutura consolidada de pesquisa:** v1.22.2-dev.1 (13/10/2026).  
**Control oficial reproduzível:** v1.22.1, congelado, sem alterações de algoritmo.  
**Pontos de entrada do projeto:** apenas `buscar_ativos.py` e `reproduzir_experimento.py`.

## Instalação e execução no Spyder

Requer Python 3.12 e `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

**1. Reproduzir o experimento oficial** (não requer rede, conta externa nem credenciais):

```python
%run reproduzir_experimento.py
```

O Control usa `dados/u67/`, com barras CSV e ações corporativas
congeladas no Git e hashes validados por `manifest.json`. São 67
ativos requisitados e 65 elegíveis após as exclusões estruturais
**CLMT** e **DOC**; esses ativos não são remendados ou reconstruídos.
O checkpoint oficial deve reproduzir capital simulado
**US$ 76.927.051,38897176**, com término em 06/10/2026,
e checkpoint intermediário já documentado. O script aborta se
os resultados divergirem do valor de referência.

**2. Buscar ativos** (pesquisa separada; pode consultar a Alpaca):

```python
%run buscar_ativos.py
```

A busca parte do universo U67, obtém candidatos por catálogo e
histórico, aplica auditoria de continuidade, resume os mesmos
52 atributos técnicos e faz classificação matemática/geométrica
de semelhança ou complementaridade (PCA, clusters, distância).
O escalonamento e a PCA são ajustados **somente na referência U67**;
candidatos são projetados nessa referência. O método NÃO utiliza
o capital final do backtest para selecionar candidatos e NÃO
insere automaticamente nenhum ativo na reprodução oficial.
Gera shortlist, CSVs e gráficos em
`output/candidate_cluster_search/`; emite sinal sonoro distinto para
conclusão ou falha; utiliza cache local em
`output/`, ignorado pelo Git. Requer credenciais Alpaca no
`.env` local:

```text
ALPACA_API_KEY=sua_api_key
ALPACA_SECRET_KEY=sua_secret_key
```

## Contrato científico preservado

- **Swing trade:** mínimo obrigatório de **dois pregões** antes
  de girar a posição. Isso não significa vender no terceiro dia:
  a posição continua se não houver vantagem suficiente.
- **Modelo:** LightGBM de utilidade ajustada ao risco, com
  horizonte temporal, risco de queda, drawdown, persistência
  e movimento incorporados ao objetivo.
- **Decisões:** prever com informação observável até o fechamento
  da sessão e executar mudanças na abertura seguinte; filtro de
  elegibilidade sem informação da sessão seguinte.
- **Validação:** folds walk-forward, purge temporal, margem de
  troca calibrada somente no histórico pré-OOS, CPU e parâmetros
  congelados para reprodução.
- **Comparador:** compra e manutenção de pesos iguais exatamente
  nos mesmos 65 ativos elegíveis e com a mesma janela/custos.
- **Custos e resultados:** comissões, taxas, slippage, curva de
  capital, drawdown, Sharpe, trades, planilhas e pacote ZIP.
  Sinal sonoro ao concluir; os resultados são de backtest,
  não promessas de lucratividade real.

## Dados, módulos e histórico

A raiz possui **dois scripts executáveis**. As pastas
`engine/` e `reproducao/` contêm implementações de apoio;
`dados/u67/` contém os snapshots imutáveis;
`tests/` mantém verificações automatizadas; `evidencias/`
e `figuras/` preservam os materiais verificáveis do TCC.
`MANIFEST.md` é o registro de decisões, experimentos e
limitações. Os executáveis exploratórios substituídos pela
estrutura limpa permanecem recuperáveis pelo histórico Git.

**Separação de campanhas:** o checkpoint experimental
**U67+10 / 75 ativos / US$ 587.165.941,13** foi estudado nas
campanhas v1.0.9–v1.0.11, mas não substitui o Control oficial
U67 neste repositório. Parte das dez séries extras estava em
cache local, fora do snapshot Git; por isso não apresentamos
esse checkpoint como reprodução oficial portátil. As técnicas
Borda, TOPSIS, Pareto e as ablações sobre hold=0 são resultados
exploratórios, **não políticas aprovadas** para o runner oficial.

A literatura acadêmica específica que fundamenta a escolha
original dos dois pregões ainda precisa ser citada explicitamente
no texto do TCC; não se afirma ótimo universal.

**Nota de segurança:** reconstruir o snapshot em vez de
reproduzi-lo requer ação explícita e separada,
`python -m reproducao.preparar_snapshot`; isso substitui o
diretório congelado local e NÃO deve ser feito durante a
reprodução oficial.

## Validação de código

```bash
python -m pytest -q
python -m ruff check engine reproducao buscar_ativos.py reproduzir_experimento.py tests --select F401,F811,F821,F841
```

O TCC mantém apenas o Control aprovado em produção científica.
Experimentações futuras só devem modificar essa configuração
mediante nova validação e checkpoint versionado.
