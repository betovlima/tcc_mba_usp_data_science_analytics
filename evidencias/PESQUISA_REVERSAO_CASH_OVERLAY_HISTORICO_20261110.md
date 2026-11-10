# Reversão para CASH | triagem histórica da Winner | 10/11/2026

**Status:** resultado exploratório reproduzido, **sem aprovação para operar**.
**Escopo:** backtest histórico com **77 ativos configurados e 75 séries elegíveis**. DOC e CLMT foram excluídos na execução histórica por mudanças de identidade societária. Este resultado não comprova uma política para os **77 ativos completos**; o estudo principal deve exigir o conjunto integral e validação causal.

## Proveniência

- Pacote: `market_cycle_trader_20261110T140411-8f086044.zip`.
- SHA-256: `1cdeda67bafccf424a75b675b5f16084b14c282ec1ba1b9599c6e234312dc0f2`.
- Período: 22/07/2020 a 09/11/2026, 1.584 sessões.
- Capital inicial: US$ 10.000; capital final do controle: US$ 122.119.681,69.
- Drawdown máximo histórico do controle: −41,68%; dias de CASH: 0.
- Validação cronológica usada nesta rodada: fold 1 (504 sessões) treino, fold 2 (504) seleção de limiar, fold 3 (576) avaliação.
- Alvo: perda da carteira Control de pelo menos 3% no pregão subsequente.
- Custo adicional ilustrativo do CASH: 0,10% por venda e recompra completas, aplicado como meia parcela por mudança de exposição.

O fold 3 foi reservado contra ajuste **nesta rodada**. Como seu histórico já foi analisado em pesquisas anteriores, não se deve apresentá-lo como holdout inteiramente virgem para todas as hipóteses futuras.

## Resultado na avaliação do fold 3

| Modelo | AUC | Dias CASH | Quedas de pelo menos 3% evitadas | Variação do capital final vs. controle | MaxDD |
|---|---:|---:|---:|---:|---:|
| Regressão logística, mercado | 0,478 | 7 | 1 de 73 | +4,74% | −41,68% |
| LightGBM, mercado | 0,497 | 0 | 0 de 73 | 0,00% | −41,68% |
| Regressão logística, posição | 0,550 | 1 | 0 de 73 | +0,15% | −41,68% |
| LightGBM, posição | 0,521 | 0 | 0 de 73 | 0,00% | −41,68% |

O ganho de +4,74% resulta de **shadow replay aproximado**, não de execuções verificadas. Seu intervalo exploratório por bootstrap em blocos de cinco sessões foi de **−8,93% a +27,27%**. Nenhuma variante reduziu a maior perda acumulada desde o topo.

### Episódio de CAPR

Na sessão de **30/07/2026**, o retorno diário do controle foi **−36,2253%**, com CAPR selecionado e `raw_best_score=0,60111`. Os quatro testes sinalizaram **permanência exposta**, ou seja, **não identificaram** esse evento como ocasião para CASH.

## Restrições e próximos testes

1. O shadow replay substitui o retorno do controle por zero nos dias de CASH e aplica custos aproximados. **Não** recria carteira alternativa, preços de execução, reentrada no novo ativo, slippage e restrições de liquidez.
2. Os modelos com atributos de posição usam o estado da **carteira controle**, que mudaria após uma saída real; seus números servem apenas para triagem de veto pontual.
3. A auditoria do manifesto aponta `split_normalization_uses_future_events=true`; é necessário verificar a causalidade do ajuste antes de afirmar ausência de informação futura.
4. O estudo de 77 ativos exige OHLCV e eventos corporativos congelados para todo o universo, com tratamento explícito de DOC/CLMT. A referência com 75 séries não substitui esse experimento.
5. Priorizar modelos probabilísticos **por ativo e estado da posição** que comparem manter, rotacionar e CASH, com validação temporal e execução factível.
6. **Não alterar o MCT Live** e não promover regras para `main` com base nestes resultados.

## Reprodução no Spyder: arquivo e pacote únicos

Abra **`pesquisar_protecao_reversao_spyder.py`** na raiz do repositório e execute com **F5**. Este é o **único script principal** desta pesquisa. A triagem de quatro modelos de CASH foi integrada como função interna desse mesmo arquivo, preservando as análises anteriores de CAPR, risco de reversão e CASH/posição.

A fonte continua sendo o ZIP de produção original com SHA-256 informado acima. O próprio script localiza, valida e reutiliza o arquivo congelado; não é necessário passar argumentos. O teste histórico sobre a carteira gravada usa 77 símbolos configurados e 75 séries elegíveis, **sem** apresentá-los como estudo completo com 77 séries.

As saídas ficam em **`out/pacote_protecao_reversao.zip`**, junto dos demais resultados da pesquisa. Arquivos próprios desta rodada têm o prefixo `overlay_cash_`:

- `overlay_cash_resultado_pesquisa.json`
- `overlay_cash_relatorio.md`
- `overlay_cash_sessoes_teste_fold3.csv`
- `overlay_cash_calibracao_fold2.csv`
- `overlay_cash_capr_fold3.csv`
- `overlay_cash_01_patrimonio_holdout.png`
- `overlay_cash_02_drawdown_holdout.png`

Não existe mais um segundo script `pesquisar_reversao_cash_overlay_historico_u75_spyder.py` nesta branch. O código foi incorporado ao original. Os demais resultados históricos continuam no ZIP principal.

**Sem mudanças em `main` ou no MCT Live.** Não versionar auditorias da conta Live nem fontes brutas privadas; os arquivos de saída permanecem locais.
