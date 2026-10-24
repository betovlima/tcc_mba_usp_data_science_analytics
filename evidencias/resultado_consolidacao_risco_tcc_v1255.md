# Consolidação de generalização e risco de seleção — TCC v1.25.5

**Data:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`, sem alterações na `main`.
**Executor:** `consolidar_evidencias_overfitting_spyder.py`.
**Método:** análise *somente leitura* dos ZIPs históricos do mesmo snapshot; não consulta Alpaca, não treina nem calibra LightGBM, não executa backtest financeiro e não altera séries.

## Contrato de entrada e evidências já verificadas

- `pacote_generalizacao_587m_v1254.zip`: SHA-256 `0b455cbdfc5dc4c83818d30ba8e868634d91e84e5ac5ab49b869c505bcc9a060`, 225 combinações ativo/fold do vintage histórico.
- `pacote_reproducao_universo(10).zip`: SHA-256 `b62982a462fe0c8b8c1ea1a3c9df5ea4d719716956d4b6680f361847eaa1685d`; 1.560 sessões, 354 rotações ativo→ativo com retorno adicional local calculável e checkpoint exato de US$ 587.165.941,13.
- Snapshot SHA `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`, que não deve ser substituído por vintage novo.
- PBO log-growth 53,57% e PBO Sharpe 59,52% vieram da avaliação `decision_validation` v1.4.1, `MANIFEST.md` §25.8; **não foram estimados novamente**.
- Capital da mesma política com dados renovados US$ 89.941.496,49 é evidência do contrafactual de maio/2018, documentado em `evidencias/auditoria_contrafactual_2018_v1237.md`. **Não confundir** com o resultado econômico RAW v1.25.0 (outro processamento).

## Sete figuras, significado e limites

1. **Treino x OOS dos 225 LightGBMs:** cada ponto equivale a um ativo/fold. Acima da diagonal está a pior generalização preditiva, não perda financeira.
2. **Distribuição de gap por fold:** `100 × (RMSE_OOS / RMSE_treino − 1)`, por ativo, dividida nos três folds. Gap não é probabilidade de overfitting. Mediana geral verificada +161,32%.
3. **Skill OOS vs média do treino:** quantidade de modelos cuja previsão supera o comparador constante que prevê **a média do alvo de treino**, não um baseline zero. Apenas 29/225 com skill positivo.
4. **Retorno local da troca vs manter posição anterior:** distribuição de `rotation_value_added` para 354 rotações. **222/354** tiveram diferença positiva, mediana de ~+2,19 pontos percentuais; diferença de retornos brutos no horizonte seguinte, **não** replay contrafactual da carteira toda, nem demonstração de robustez após seleção.
5. **Curva de patrimônio do replay vs buy-and-hold:** OOS interno em folds, eixo logarítmico. O universo e as regras foram selecionados e analisados retrospectivamente, logo o gráfico NÃO é teste prospectivo independente.
6. **PBO da seleção retrospectiva:** log-growth 53,57%, Sharpe 59,52%, 180 subconjuntos, CSCV 252 partições. Reutiliza a métrica da pesquisa anterior: representa risco do processo de escolher um vencedor, não taxa de erros nas ordens.
7. **Capital por versão de histórico:** checkpoint congelado US$ 587,17 milhões vs atualizado US$ 89,94 milhões, mantendo motor comparável; 73 barras de maio/2018 explicam a divergência entre esses vintages. Não implica revisão de transações financeiras já executadas.

**Limitação metodológica essencial:** as janelas OOS internas do treinamento não são economicamente independentes do *processo de pesquisa completo* quando universo e hiperparâmetros foram ajustados após olhar o histórico. O script **não fabrica** um novo período intocado de 2020 a 2026. Para evidência prospectiva autêntica, registrar previamente uma estratégia congelada, custos, benchmark, decisões diárias e critérios de teste, antes de observar novas sessões. Isso exige observações que ainda não participaram do desenvolvimento. As figuras atuais documentam a pesquisa já concluída.

## Execução e estado

No Spyder, abrir `consolidar_evidencias_overfitting_spyder.py`, ajustar `PACOTE_GENERALIZACAO` e `PACOTE_REPRODUCAO` somente se a busca automática em `output/` não localizar os ZIPs, e pressionar F5. A versão gera `output/consolidacao_overfitting_tcc/<id>/pacote_consolidacao_overfitting_v1255.zip`, contendo PNGs, `LEIA_ME_GRAFICOS.md` e `resumo_consolidado.json`. Valida hashes/identidade do vintage e reconcilia 225 ajustes e 354 rotações.

**Teste local antes da publicação:** executor exercitado contra os dois pacotes acima, 7 PNGs gerados, 225 modelos e 354 rotações reconciliados. O código foi publicado no GitHub; a execução no Spyder do usuário ainda não foi verificada. Não foram realizadas novas operações financeiras.

**Conclusão científica:** overfitting preditivo e instabilidade na seleção já quantificados; há simultaneamente valor adicional local em várias rotações e forte desempenho simulado histórico, mas seleção retrospectiva e deriva de fonte impedem proclamar vantagem econômica prospectiva ou atribuir uma porcentagem do capital ao overfitting.
