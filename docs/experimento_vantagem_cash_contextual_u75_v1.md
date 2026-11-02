# Pesquisa: vantagem contextual de CASH no U75

**Versão:** \`u75-contextual-cash-advantage-v1.0.0\`  
**Branch:** \`research/u75-contextual-cash-advantage-v1\`  
**Executor Spyder:** \`pesquisar_vantagem_cash_contextual_u75_spyder.py\`  
**Motor matemático:** \`pesquisa/risco_contextual_cash_u75.py\`  
**Ambiente:** somente pesquisa; não alterar o sistema operacional, o Control, os CSVs RAW/SIP ou o universo histórico.

## Pergunta científica

Conseguimos identificar, **antes da abertura seguinte**, configurações de queda, amplitude e volatilidade que historicamente favoreceram CASH em vez de manter o ativo escolhido pelo Control?

O resultado econômico de maior interesse é **menor perda nas piores sessões e drawdown**, sem custo desproporcional de recuperações perdidas. Capital final não é objetivo de ajuste. As grandes quedas em AMS/CAPR e o episódio VNCE 2023 já foram inspecionados, e **não são holdout virgem** deste novo piloto.

## Contrato da referência e séries

Preservar o ZIP original Strategy #30 e o snapshot congelado validado por SHA, para 75 ativos e o período de referência definido no próprio manifesto. A quantidade de ativos, folds e decisões vem do **manifesto da referência**, sem números repetidos no código do novo executor. O motor matemático aceita qualquer universo com nomes únicos, histórico OHLC coerente e cobertura mínima percentual.

O executor reproduz os modelos LightGBM, a calibração original da margem e **verifica a igualdade diária da curva e de cada ativo com o Control**, antes de qualquer política alternativa. A referência apresenta limitação conhecida de elegibilidade futura (t+1), que esta etapa **não corrige**.

## Os estados observáveis em cada fechamento

No fechamento do dia t são calculadas:

| Canal | Variável | Definição |
|---|---|---|
| U75 | market_log_return_5 e market_log_return_20 | Log-retorno acumulado do índice diagnóstico equal-weight em 5/20 pregões |
| U75 | market_log_volatility_20 | Desvio padrão dos log-retornos diários do índice, janela de 20 sessões |
| U75 | market_drawdown_20 | Diferença logarítmica em relação ao maior nível do índice nos 20 últimos pregões |
| U75 | breadth_positive_20 | Fração dos ativos com retorno positivo nos últimos 20 pregões |
| U75 | breadth_below_ma20 | Fração dos ativos que fecharam abaixo de sua média móvel de 20 pregões |
| U75 | dispersion_return_20 | Desvio padrão transversal dos log-retornos de 20 pregões dos ativos |
| Ativo candidato | asset_log_return_5 / 20 | Log-retornos de fechamento em 5/20 sessões |
| Ativo candidato | asset_log_volatility_20 | Desvio padrão dos log-retornos diários do próprio ativo |
| Ativo candidato | asset_drawdown_20 | Distância logarítmica do fechamento à máxima de 20 pregões |
| Ativo candidato | asset_volatility_ratio_5_20 | Razão da volatilidade recente de 5 sessões pela de 20 sessões |
| Forma suplementar | janela z-normalizada de 20 retornos acumulados | Formato recente da curva, refinado por DTW com banda de Sakoe–Chiba |

A **amplitude absoluta de retornos** e a volatilidade permanecem como variáveis de estado; não são eliminadas pela padronização por janela usada apenas no canal de formato. Para equiparar variáveis em escalas diferentes, é usada mediana e MAD robusto calculado **somente com as observações históricas até t**. Exige-se pelo menos 90% de cobertura do universo para os indicadores coletivos. Preços ausentes não recebem preenchimento artificial.

## Três hipóteses pré-declaradas

| Cenário | Como se escolhem os episódios semelhantes |
|---|---|
| \`contexto_mercado\` | Apenas indicadores coletivos do U75; desfecho financeiro estimado para o ativo efetivamente indicado pelo Control |
| \`contexto_mercado_ativo\` | Indicadores coletivos + retornos, amplitude e volatilidade do ativo candidato |
| \`contexto_mercado_ativo_forma\` | Mesmo estado coletivo/individual + refinamento por DTW da forma das trajetórias |

Todos usam o mesmo horizonte de 5 pregões, janela 20, 96 vizinhos candidatos no pré-filtro, até 16 episódios distintos e mínimo 8. A distância entre episódios escolhidos deve ser de **h+1** pregões ou mais. A janela DTW tem raio fixo 4 e peso de forma 0,35.

**Não se trata de treinamento supervisionado com rótulos futuros no teste.** É uma seleção expandida de análogos históricos: a data final j de um padrão tem resultado elegível se \`j+h+1 <= t\`, porque a alternativa de manter posição é medida desde a abertura de \`j+1\` até a abertura de \`j+h+1\`. A abertura futura do próprio dia \`t+1\` nunca entra na previsão de t. Nenhuma decisão posterior pode mudar os análogos disponíveis numa data anterior.

## Regra econômica de proteção

Para cada análogo histórico elegível, seja \`r\` o log-retorno de permanecer no **mesmo ativo candidato** entre aberturas, por 5 pregões. Definir:

\[
U_{\mathrm{investido}}(r)=r-\lambda \max(-r,0)
\]

\[
U_{\mathrm{CASH}}=-c \quad\text{e}\quad
\Delta U=U_{\mathrm{CASH}}-U_{\mathrm{investido}}.
\]

Com parâmetros científicos piloto \`lambda=1\`, \`c=15\` pontos-base (0,15% como margem sintética de atrito de saída e reentrada), calcula-se a média dos ganhos relativos de CASH nos episódios históricos, \`m=mean(ΔU)\`, e seu limite inferior unilateral exploratório \`L=m-1,645*s(ΔU)/sqrt(n)\`.

O alvo CASH só é escolhido quando, simultaneamente:

1. \`L > 0\`;
2. pelo menos **65%** dos episódios semelhantes apresentaram retorno futuro negativo;
3. média de retorno futuro do ativo também é negativa.

Não se deve considerar \`L\` um limite com cobertura nominal certificada: episódios podem continuar economicamente dependentes, e grupos extremos têm poucas observações. Os três limites são **hipóteses piloto fixas**, não resultado de calibragem sobre 2026.

A vantagem estimada é um **proxy** de risco/retorno para um horizonte fixo, e não o patrimônio que o Control teria obtido ao rodar cinco dias sem girar. O replay financeiro efetivo continua a cargo de \`_simular_exato\`, que usa as taxas e slippage originais e recalcula o estado todos os dias. O buffer de 15 pontos-base serve **apenas para gerar sinais**, e não é cobrado uma segunda vez na simulação.

## Replay de carteira e avaliação posterior

Depois da paridade, cada política decide ao fechamento t e efetiva na abertura t+1. O Control recebe a **posição real de sua própria carteira**, inclusive CASH, quando o alerta é emitido ou cessa. Não há período fixo de CASH, nem múltiplas rotações no mesmo dia.

Arquivos por \`output/vantagem_cash_contextual_u75/<timestamp>/\`:

- \`control_paridade.json\`, \`control_posicoes.csv\`, \`snapshot_identidade.csv\`: identidade da referência;
- \`contexto_u75_cobertura.csv\`: indicadores de cobertura e market context;
- \`<cenario>_estimativas_risco.csv\`: estimativa disponível no fechamento, número de episódios, fração de perdas, limite inferior de vantagem, decisão CASH;
- \`<cenario>_decisoes_posicoes.csv\` e \`<cenario>_operacoes.csv\`: posições e custos reais do simulador;
- \`<cenario>_avaliacao_futura.csv\`: **apenas depois da simulação**, resultados observados do ativo após cada decisão, para verificar acertos e alarmes sem resposta negativa. Esse arquivo não é usado na decisão;
- \`patrimonio_cenarios.csv\`, \`patrimonio_comparado.png\`, \`metricas_pareadas.csv\`: comparação do Control com os três overlays;
- \`andamento_atual.json\`, \`andamento.jsonl\`, \`ultimo_erro.txt\` (se houver): diagnósticos de execução;
- \`resultado.json\`, \`resumo.md\` e o ZIP final \`pacote_vantagem_cash_contextual_u75_*.zip\`.

Os labels de \`avaliacao_futura.csv\` são dados de auditoria com informação posterior e não devem ser reincorporados às features ou usados para escolher a variante “vencedora” com alegação de teste independente.

## O que o piloto NÃO prova

- Não prova previsão antecipada de um choque individual imprevisível, como CAPR em 30/07/2026, nem ausência de sobreajuste;
- não corrige a elegibilidade da política Control original dependente de t+1;
- não executa busca nem rotação para ativos externos ao U75;
- não melhora ou seleciona automaticamente parâmetros no histórico já inspecionado;
- não incorpora ainda novos pregões após o corte do Control congelado;
- não indica uma estratégia adequada para operação sem comparação em períodos prospectivos e independentes.

## Executar

Na branch \`research/u75-contextual-cash-advantage-v1\`, abrir o script \`pesquisar_vantagem_cash_contextual_u75_spyder.py\` no Spyder e pressionar F5. A execução exige os mesmos 75 CSVs brutos e corporate actions, o ZIP original do Control, Python científico e o ambiente já utilizado nas pesquisas anteriores. Não baixa dados nem usa banco.

A pasta de saída é configurável via \`TCC_CONTEXT_CASH_OUTPUT_DIR\` (somente localização de resultados). As etapas são registradas nos arquivos de andamento e no console. Enviar o novo ZIP de resultados para auditoria econômica e dos episódios observados.
