# Diagnóstico de execução longa: primeiro arquivo \`snapshot_identidade.csv\`

**Executor:** \`u75-historical-motifs-cash-v1.0.2\`  
**Alteração:** apenas observabilidade do processo de reprodução, sem mudança no modelo DTW, no LightGBM, no universo, no Control ou nas decisões CASH.  
**Branch:** \`fix/u75-motifs-progress-diagnostics-v1\`.

## O que ocorreu

O usuário executou a versão anterior e informou que só foi criado \`snapshot_identidade.csv\` (75 linhas de ativos + cabeçalho). Isso **não demonstra erro**: esse arquivo é escrito antes de treinar novamente os três folds, cada um com a fase de calibração e o ajuste final dos modelos LightGBM, e antes de reproduzir o Control. Os arquivos \`control_paridade.json\`, \`control_posicoes.csv\` e os cenários CASH **só começam a ser gravados depois desses treinamentos**.

Não é possível determinar, a partir do único arquivo, se o processo continuava rodando, foi interrompido pelo usuário, perdeu o kernel, encontrou uma exceção ou ficou sem recursos. Não atribuir uma causa específica sem log.

## Correção de diagnóstico

A partir de \`v1.0.2\` o script cria, junto com o primeiro arquivo:

- \`andamento_atual.json\`: última etapa com horário em UTC, estado \`running\`, \`completed\` ou \`failed\`, e número do fold/cenário;
- \`andamento.jsonl\`: histórico permanente de etapas, acrescentado e sincronizado após cada marco.

O histórico informa, entre outros marcos:

\`\`\`text
execucao_iniciada
snapshot_validado
calibracao_inicio [fold 1]
calibracao_modelos_concluida [fold 1]
treinamento_final_inicio [fold 1]
treinamento_final_concluido [fold 1]
utilidades_calculadas [fold 1]
... repete folds 2 e 3 ...
treinamento_tres_folds_concluido
reproducao_control_inicio
reproducao_control_concluida
paridade_control_validada
indice_coletivo_construido
simulacao_cash_inicio / simulacao_cash_concluida [3 cenarios]
relatorio_financeiro_concluido
evidencias_verificadas
pacote_gerado
\`\`\`

A criação de checkpoints **não acelera** os treinamentos. Não há retreinamento adicional, cache de modelos ou relaxamento da paridade. Uma queda abrupta do kernel deixa o último marco como \`running\`, sem afirmar \`completed\`. Exceções Python são marcadas como \`failed\` e salvas no \`ultimo_erro.txt\`/ZIP de falha.

## Execução

No Git Bash, estando em qualquer branch:

\`\`\`bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch --track origin/fix/u75-motifs-progress-diagnostics-v1
\`\`\`

Reiniciar o kernel Spyder e executar **\`pesquisar_motivos_historicos_cash_u75_spyder.py\`** com F5.

**Se a execução antiga ainda estiver em andamento, não executá-la em paralelo nem interrompê-la apenas porque há só um arquivo.** O console do Spyder deve informar o último \`[modelo] fold ...\`. Se a execução antiga terminou ou o kernel foi reiniciado, usar esta nova versão para obter etapas persistidas.

Arquivos estarão em \`output/motivos_historicos_cash_u75/<timestamp>/\`. Se definida, a variável \`TCC_MOTIFS_OUTPUT_DIR\` substitui só a raiz de **resultados**; não altera o snapshot. A pasta de cada execução possui timestamp próprio. A falha de uma pasta removida durante o processo é tratada na versão \`v1.0.1\` herdada, e a identidade dos arquivos é verificada antes do status de sucesso.

## Testes

\`\`\`bash
python -m pytest -q tests/test_motivos_historicos_cash_u75.py tests/test_saida_resiliente_motivos_u75.py tests/test_andamento_motivos_u75.py
\`\`\`

Os novos testes verificam histórico de marcos por fold, persistência após interrupção simulada, recuperação de pasta excluída e rejeição de sinal falso de sucesso. O workflow \`historical-motifs-cash-tests.yml\` verifica esses testes sem modificar a suíte antiga.

Esta correção não gera resultados financeiros novos. O ZIP completo anteriormente recebido do estudo de padrões históricos permanece referência para o estudo retrospectivo.
