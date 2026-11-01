# Correção de saída do executor de motivos históricos U75

**Versão do executor:** \`u75-historical-motifs-cash-v1.0.1\`  
**Branch:** \`fix/u75-motifs-output-directory-v1\`  
**Sem alterações** no motor matemático (que mantém \`u75-historical-motifs-cash-v1.0.0\`), no Control, nos dados RAW/SIP, nos custos ou nas três hipóteses de CASH.

## Erro observado

Em uma nova execução no Spyder, o usuário recebeu \`FileNotFoundError\` ao gravar \`control_paridade.json\`, seguido por outra exceção ao gravar \`ultimo_erro.txt\`, no caminho:

\`output/motivos_historicos_cash_u75/20261101T230102194205Z/\`

O \`main()\` tinha criado a pasta antes do longo treinamento; arquivos iniciais chegaram a ser gravados, mas ela **não existia mais quando a reprodução do Control terminou**. O código da pesquisa não contém rotina intencional de exclusão desse diretório. A origem do desaparecimento não está comprovada; não atribuir automaticamente a antivírus, limpeza externa ou Spyder sem evidência.

## Correção

- As funções de gravação revalidam/criam a pasta pai **imediatamente antes de cada arquivo**. Se a pasta desaparecer entre a criação e a gravação, uma segunda tentativa recupera a operação.
- A gravação da imagem, resumo, CSVs, JSON, último erro e ZIP utiliza o mesmo mecanismo.
- O executor **não anuncia \`completed\`** se as evidências produzidas anteriormente tiverem sumido: verifica todos os arquivos essenciais do Control e dos três replays antes de finalizar. Recriar diretório não equivale a recuperar arquivos antigos.
- \`ultimo_erro.txt\` passa a ser gravado com criação automática de pastas, e existe caminho de fallback \`output/ultimo_erro_motivos_historicos_<data>.txt\` se a primeira pasta falhar persistentemente.
- Saída impressa com caminho completo no console, facilitando diagnosticar se o diretório foi movido/removido.
- Uma variável opcional **\`TCC_MOTIFS_OUTPUT_DIR\`** permite escolher uma pasta de saída estável, em outro lugar do disco. Essa variável altera somente o local de arquivos gerados, **não** o snapshot nem o resultado econômico.

## Rodar novamente no Git Bash / Spyder

Atualize a branch:

\`\`\`bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch --track origin/fix/u75-motifs-output-directory-v1
\`\`\`

No Spyder, reinicie o kernel e execute **\`pesquisar_motivos_historicos_cash_u75_spyder.py\`** usando F5.

Caso haja processo externo removendo a árvore \`output\`, configure no console do Spyder **antes de executar F5 no mesmo kernel**:

\`\`\`python
import os
os.environ["TCC_MOTIFS_OUTPUT_DIR"] = r"C:\tcc_motivos_saida"
\`\`\`

Se o Spyder estiver configurado para usar um kernel novo a cada execução, defina a variável no ambiente de inicialização ou no Windows antes de abrir o Spyder; uma variável criada em outro kernel não persiste automaticamente.

O ZIP é criado dentro de \`<TCC_MOTIFS_OUTPUT_DIR>/<timestamp>/\` quando a variável estiver ativa; senão no caminho padrão original. Nenhum dado de preços é baixado. Os tempos de treino do Control continuam sendo necessários, pois esta versão não introduz cache novo de modelos.

## Testes de regressão

\`python -m pytest -q tests/test_saida_resiliente_motivos_u75.py tests/test_motivos_historicos_cash_u75.py\`

Os casos adicionais simulam exclusão da pasta após a criação, exclusão entre mkdir e escrita, falha repetida e integridade da lista de evidências. O workflow dedicado executa testes estáticos e sintéticos em GitHub Actions. A suíte geral do repositório pode acusar regra antiga de nomenclatura em scripts herdados e deve ser avaliada separadamente, sem editar a história científica para tornar artificialmente o teste verde.

Esta correção **não gera** resultados financeiros novos nem corrige a limitação histórica de elegibilidade t+1 da referência. O último pacote completo \`pacote_motivos_historicos_cash_u75_20261101T205201014162Z.zip\` já foi auditado; repetir o estudo é útil para testar a estabilidade da execução, não para chamar os mesmos eventos de 2026 de validação independente.
