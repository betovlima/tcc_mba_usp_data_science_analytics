# CONTEXTO_MESTRE — reprodução científica do TCC

**Atualização:** 21/10/2026. **Projeto:** `betovlima/tcc_mba_usp_data_science_analytics`. Este documento é a referência inicial para um novo assistente ou nova sessão. **Separação de ambientes:** o TCC utiliza dados congelados em pesquisa; a aplicação operacional independente utiliza cotações Alpaca atuais.

## 1. Git, estado e identidade

- **Branch principal:** `main`. O commit da pesquisa histórica `13e5ed39ed0e700d2eb8c90bc8bee64adc7fdf9a` já foi incorporado à `main` pelo merge **`9d2e279938a2ad271f3ef5e0ac86a18e130c7e4f`**, de 18/10/2026 UTC. Seu HEAD histórico era `13e5ed39ed0e700d2eb8c90bc8bee64adc7fdf9a`. **Não fazer novo merge nem abrir branches duplicadas.**
- **Motor Control de referência:** versão `1.22.1`. **Programa de pesquisa:** `pesquisar_expansao_u67_spyder.py`, versão `u67-asset-expansion-v1.4.1`, esquema `u67-asset-expansion-v5`.
- **Tag de congelamento acordada:** `research-u67plus10-challenger-v1.4.1`. A tag deve apontar ao estado documentado da `main`; **verificar com `git ls-remote --tags origin` se foi efetivamente publicada**. Não afirmar que existe antes da confirmação.
- **Reprodução já atestada:** `reproduzir_experimento.py` reproduziu Control base; `assurance_validation` confirmou os cenários de expansão em 21/10/2026. Não mudar código, limites financeiros, folds, modelos ou fonte de dados para alcançar o checkpoint.

## 2. Dados congelados que produzem o resultado histórico

| Item | Contrato obrigatório |
|---|---|
| Base | `dados/u67/manifest.json` e 67 arquivos RAW/Corporate Actions versionados |
| Identidade da base | `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575` |
| Fonte / datas | Alpaca SIP, RAW, 1Day; início 01/01/2016, corte **06/10/2026** |
| Base efetiva | 65 ativos; excluir DOC e CLMT por identidade estrutural, sem reparar/concatenar séries |
| Expansão testada | 15: CRT, NBR, CAPR, PNBK, PENN, JBL, LEU, JBSS, STAG, MCHB, EML, MAR, LAD, TOWN, ACIW |
| Dez selecionados | **CRT, NBR, CAPR, PNBK, PENN, JBL, STAG, MCHB, LAD, TOWN** |
| Cache dos 15 | `dados/.u67_expansion_cache/full_raw_20261006/raw_bars/*.csv` e `corporate_actions/*.csv` (ignorado pelo Git) |
| Evidência externa | `pacote_analise_u67_assurance_validation(1).zip`, SHA-256 `0f06561e9685b3e84194580c50a6808e4c84d30a55584b97bd19dae2fafd6c37` |

**Risco crítico:** houve execução com um snapshot local diferente (`8183a2b718bb...`) que retornou **US$ 45.493.884,10360689** no Control inicial; após restaurar o snapshot original versionado, voltou a **US$ 76.927.051,38897176**. Validar um manifesto local não prova que ele é o manifesto histórico: comparar o SHA exato. **Não executar `preparar_snapshot_u67.py` durante a reprodução e não baixar novamente barras/eventos dos 15 ativos.** Antes de restaurar arquivos, criar backup fora do Git.

## 3. Capitais e evidência recebida

| Cenário no cutoff 06/10/2026 | Capital (US$) |
|---|---:|
| Control base, 65 efetivos | **76.927.051,38897176** |
| Base + 15 positivos | **290.727.954,41339606** |
| Base + 10 selecionados, 75 efetivos | **587.165.941,1330212** |

A execução `assurance_validation` enviada pelo autor declarou `status=completed`, `all_15_exact_to_cent=true` e `selected_10_exact_to_cent=true`. `reproducibility.json` confirmou `exact_to_cent=true`, `selected_asset_sequence_identical=true`, `max_equity_absolute_difference=0.0` **entre dois replays da campanha restaurada**. O Sharpe dos dez foi **2,7285355362** e o MaxDD **-41,68283638%**; com 10 bps de slippage, capital caiu para **US$ 288.628.398,50**.

**O ZIP e o cache de candidatos não foram anexados ao Git**; o ZIP inclui os SHA-256 de cada barra e arquivo de Corporate Actions dos 15 candidatos. Clonar o repositório sozinho **não garante** a reprodução do cenário expandido. A evidência histórica deve ser preservada de forma separada e seu checksum conferido.

## 4. Como executar sem alterar os dados (Spyder)

**Git é usado no console do Git; Python exclusivamente no console IPython do Spyder.** Na reprodução histórica, trabalhar na `main` atualizada e preservar alterações locais antes de usar `git restore`.

No **console Git** (somente consulta/atualização de referências e estado):

```bash
git status --short
git fetch origin
git branch --show-current
git ls-remote --tags origin research-u67plus10-challenger-v1.4.1
```

Para restaurar **somente se necessário**, após backup externo de `dados/u67`, usar no console Git:

```bash
git restore --source=origin/main --worktree -- dados/u67
```

No **console Spyder** (diretório de trabalho é a raiz do TCC), reiniciar kernel e executar:

```python
from pathlib import Path
from datetime import datetime
import runpy
import shutil
import pandas as pd
from reproducao.dados import SnapshotPaths, validate_snapshot

ROOT = Path.cwd().resolve()
assert (ROOT / "pesquisar_expansao_u67_spyder.py").is_file(), "Abra o Spyder na raiz do TCC"
assert validate_snapshot(SnapshotPaths.u67(ROOT))["snapshot_sha256"] == "e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575"

namespace = runpy.run_path(str(ROOT / "pesquisar_expansao_u67_spyder.py"), run_name="assurance_offline")
env = namespace["main"].__globals__
env["RUN_MODE"] = "assurance_validation"
env["OUT"] = ROOT / "output" / "u67_assurance_validation"
if env["OUT"].exists():
    backup = env["OUT"].with_name(env["OUT"].name + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copytree(env["OUT"], backup)
    print("Backup da saída:", backup)

def usar_somente_cache(_credentials, ativos):
    rows = []
    for asset in ativos:
        valid_bars = env["_full_raw_cache_valid"](asset)
        valid_actions = (env["FULL_DATA"].corporate_actions / f"{asset}.csv").is_file()
        if not (valid_bars and valid_actions):
            raise RuntimeError(f"Cache histórico faltante/incompleto: {asset}")
        rows.append({"asset": asset, "status": "cached", "reason": None})
    return pd.DataFrame(rows)

def bloquear_download(*args, **kwargs):
    raise RuntimeError("Download proibido na reprodução histórica")

env["load_alpaca_credentials"] = lambda *_: None
env["_prepare_full_candidate_files"] = usar_somente_cache
env["download_raw_bars"] = bloquear_download
env["download_corporate_actions"] = bloquear_download
result = env["main"]()
```

**Observação:** esse bloqueio cobre as funções de download do script histórico v1.4.1 e não substitui validação de hash dos arquivos do cache. O programa grava em `output/u67_assurance_validation/` e pode demorar após `[assurance-train] 15/15 ACIW` sem mostrar novos logs: trabalha em simulações de subconjuntos. Um primeiro arquivo de progresso esperado é `assurance_scenarios_partial.csv`. Não reiniciar o kernel enquanto houver CPU trabalhando.

## 5. Como interpretar e continuar a pesquisa

- Os checkpoints são **reprodução histórica**, não evidência de lucro futuro. A rotina original selecionou candidatos com informação de toda a janela; a busca de subconjuntos apresentou **PBO 53,57% (log-growth)** e **59,52% (Sharpe)**.
- A pesquisa temporal em pool fixo selecionou `NBR,PENN,EML` usando fold 1 para avaliar fold 2; e `CRT,NBR,PNBK,PENN,JBL,STAG,MCHB,TOWN` usando folds 1+2 para avaliar fold 3. O pool já era conhecido: **não é descoberta nested prospectiva**.
- A auditoria registrou **38 alertas de overfit em 45 candidato-fold**, segundo `model_generalization.csv`. Não esconder isso nas redações.
- Priorizar uma **assinatura contextual/temporal de pertencimento** `S(ativo, U_t, contexto_t)` para ADICIONAR/MANTER/REMOVER sem olhar capital posterior à decisão. Diferenciar a pesquisa matemática posterior v1.18/v1.19, que teve outro baseline (U59), antes de tentar transferir o modelo.
- **Separação permanente:** o TCC utiliza CSVs congelados e a plataforma operacional independente utiliza preços atuais da Alpaca. O capital do experimento histórico não é previsão nem condição automática para decisões novas.

## 6. Documentos para releitura antes de qualquer modificação

1. [MANIFEST.md](MANIFEST.md), seção 26, com resultados e contexto da promoção.
2. [PESQUISA_EXPANSAO_U67.md](PESQUISA_EXPANSAO_U67.md), metodologia da campanha e interpretação da assinatura temporal.
3. [README.md](README.md), instruções gerais e requisitos.

**Regra para a próxima sessão:** verificar branch/commit/snapshot/cache/hash antes de propor alteração; não criar branches nem mudar `main` por suposição. Alterações futuras só mediante solicitação explícita do autor; documentar versão, causa, testes e limites.


## 7. Mudança de entrada oficial (v1.23.0)

A entrada `reproduzir_experimento.py` usa a função reutilizável `reproduzir_universo_congelado` em `reproducao/experimento.py` e deve calcular o checkpoint histórico do **universo selecionado**, US$ 587.165.941,1330212, com 77 ativos solicitados (75 efetivos). Evitar reintroduzir marcadores de contagem no nome do executor. O programa de pesquisa histórico mantém suas funções originais para garantir paridade numérica. `evidencias/manifest_universo_congelado.json` fixa os SHA-256 das barras e eventos corporativos dos dez candidatos; o executor não consulta Alpaca. O pacote gerado é `output/reproducao_universo/pacote_reproducao_universo.zip`. O usuário usa comandos Git em seu console Git e Python no Spyder.

O checkpoint de US$ 587 milhões é **meta de reprodução na mesma amostra congelada**, não inferência prospectiva. **A verificação integral foi concluída com sucesso em 21/10/2026**, no ambiente Spyder do autor: capital final observado de US$ 587.165.941,1330212, diferença US$ 0,00, `status=completed`, 1.560 sessões e 354 rotações. Pacote recebido `pacote_reproducao_universo.zip`, SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`. Os dez pares de arquivos adicionais continuam locais e não versionados.


## 8. Ponto de recuperação oficial — release v1.23.0

**Estado aprovado para congelamento da release em 21/10/2026:** a nova entrada `reproduzir_experimento.py` é o executor oficial do universo selecionado, com nomes genéricos, delegando a rotina financeira ao código validado na campanha histórica. Foram confirmados a identidade do snapshot `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`, os 20 hashes dos dez arquivos de barras e dez eventos, 77 ativos solicitados/75 efetivos e capital de **US$ 587.165.941,1330212**, exato ao centavo. `reproducao_universo.json` do ZIP declarou `reproduced_exact_to_cent=true`. O arquivo de evidência é `pacote_reproducao_universo.zip` (SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`); não foi adicionado ao Git.

A tag de referência da versão é **`v1.23.0`**, criada somente após sucesso dos testes do commit documental. **Antes de afirmar que a tag foi publicada, verificá-la no GitHub**. O código da versão já estava na `main`; não é necessário fazer novo merge. A tag histórica `research-u67plus10-challenger-v1.4.1` não deve ser movida. A CI cobre 94 testes/checagens unitárias e estáticas, mas não roda o replay financeiro que depende do cache não versionado.

**Próxima sessão:** usar `git fetch origin`, `git switch main`, `git pull --ff-only origin main` em console Git; executar `reproduzir_experimento.py` com F5 no Spyder sem download de histórico; comparar o SHA-256 do ZIP local contra a evidência desta seção. Não misturar os comandos Git em células Python. A pesquisa futura de assinatura de pertencimento temporal deve tratar os US$ 587 milhões como checkpoint histórico com seleção retrospectiva, não como validação prospectiva de descoberta automática.

## Atualização de pesquisa: sensibilidade de margens (12/11/2026)

A branch `research/permanencia-minima-contrafactual-v1` preserva um único arquivo
de pesquisa, `pesquisar_permanencia_minima_spyder.py`, executado com F5 no
Spyder. O modo `sensibilidade_margens` compara o Control congelado com um
cenário contrafactual no **mesmo snapshot científico**, sem retreinamento,
download, acesso a contas ou alteração do motor oficial.

- Fonte: `output/pesquisa_permanencia_minima.zip`, 75 ativos efetivos, 3
  folds e 1.560 sessões; primeiro é reproduzido o Control histórico.
- Margens calibradas originais por fold: `{1: 0.0, 2: 0.01, 3: 0.0}`.
  As margens efetivas são `0.0005 / 0.01 / 0.0005`.
- Cenário de sensibilidade: candidatos por fold
  `{1: 0.01, 2: 0.0, 3: 0.01}`, margens efetivas
  `0.01 / 0.0005 / 0.01`.
- Com a mesma base e as mesmas previsões, a reprodução original resultou
  em **US$ 587.165.941,13**, 354 rotações. O cenário contrafactual resultou
  em **US$ 58.862.373,39**, 334 rotações.
- A reconstrução da política com margens originais reproduziu exatamente a
  trajetória Control (erro máximo de US$ 0,00).
- O relatório continua usando apenas
  `output/pesquisa_vantagem_rotacao.zip` (substituível com verificação CRC
  e SHA-256), sem sufixos de versão.

**Limite:** este é um teste de sensibilidade *ex post*, não uma nova calibração,
um resultado prospectivo ou comprovação de identidade com outras bases.
O checkpoint de US$ 587 milhões serve somente à reprodução do snapshot
histórico associado. É proibido substituir o assert por outro valor sem
reconstruir todas as entradas, o treinamento, a calibração e o replay.
Não promover essa política para execução real.

## Auditoria descritiva de permanencia no replay atualizado (12/11/2026)

O arquivo unico de pesquisa `pesquisar_permanencia_minima_spyder.py` possui
modo `analisar_permanencia` (padrao). Ele localiza na pasta `output/`, na raiz
ou em `Downloads` um ZIP de backtest com `*_predictions.csv` e
`*_metrics.json`. Nao importa nem depende de outra aplicacao. A pesquisa
somente **le** o replay exportado, nao usa dados futuros nas variaveis de
decisao, nao modifica o motor, nao retreina e nao executa ordens.

Referencia auditada:
- Fonte SHA-256:
  `46fe00a978418ab10d742dee855136ac6d5fbb7e726a9ee928f3109c178c06d9`.
- Periodo: 22/07/2020 a 11/11/2026, **1.586** sessoes.
- Capital inicial: US$ 10.000; capital final exportado:
  **US$ 139.871.072,30749363**; 350 rotacoes.
- **351** episodios de exposicao, **884** decisoes de permanencia
  (idade da posicao >= 2) e retornos observaveis de 1, 3 e 5 pregoes em
  884, 583 e 426 casos, respectivamente.
- Em oito episodios houve drawdown da posicao >= 20% enquanto ela
  permanecia investida; cinco terminaram com retorno positivo no episodio.
- Caso de teste: posicao CAPR de 28/07 a 27/08/2026. Drawdown no episodio
  de **-41,6828%**, mas retorno do capital no episodio completo
  **+52,3820%**. No fechamento anterior ao tombo inicial, os sinais
  de drawdown estavam em torno de 9% e a utilidade havia subido.
- Condicao exploratoria `drawdown >= 10% E score desde a entrada caiu`:
  20 observacoes, 12 seguidas por perda no dia seguinte; amostras
  dependentes e **insuficientes** para implantar um gatilho.
- **Nao confundir** auditoria observacional com contrafactual executavel.
  Rotacoes censuram o horizonte e a saida para CASH exigiria precos
  de abertura/fechamento e custos para reconstruir ordens.

O pacote de resultados segue exatamente o nome
`output/pesquisa_vantagem_rotacao.zip`, sem v1/v2/v3. Ele contem
`sinais_por_decisao.csv`, `episodios_posicao.csv`,
`associacoes_sinais_e_retornos_observados.csv`, estudo do periodo critico,
manifesto e SHA-256. O capital de US$ 587 milhoes permanece apenas como
checkpoint do snapshot historico congelado, nao como referencia desse replay
atualizado. Nenhuma hipotese de melhoria foi validada ainda fora da amostra
apos a escolha dessas condicoes.

## 12/11/2026: impacto financeiro teorico dos alertas de cauda

Fonte: pacote `pesquisa_vantagem_rotacao(6).zip`, CRC e 11 hashes internos
conferidos. O manifesto do pacote vincula o replay ao capital final de
**US$ 139.871.072,31**, 1.586 sessoes, 351 episodios e 884 decisoes de
manutencao; nao se trata do antigo checkpoint de US$ 587 milhoes.

Modelo exploratorio de risco de perda >=5% na sessao seguinte, treinado
somente em folds cronologicamente anteriores:
- Fold 2: AUC 0,6164; 42 alertas, 5 quedas >=5% identificadas
  em 12 perdas observadas; 27 sessoes com alta apesar do alerta.
- Fold 3: AUC 0,7218; 39 alertas, 8 quedas >=5% identificadas
  em 16 perdas observadas; 22 sessoes com alta apesar do alerta.
- Total: 81 alertas, 13 quedas >=5% identificadas, apenas 17 episodios
  distintos alertados; 49 das 81 sessoes sinalizadas tiveram retorno positivo.
- Em 13/08/2026, o alerta de CAPR precedeu valorizacao de **57,9572%**;
  o alerta de 29/07/2026 NAO foi acionado antes da queda de **36,2253%**.

**Teste economico limitado, sem executar ordens:** manter a curva de capital
original, mas substituir por 0% os retornos close-to-close das 81 sessoes
sinalizadas, preservando todos os demais fatores diarios, leva a
**US$ 50.289.622,51** de capital final, contra **US$ 139.871.072,31** do
Control (queda de aproximadamente 64,05% em relacao ao Control).
O max drawdown desta curva teorica fica **-48,37%**, frente a **-41,68%**
do Control. Isso NAO e um replay de ordens em abertura, nao inclui venda,
recompra, custo extra, reentrada real ou alteracao de politica apos sair.
Resultados sensiveis a sinais previamente escolhidos examinando a amostra.

No mesmo arquivo `pesquisar_permanencia_minima_spyder.py`, modo
`analisar_permanencia`, a auditoria exporta adicionalmente
`impacto_alertas_curva_shadow.csv`,
`impacto_alertas_por_decisao.csv` e
`impacto_alertas_resumo_por_fold.csv` para o **mesmo**
`output/pesquisa_vantagem_rotacao.zip`. Nao criar script, branch,
sufixo versionado ou estrategia de negociacao sem validacao economica
e causal independente. A estrategia operativa segue intocada.

## 13/11/2026: alertas repetidos versus primeiro alerta da posicao

Analisado o pacote seguinte, identificado pelo SHA-256
`4a944f808c21af247cef00b181b110909679995fdffe58651789b9f636a2f178`.
Os 14 checksums internos, CRC, manifesto e serie de capital conferem.
A referencia continua US$ 139.871.072,31, API 10.8.149.

- 81 alertas emitidos em 17 episodios diferentes; 49 dos 81 dias
  apresentaram valorizacao do ativo.
- O somatorio dos log-retornos close-to-close dos dias alertados foi
  +1,0229223424. Neutralizar esses dias gera US$ 50.289.622,51
  na curva teorica, pior que o Control.
- Grande concentracao de oportunidades: um alerta antes de CAPR subir
  +57,9572% (14/08/2026) e outro antes de YANG subir +35,9833%
  (08/10/2024). **Nao remover tais casos por conhecer o futuro.**
- Nova analise diagnostica predefinida: neutralizar SOMENTE o dia
  posterior ao primeiro alerta cronologico por episodio historico,
  desconsiderando alertas posteriores ate a proxima troca de ativo.
  Isso produz **17 dias** neutralizados, com 8 positivos e 9 negativos,
  e capital teorico de **US$ 118.757.224,48**, drawdown -48,37%.
  E melhor que neutralizar 81 dias, mas ainda inferior ao Control.
- O principal alerta isolado que prejudica essa nova curva e o
  primeiro alerta do episodio YANG em 07/10/2024, antes da alta
  de +35,9833%. Nao ha regra causal demonstrada para filtrar o caso.

O mesmo script `pesquisar_permanencia_minima_spyder.py` no modo
`analisar_permanencia` passa a registrar os arquivos
`impacto_primeiro_alerta_curva_shadow.csv`,
`impacto_primeiro_alerta_por_episodio.csv`,
`impacto_primeiro_alerta_resumo_por_fold.csv` e
`concentracao_alertas_por_episodio.csv` no mesmo
`output/pesquisa_vantagem_rotacao.zip`.

**Limites:** nao e simulacao operacional de CASH com ordens em abertura,
custos, diferenca entre abertura e fechamento, nem nova politica refitada.
Os resultados sao diagnosticos de um comportamento observado e o
episodio e definido pela trajetoria de posicao original. Repetir avaliacoes
sobre esta amostra nao transforma essas regras em evidencias prospectivas.

## 13/11/2026: generalizacao temporal e influencia dos primeiros alertas

Novo pacote de saida recebido `pesquisa_vantagem_rotacao(8).zip`,
com CRC e 18 hashes internos validados. Continua o replay concluido
com 1.586 sessoes, 351 episodios, patrimonio final
US$ 139.871.072,31. Nao usar os US$ 587 milhoes do snapshot de
reproducao historica como referencial financeiro desta pesquisa.

O classificador de risco de perda diaria >=5% foi REAJUSTADO, reproduzindo
exatamente (erro de score numerico maximo <=2.3e-16) os scores exportados:
- Fold 2: treino 328 obs / 13 eventos, AUC in-sample **0,796337**;
  teste 221 obs / 12 eventos, AUC **0,616427**, gap **0,179910**.
  Eventos de perda concentrados em apenas **9 episodios** distintos
  dentre 67 episodios testados.
- Fold 3: treino 550 obs / 25 eventos, AUC in-sample **0,776610**;
  teste 328 obs / 16 eventos, AUC **0,721755**, gap **0,054855**.
  Eventos em apenas **9 episodios** dentre 67 testados.
- Bootstrap exploratorio de episodios, 1.200 replicacoes e seed
  por fold 42+fold: intervalo empirico de 95% para AUC OOS
  **fold 2: 0,348 a 0,892**, **fold 3: 0,446 a 0,877**.
  Ambos os intervalos incluem AUC 0,5; nao ha comprovacao de
  generalizacao robusta so com estes episodios.
- Os scores logistic class_weight=balanced NAO sao probabilidades
  calibradas: media prevista 0,492/0,458 versus incidencia
  0,0543/0,0488 (folds 2/3).
  Brier OOS do modelo 0,27166 / 0,21966, muito pior que o
  preditor constante da prevalencia de treino 0,05157 / 0,04641.
  Os limiares de score percentil 90 de treino sao apenas regras
  de ranking, nao probabilidades de 63%-65% de queda.
- Primeiro alerta por episodio: 17 intervencoes idealizadas,
  capital teorico US$ 118.757.224,48 contra Control de
  US$ 139.871.072,31. Analise leave-one-episode-out:
  retirar o primeiro alerta de YANG em **07/10/2024** transforma
  o resultado teorico em **US$ 161.489.949,61**, invertendo a
  conclusao sobre vencer ou nao o Control. Retirar qualquer
  dos outros 16 alertas individualmente nao supera o Control.
  Isso e medida de fragilidade ex post, NAO autoriza remover
  YANG da regra por conhecer a valorizacao posterior.

**Diagnostico:** a queda do AUC do treino para OOS no fold 2 e
compativel com dificuldade de generalizacao (risco de overfitting);
underfitting nao ficou demonstrado. A variacao de resultados entre
folds e episodios, o risco de seletividade da pesquisa e a
calibracao ruim impedem uso em producao. Nem o primeiro alerta
nem o shadow close-to-close simulam negociações reais, custo,
abertura, ou o reposicionamento da carteira em CASH.

A branch de pesquisa conserva `pesquisar_permanencia_minima_spyder.py`
e o mesmo artefato `output/pesquisa_vantagem_rotacao.zip`.
O modo `analisar_permanencia` inclui agora treino-vs-teste,
bootstrap de AUC por episodio e leave-one-out de capital no
manifesto e nos CSVs de qualidade e sensibilidade. Sem nova
branch, sem nova versao de arquivo e sem alteracao da operacao.

## 12/11/2026: primeira avaliacao economica cronologica de HOLD, ROTATE e CASH

Execucao imediata sobre o replay exportado atual, com 1.586 sessoes,
351 episodios e Control de **US$ 139.871.072,30749363**.
O valor historico antigo permanece apenas como checkpoint de sua
base antiga, nao e o alvo dos novos estudos.

**Regras de validacao:** nenhum retreinamento do motor original e nenhum
acesso a sistemas operacionais. Ridge(alpha=80), 12 sinais conhecidos
no fechamento da decisao, fold 1 somente treino, folds 2 e 3 avaliados
cronologicamente. Embargo de 30 dias-calendario na amostra diaria.
Na comparacao de rotacoes, a amostra de treino tambem exige que o
episodio tenha TERMINADO antes do embargo, evitando o rotulo futuro
entrar no treinamento. Paridade da curva original melhor que US$ 0,01.

**Teste A: CASH de um pregão com retorno close-to-close idealizado.**
Classificador anterior de perda substituido por regressao Ridge direta
do log-retorno do Control. CASH apenas se o log-retorno previsto de
seguir a operacao original e inferior a -0,002. Um dia em CASH recebe
retorno log -0,002 (custo hipotetico, 10bps por perna);
no pregão seguinte a politica volta automaticamente a reproduzir a
posicao Control, SEM simular o gap de abertura nem recalcular ordens.
- Fold 2: 482 sessoes de treino, 504 de avaliacao, **249** dias em
  CASH, dos quais **139** apresentaram alta originalmente.
  Fator sombra/control no fold **0,06407346**.
- Fold 3: 986 treino, 578 avaliacao, **6** dias em CASH, 3 com alta
  original. Fator no fold **0,88905436**.
- Acumulado: **255** dias em CASH; final da curva sombra
  **US$ 7.967.726,13** vs Control **US$ 139.871.072,31**
  (perda relativa ~94,30%). Sem custo adicional, a curva ainda terminaria\n  em **US$ 13.268.584,17** (perda relativa ~90,51%).\n  Resultado NEGATIVO mesmo sem atrito de negociacao.
- NAO e simulacao executavel nem politica valida: sem preco das
  aberturas, ordem, estado alternativo de carteira, impacto de mercado
  e regresso operacional a posicao.

**Teste B: rotacoes emparelhadas pelo mesmo intervalo historico.**
O replay contem **350** pares de retornos do ativo comprado e do ativo
anterior durante o intervalo em que a rotacao original foi mantida.
Tres opcoes: ROTATE=retorno do escolhido, HOLD=retorno do anterior,
CASH=zero bruto menos 0,002 hipotetico. Modelo Ridge multi-saida
treinado nas rotacoes finalizadas dos folds anteriores, embargo de
30 dias e teste por fold; nao encadeia alternativas num portfolio.
- Fold 2: 141 rotacoes avaliadas; a escolha original superou HOLD em
  **59,57%** dos casos, com vantagem log media **+0,027082**.
  O novo modelo recomendou 62 ROTATE, 71 HOLD, 8 CASH;
  sua vantagem log MEDIA pareada sobre a rotacao original foi
  **-0,012079** por episodio.
- Fold 3: 125 rotacoes, a original superou HOLD em **59,20%**,
  vantagem log media **+0,024723**. O novo modelo recomendou
  116 ROTATE, 9 HOLD, 0 CASH; vantagem log media **-0,002578**.
- As somas de vantagens log de episodios nao sao capitalizaveis,
  pois uma decisao alternativa alteraria o ativo de entrada na proxima
  operacao; o replay nao contem todos os precos para recomputar
  o estado posterior. NAO afirmar capital alternativo HOLD/ROTATE.

**Conclusao provisoria:** o modelo economico exploratorio nao superou
as escolhas originais, no shadow de caixa nem na comparacao pareada
de rotacoes. A avaliacao fold2/3 ja participou de multiplas analises,
logo nao constitui teste prospectivo independente. Nao usar em
negociacao real. Avancar so com reproducoes de trajetorias executaveis
sobre precos completos e novas janelas temporais genuinamente nao
usadas na escolha da politica.

**Entrega:** mesmo arquivo `pesquisar_permanencia_minima_spyder.py`,
modo-padrao `vantagem_economica` (modos anteriores preservados).
Mesmo `output/pesquisa_vantagem_rotacao.zip` com relatorios anteriores
e quatro arquivos `vantagem_economica_*.csv`, manifesto,
codigo, hashes CRC e SHA-256. Branch de pesquisa existente, sem
deploy e sem integracao na principal.

## 12/11/2026: auditoria de generalizacao economica do replay 015245

O arquivo recebido `pesquisa_vantagem_rotacao(20261113-015245).zip`,
SHA-256 `96df3f85b4e4759779ea66541fe6775c7cf3259e459579488c72481df5ad761e`,
foi validado (CRC e 24 hashes internos); possui 25 arquivos totais,
incluindo o manifesto e codigo original. Refere-se a 1.586 pregoes,
Control de **US$ 139.871.072,31** e API 10.8.149. O checkpoint antigo
de outro snapshot nao e meta nem benchmark atual.

O modelo economico Ridge(alpha=80) de retorno log diario, com 12
variaveis conhecidas no fechamento, foi retreinado para auditoria
com exatamente os mesmos intervalos e embargo de 30 dias. As previsoes
reproduziram o ZIP com erro maximo ~1.1e-16. Os indicadores:

| Metrica | Fold 2 | Fold 3 |
| --- | ---: | ---: |
| Observacoes de treino | 482 | 986 |
| Avaliacao temporal | 504 | 578 |
| R2 no treino | 0,03230 | 0,01196 |
| R2 na avaliacao contra media do treino | -0,07031 | +0,00225 |
| Correlacao da previsao com o retorno real | -0,03571 | +0,06705 |
| Retorno log medio previsto/dia | -0,0005943 | +0,0036109 |
| Retorno log medio realizado/dia | +0,0087214 | +0,0060813 |
| Dias em CASH idealizado | 249 (49,40%) | 6 (1,04%) |

O modelo quase nao explica o retorno diario mesmo no ajuste, e
na avaliacao do fold 2 perde para o preditor constante da media
disponivel no treino. Isso e compativel com capacidade limitada ou
pouco sinal nas variaveis; nao demonstra isoladamente underfitting
nem overfitting de parametros. A busca repetida sobre estes mesmos
folds agrava sobreajuste de pesquisa nas regras escolhidas.

No fold 2, dos 249 dias em CASH, 139 foram de alta original. O
somatorio log-retorno do ativo durante os dias em CASH foi
**+2,249725**, sendo +4,477487 de altas perdidas e
-2,227762 de perdas evitadas. A posicao VNCE concentra +0,849070
deste total, cerca de 37,74%; por exemplo, as decisoes de CASH
antes de 26/12 e 27/12/2023 perderam retornos diarios log
de +0,320695 e +0,396654, respectivamente.
No fold 3 os 6 dias de CASH ocorreram em CAPR, com +0,105597
de retorno log original perdido.

Entre os 266 intervalos pareados de rotacao, as alternativas
sugeridas (HOLD/CASH) tiveram vantagem log media NEGATIVA frente
ao ROTATE original: -0,012079 no fold 2 e -0,002578 no fold 3.
O pior caso foi a sugestao HOLD (AMD) em lugar de ROTATE (VNCE)
em 26/12/2023, vantagem log -0,643219, aproximadamente 37,77%
da soma das desvantagens locais do fold 2. Os pares nao podem
ser concatenados em patrimonio alternativo por dependerem de
trajetorias futuras nao reconstruidas.

A branch atualizou o **mesmo**
`pesquisar_permanencia_minima_spyder.py`, modo
`vantagem_economica`, com quatro CSVs adicionais:
`vantagem_economica_generalizacao_diaria.csv`,
`vantagem_economica_cash_por_ano.csv`,
`vantagem_economica_cash_por_ativo.csv` e
`vantagem_economica_influencia_rotacoes.csv`.
Os dados sao somente auditoria diagnostica, sem alterar modelo,
limiar ou politica. Output permanece com o mesmo nome
`output/pesquisa_vantagem_rotacao.zip`, sem nova branch ou deploy.

**Proxima investigacao cientifica:** nao otimizar novo limiar nos
folds 2/3 ja examinados. Primeiro obter historico completo,
com calendario e custos suficientes para uma politica alternativa
executavel. Confirmar generalizacao em nova janela ainda nao
usada para selecionar hipoteses antes de considerar producao.

## 12/11/2026: estudo direto da trava de dois pregoes

O objetivo cientifico volta a ser tornar a trava da permanencia minima
de dois pregoes mais responsiva ao estado da propria posicao, sem
substituir a rede de utilidade da estrategia principal nem acrescentar
um previsor externo de retorno.

A exportacao do replay atualizado de 12/11/2026, com capital de
**US$ 139.871.072,31**, contem 1.586 sessoes e **351** decisoes em
que a protecao do primeiro pregao atuou, inclusive quando o ativo
continuava no topo. Em **149** delas o topo estimado era outro ativo;
**134** tinham gap de utilidade superior a margem efetiva.
A regra exploratoria predefinida para uma excecao no primeiro pregao,
com gap >= max(0,045, 3*margem) e queda do score desde a entrada
<=-0,04, apontaria **38** casos potenciais na trajetoria atual
(incluindo folds 1=8, 2=16, 3=14). Isto e **diagnostico**, nao
simulacao de ordens ou ganho de capital, pois a exportacao nao contem
todos os precos para recalcular as posicoes alternativas.

Para testar a logica sem inventar trajetorias, as mesmas regras
foram simuladas com ordens na abertura seguinte, taxas e slippage
sobre a base cientifica CONGELADA que possui OHLCV dos 75 ativos.
Esse backtest antigo e metodologicamente util, mas **NAO** e o
Control financeiro atualizado de producao; fold1 preserva o Control
e somente folds 2 e 3 sao alterados.

Cenarios avaliados no replay de historico congelado:
- Control com 2 pregoes: US$ 587.165.941,13; 354 rotacoes.
- Excecao de troca apos o primeiro pregao com queda de score + gap
  muito forte: US$ **280.599.893,14**; 373 rotacoes e 41 excecoes.
- Exigir persistencia de vantagem em dois fechamentos consecutivos
  para a troca de gap pequeno, com liberacao imediata se gap forte:
  US$ **100.124.258,54**, 285 rotacoes e 110 esperas de confirmacao.
- Confirmacao mais excecao de primeiro pregao: US$ **56.445.854,13**,
  292 rotacoes e 142 intervencoes registradas.
- Um terceiro pregao SOMENTE se a utilidade de posicao subiu
  +0,03 e o gap concorrente e pequeno: US$ **606.778.867,73**,
  com apenas **uma** intervencao, logo nao generalizavel.

As regras simples de alteracao da trava pioraram muito o capital
na base historica na maioria dos casos. A melhora isolada de um dia
extra nao autoriza trocar a politica porque ocorreu apenas uma vez.
Os limiares foram predefinidos sem reajuste nesta rodada; contudo,
todos os folds antigos ja foram explorados por muitas hipoteses,
portanto o estudo e EXPLORATORIO, nao confirmacao independente.

O arquivo unico `pesquisar_permanencia_minima_spyder.py` tem novo
modo-padrao `permanencia_inteligente` que exporta no MESMO ZIP
`output/pesquisa_vantagem_rotacao.zip`: controle de dois pregoes,
quatro variantes, resultados por fold, ordens e decisoes completas
da base congelada, intervencoes e auditoria sem ordens do replay
atualizado quando este esta disponivel localmente. A base congelada
nao e rebaixada nem falsamente declarada como capital atual.
Nenhum arquivo/branch adicional, deploy ou alteracao em Live.

## Correcao do F5: recuperacao automatica da base historica

Problema relatado na execucao local do Spyder:
`FileNotFoundError: ... pesquisa_permanencia_minima.zip`, no
`_pi_localizar_base`. O modo `permanencia_inteligente` precisava do
pacote com 75 cotacoes, utilidades e serie de patrimonio congelados,
mas o localizador anterior procurava apenas tres caminhos fixos.
Assim, falhava mesmo quando a fonte estava embutida em um ZIP de
pesquisa anterior.

**Resolvido na mesma branch e no mesmo arquivo:**
1. Procura automaticamente os ZIPs `pesquisa_permanencia_minima*.zip`
   em `output`, raiz do projeto, pasta pai e Downloads/Documentos/Desktop.
2. Se a pasta `output/pesquisa_permanencia_minima` existir com os 75
   CSVs, os arquivos obrigatorios e o registro de SHA-256 intacto,
   recompacota a base sem refazer treinamento ou alterar cotacoes.
3. Caso contrario, procura em `pesquisa_vantagem_rotacao*.zip` de
   execucoes anteriores pelo membro
   `dados_origem/pesquisa_permanencia_minima.zip`, valida CRC,
   arquivos obrigatorios, 75 series e manifesto concluido (3 folds),
   e salva automaticamente a copia recuperada com o nome canonico
   `output/pesquisa_permanencia_minima.zip`.
4. Quando nao existe fonte completa, mas ha um ZIP de backtest
   atualizado, executa apenas a auditoria observacional da trava,
   publica `output/pesquisa_vantagem_rotacao.zip` e grava
   `simulacao_contrafactual_executada=false` no manifesto; NAO
   inventa resultado de capital com novas rotacoes.
5. Se nenhum desses dados estiver disponivel, informa precisamente
   que nao ha dados suficientes, sem sobrescrever a pesquisa anterior.

Teste automatizado de ZIP aninhado, checksum invalidado, diretorio
completo, fallback do replay e modos preexistentes. **Nenhuma**
alteracao no modelo Winner, na operacao Live ou na branch principal.

Nota metodologica: a base congelada antiga nao e igual ao snapshot
atualizado. Ela permite testar logicamente o efeito de ordens
alternativas, mas seus capitais historicos nao substituem o Control
atualizado de US$ 139.871.072,31.
