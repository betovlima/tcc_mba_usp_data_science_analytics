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


## 9. Pesquisa diagnóstica OOS v1.25.2 (23/10/2026, branch isolada)

- Branch de trabalho: `research/v1.25.2-diagnosticos-preditivos`, sem merge ou alteração de `main`.
- Escopo: exportação de previsão e alvo por ativo/sessão/fold, métricas MAE/RMSE da utilidade prevista, ranking temporal Spearman/NDCG, auditoria de normalização de Corporate Actions e limites de rótulos entre treino/calibração/teste. A rotina financeira, parâmetros e dados históricos NÃO foram alterados.
- Base pública disponível: executor oficial de reprodução `v1.23.0`, que usa o snapshot U67 congelado e o cache local dos dez candidatos. Os pacotes `v1.25.0` e `v1.25.1` são evidências de outra execução; **o código correspondente não estava disponível no GitHub consultado**, e esta branch NÃO integra aquele executor.
- Pacotes recentes `pacote_reproducao_universo(7).zip` e `(8).zip` são byte a byte idênticos (SHA-256 `f71ebf46b6fbcad1d97e75b720c78e8edad32705fe2d0b31cb5c58d41706ff5d`). Ambos exibem `predictive_audit.status=failed`, mensagem `Cannot losslessly convert units`, sem traceback; logo, não validam o hotfix subsequente nem constituem nova execução.
- Auditorias desses pacotes: 75/75 ativos com consistência interna de normalização das barras e eventos locais; 3 alertas de possível sobreposição de maturação de rótulos nos 3 folds finais, sem comprovação automática de vazamento efetivo. Diagnósticos MAE/RMSE e ranking ainda indisponíveis.
- Hotfix posterior substitui joins e buscas de data com unidades nativas distintas por chaves UTC inteiras e adiciona teste micro/nanossegundos; salva traceback quando ocorrer falha de diagnóstico. Não há evidência de execução completa deste hotfix no Spyder até este registro.
- Para validar a instalação local antes de novo replay, consultar `LEIA_ME_VALIDACAO.md`, conferir o commit Git com `git rev-parse --short HEAD`, reiniciar o kernel do Spyder e testar a função de exportação antes de executar `reproduzir_experimento.py`. Nunca baixar os CSVs congelados só para obter os diagnósticos.
- Critério científico: não usar o capital como prova de qualidade preditiva; não alegar ausência de overfitting sem confirmar generalização OOS e origem temporal dos candidatos.
