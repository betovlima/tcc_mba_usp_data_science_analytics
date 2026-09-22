"""Reproducao oficial do experimento do TCC MBA USP.

Este arquivo foi desenhado para o Spyder:
- cada bloco "# %%" e uma sessao independente;
- execute de cima para baixo;
- as variaveis permanecem disponiveis no Variable Explorer.

Executado como script, os mesmos blocos rodam sequencialmente.

Fluxo oficial:
Alpaca RAW/SIP -> CSV por ativo -> Corporate Actions CSV por ativo ->
integridade -> exclusao estrutural -> normalizacao de splits -> folds ->
LightGBM CPU -> Control -> Soft Horizon Consensus -> comparacao.
"""

# %% 0 - Imports, caminhos e configuracao congelada
from pathlib import Path
import time

from reproducao.artefatos import save_results
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.experimento import (
    build_folds,
    build_variant_configs,
    compare_variants,
    run_variant,
)
from reproducao.preparacao import prepare_model_frames
from engine.config import (
    ANALYSIS_END_DATE,
    ASSETS,
    BAR_SNAPSHOT_AS_OF_END,
    CONFIG,
    EXPERIMENT_VERSION,
    SOFT_HORIZON_CONSENSUS_PENALTY,
    START_DATE,
)

RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHOS_PESQUISA = SnapshotPaths.research(RAIZ_PROJETO)
CAMINHOS_TEMPORARIOS = SnapshotPaths.temporary(RAIZ_PROJETO)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "reproducao_v1"

# True = usa exatamente os CSVs versionados em dados/pesquisa_v1.
USAR_DADOS_PESQUISA_CONGELADOS = False

# Quando USAR_DADOS_PESQUISA_CONGELADOS=False:
# False = reutiliza arquivos existentes em dados/temporario/ quando disponiveis.
# True = apaga somente dados/temporario/ e baixa novamente todos os arquivos.
# Esta chave nunca altera dados/pesquisa_v1.
FORCAR_DOWNLOAD = False

print("=" * 78, flush=True)
print("TCC MBA USP - reproducao cientifica independente", flush=True)
print(f"versao={EXPERIMENT_VERSION} backend=CPU", flush=True)
print(f"periodo={START_DATE} -> {ANALYSIS_END_DATE}", flush=True)
print(f"bar_snapshot_as_of={BAR_SNAPSHOT_AS_OF_END}", flush=True)
print(f"ativos_solicitados={len(ASSETS)}", flush=True)
print("banco_de_dados=NAO", flush=True)
print("comparacao=CONTROL vs SOFT_HORIZON_CONSENSUS", flush=True)
print("=" * 78, flush=True)


# %% 1 - Origem dos dados
if USAR_DADOS_PESQUISA_CONGELADOS and FORCAR_DOWNLOAD:
    raise RuntimeError(
        "Nao use FORCAR_DOWNLOAD=True junto com "
        "USAR_DADOS_PESQUISA_CONGELADOS=True."
    )

if USAR_DADOS_PESQUISA_CONGELADOS:
    CAMINHOS = CAMINHOS_PESQUISA
    credenciais = None
    print("[snapshot] modo=pesquisa-versionada", flush=True)
    validate_snapshot(CAMINHOS)
else:
    CAMINHOS = CAMINHOS_TEMPORARIOS
    credenciais = load_alpaca_credentials(RAIZ_PROJETO)
    if FORCAR_DOWNLOAD:
        print("[snapshot] modo=download-temporario-forcado", flush=True)
        CAMINHOS.clear_generated()
    else:
        print("[snapshot] modo=download-temporario-reutilizavel", flush=True)
        CAMINHOS.ensure()


# %% 2 - Download OHLCV RAW/SIP: um CSV por ativo
inicio_barras = time.perf_counter()
if credenciais is None:
    arquivos_raw = {
        ativo: CAMINHOS.raw_bars / f"{ativo}.csv"
        for ativo in ASSETS
    }
    manifesto_existente = validate_snapshot(CAMINHOS)
else:
    arquivos_raw = download_raw_bars(
        credenciais,
        CAMINHOS,
        assets=ASSETS,
        replace=FORCAR_DOWNLOAD,
    )
print(
    f"[stage] raw-bars completed seconds={time.perf_counter() - inicio_barras:.3f}",
    flush=True,
)


# %% 3 - Corporate Actions: um CSV por ativo
inicio_ca = time.perf_counter()
if credenciais is None:
    arquivos_eventos = {
        ativo: CAMINHOS.corporate_actions / f"{ativo}.csv"
        for ativo in ASSETS
    }
else:
    arquivos_eventos = download_corporate_actions(
        credenciais,
        CAMINHOS,
        assets=ASSETS,
        replace=FORCAR_DOWNLOAD,
    )
print(
    f"[stage] corporate-actions completed seconds={time.perf_counter() - inicio_ca:.3f}",
    flush=True,
)


# %% 4 - Congelamento e SHA-256 do snapshot
if credenciais is None:
    manifesto = validate_snapshot(CAMINHOS)
else:
    manifesto = build_snapshot_manifest(
        CAMINHOS,
        arquivos_raw,
        arquivos_eventos,
        credentials=credenciais,
    )
    manifesto = validate_snapshot(CAMINHOS)


# %% 5 - Preparacao dos dados
# Regras:
# 1) problema estrutural de identidade => exclusao, sem bridge/reconstrucao;
# 2) splits => normalizacao local equivalente a linha 10.8.74/10.8.84;
# 3) somente OHLCV entra no modelo.
inicio_preparacao = time.perf_counter()
frames, exclusoes, diagnosticos_dados, auditoria_dados = prepare_model_frames(
    CAMINHOS,
    assets=ASSETS
)
print(
    f"[stage] preparation completed eligible={len(frames)} "
    f"seconds={time.perf_counter() - inicio_preparacao:.3f}",
    flush=True,
)

# Variavel util no Spyder.
ativos_elegiveis = tuple(frames)


# %% 6 - Configuracoes experimentais e folds walk-forward
config_control, config_soft = build_variant_configs(frames, CONFIG)

# Inspecione estas duas variaveis no Spyder.
control_soft_flag = config_control.research_model_settings["soft_horizon_consensus"]
soft_soft_flag = config_soft.research_model_settings["soft_horizon_consensus"]

print(f"[variant] CONTROL soft={control_soft_flag}", flush=True)
print(f"[variant] SOFT soft={soft_soft_flag}", flush=True)
print(
    f"[variant] horizons={config_soft.rotation_target_horizons} "
    f"weights={config_soft.rotation_target_horizon_weights} "
    f"penalty={SOFT_HORIZON_CONSENSUS_PENALTY}",
    flush=True,
)

datas_comuns, folds = build_folds(frames, config_control)
print(
    f"[folds] common_dates={len(datas_comuns)} folds={len(folds)} "
    f"first={datas_comuns.min().date()} last={datas_comuns.max().date()}",
    flush=True,
)


# %% 7 - CONTROL
# Control = LightGBM + politica-base, SEM Soft Horizon Consensus.
inicio_control = time.perf_counter()
control_result, control_metrics = run_variant(
    "CONTROL",
    frames,
    config_control,
    folds,
)
print(
    f"[stage] CONTROL total_seconds={time.perf_counter() - inicio_control:.3f}",
    flush=True,
)


# %% 8 - SOFT HORIZON CONSENSUS
# Soft = mesmo Control + modificador continuo multi-horizonte.
inicio_soft = time.perf_counter()
soft_result, soft_metrics = run_variant(
    "SOFT_HORIZON_CONSENSUS",
    frames,
    config_soft,
    folds,
)
print(
    f"[stage] SOFT_HORIZON_CONSENSUS "
    f"total_seconds={time.perf_counter() - inicio_soft:.3f}",
    flush=True,
)


# %% 9 - Comparacao Control vs Soft
comparacao = compare_variants(
    control_metrics,
    soft_metrics,
)

print(
    "[comparison] "
    f"CONTROL={comparacao['control_ending_capital']:,.2f} "
    f"SOFT={comparacao['soft_ending_capital']:,.2f} "
    f"delta={comparacao['soft_minus_control_capital']:,.2f} "
    f"ratio={comparacao['soft_vs_control_ratio']:+.4%}",
    flush=True,
)
print(
    "[comparison] "
    f"soft_changed_base_actions={comparacao['soft_changed_base_actions']} "
    f"device={comparacao['effective_compute_device']}",
    flush=True,
)


# %% 10 - Exportacao dos artefatos finais
arquivos_resultado = save_results(
    DIRETORIO_RESULTADOS,
    manifest=manifesto,
    exclusions=exclusoes,
    diagnostics=diagnosticos_dados,
    audit=auditoria_dados,
    folds=folds,
    control_result=control_result,
    control_metrics=control_metrics,
    soft_result=soft_result,
    soft_metrics=soft_metrics,
    comparison=comparacao,
)

print("[done] reproducao concluida", flush=True)
