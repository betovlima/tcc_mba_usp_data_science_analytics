"""Ablacao cientifica das penalidades da utilidade LightGBM (TCC v1.23.0).

Spyder: execute este arquivo com F5, a partir da branch de pesquisa.
Nao acessa Alpaca, nao atualiza series e nao altera a reproducao oficial.
Requer os MESMOS arquivos congelados e o cache local do experimento oficial.

Desenho: os alvos sao recalculados para cada cenario; treino e previsoes
sao refeitos. Os coeficientes e a regra de calibracao/execucao permanecem
fixos na configuracao ORIGINAL, isolando a mudanca da variavel-alvo.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.rotacao import _drawdown_maximo
from reproducao.dados import validate_snapshot
from reproducao.preparacao import prepare_model_frames
from reproducao.experimento import build_control_config
from reproducao.artefatos import sinal_sonoro_conclusao

# Reuso EXATO das rotinas de treino, calibracao e replay do estudo congelado.
# Importar este modulo nao executa a pesquisa de busca de ativos.
import pesquisar_expansao_u67_spyder as pesquisa


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "ablacao_utilidade_risco"
SCHEMA = "tcc-risk-target-ablation-v1"
TOLERANCIA_CAPITAL = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def salvar_json(path: Path, dados: dict) -> None:
    path.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def verificar_entradas() -> tuple[dict, dict, tuple[str, ...]]:
    contrato = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json")
        .read_text(encoding="utf-8")
    )
    selecionados = tuple(contrato["selected_assets"])
    if selecionados != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES):
        raise RuntimeError("Lista de ativos diferente da reproducao oficial.")
    snapshot = validate_snapshot(pesquisa.BASE_DATA)
    if snapshot.get("snapshot_sha256") != contrato["snapshot_sha256"]:
        raise RuntimeError("Snapshot base diferente. NAO faca download.")
    if tuple(snapshot.get("assets") or ()) != tuple(pesquisa.U67_REQUESTED_ASSETS):
        raise RuntimeError("Ordem dos ativos base diferente.")
    for ativo in selecionados:
        esperado = contrato["candidate_file_hashes"][ativo]
        arquivos = {
            "raw_sha256": pesquisa.FULL_DATA.raw_bars / f"{ativo}.csv",
            "corporate_actions_sha256":
                pesquisa.FULL_DATA.corporate_actions / f"{ativo}.csv",
        }
        for campo, caminho in arquivos.items():
            if not caminho.is_file():
                raise FileNotFoundError(
                    f"Cache congelado ausente: {caminho}. NAO faca download."
                )
            if sha256(caminho) != esperado[campo]:
                raise RuntimeError(
                    f"SHA-256 divergente: {ativo} {campo}. NAO faca download."
                )
    print(
        f"[integridade] base e {len(selecionados)} ativos adicionais "
        "verificados por SHA-256.",
        flush=True,
    )
    return contrato, snapshot, selecionados


def episodios_da_curva(curva: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Episodios contiguos abaixo do pico anterior, em sessoes observadas."""
    serie = curva.astype(float)
    pico = serie.cummax()
    dd = serie / pico - 1.0
    linhas = []
    em_queda = False
    inicio = vale = None
    menor = 0.0
    duracao = 0
    for momento, valor in dd.items():
        if valor < -1e-12:
            if not em_queda:
                em_queda = True
                inicio, vale, menor, duracao = momento, momento, float(valor), 0
            duracao += 1
            if float(valor) < menor:
                menor, vale = float(valor), momento
        elif em_queda:
            linhas.append({
                "inicio": inicio, "fundo": vale, "recuperacao": momento,
                "drawdown_minimo": menor, "sessoes_abaixo_pico": duracao,
                "recuperado": True,
            })
            em_queda = False
    if em_queda:
        linhas.append({
            "inicio": inicio, "fundo": vale, "recuperacao": pd.NaT,
            "drawdown_minimo": menor, "sessoes_abaixo_pico": duracao,
            "recuperado": False,
        })
    resultado = pd.DataFrame(linhas)
    stats = {
        "episodios_total": len(linhas),
        "episodios_recuperados": sum(int(item["recuperado"]) for item in linhas),
        "sessoes_abaixo_pico": int((dd < -1e-12).sum()),
        "max_sessoes_abaixo_pico": max(
            (int(item["sessoes_abaixo_pico"]) for item in linhas), default=0
        ),
    }
    return resultado, stats


def preparar_cenario(
    nome: str,
    coef_downside: float,
    coef_drawdown: float,
    contrato: dict,
    selecionados: tuple[str, ...],
    raw_base: dict[str, pd.DataFrame],
    config_original: object,
    calendario_referencia: pd.DatetimeIndex | None,
) -> tuple[pd.Series, pd.Series, dict]:
    # Somente a construcao dos ROTULOS usa os coeficientes experimentais.
    config_alvo = config_original.copiar_modelo(update={
        "rotation_downside_penalty": coef_downside,
        "rotation_drawdown_penalty": coef_drawdown,
    })
    (
        quadros_base, datas_comuns, fonte_calendario, simbolos_base,
        folds, datas_decisao, decisao_fold, metadados_decisao,
    ) = pesquisa._construir_contexto_execucao(raw_base, config_alvo)
    if calendario_referencia is not None and not datas_decisao.equals(
        calendario_referencia
    ):
        raise RuntimeError(f"{nome}: calendario de teste mudou.")

    # Os treinos usam os rotulos dos quadros_base, mas as regras de calibracao
    # continuam com o config_original em todos os quatro cenarios.
    estados, _, _ = pesquisa._train_baseline_states(
        quadros_base, datas_comuns, simbolos_base, folds, config_original
    )
    adicionais, continuidade, hashes = pesquisa._prepare_candidate_frames(
        list(selecionados), datas_comuns, datas_decisao, config_alvo
    )
    if set(adicionais) != set(selecionados):
        raise RuntimeError(
            f"{nome}: faltam ativos no cache: "
            f"{sorted(set(selecionados) - set(adicionais))}"
        )
    for ativo in selecionados:
        if hashes.get(ativo) != contrato["candidate_file_hashes"][ativo]:
            raise RuntimeError(f"{nome}: hash alterado no processamento: {ativo}")

    modelos_adicionais = {}
    for posicao, ativo in enumerate(selecionados, start=1):
        print(
            f"[{nome}] treino adicional {posicao}/{len(selecionados)}: {ativo}",
            flush=True,
        )
        modelos_adicionais[ativo] = pesquisa._train_candidate_caches(
            ativo, adicionais[ativo], estados, config_original
        )

    resultado, indicadores, calibracao = pesquisa._evaluate_scenario(
        f"ABLACAO_{nome}", list(selecionados),
        quadros_base, adicionais, simbolos_base, estados,
        modelos_adicionais, folds, datas_decisao, decisao_fold,
        metadados_decisao, config_original
    )
    destino = OUT / nome
    destino.mkdir(parents=True, exist_ok=True)

    resultado.predictions.to_csv(
        destino / "predictions.csv", float_format="%.17g", index=True
    )
    resultado.trades.to_csv(
        destino / "trades.csv", float_format="%.17g", index=False
    )
    pd.DataFrame(indicadores.get("folds") or []).to_csv(
        destino / "folds.csv", index=False
    )
    pd.DataFrame(calibracao).to_csv(
        destino / "calibracao_margens.csv", index=False
    )
    continuidade.to_csv(destino / "auditoria_continuidade.csv", index=False)
    curva = resultado.predictions["strategy_equity"].astype(float).rename(nome)
    ativos = resultado.predictions["selected_asset"].fillna("CASH").astype(str).rename(nome)
    episodios, duracoes = episodios_da_curva(curva)
    episodios.to_csv(destino / "episodios_drawdown.csv", index=False)

    dados = {
        "cenario": nome,
        "coef_downside_no_rotulo": coef_downside,
        "coef_drawdown_no_rotulo": coef_drawdown,
        "coeficientes_da_calibracao": {
            "downside": float(config_original.rotation_downside_penalty),
            "drawdown": float(config_original.rotation_drawdown_penalty),
        },
        "capital_final": float(indicadores["ending_capital"]),
        "cagr": float(indicadores["cagr"]),
        "sharpe": float(indicadores["sharpe"]),
        "drawdown_maximo": float(indicadores["maximum_drawdown"]),
        "drawdown_recalculado": float(_drawdown_maximo(curva)),
        "rotacoes": int(indicadores["capital_rotations"]),
        "compras": int(indicadores["simulated_buys"]),
        "vendas": int(indicadores["simulated_sells"]),
        "taxas_total": float(indicadores["total_transaction_fees"]),
        "folds": indicadores.get("folds") or [],
        **duracoes,
        "fonte_calendario": fonte_calendario,
        "sessoes": len(curva),
    }
    salvar_json(destino / "indicadores.json", dados)
    print(
        f"[{nome}] capital=US$ {dados['capital_final']:,.2f} "
        f"sharpe={dados['sharpe']:.4f} dd={dados['drawdown_maximo']:.2%}",
        flush=True,
    )
    return curva.copy(), ativos.copy(), dados


def graficos_reais(curvas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for coluna in curvas:
        ax.plot(curvas.index, curvas[coluna], label=coluna, lw=1.4)
    ax.set_yscale("log")
    ax.set_title("Evolucao patrimonial por cenario - escala logaritmica")
    ax.set_ylabel("Patrimonio (US$, escala log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.22)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "figura_01_capital_comparado.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for coluna in curvas:
        dd = curvas[coluna] / curvas[coluna].cummax() - 1.0
        ax.plot(curvas.index, dd * 100.0, label=coluna, lw=1.2)
    ax.set_title("Drawdowns de patrimonio por cenario")
    ax.set_ylabel("Queda em relacao ao pico (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.22)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "figura_02_drawdown_comparado.png", dpi=170)
    plt.close(fig)


def main() -> None:
    inicio = time.perf_counter()
    contrato, snapshot, selecionados = verificar_entradas()
    raw_base, exclusoes, _, _ = prepare_model_frames(
        pesquisa.BASE_DATA,
        assets=tuple(pesquisa.U67_REQUESTED_ASSETS),
        csv_float_precision="round_trip",
    )
    if {item["symbol"] for item in exclusoes} != set(
        pesquisa.U67_EXPECTED_EXCLUSIONS
    ):
        raise RuntimeError("Exclusoes de identidade divergentes.")
    config_original = build_control_config(raw_base, pesquisa.CONFIG).copiar_modelo(
        update={"analysis_end_date": contrato["analysis_end_date"]}
    )
    d = float(config_original.rotation_downside_penalty)
    m = float(config_original.rotation_drawdown_penalty)
    cenarios = [
        ("A_ORIGINAL", d, m),
        ("B_SEM_DOWNSIDE", 0.0, m),
        ("C_SEM_DRAWDOWN", d, 0.0),
        ("D_SEM_AMBAS", 0.0, 0.0),
    ]
    # Nenhuma pesquisa de ativos, nenhuma chamada Alpaca.
    OUT.mkdir(parents=True, exist_ok=True)
    comparacao = []
    curvas = {}
    escolhas = {}
    calendario_referencia = None
    capital_esperado = float(contrato["expected_ending_capital"])

    for nome, coef_d, coef_m in cenarios:
        print(
            f"\n[ablacao] {nome} alvo: downside={coef_d} drawdown={coef_m}",
            flush=True,
        )
        curva, ativos, dados = preparar_cenario(
            nome, coef_d, coef_m, contrato, selecionados, raw_base,
            config_original, calendario_referencia
        )
        if nome == "A_ORIGINAL":
            if not math.isclose(
                dados["capital_final"], capital_esperado,
                rel_tol=0.0, abs_tol=TOLERANCIA_CAPITAL
            ):
                raise RuntimeError(
                    "Checkpoint oficial divergente: "
                    f"{dados['capital_final']:.8f} vs {capital_esperado:.8f}. "
                    "ABORTADO: nao interprete as ablacoes."
                )
            calendario_referencia = pd.DatetimeIndex(curva.index)
            print("[controle] capital oficial reproduzido a centavos.", flush=True)
        else:
            if not curva.index.equals(calendario_referencia):
                raise RuntimeError(f"{nome}: eixo temporal divergente.")
        curvas[nome] = curva
        escolhas[nome] = ativos
        comparacao.append({k: v for k, v in dados.items() if k != "folds"})
        pd.DataFrame(comparacao).to_csv(
            OUT / "comparacao_parcial.csv", index=False
        )
        del curva, ativos
        gc.collect()

    curvas_df = pd.DataFrame(curvas)
    curvas_df.to_csv(
        OUT / "curvas_patrimoniais.csv", float_format="%.17g"
    )
    original = escolhas["A_ORIGINAL"]
    mudancas = pd.DataFrame(index=original.index)
    mudancas["ativo_original"] = original
    for nome, _, _ in cenarios[1:]:
        mudancas[f"ativo_{nome}"] = escolhas[nome]
        mudancas[f"divergiu_{nome}"] = original.ne(escolhas[nome])
    mudancas.to_csv(OUT / "decisoes_comparadas.csv", index_label="timestamp")
    tabela = pd.DataFrame(comparacao)
    base = next(row for row in comparacao if row["cenario"] == "A_ORIGINAL")
    tabela["diferenca_capital_vs_original"] = (
        tabela["capital_final"] - base["capital_final"]
    )
    tabela["diferenca_sharpe_vs_original"] = (
        tabela["sharpe"] - base["sharpe"]
    )
    tabela["diferenca_drawdown_vs_original"] = (
        tabela["drawdown_maximo"] - base["drawdown_maximo"]
    )
    tabela["delta_crescimento_log"] = np.log(
        tabela["capital_final"] / base["capital_final"]
    )
    tabela["sessoes_decisao_diferente"] = [
        0 if nome == "A_ORIGINAL"
        else int(mudancas[f"divergiu_{nome}"].sum())
        for nome in tabela["cenario"]
    ]
    tabela.to_csv(OUT / "comparacao_cenarios.csv", index=False)
    graficos_reais(curvas_df)
    salvar_json(OUT / "manifesto_execucao.json", {
        "schema": SCHEMA,
        "desenho": "ablacao SOMENTE do rotulo; calibracao e execucao fixas",
        "reproducao_oficial_preservada": True,
        "checkpoint_exato_centavos": True,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "candidate_file_hashes": contrato["candidate_file_hashes"],
        "analysis_end_date": contrato["analysis_end_date"],
        "capital_oficial_esperado": capital_esperado,
        "cenarios": [
            {"nome": nome, "downside": coefd, "drawdown": coefm}
            for nome, coefd, coefm in cenarios
        ],
        "tempo_total_segundos": time.perf_counter() - inicio,
        "avisos": [
            "Diferencas nao provam validade prospectiva.",
            "Universo retrospectivamente fixado e execucao simulada.",
            "Os componentes de captura e persistencia continuam no rotulo.",
        ],
    })
    pacote = OUT.parent / "pacote_ablacao_utilidade_risco.zip"
    with zipfile.ZipFile(pacote, "w", zipfile.ZIP_DEFLATED) as zipf:
        for caminho in sorted(OUT.rglob("*")):
            if caminho.is_file():
                zipf.write(caminho, caminho.relative_to(OUT.parent))
    print("\n[concluido] comparacao dos quatro cenarios:", flush=True)
    print(tabela[[
        "cenario", "capital_final", "sharpe", "drawdown_maximo",
        "sessoes_decisao_diferente",
    ]].to_string(index=False), flush=True)
    print(f"[pacote] {pacote}", flush=True)
    sinal_sonoro_conclusao()


if __name__ == "__main__":
    main()
