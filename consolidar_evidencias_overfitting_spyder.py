"""Consolidacao retrospectiva do TCC; NAO e um holdout prospectivo.

Spyder: ajustar os dois caminhos abaixo se necessario e executar F5.
Usa SOMENTE artefatos de auditorias anteriores, sem consultas Alpaca,
fit, calibracao, replay ou modificacoes de dados/estrategia.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERSAO = "1.25.5"
PACOTE_GENERALIZACAO = None  # Ou Path(r"C:\...\pacote_generalizacao_587m_v1254.zip")
PACOTE_REPRODUCAO = None     # Ou Path(r"C:\...\pacote_reproducao_universo.zip")
SAIDA = ROOT / "output" / "consolidacao_overfitting_tcc"
SNAPSHOT_ORIGINAL = "e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575"
CAPITAL_HISTORICO = 587165941.1330212
CAPITAL_RENOVADO = 89941496.49095525
PBO_LOG_GROWTH = 53.57
PBO_SHARPE = 59.52


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encontrar(configurado, padroes, rotulo):
    if configurado is not None:
        path = Path(configurado).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"{rotulo}: caminho configurado ausente: {path}")
    for base in (ROOT / "output", ROOT):
        if not base.exists():
            continue
        for pattern in padroes:
            candidatos = sorted(base.rglob(pattern), key=lambda p: p.stat().st_mtime,
                                reverse=True)
            if candidatos:
                return candidatos[0]
    raise FileNotFoundError(
        f"{rotulo}: pacote ausente. Configure PACOTE_{rotulo} no inicio do script."
    )


def ler_csv(z, nome):
    return pd.read_csv(io.BytesIO(z.read(nome)))


def verificar(pgen, prep):
    with ZipFile(pgen) as zg, ZipFile(prep) as zr:
        gmeta = json.loads(zg.read("resumo_generalizacao.json"))
        rmeta = json.loads(zr.read("reproducao_universo.json"))
        g = ler_csv(zg, "model_generalization_225.csv")
        pred = ler_csv(zr, "predictions.csv")
        rot = ler_csv(zr, "graficos/capital_rotations_control.csv")

    if gmeta.get("status") != "completed" or gmeta.get("model_count") != 225:
        raise RuntimeError("Auditoria de generalizacao incompleta.")
    if gmeta.get("historical_snapshot_sha256") != SNAPSHOT_ORIGINAL:
        raise RuntimeError("Vintage historico incorreto na generalizacao.")
    if rmeta.get("data", {}).get("snapshot_sha256") != SNAPSHOT_ORIGINAL:
        raise RuntimeError("Vintage historico incorreto na reproducao.")
    if not np.isclose(float(rmeta["checkpoint"]["observed_capital"]),
                      CAPITAL_HISTORICO, atol=0.01, rtol=0):
        raise RuntimeError("O capital nao reproduziu o checkpoint historico.")
    if len(g) != 225 or g.duplicated(["asset", "fold_id"]).any():
        raise RuntimeError("Esperados 225 pares exclusivos de ativo/fold.")
    if g.asset.nunique() != 75 or set(g.fold_id) != {1, 2, 3}:
        raise RuntimeError("Universo/folds de generalizacao divergentes.")
    if not {"train_rmse", "oos_rmse", "oos_skill_vs_naive",
            "overfit_warning"}.issubset(g):
        raise RuntimeError("Colunas preditivas essenciais ausentes.")
    if g[["train_rmse", "oos_rmse"]].isna().any().any():
        raise RuntimeError("RMSE ausente; nao preencher valores.")
    if not (g["train_rmse"] > 0).all() or not (g["oos_rmse"] >= 0).all():
        raise RuntimeError("RMSE negativo ou treino zero.")
    if len(pred) != 1560 or pred[["strategy_equity", "buy_hold_equity"]].isna().any().any():
        raise RuntimeError("Serie financeira incompleta.")
    if not np.isclose(pred["strategy_equity"].iloc[-1], CAPITAL_HISTORICO,
                      rtol=0, atol=0.01):
        raise RuntimeError("Capital da curva nao reconcilia ao checkpoint.")
    if len(rot) != 355 or rot["rotation_value_added"].notna().sum() != 354:
        raise RuntimeError("Contagem de rotacoes ou contrafactuais divergente.")
    return g, pred, rot, rmeta


def salvar(fig, pasta, nome):
    fig.tight_layout()
    fig.savefig(pasta / nome, dpi=165, bbox_inches="tight")
    plt.close(fig)


def gerar(g, pred, rot, pasta):
    filenames = []
    # Figura 1: diagnostico treino-teste por modelo.
    fig, ax = plt.subplots(figsize=(8, 7))
    for fold, part in g.groupby("fold_id"):
        ax.scatter(part.train_rmse, part.oos_rmse, label=f"Fold {int(fold)}",
                   s=26, alpha=.65)
    lim = max(float(g.train_rmse.max()), float(g.oos_rmse.max())) * 1.04
    ax.plot([0, lim], [0, lim], "--", linewidth=1, label="Treino = OOS")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="RMSE no treino",
           ylabel="RMSE fora da amostra (OOS)",
           title="1. Generalizacao dos 225 modelos LightGBM")
    ax.legend()
    filenames.append("01_treino_vs_oos.png"); salvar(fig, pasta, filenames[-1])

    # Figura 2: distribuicao real dos gaps, nao porcentagem de overfitting.
    g = g.copy()
    g["gap_percentual"] = 100 * (g.oos_rmse / g.train_rmse - 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    parts = [g.loc[g.fold_id == fold, "gap_percentual"] for fold in (1, 2, 3)]
    ax.boxplot(parts, tick_labels=["Fold 1", "Fold 2", "Fold 3"], showfliers=True)
    ax.axhline(0, ls="--", linewidth=1)
    ax.set(ylabel="Gap percentual = (RMSE OOS / treino - 1) x 100",
           title="2. Distribuicao do gap de generalizacao por fold")
    filenames.append("02_gap_por_fold.png"); salvar(fig, pasta, filenames[-1])

    # Figura 3: comparador especifico: a media do alvo de treino, NAO zero.
    fig, ax = plt.subplots(figsize=(9, 5))
    positive = [int((g.loc[g.fold_id == f, "oos_skill_vs_naive"] > 0).sum())
                for f in (1, 2, 3)]
    negative = [75-x for x in positive]
    positions = np.arange(3)
    ax.bar(positions, positive, label="LightGBM melhor que media do treino")
    ax.bar(positions, negative, bottom=positive,
           label="LightGBM igual/pior que media do treino")
    ax.set(xticks=positions, xticklabels=["Fold 1", "Fold 2", "Fold 3"],
           ylim=(0, 83), ylabel="Quantidade de modelos (de 75)",
           title="3. Habilidade preditiva fora da amostra")
    ax.legend(fontsize=9)
    filenames.append("03_skill_oos_vs_media.png"); salvar(fig, pasta, filenames[-1])

    # Figura 4: evidencia economica local (mesma janela da posicao apos a troca).
    valid = rot.loc[rot.rotation_value_added.notna(), "rotation_value_added"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valid * 100, bins=35)
    ax.axvline(0, ls="--", linewidth=1, label="Empate com manter")
    ax.set(xlabel="Retorno adicional bruto da troca (pontos percentuais)",
           ylabel="Rotacoes", title="4. Trocar ativo versus manter o anterior")
    ax.legend()
    filenames.append("04_valor_adicional_das_rotacoes.png"); salvar(fig, pasta, filenames[-1])

    # Figura 5: capital do backtest com benchmark, SEM selo de independencia.
    fig, ax = plt.subplots(figsize=(11, 5))
    datas = pd.to_datetime(pred["timestamp"], utc=True)
    ax.plot(datas, pred.strategy_equity, label="Politica LightGBM")
    ax.plot(datas, pred.buy_hold_equity, label="Comprar/manter igual peso")
    ax.set_yscale("log")
    ax.set(xlabel="Datas das decisoes OOS internas", ylabel="Capital simulado (US$, log)",
           title="5. Capital historico composto (universo escolhido retrospectivamente)")
    ax.legend()
    filenames.append("05_capital_backtest_e_benchmark.png"); salvar(fig, pasta, filenames[-1])

    # Figura 6: probabilidades estimadas na pesquisa ANTERIOR, nao nova inferencia.
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(["Log-growth", "Sharpe"], [PBO_LOG_GROWTH, PBO_SHARPE])
    ax.axhline(50, ls="--", linewidth=1, label="50% (referencia visual)")
    for bar, val in zip(bars, [PBO_LOG_GROWTH, PBO_SHARPE]):
        ax.text(bar.get_x()+bar.get_width()/2, val+1, f"{val:.2f}%",
                ha="center")
    ax.set(ylim=(0, 100), ylabel="PBO do processo de selecao (%)",
           title="6. Risco de sobreajuste na selecao de subconjuntos (v1.4.1)")
    ax.legend()
    filenames.append("06_pbo_historico_selecao.png"); salvar(fig, pasta, filenames[-1])

    # Figura 7: mudanca de versao do historico, NAO cambio de mercado em 2020-26.
    fig, ax = plt.subplots(figsize=(9, 5))
    cap = np.array([CAPITAL_HISTORICO, CAPITAL_RENOVADO])/1e6
    bars = ax.bar(["Vintage congelado", "Historico Alpaca renovado"], cap)
    for bar, val in zip(bars, cap):
        ax.text(bar.get_x()+bar.get_width()/2, val+8,
                f"US$ {val:.2f} mi", ha="center")
    ax.set(ylim=(0, 650), ylabel="Capital final simulado (US$ milhoes)",
           title="7. Sensibilidade a revisao de 73 barras de maio/2018")
    filenames.append("07_sensibilidade_vintage.png"); salvar(fig, pasta, filenames[-1])
    return filenames, g, valid


def relatorio(g, valid, arquivos, meta, pgen, prep):
    positive = int((g.oos_rmse > g.train_rmse).sum())
    warns = int(g.overfit_warning.astype(str).str.lower().eq("true").sum())
    wins = int((valid > 0).sum())
    median = float((100*(g.oos_rmse / g.train_rmse - 1)).median())
    descricoes = [
        ("01_treino_vs_oos.png", "Cada ponto e um ativo num fold. Acima da diagonal, RMSE OOS maior que treino. Nao mede retorno financeiro."),
        ("02_gap_por_fold.png", "Cada caixa representa 75 gaps percentuais; linha horizontal zero = erros iguais. A mediana e uma variacao relativa de erro, NAO a probabilidade de sobreajuste."),
        ("03_skill_oos_vs_media.png", "Conta modelos que superaram ou nao a previsao constante da MEDIA dos rotulos do treino em OOS. Nao usa baseline zero."),
        ("04_valor_adicional_das_rotacoes.png", "Compara o retorno do ativo escolhido com manter o ativo anterior pelo mesmo tempo. Valores positivos favorecem a troca; calculo contrafactual local bruto, nao causalidade global nem robustez prospectiva."),
        ("05_capital_backtest_e_benchmark.png", "Capital composto historico versus buy-and-hold de igual peso nos mesmos 75 ativos. Eixo vertical logaritmico. Os ativos foram selecionados usando a janela historica: NAO e validacao prospectiva independente."),
        ("06_pbo_historico_selecao.png", "Reproduz as probabilidades historicas estimadas por CSCV sobre 180 subconjuntos e 252 particoes. Quantifica fragilidade do processo de selecao, NAO chance de cada ordem estar errada."),
        ("07_sensibilidade_vintage.png", "Compara capital final com dois vintages historicos: 73 OHLCV de maio/2018 mudaram, mas as operacoes OOS seguiram outras rotas. Nao mostra dinheiro real mudando retroativamente."),
    ]
    if [a for a,_ in descricoes] != arquivos:
        raise RuntimeError("Descricoes de grafico nao correspondem aos arquivos.")
    lines = [
        "# Consolidacao da evidencia de overfitting — v1.25.5", "",
        "**Natureza da evidencia:** auditoria retrospectiva; nao foi criado novo holdout independente.",
        "**Fontes:** snapshots e relatorios ja congelados; sem retreino/backtest ou consultas externas.", "",
        "## Numeros medidos", "",
        f"- LightGBM: {len(g)} modelos; {positive} com RMSE OOS superior ao treino; gap mediano {median:.2f}%.",
        f"- Alertas de overfit segundo regra pre-registrada: {warns}/{len(g)}.",
        f"- Rotacoes com retorno bruto superior a manter: {wins}/{len(valid)} ({wins/len(valid):.1%}); mediana {valid.median():+.2%} de retorno adicional. Comparacao local de horizonte, nao simulacao de uma estrategia alternativa de manter sempre.",
        f"- Capital historico: US$ {CAPITAL_HISTORICO:,.2f}; snapshot renovado (processamento legado): US$ {CAPITAL_RENOVADO:,.2f} (documentos de auditoria de vintages distintos).",
        f"- PBO historico (v1.4.1): log-growth {PBO_LOG_GROWTH:.2f}% e Sharpe {PBO_SHARPE:.2f}%.", "",
        "## Leitura de cada grafico", ""
    ]
    for nome,explicacao in descricoes:
        lines.extend([f"### {nome}", "", explicacao, "", f"![{nome}]({nome})", ""])
    lines += [
        "## Resposta sobre a validacao independente", "",
        "As janelas 2020–2026 sao OOS INTERNAS em cada fold de treinamento. O universo de ativos, a configuracao e a busca de subconjuntos foram avaliados retrospectivamente nessa janela. Portanto esta consolidacao NAO certifica um teste economicamente independente do processo completo de desenvolvimento.",
        "", "Para responder prospectivamente, e necessario congelar por escrito, ANTES de novas decisoes, modelos, dados de entrada, politica de rotacao, custos e benchmark; acumular decisoes e execucoes em um periodo futuro nao usado na pesquisa; so entao calcular capital, drawdown, excessos e incerteza. Nenhum resultado futuro e presumido aqui.",
        "", "## Procedencia", "",
        f"- {pgen.name}: SHA256 {sha256(pgen)}.",
        f"- {prep.name}: SHA256 {sha256(prep)}.",
        "- PBO: MANIFEST.md, secoes 25.5 e 25.8, versao historica v1.4.1.",
        "- Vintage atualizado: evidencias/auditoria_contrafactual_2018_v1237.md.",
        "- Gap: evidencias/resultado_generalizacao_587m_v1254.md.",
        "", "**Nenhuma porcentagem do patrimonio pode ser atribuida causalmente a overfitting com estas medidas.**", ""
    ]
    return "\n".join(lines)


def main():
    pgen = encontrar(PACOTE_GENERALIZACAO,
                     ["pacote_generalizacao_587m_v1254.zip"], "GENERALIZACAO")
    prep = encontrar(PACOTE_REPRODUCAO,
                     ["pacote_reproducao_universo.zip", "pacote_reproducao_universo(*).zip"],
                     "REPRODUCAO")
    print(f"[inputs] generalizacao={pgen}\n[inputs] reproducao={prep}", flush=True)
    g, pred, rot, meta = verificar(pgen, prep)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pasta = SAIDA / run
    pasta.mkdir(parents=True, exist_ok=False)
    arquivos, g, valid = gerar(g, pred, rot, pasta)
    (pasta / "LEIA_ME_GRAFICOS.md").write_text(
        relatorio(g, valid, arquivos, meta, pgen, prep), encoding="utf-8")
    info = {
        "version": VERSAO, "status": "completed", "prospective_independent_test": False,
        "data_vintage": SNAPSHOT_ORIGINAL, "models": len(g),
        "rmse_gap_median_percent": float(g.gap_percentual.median()),
        "rmse_oos_gt_train_count": int((g.oos_rmse > g.train_rmse).sum()),
        "rotation_count": len(valid), "rotation_better_than_hold": int((valid > 0).sum()),
        "historical_pbo_log_growth_percent": PBO_LOG_GROWTH,
        "historical_pbo_sharpe_percent": PBO_SHARPE,
        "sha256_generalization_zip": sha256(pgen),
        "sha256_reproduction_zip": sha256(prep),
        "charts": arquivos,
    }
    (pasta / "resumo_consolidado.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    package = pasta / "pacote_consolidacao_overfitting_v1255.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as z:
        for path in sorted(pasta.iterdir()):
            if path.is_file() and path != package:
                z.write(path, path.name)
    print(f"[done] pacotes verificados; {len(arquivos)} graficos; pasta={pasta}")
    print(f"[done] pacote={package}", flush=True)
    return info


if __name__ == "__main__":
    main()
