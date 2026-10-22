"""Diagnostico cientifico de generalizacao LightGBM, TCC 2026.

Execucao Spyder: abra este arquivo e pressione F5, na branch de pesquisa.
Leitura EXCLUSIVA dos snapshots congelados, sem rede/Alpaca ou banco de dados.
Nao altera treino oficial, politica, parametros, dados nem a pasta de reproducao.

O teste final nao e usado para escolher parametros. Curvas por arvores e
tamanho de treino usam exclusivamente a calibracao cronologica anterior.
O alvo observado pode estar ausente nas ultimas 60 sessoes: nunca preencher.
"""
from __future__ import annotations

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

from engine.rotacao import ROTATION_FEATURES
from engine.modelo_lightgbm import _ajustar_modelos_lightgbm
from reproducao.dados import validate_snapshot
from reproducao.preparacao import prepare_model_frames
from reproducao.experimento import build_control_config
from reproducao.artefatos import sinal_sonoro_conclusao
import pesquisar_expansao_u67_spyder as pesquisa

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "diagnosticos_generalizacao"
PACOTE = ROOT / "output" / "pacote_diagnosticos_generalizacao.zip"
ALVO = "forward_risk_adjusted_utility"
AMOSTRA_GRAFICOS = ("AAPL", "NVDA", "SPY", "TSLA")
ESTAGIOS = (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 329)
FRACOES_TREINO = (0.65, 0.80, 1.0)
SCHEMA = "tcc-lightgbm-generalization-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def validar_snapshot() -> tuple[dict, dict, tuple[str, ...]]:
    contrato = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json")
        .read_text(encoding="utf-8")
    )
    selecionados = tuple(contrato["selected_assets"])
    if selecionados != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES):
        raise RuntimeError("Identidade dos ativos diferente do contrato oficial.")
    snapshot = validate_snapshot(pesquisa.BASE_DATA)
    if (snapshot.get("snapshot_sha256") != contrato["snapshot_sha256"]
            or tuple(snapshot.get("assets") or ()) != tuple(pesquisa.U67_REQUESTED_ASSETS)):
        raise RuntimeError("Snapshot base divergente; nao faca download.")
    for ativo in selecionados:
        esperado = contrato["candidate_file_hashes"][ativo]
        for tipo, caminho in {
            "raw_sha256": pesquisa.FULL_DATA.raw_bars / f"{ativo}.csv",
            "corporate_actions_sha256": pesquisa.FULL_DATA.corporate_actions / f"{ativo}.csv",
        }.items():
            if not caminho.is_file() or sha256(caminho) != esperado[tipo]:
                raise RuntimeError(f"Arquivo congelado ausente/divergente: {ativo}, {tipo}")
    print("[integridade] dados congelados confirmados por SHA-256", flush=True)
    return contrato, snapshot, selecionados


def medidas(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return {"n": 0, "mae": np.nan, "rmse": np.nan}
    erro = p[ok] - y[ok]
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(erro))),
        "rmse": float(np.sqrt(np.mean(erro * erro))),
    }


def amostra(frame: pd.DataFrame, datas: pd.DatetimeIndex) -> pd.DataFrame:
    cols = [*ROTATION_FEATURES, ALVO]
    x = frame.reindex(datas)[cols]
    return x.dropna(subset=cols)


def ranking_sessao(data: pd.DataFrame) -> dict:
    """Comparacao transversal; alvo futuro usado SOMENTE para diagnostico."""
    vals = data[["previsto", "observado"]].to_numpy(dtype=float)
    n = len(vals)
    if n < 3:
        return {"n_ativos": n, "spearman": np.nan,
                "pares_validos": 0, "concordantes": 0, "concordancia": np.nan}
    rho = pd.Series(vals[:, 0]).rank().corr(pd.Series(vals[:, 1]).rank())
    p = vals[:, 0]
    y = vals[:, 1]
    tri = np.triu_indices(n, k=1)
    vp = p[tri[0]] - p[tri[1]]
    vy = y[tri[0]] - y[tri[1]]
    ok = (np.abs(vp) > 1e-12) & (np.abs(vy) > 1e-12)
    concordantes = int(((vp[ok] * vy[ok]) > 0).sum())
    total = int(ok.sum())
    return {
        "n_ativos": n, "spearman": float(rho) if pd.notna(rho) else np.nan,
        "pares_validos": total, "concordantes": concordantes,
        "concordancia": concordantes / total if total else np.nan,
    }


def curvas_calibracao(
    frame: pd.DataFrame, simbolo: str, fold: int,
    datas_treino: pd.DatetimeIndex,
    datas_calibracao: pd.DatetimeIndex,
    config: object,
) -> tuple[list[dict], list[dict]]:
    """Modelos diagnosticos; calibracao nao interfere no teste."""
    linhas_estagios, linhas_tamanho = [], []
    for fracao in FRACOES_TREINO:
        n = min(len(datas_treino), max(1, int(round(len(datas_treino) * fracao))))
        subdatas = datas_treino[:n]
        modelos = _ajustar_modelos_lightgbm(
            {simbolo: frame}, [simbolo], subdatas, config,
            phase=f"diagnostic_{simbolo}_fold{fold}_frac{fracao}",
        )
        model = modelos.get(simbolo)
        if model is None:
            linhas_tamanho.append({
                "ativo": simbolo, "fold": fold, "fracao": fracao,
                "datas_treino": len(subdatas), "linhas_validas": 0,
                "rmse_treino": np.nan, "rmse_calibracao": np.nan,
                "observacao": "treino_insuficiente",
            })
            continue
        treino = amostra(frame, subdatas)
        cal = amostra(frame, datas_calibracao)
        y_treino = treino[ALVO].to_numpy(dtype=float)
        y_cal = cal[ALVO].to_numpy(dtype=float)
        final_treino = medidas(y_treino, model.predict(treino[ROTATION_FEATURES]))
        final_cal = medidas(y_cal, model.predict(cal[ROTATION_FEATURES])) if len(cal) else {"rmse": np.nan}
        linhas_tamanho.append({
            "ativo": simbolo, "fold": fold, "fracao": fracao,
            "datas_treino": len(subdatas), "linhas_validas": len(treino),
            "rmse_treino": final_treino["rmse"],
            "rmse_calibracao": final_cal["rmse"],
            "observacao": "ok" if len(cal) else "calibracao_sem_rotulo",
        })
        # Apenas o modelo de treino completo serve para curva por arvores.
        if fracao == 1.0:
            for n_arvores in ESTAGIOS:
                if n_arvores > int(model.n_estimators):
                    continue
                p_t = model.predict(treino[ROTATION_FEATURES], num_iteration=n_arvores)
                rmse_t = medidas(y_treino, p_t)["rmse"]
                rmse_c = (
                    medidas(y_cal, model.predict(cal[ROTATION_FEATURES],
                                                num_iteration=n_arvores))["rmse"]
                    if len(cal) else np.nan
                )
                linhas_estagios.append({
                    "ativo": simbolo, "fold": fold, "arvores": n_arvores,
                    "rmse_treino": rmse_t, "rmse_calibracao": rmse_c,
                    "linhas_treino": len(treino), "linhas_calibracao": len(cal),
                })
    return linhas_estagios, linhas_tamanho


def graficos(qualidade: pd.DataFrame, rank: pd.DataFrame,
             estagios: pd.DataFrame, tamanho: pd.DataFrame,
             predicoes: pd.DataFrame) -> None:
    # 1: mapa de calor do skill fora da amostra, todas as 75 series.
    mat = qualidade.pivot(index="ativo", columns="fold", values="skill_teste")
    fig, ax = plt.subplots(figsize=(10, 16))
    arr = mat.to_numpy(dtype=float)
    dados_validos = arr[np.isfinite(arr)]
    limite = max(0.10, float(np.quantile(np.abs(dados_validos), .95))) if len(dados_validos) else 1.0
    img = ax.imshow(np.ma.masked_invalid(arr), cmap="RdBu", vmin=-limite, vmax=limite,
                    aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(mat.index)), labels=mat.index, fontsize=6.5)
    ax.set_xticks(range(len(mat.columns)), labels=[f"Fold {f}" for f in mat.columns])
    ax.set_title("Skill RMSE OOS vs previsao constante por ativo\n"
                 "Skill = 1 - RMSE_modelo / RMSE_media_treino")
    fig.colorbar(img, ax=ax, label="Positivo: superior ao preditor de referencia")
    fig.tight_layout()
    fig.savefig(OUT / "figura_01_skill_por_ativo_fold.png", dpi=160)
    plt.close(fig)

    # 2: erros de treinamento e teste, comparacao por ativo/fold.
    df = qualidade.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["rmse_treino", "rmse_teste"])
    fig, ax = plt.subplots(figsize=(8, 7))
    for fold, sub in df.groupby("fold"):
        ax.scatter(sub["rmse_treino"], sub["rmse_teste"], label=f"Fold {fold}",
                   s=19, alpha=.65)
    limite = float(max(df[["rmse_treino", "rmse_teste"]].max())) if len(df) else 1.0
    ax.plot([0, limite], [0, limite], linestyle="--", linewidth=1,
            color="grey", label="Igualdade")
    ax.set(xlabel="RMSE de treinamento", ylabel="RMSE fora da amostra",
           title="Generalizacao: treinamento vs teste por ativo e fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figura_02_rmse_treino_teste.png", dpi=170)
    plt.close(fig)

    # 3: curva por arvores, sem consultar os rotulos do teste.
    ativos = sorted(estagios["ativo"].unique()) if not estagios.empty else []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, ativo in zip(axes.flat, ativos):
        sub = estagios[estagios["ativo"] == ativo]
        for fold, grupo in sub.groupby("fold"):
            ax.plot(grupo["arvores"], grupo["rmse_calibracao"],
                    label=f"Validacao fold {fold}", linewidth=1.5)
            ax.plot(grupo["arvores"], grupo["rmse_treino"],
                    linestyle=":", linewidth=1, alpha=.7)
        ax.set(title=ativo, xlabel="Arvores no boosting",
               ylabel="RMSE (solida: calibracao; pontilhada: treino)")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(ativos):]:
        ax.axis("off")
    fig.suptitle("Erro por numero de arvores: treino e calibracao cronologica")
    fig.tight_layout()
    fig.savefig(OUT / "figura_03_erro_por_arvore.png", dpi=170)
    plt.close(fig)

    # 4: curvas de aprendizagem sobre prefixos crescentes do TREINO original.
    ativos = sorted(tamanho["ativo"].unique()) if not tamanho.empty else []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, ativo in zip(axes.flat, ativos):
        sub = tamanho[tamanho["ativo"] == ativo].dropna(subset=["rmse_calibracao"])
        for fold, grupo in sub.groupby("fold"):
            grupo = grupo.sort_values("linhas_validas")
            ax.plot(grupo["linhas_validas"], grupo["rmse_calibracao"],
                    "-o", label=f"Calibracao fold {fold}")
            ax.plot(grupo["linhas_validas"], grupo["rmse_treino"],
                    ":", alpha=.65)
        ax.set(title=ativo, xlabel="Amostras validas no treinamento", ylabel="RMSE")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(ativos):]:
        ax.axis("off")
    fig.suptitle("Curvas de aprendizagem: prefixos historicos do treino")
    fig.tight_layout()
    fig.savefig(OUT / "figura_04_curvas_aprendizagem.png", dpi=170)
    plt.close(fig)

    # 5: qualidade da ordenacao transversal por fold.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    folds = sorted(rank["fold"].unique())
    agregados = rank.groupby("fold", as_index=False).agg(
        spearman=("spearman", "mean"),
        concordantes=("concordantes", "sum"),
        pares=("pares_validos", "sum"),
    )
    agregados["concordancia"] = agregados["concordantes"] / agregados["pares"].replace(0, np.nan)
    for ax, col, titulo in zip(axes, ["spearman", "concordancia"],
                               ["Spearman medio por sessao", "Concordancia entre pares"]):
        ax.bar(agregados["fold"].astype(str), agregados[col])
        ax.set(xlabel="Fold", ylabel=titulo, title=titulo)
        ax.axhline(0.0 if col == "spearman" else .5, linestyle="--",
                   color="grey", linewidth=1)
    fig.suptitle("Ranking de utilidades no teste (somente alvos completos)")
    fig.tight_layout()
    fig.savefig(OUT / "figura_05_ranking_por_fold.png", dpi=170)
    plt.close(fig)

    # 6: previsto x observado no teste, para quatro exemplos predefinidos.
    filtrado = predicoes[predicoes["ativo"].isin(AMOSTRA_GRAFICOS)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, ativo in zip(axes.flat, AMOSTRA_GRAFICOS):
        sub = filtrado[filtrado["ativo"] == ativo]
        ax.scatter(sub["observado"], sub["previsto"], s=8, alpha=.3)
        if len(sub):
            mn = min(sub["observado"].min(), sub["previsto"].min())
            mx = max(sub["observado"].max(), sub["previsto"].max())
            ax.plot([mn, mx], [mn, mx], color="grey", linestyle="--")
        ax.set(title=ativo, xlabel="Utilidade observada completa",
               ylabel="Utilidade prevista OOS")
    fig.suptitle("Utilidade prevista versus observada, teste cronologico")
    fig.tight_layout()
    fig.savefig(OUT / "figura_06_previsto_observado.png", dpi=170)
    plt.close(fig)


def main() -> None:
    inicio = time.perf_counter()
    contrato, snapshot, selecionados = validar_snapshot()
    base_raw, exclusoes, _, _ = prepare_model_frames(
        pesquisa.BASE_DATA, assets=tuple(pesquisa.U67_REQUESTED_ASSETS),
        csv_float_precision="round_trip"
    )
    if {r["symbol"] for r in exclusoes} != set(pesquisa.U67_EXPECTED_EXCLUSIONS):
        raise RuntimeError("Exclusoes de integridade divergentes.")
    config = build_control_config(base_raw, pesquisa.CONFIG).copiar_modelo(
        update={"analysis_end_date": contrato["analysis_end_date"]}
    )
    (base, calendario, fonte, base_symbols, folds, datas_decisao,
     _, _) = pesquisa._construir_contexto_execucao(base_raw, config)
    extras, continuidade, hashes = pesquisa._prepare_candidate_frames(
        list(selecionados), calendario, datas_decisao, config
    )
    if set(extras) != set(selecionados):
        raise RuntimeError("Nem todos os ativos congelados estao disponiveis.")
    if hashes != contrato["candidate_file_hashes"]:
        raise RuntimeError("SHA-256 dos arquivos congelados mudou ao processar.")
    frames = {**base, **extras}
    simbolos = sorted(frames)
    OUT.mkdir(parents=True, exist_ok=True)
    qualidade, rank, estagios, tamanho, predicoes = [], [], [], [], []
    folds_info = []

    for fold in folds:
        fid = int(fold["fold_id"])
        treino_ini = pd.DatetimeIndex(calendario[:int(fold["train_end_index"])])
        cal_ini = pd.DatetimeIndex(
            calendario[int(fold["calibration_start_index"]):
                      int(fold["calibration_end_index"])]
        )
        treino_final = pd.DatetimeIndex(calendario[:int(fold["final_fit_end_index"])])
        teste = pd.DatetimeIndex(fold["decision_dates"])
        print(f"[fold {fid}] treino final: {len(treino_final)} datas; "
              f"calibracao: {len(cal_ini)}; teste: {len(teste)}", flush=True)
        folds_info.append({
            "fold": fid, "n_treino_inicial_datas": len(treino_ini),
            "n_calibracao_datas": len(cal_ini),
            "n_treino_final_datas": len(treino_final),
            "n_teste_datas": len(teste),
            "inicio_teste": str(teste[0]), "fim_teste": str(teste[-1]),
        })
        # Modelos de ajuste FINAL com os mesmos parametros da reproducao oficial.
        modelos = _ajustar_modelos_lightgbm(
            frames, simbolos, treino_final, config,
            phase=f"diagnostico_generalizacao_fold_{fid}_final"
        )
        scores_por_data = {}
        for j, ativo in enumerate(simbolos, 1):
            if ativo not in modelos:
                print(f"[fold {fid}] sem modelo por falta de dados: {ativo}", flush=True)
                continue
            frame = frames[ativo]
            model = modelos[ativo]
            tr = amostra(frame, treino_final)
            te = amostra(frame, teste)
            y_tr = tr[ALVO].to_numpy(dtype=float)
            pred_tr = model.predict(tr[ROTATION_FEATURES])
            metric_tr = medidas(y_tr, pred_tr)
            y_te = te[ALVO].to_numpy(dtype=float)
            pred_te = (model.predict(te[ROTATION_FEATURES]) if len(te)
                       else np.array([], dtype=float))
            metric_te = medidas(y_te, pred_te)
            mu = float(np.mean(y_tr))
            naive_tr = medidas(y_tr, np.full(y_tr.shape, mu))
            naive_te = medidas(y_te, np.full(y_te.shape, mu))
            skill_tr = (1 - metric_tr["rmse"] / naive_tr["rmse"]
                        if naive_tr["rmse"] > 0 else np.nan)
            skill_te = (1 - metric_te["rmse"] / naive_te["rmse"]
                        if naive_te["rmse"] > 0 and metric_te["n"] else np.nan)
            gap = (metric_te["rmse"] / metric_tr["rmse"] - 1
                   if metric_tr["rmse"] > 0 and metric_te["n"] else np.nan)
            qualidade.append({
                "ativo": ativo, "fold": fid,
                "treino_n": metric_tr["n"], "teste_n": metric_te["n"],
                "rmse_treino": metric_tr["rmse"], "rmse_teste": metric_te["rmse"],
                "mae_treino": metric_tr["mae"], "mae_teste": metric_te["mae"],
                "rmse_naive_treino": naive_tr["rmse"],
                "rmse_naive_teste": naive_te["rmse"],
                "skill_treino": skill_tr, "skill_teste": skill_te,
                "gap_rmse_relativo": gap,
                "alerta_sobreajuste": bool(skill_tr > 0 and skill_te <= 0)
                    if np.isfinite(skill_te) else False,
                "alerta_subajuste": bool(skill_tr <= 0 and skill_te <= 0)
                    if np.isfinite(skill_te) else False,
            })
            for data, observado, previsto in zip(te.index, y_te, pred_te):
                dt = pd.Timestamp(data)
                item = {"timestamp": dt, "fold": fid, "ativo": ativo,
                        "observado": float(observado), "previsto": float(previsto)}
                predicoes.append(item)
                scores_por_data.setdefault(dt, []).append(item)
            if j % 15 == 0 or j == len(simbolos):
                print(f"[fold {fid}] modelos avaliados: {j}/{len(simbolos)}", flush=True)

        for data, linhas in scores_por_data.items():
            metrica = ranking_sessao(pd.DataFrame(linhas))
            rank.append({"timestamp": data, "fold": fid, **metrica})

        del modelos
        # Curvas suplementares: quatro ativos fixos, apenas treino/calibracao.
        for ativo in AMOSTRA_GRAFICOS:
            if ativo not in frames:
                continue
            est, tam = curvas_calibracao(
                frames[ativo], ativo, fid, treino_ini, cal_ini, config
            )
            estagios.extend(est)
            tamanho.extend(tam)
        print(f"[fold {fid}] diagnosticos completos", flush=True)

    df_qual = pd.DataFrame(qualidade)
    df_rank = pd.DataFrame(rank)
    df_est = pd.DataFrame(estagios)
    df_tam = pd.DataFrame(tamanho)
    df_pred = pd.DataFrame(predicoes)
    if df_qual.empty or df_rank.empty or df_pred.empty:
        raise RuntimeError("Diagnosticos incompletos; resultados NAO publicados.")
    df_qual.to_csv(OUT / "qualidade_por_ativo_fold.csv", index=False)
    df_rank.to_csv(OUT / "ranking_por_sessao.csv", index=False)
    df_est.to_csv(OUT / "erro_por_arvore.csv", index=False)
    df_tam.to_csv(OUT / "curva_aprendizagem.csv", index=False)
    df_pred.to_csv(OUT / "previsoes_e_alvos_teste.csv", index=False,
                   float_format="%.17g")
    pd.DataFrame(folds_info).to_csv(OUT / "contrato_folds.csv", index=False)
    resumo = df_qual.groupby("fold", as_index=False).agg(
        ativos=("ativo", "nunique"), pares_teste=("teste_n", "sum"),
        rmse_medio_treino=("rmse_treino", "mean"),
        rmse_medio_teste=("rmse_teste", "mean"),
        skill_medio_teste=("skill_teste", "mean"),
        alertas_sobreajuste=("alerta_sobreajuste", "sum"),
        alertas_subajuste=("alerta_subajuste", "sum"),
    )
    resumo.to_csv(OUT / "resumo_por_fold.csv", index=False)
    graficos(df_qual, df_rank, df_est, df_tam, df_pred)
    manifesto = {
        "schema": SCHEMA,
        "status": "completed",
        "design": "diagnostico sem selecao/otimizacao de parametros",
        "cutoff": contrato["analysis_end_date"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "candidate_file_hashes": contrato["candidate_file_hashes"],
        "target": ALVO, "features": list(ROTATION_FEATURES),
        "sample_examples_predeclared": list(AMOSTRA_GRAFICOS),
        "n_estimators_official": int(config.research_model_settings["lightgbm"]["n_estimators"]),
        "folds": folds_info, "tempo_segundos": time.perf_counter() - inicio,
        "cautions": [
            "Rótulos futuros completos somente na avaliacao, nunca nas features.",
            "Curvas de aprendizado e estagios usam apenas calibracao cronologica.",
            "O teste cronologico foi usado para diagnostico, nao otimizacao.",
            "Tres folds do mesmo historico nao sao replicacoes independentes.",
            "Sobreposicao de rotulos multi-horizonte cria dependencia temporal.",
            "Skill positivo nao elimina vies por escolhas historicas da pesquisa.",
        ],
    }
    (OUT / "manifesto_diagnosticos.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    with zipfile.ZipFile(PACOTE, "w", zipfile.ZIP_DEFLATED) as z:
        for arquivo in sorted(OUT.rglob("*")):
            if arquivo.is_file():
                z.write(arquivo, arquivo.relative_to(OUT.parent))
    print("\n[resultado] diagnosticos por fold:", flush=True)
    print(resumo.to_string(index=False), flush=True)
    print(f"[pacote] {PACOTE}", flush=True)
    sinal_sonoro_conclusao()


if __name__ == "__main__":
    main()
