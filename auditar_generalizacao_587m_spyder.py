"""Auditoria de generalizacao preditiva do TCC historico de US$ 587 mi.

F5 no Spyder. Primeiro tenta reutilizar a auditoria assurance original,
desde que sua procedencia seja comprovada por hashes. Se ela nao existir,
refaz SOMENTE os ajustes LightGBM do snapshot congelado e calcula os erros
treino/OOS. Nao consulta Alpaca, nao simula carteira e nao otimiza parametros.

Os resultados medem sobreajuste PREDITIVO; nao atribuem parcela do capital
historico a overfitting nem validam selecao prospectiva do universo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERSION = "1.25.3"
OUT_ROOT = ROOT / "output" / "auditoria_generalizacao_587m"
EXPECTED_CAPITAL = 587_165_941.1330212
EXPECTED_FOLDS = (1, 2, 3)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _inputs():
    import pesquisar_expansao_u67_spyder as pesquisa
    from reproducao.dados import validate_snapshot

    contract = _json(ROOT / "evidencias" / "manifest_universo_congelado.json")
    selected = tuple(contract["selected_assets"])
    if selected != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES):
        raise RuntimeError("Os dez candidatos nao coincidem com o manifesto.")
    if not math.isclose(
        float(contract["expected_ending_capital"]), EXPECTED_CAPITAL,
        rel_tol=0, abs_tol=0.01,
    ):
        raise RuntimeError("Checkpoint historico inesperado.")

    base_symbols = tuple(
        symbol for symbol in pesquisa.U67_REQUESTED_ASSETS
        if symbol not in pesquisa.U67_EXPECTED_EXCLUSIONS
    )
    expected_assets = set(base_symbols) | set(selected)
    if len(base_symbols) != 65 or len(selected) != 10 or len(expected_assets) != 75:
        raise RuntimeError("Universo historico nao corresponde a 65+10=75.")

    snapshot = validate_snapshot(pesquisa.BASE_DATA)
    if snapshot.get("snapshot_sha256") != contract["snapshot_sha256"]:
        raise RuntimeError(
            "Snapshot base nao corresponde ao vintage dos US$ 587 mi. "
            "Nao atualize as series para forcar a reproducao."
        )
    actual_hashes = {}
    for symbol in selected:
        expected = contract["candidate_file_hashes"][symbol]
        files = {
            "raw_sha256": pesquisa.FULL_DATA.raw_bars / (symbol + ".csv"),
            "corporate_actions_sha256":
                pesquisa.FULL_DATA.corporate_actions / (symbol + ".csv"),
        }
        actual_hashes[symbol] = {}
        for key, file in files.items():
            if not file.is_file():
                raise FileNotFoundError(
                    f"Cache historico ausente: {file}. "
                    "Nao e permitido substitui-lo por download."
                )
            observed = _sha256(file)
            if observed != expected[key]:
                raise RuntimeError(f"Hash divergente: {symbol}, {key}.")
            actual_hashes[symbol][key] = observed

    return pesquisa, contract, base_symbols, selected, actual_hashes


def _reutilizar_diagnostico(
    contract: dict, base_symbols: tuple, selected: tuple, hashes: dict,
) -> pd.DataFrame | None:
    folder = ROOT / "output" / "u67_assurance_validation"
    metrics = folder / "model_generalization.csv"
    manifest = folder / "pesquisa_u67_assurance_validation.json"
    if not (metrics.is_file() and manifest.is_file()):
        return None
    try:
        metadata = _json(manifest)
        data = metadata.get("data") or {}
        if data.get("baseline_snapshot_sha256") != contract["snapshot_sha256"]:
            return None
        recorded_hashes = data.get("candidate_file_hashes") or {}
        if any(recorded_hashes.get(symbol) != hashes[symbol] for symbol in selected):
            return None
        frame = pd.read_csv(metrics)
        if not {"asset", "fold_id", "train_rmse", "oos_rmse"}.issubset(frame):
            return None
        universe = set(base_symbols) | set(selected)
        frame = frame.loc[frame["asset"].isin(universe)].copy()
        # A campanha assurance inclui tambem cinco candidatos nao selecionados.
        if set(frame["asset"]) != universe or set(frame["fold_id"]) != set(EXPECTED_FOLDS):
            return None
        if frame.duplicated(["asset", "fold_id"]).any() or len(frame) != 225:
            return None
        return frame
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _recalcular_apenas_modelos(pesquisa, contract, base_symbols, selected):
    """Mesmo ajuste por fold do executor oficial, mas sem replay financeiro."""
    from reproducao.preparacao import prepare_model_frames
    from reproducao.experimento import build_control_config

    raw, exclusions, _, _ = prepare_model_frames(
        pesquisa.BASE_DATA,
        assets=tuple(pesquisa.U67_REQUESTED_ASSETS),
        csv_float_precision="round_trip",
    )
    excluded = {row["symbol"] for row in exclusions}
    if excluded != set(pesquisa.U67_EXPECTED_EXCLUSIONS):
        raise RuntimeError("Exclusoes estruturais nao coincidem com a pesquisa.")
    if set(raw) != set(base_symbols):
        raise RuntimeError("Conjunto de ativos base divergente.")
    frames_base = {s: raw[s] for s in base_symbols}
    config = build_control_config(frames_base, pesquisa.CONFIG).copiar_modelo(
        update={"analysis_end_date": contract["analysis_end_date"]}
    )
    (
        frames, common_dates, _, symbols, folds, dates, _, _
    ) = pesquisa._construir_contexto_execucao(frames_base, config)
    if set(symbols) != set(base_symbols) or len(folds) != 3:
        raise RuntimeError("Contexto de folds/ativos diferente do historico.")

    print("[generalizacao] ajustando LightGBM base; SEM backtest...", flush=True)
    states, _, _ = pesquisa._train_baseline_states(
        frames, common_dates, symbols, folds, config
    )
    rows = []
    for state in states:
        for symbol in symbols:
            rows.append(pesquisa._fit_quality_row(
                universe_role="base",
                symbol=symbol,
                fold_id=state.fold_id,
                frame=frames[symbol],
                train_dates=state.final_fit_dates,
                decision_dates=state.decision_dates,
                fit_diagnostics=state.fit_diagnostics.get(symbol),
                oos_diagnostics=state.oos_diagnostics.get(symbol),
            ))

    candidates, _, candidate_hashes = pesquisa._prepare_candidate_frames(
        list(selected), common_dates, dates, config, paths=pesquisa.FULL_DATA
    )
    if set(candidates) != set(selected) or not candidate_hashes:
        raise RuntimeError("Candidatos historicos incompletos.")

    for position, symbol in enumerate(selected, start=1):
        print(f"[generalizacao] candidato {position}/10: {symbol}", flush=True)
        caches = pesquisa._train_candidate_caches(
            symbol, candidates[symbol], states, config
        )
        for state in states:
            cache = caches[state.fold_id]
            rows.append(pesquisa._fit_quality_row(
                universe_role="selected",
                symbol=symbol,
                fold_id=state.fold_id,
                frame=candidates[symbol],
                train_dates=state.final_fit_dates,
                decision_dates=state.decision_dates,
                fit_diagnostics=cache.fit_diagnostics,
                oos_diagnostics=cache.oos_diagnostics.get(symbol),
            ))
    return pd.DataFrame(rows)


def _validar_e_calcular(frame: pd.DataFrame, base_symbols, selected):
    required = {
        "asset", "fold_id", "train_rmse", "oos_rmse",
        "train_rows", "oos_rows",
    }
    if not required.issubset(frame):
        raise RuntimeError("Colunas obrigatorias ausentes: " + str(required-set(frame)))
    frame = frame.copy()
    assets = set(base_symbols) | set(selected)
    if (
        len(frame) != 225 or set(frame["asset"]) != assets
        or set(frame["fold_id"]) != set(EXPECTED_FOLDS)
        or frame.duplicated(["asset", "fold_id"]).any()
    ):
        raise RuntimeError("Nao sao exatamente 75 ativos x 3 folds.")
    for column in ("train_rmse", "oos_rmse", "train_rows", "oos_rows"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (
        frame[["train_rmse", "oos_rmse", "train_rows", "oos_rows"]].isna().any().any()
        or (frame["train_rmse"] <= 0).any()
        or (frame["oos_rmse"] < 0).any()
        or (frame[["train_rows", "oos_rows"]] <= 0).any().any()
    ):
        raise RuntimeError("Metricas incompletas ou invalidas; nao imputar dados.")
    frame["gap_rmse_percent"] = (
        100.0 * (frame["oos_rmse"] / frame["train_rmse"] - 1.0)
    )
    return frame.sort_values(["asset", "fold_id"]).reset_index(drop=True)


def _graficos(frame: pd.DataFrame, output: Path):
    # 1. Diagrama diagnostico: train vs OOS; diagonal significa erro igual.
    fig, ax = plt.subplots(figsize=(8, 7))
    for fold in EXPECTED_FOLDS:
        part = frame.loc[frame["fold_id"] == fold]
        ax.scatter(part["train_rmse"], part["oos_rmse"],
                   label=f"Fold {fold}", s=24, alpha=.7)
    limit = 1.05 * float(
        max(frame["train_rmse"].max(), frame["oos_rmse"].max())
    )
    ax.plot([0, limit], [0, limit], "--", lw=1, label="Treino = teste")
    ax.set(xlim=(0, limit), ylim=(0, limit),
           xlabel="RMSE treino", ylabel="RMSE fora da amostra",
           title="Generalizacao LightGBM — 75 ativos, 3 folds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "01_rmse_treino_versus_oos.png", dpi=175)
    plt.close(fig)

    # 2. Mapa de calor: todos os 75 ativos, ordenados pelo gap mediano.
    matrix = frame.pivot(index="asset", columns="fold_id", values="gap_rmse_percent")
    ordered = matrix.median(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[ordered, list(EXPECTED_FOLDS)]
    fig, ax = plt.subplots(figsize=(9, 19))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(3), [f"Fold {f}" for f in EXPECTED_FOLDS])
    ax.set_yticks(range(len(matrix)), list(matrix.index), fontsize=7)
    ax.set_title("Gap percentual RMSE por ativo e fold\n"
                 "(positivo = teste com erro maior)")
    fig.colorbar(image, ax=ax, label="(RMSE OOS / RMSE treino − 1) × 100 (%)",
                 shrink=.55)
    fig.tight_layout()
    fig.savefig(output / "02_mapa_gap_por_ativo_fold.png", dpi=180)
    plt.close(fig)

    # 3. Distribuicao, sem limiar arbitrario de classificacao.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame["gap_rmse_percent"], bins=35, edgecolor="white")
    ax.axvline(0, ls="--", lw=1, label="Gap zero")
    ax.set(xlabel="Gap percentual de RMSE (teste versus treino)",
           ylabel="Quantidade de modelos",
           title="Distribuicao de generalizacao — 225 ajustes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "03_distribuicao_gap_rmse.png", dpi=175)
    plt.close(fig)


def main():
    pesquisa, contract, base, selected, hashes = _inputs()
    original = ROOT / "output" / "reproducao_universo" / "reproducao_universo.json"
    if original.is_file():
        payload = _json(original)
        if (
            payload.get("data", {}).get("snapshot_sha256")
            != contract["snapshot_sha256"]
        ):
            raise RuntimeError("Reproducao local pertence a outro snapshot.")
        if not math.isclose(
            float(payload.get("checkpoint", {}).get("observed_capital", 0)),
            EXPECTED_CAPITAL, rel_tol=0, abs_tol=0.01
        ):
            raise RuntimeError("Capital da reproducao local nao confere.")

    frame = _reutilizar_diagnostico(contract, base, selected, hashes)
    origem = "assurance_historica_verificada" if frame is not None else "somente_modelos_reajustados"
    if frame is None:
        print("[generalizacao] CSV original indisponivel; recomputando "
              "APENAS modelos sobre o snapshot imutavel...", flush=True)
        frame = _recalcular_apenas_modelos(pesquisa, contract, base, selected)
    frame = _validar_e_calcular(frame, base, selected)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUT_ROOT / run_id
    if output.exists():
        raise RuntimeError("Diretorio de saida ja existe; nao sobrescrever.")
    output.mkdir(parents=True)
    frame.to_csv(output / "model_generalization_225.csv", index=False)
    _graficos(frame, output)

    by_fold = []
    for fid, part in frame.groupby("fold_id"):
        by_fold.append({
            "fold": int(fid),
            "n": len(part),
            "train_rmse_mediana": float(part["train_rmse"].median()),
            "oos_rmse_mediana": float(part["oos_rmse"].median()),
            "gap_rmse_mediana_percent": float(part["gap_rmse_percent"].median()),
            "oos_rmse_maior_que_treino": int(
                (part["oos_rmse"] > part["train_rmse"]).sum()
            ),
        })
    summary = {
        "script_version": VERSION,
        "status": "completed",
        "source": origem,
        "historical_checkpoint_reference": EXPECTED_CAPITAL,
        "historical_snapshot_sha256": contract["snapshot_sha256"],
        "model_count": len(frame),
        "asset_count": len(set(frame["asset"])),
        "folds": by_fold,
        "gap_rmse_mediano_percent": float(frame["gap_rmse_percent"].median()),
        "gap_rmse_maior_que_zero": int((frame["gap_rmse_percent"] > 0).sum()),
        "interpretacao": (
            "Gap mede desempenho preditivo; nao e uma probabilidade de "
            "overfitting nem prova atribuicao causal do capital. "
            "Selecao retrospectiva de ativos e hiperparametros nao foi "
            "validada prospectivamente."
        ),
        "input_file_hash": _sha256(ROOT / "evidencias" /
                                   "manifest_universo_congelado.json"),
    }
    (output / "resumo_generalizacao.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    package = output / "pacote_generalizacao_587m_v1253.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as archive:
        for item in sorted(output.iterdir()):
            if item.is_file() and item != package:
                archive.write(item, arcname=item.name)
    print(f"[generalizacao] pacote={package}", flush=True)
    print(f"[generalizacao] modelos={len(frame)}, "
          f"gap_mediano={summary['gap_rmse_mediano_percent']:.2f}%, "
          f"origem={origem}", flush=True)
    print(f"[generalizacao] resultado={output}", flush=True)
    print("[generalizacao] Sem download, sem simulacao financeira e sem alteracao"
          " do historico.", flush=True)
    return summary


if __name__ == "__main__":
    main()
