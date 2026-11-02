"""TCC | Contribuicao marginal de ativos FORA do U75 no Control completo.

Spyder: executar este arquivo com F5. Usa apenas os 75 CSVs Strategy #30 e
a shortlist congelada de 24 candidatos, sem novo download, banco ou MCT.
O Control precisa ser reproduzido EXATAMENTE antes de qualquer comparacao.

Cada candidato eh ADICIONADO isoladamente ao U75, os modelos por ativo sao
retreinados nos folds originais, a margem eh calibrada no passado e a carteira
inteira eh simulada com a mesma regra de rotacao, custos e reinvestimento.
A shortlist eh retrospectiva. Resultados sao EXPLORATORIOS e nao indicam
que o candidato poderia ter sido descoberto prospectivamente.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from pesquisa.control_operacional_strategy30 import (
    ROOT, carregar_contrato, validar_control_strategy30,
)
from pesquisa.paridade_strategy30 import (
    compare_control_trajectories, config_from_frozen_manifest,
)
from pesquisa.referencia_strategy30.execucao import (
    calcular_taxas_referencia, aplicar_deslizamento,
)
from pesquisa.referencia_strategy30.modelo_lightgbm import executar_lightgbm
from pesquisa.referencia_strategy30.rotacao import preparar_painel_rotacao
from pesquisa.contribuicao_marginal_externa_u75 import comparar_trajetorias
from pesquisa.persistencia_resultados import salvar_csv, salvar_json, salvar_texto
from pesquisar_controle_strategy30_spyder import localizar_zip_original
from pesquisar_busca_externa_lider_u75_spyder import (
    load_snapshots, CANDIDATE_SNAPSHOT,
)
from pesquisar_expansao_u67_spyder import _write_json


# %% Parametros OPERACIONAIS de processamento, nao hiperparametros do modelo.
SCRIPT_VERSION = "u75-external-marginal-capital-v1.0.0"
OUTPUT = ROOT / "output" / "contribuicao_marginal_externa_u75"
START_INDEX = int(os.getenv("U75_EXTERNAL_BATCH_START", "0"))
BATCH_SIZE = int(os.getenv("U75_EXTERNAL_BATCH_SIZE", "3"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _empacotar_resultados(out: Path, status: str) -> Path:
    """Pacote proprio com manifesto SHA-256, sem copiar RAW nem credenciais."""
    _require((out / "resultado.json").is_file(), "Resultado JSON ausente")
    path = out.parent / f"pacote_contribuicao_marginal_externa_u75_{out.name}.zip"
    _require(not path.exists(), "Pacote de resultados ja existe")
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}
    )
    manifest = {
        "schema_version": 1,
        "experiment": "u75-external-marginal-capital",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "run_id": out.name,
        "raw_data_included": False,
        "files": [
            {"name": p.relative_to(out).as_posix(),
             "size": p.stat().st_size, "sha256": _hash_file(p)}
            for p in files
        ],
    }
    tmp = path.with_suffix(".zip.tmp")
    try:
        with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("manifesto_pacote.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            for p in files:
                z.write(p, arcname=p.relative_to(out).as_posix())
        with ZipFile(tmp) as z:
            _require(z.testzip() is None, "Pacote corrompido")
            for file in manifest["files"]:
                _require(sha256(z.read(file["name"])).hexdigest() == file["sha256"],
                         f"Hash divergente: {file['name']}")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _metrics(run) -> dict:
    names = (
        "strategy_ending_capital", "strategy_sharpe",
        "strategy_maximum_drawdown", "strategy_cagr",
        "capital_rotations", "total_transaction_fees", "cash_days",
        "market_exposure", "turnover_ratio",
    )
    return {k: (float(run.metrics[k]) if run.metrics.get(k) is not None else None)
            for k in names}


def _run_engine(frames: dict, config, label: str):
    _require(set(frames) == set(config.assets),
             f"{label}: universo diferente dos ativos da configuracao")
    print(f"[fit] {label}: {len(frames)} ativos", flush=True)
    results = executar_lightgbm(
        frames, config, calcular_taxas_referencia, aplicar_deslizamento,
        technical_log_callback=lambda s: print(f"[{label}] {s}", flush=True),
    )
    _require(len(results) == 1, f"{label}: esperado exatamente um replay")
    return results[0]


def _preflight_calendar(base: dict, challenger: dict, base_config, base_calendar):
    changed = base_config.copiar_modelo(update={"assets": tuple(sorted(challenger))})
    _, new_calendar, source = preparar_painel_rotacao(challenger, changed)
    if not pd.DatetimeIndex(new_calendar).equals(pd.DatetimeIndex(base_calendar)):
        raise RuntimeError(
            "Candidato alterou o calendario do motor. "
            "Nao comparar capitais com folds ou janelas diferentes."
        )
    return changed, source


def main() -> dict:
    _require(START_INDEX >= 0 and BATCH_SIZE >= 1,
             "U75_EXTERNAL_BATCH_START deve ser >=0 e BATCH_SIZE >=1")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUTPUT / run_id
    out.mkdir(parents=True, exist_ok=False)
    status = "failed"
    data = {"version": SCRIPT_VERSION, "status": "running",
            "research_only": True,
            "benchmark": "control_operational_strategy30_20261028",
            "selection_not_prospectively_validated": True,
            "uses_market_downloads": False,
            "modifies_cash_logic": False}
    try:
        # %% 1. Control assinado e configuracao original
        reference_zip = localizar_zip_original()
        contract = carregar_contrato()
        audit, reference_preds, _ = validar_control_strategy30(
            reference_zip, contrato=contract,
        )
        with ZipFile(reference_zip) as archive:
            strategy = json.loads(archive.read("strategy_manifest.json"))
            experiment = json.loads(archive.read("experiment_manifest.json"))
        config = config_from_frozen_manifest(strategy, contract)
        _require(int(config.rotation_model_repetitions) == 1,
                 "Paridade exige uma repeticao congelada")
        _require(experiment["market_data_signature_sha256"] ==
                 contract["market_data_signature_sha256"],
                 "Assinatura da fonte Strategy #30 divergente")

        # %% 2. Snapshots 75+24 validados pelo teste externo anterior
        base, external, universe, candidate_config, source = load_snapshots()
        _require(universe == set(contract["assets"]), "Universo U75 mudou")
        _require(set(base) == universe, "75 barras de referencia incompletas")
        _require(set(external).isdisjoint(universe),
                 "Candidato externo ja pertence ao U75")
        _require(candidate_config.analysis_end_date == config.analysis_end_date,
                 "Configuracao da shortlist e Control possuem datas diferentes")
        choices = sorted(external)
        chosen = choices[START_INDEX:START_INDEX + BATCH_SIZE]
        _require(bool(chosen),
                 f"Lote vazio: {len(choices)} candidatos, inicio {START_INDEX}")
        data["source"] = source
        data["source"]["control_zip_sha256"] = audit["source_zip_sha256"]
        data["batch"] = {"start_index": START_INDEX, "batch_size": BATCH_SIZE,
                         "candidate_total": len(choices), "symbols": chosen,
                         "next_index": START_INDEX + len(chosen)}
        print(f"[lote] {len(chosen)} de {len(choices)} candidatos: {chosen}", flush=True)
        salvar_json(out / "contrato_e_lote.json", data, _write_json)

        # %% 3. Calendario do Control fixo. Proibido mudar folds.
        _, base_calendar, source_calendar = preparar_painel_rotacao(base, config)
        data["calendar_source"] = source_calendar
        # %% 4. Executa de verdade o mesmo Control e exige paridade financeira.
        control = _run_engine(base, config, "CONTROL")
        parity, parity_detail = compare_control_trajectories(
            reference_preds, control.predictions, contract=contract,
        )
        salvar_json(out / "auditoria_paridade.json", parity, _write_json)
        salvar_csv(parity_detail, out / "auditoria_paridade_diaria.csv", index=False)
        _require(bool(parity["fully_reproduced"]),
                 "Paridade do Control falhou: resultado dos candidatos BLOQUEADO. "
                 f"Primeira divergencia: {parity['first_mismatch_date']}")
        data["parity"] = parity
        data["baseline_metrics"] = _metrics(control)
        salvar_json(out / "baseline_metricas.json", data["baseline_metrics"], _write_json)
        salvar_csv(control.predictions, out / "control_predictions.csv",
                   index=True, index_label="timestamp", float_format="%.17g")

        # %% 5. Adicionar um candidato por vez. Nao fazer busca ex post de subconjuntos.
        rows, all_folds, errors = [], [], []
        for n, symbol in enumerate(chosen, start=1):
            print(f"[candidato] {n}/{len(chosen)} {symbol}", flush=True)
            try:
                raw_path = CANDIDATE_SNAPSHOT.raw_bars / f"{symbol}.csv"
                actions_path = CANDIDATE_SNAPSHOT.corporate_actions / f"{symbol}.csv"
                _require(raw_path.is_file() and actions_path.is_file(),
                         f"{symbol}: arquivos RAW/Corporate Actions faltantes")
                source_identity = {
                    "symbol": symbol,
                    "raw_sha256": _hash_file(raw_path),
                    "corporate_actions_sha256": _hash_file(actions_path),
                }
                extended = dict(base)
                extended[symbol] = external[symbol]
                challenger_config, calendar_source = _preflight_calendar(
                    base, extended, config, base_calendar,
                )
                run = _run_engine(extended, challenger_config, symbol)
                detail, summary, fold_table = comparar_trajetorias(
                    control.predictions, run.predictions, symbol,
                )
                summary.update({
                    "candidate_raw_sha256": source_identity["raw_sha256"],
                    "candidate_actions_sha256":
                        source_identity["corporate_actions_sha256"],
                    "calendar_source": calendar_source,
                })
                for key, value in _metrics(run).items():
                    summary[key] = value
                rows.append(summary)
                all_folds.append(fold_table)
                salvar_csv(detail, out / f"decisoes_{symbol}.csv",
                           index=False, float_format="%.17g")
                salvar_csv(run.trades, out / f"trades_{symbol}.csv", index=False)
                salvar_json(out / f"identidade_{symbol}.json",
                            source_identity, _write_json)
            except Exception as exc:
                info = {"symbol": symbol, "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc(limit=12)}
                errors.append(info)
                print(f"[reprovado] {symbol}: {exc}", flush=True)
            finally:
                # Checkpoint cumulativo mesmo se alguns ativos falharem.
                if rows:
                    salvar_csv(pd.DataFrame(rows),
                               out / "comparacao_capital.csv", index=False)
                if all_folds:
                    salvar_csv(pd.concat(all_folds, ignore_index=True),
                               out / "comparacao_por_fold.csv", index=False)
                salvar_json(out / "candidatos_reprovados.json", errors, _write_json)

        _require(bool(rows), "Nenhum candidato concluiu o backtest completo")
        data["status"] = "completed_with_exclusions" if errors else "completed"
        data["candidate_results"] = rows
        data["candidate_errors"] = errors
        data["interpretation"] = (
            "Busca marginal exploratoria: 24 candidatos foram congelados apos "
            "inspecao historica. Nenhuma selecao de ativo com base nestes "
            "resultados constitui avaliacao de descoberta prospectiva. "
            "Dependencia de elegibilidade t+1 herdada do Control permanece."
        )
        status = data["status"]
        salvar_json(out / "resultado.json", data, _write_json)
        best = sorted(rows, key=lambda v: v["delta_compound_log_growth"],
                      reverse=True)
        text = [
            f"# Contribuicao marginal externa U75 ({SCRIPT_VERSION})",
            "", f"Control: US$ {parity['reference_final_capital']:,.2f}",
            "Teste: retreino LightGBM e replay integral, um ativo externo por vez.",
            "Sem nova regra de CASH. Sem download. Sem promocao de estrategia.",
            "As comparacoes preservam os 75 ativos, calendario e folds.",
            "A shortlist de 24 ja foi preselecionada retrospectivamente.",
            "", "## Patrimonio por candidato (exploratorio)", "",
        ]
        for row in best:
            text.append(
                f"- {row['symbol']}: US$ {row['challenger_ending_capital']:,.2f}; "
                f"delta US$ {row['delta_ending_capital']:,.2f}; "
                f"posicoes alteradas={row['changed_positions']}; "
                f"dias no candidato={row['direct_candidate_days']}; "
                f"alteracoes indiretas={row['indirect_changed_days']}."
            )
        if errors:
            text.extend(["", "## Candidatos com erro", ""])
            text.extend(f"- {i['symbol']}: {i['error']}" for i in errors)
        salvar_texto(out / "resumo.md", "\n".join(text) + "\n")
        print(f"[concluido] {out}", flush=True)
        return data
    except Exception as exc:
        data["status"] = "failed"
        data["failure"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=20)}
        salvar_json(out / "resultado.json", data, _write_json)
        raise
    finally:
        zip_path = _empacotar_resultados(out, status)
        print(f"[pacote] {zip_path}", flush=True)


if __name__ == "__main__":
    main()
