"""TCC | Experimento contrafactual da permanencia minima: 0, 1, 2, 3 pregoes.

Este e o arquivo UNICO para todas as rodadas da pesquisa nesta branch:
    research/permanencia-minima-contrafactual-v1

Spyder: abra este arquivo e execute com F5.
Terminal: python pesquisar_permanencia_minima_spyder.py

Modos: MODO = "verificar" (hashes, sem treino) ou "comparar" (padrao).

A reproduçao utiliza o motor cientifico congelado do TCC (v1.23.0),
o mesmo conjunto de 77 ativos solicitados (75 efetivos), os mesmos
modelos, folds, margem calibrada por fold, custos e execucoes exatas.
O unico parametro alterado e rotation_min_holding_days.

Nao acessa MongoDB, nao busca novas cotacoes, nao reconstrói ativos
descontinuados, nao altera a Winner Live nem escreve em dados/u67.
Os 10 candidatos requerem o cache historico local e seus SHA-256
conferidos contra evidencias/manifest_universo_congelado.json.
Resultados completos: output/pesquisa_permanencia_minima/
Pacote unico para analise: output/pesquisa_permanencia_minima.zip
O ZIP inclui os quatro backtests, ranking integral, curvas, ordens, metricas
por fold, cotacoes OHLCV congeladas, parametros, hashes e integridade verificada.
So cria ZIP concluido se os 4 cenarios e a paridade do controle passarem.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

# %% 0. Um arquivo, uma branch; somente este seletor muda entre modos.
MODO = "comparar"  # "verificar" ou "comparar"
PERMANENCIAS = (0, 1, 2, 3)
PERMANENCIA_CONTROLE = 2
TOLERANCIA_CENTAVO = 0.01
ROOT = Path(__file__).resolve().parent
SAIDA = ROOT / "output" / "pesquisa_permanencia_minima"
SAIDA_TEMP = ROOT / "output" / ".pesquisa_permanencia_minima_em_execucao"
ARQUIVO_ZIP = ROOT / "output" / "pesquisa_permanencia_minima.zip"
MANIFESTO_CONGELADO = ROOT / "evidencias" / "manifest_universo_congelado.json"


def _hash_arquivo(arquivo: Path) -> str:
    digest = hashlib.sha256()
    with arquivo.open("rb") as stream:
        for bloco in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _json_seguro(valor: Any) -> Any:
    """Converte valores numpy/pandas e NaN em JSON portavel e valido."""
    if isinstance(valor, np.generic):
        return _json_seguro(valor.item())
    if isinstance(valor, float):
        return valor if math.isfinite(valor) else None
    if isinstance(valor, (str, bool, int)) or valor is None:
        return valor
    if isinstance(valor, (pd.Timestamp, datetime, Path)):
        return str(valor)
    if isinstance(valor, np.ndarray):
        return [_json_seguro(item) for item in valor.tolist()]
    if isinstance(valor, dict):
        return {str(chave): _json_seguro(dado) for chave, dado in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [_json_seguro(item) for item in valor]
    return str(valor)


def _salvar_json(caminho: Path, dados: Any) -> None:
    caminho.write_text(
        json.dumps(_json_seguro(dados), indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8",
    )


def _versao_pacote(nome: str) -> str | None:
    try:
        return version(nome)
    except PackageNotFoundError:
        return None


# %% 1. Verificar snapshot e universo exatos antes de qualquer treinamento.
def verificar_fontes() -> tuple[Any, dict[str, Any], tuple[str, ...]]:
    from reproducao.dados import validate_snapshot
    import pesquisar_expansao_u67_spyder as pesquisa

    contrato = json.loads(MANIFESTO_CONGELADO.read_text(encoding="utf-8"))
    selecionados = tuple(contrato["selected_assets"])
    if len(selecionados) != 10 or len(set(selecionados)) != 10:
        raise RuntimeError("Os dez candidatos historicos devem ser distintos.")
    if selecionados != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES):
        raise RuntimeError("Universo selecionado diferente do executor oficial.")

    snapshot = validate_snapshot(pesquisa.BASE_DATA)
    if snapshot["snapshot_sha256"] != contrato["snapshot_sha256"]:
        raise RuntimeError("SHA-256 do U67 nao corresponde ao manifesto congelado.")
    if tuple(snapshot.get("assets") or ()) != tuple(pesquisa.U67_REQUESTED_ASSETS):
        raise RuntimeError("Ordem dos 67 ativos da base mudou.")
    if contrato["analysis_end_date"] != pesquisa.ANALYSIS_END_DATE:
        raise RuntimeError("Data final da pesquisa mudou.")
    if int(pesquisa.CONFIG.rotation_min_holding_days) != PERMANENCIA_CONTROLE:
        raise RuntimeError("Controle oficial nao esta configurado com 2 pregoes.")

    # Os CSVs adicionais nao estao no Git. Recusar qualquer substituicao,
    # download implicito ou arquivo local que nao corresponda a fonte original.
    for ativo in selecionados:
        esperados = contrato["candidate_file_hashes"][ativo]
        arquivos = {
            "raw_sha256": pesquisa.FULL_DATA.raw_bars / f"{ativo}.csv",
            "corporate_actions_sha256":
                pesquisa.FULL_DATA.corporate_actions / f"{ativo}.csv",
        }
        for campo, arquivo in arquivos.items():
            if not arquivo.is_file():
                raise FileNotFoundError(
                    f"Arquivo historico do TCC ausente: {arquivo}\n"
                    "Restaure o cache original da pesquisa v1.4.1. "
                    "Nao faremos novo download nem usaremos dados do outros sistemas."
                )
            if _hash_arquivo(arquivo) != esperados[campo]:
                raise RuntimeError(
                    f"SHA-256 divergente para {ativo} ({campo}). "
                    "A pesquisa foi interrompida para evitar dados diferentes."
                )
    print(f"[DADOS] snapshot={snapshot['snapshot_sha256']} "
          f"ativos_base=67 selecionados={len(selecionados)}", flush=True)
    return pesquisa, contrato, selecionados


# %% 2. Treinar o controle e os candidatos UMA unica vez; guardar caches por fold.
def preparar_contexto(pesquisa: Any, contrato: dict, selecionados: tuple[str, ...]) -> dict:
    from reproducao.preparacao import prepare_model_frames
    from reproducao.experimento import build_control_config

    requisitados = tuple(pesquisa.U67_REQUESTED_ASSETS)
    barras, exclusoes, _, _ = prepare_model_frames(
        pesquisa.BASE_DATA, assets=requisitados, csv_float_precision="round_trip",
    )
    excluidos = {str(item.get("symbol", "")).upper() for item in exclusoes}
    if excluidos != pesquisa.U67_EXPECTED_EXCLUSIONS:
        raise RuntimeError(f"Exclusoes estruturais inesperadas: {excluidos}")
    ativos_base = tuple(ativo for ativo in requisitados if ativo in barras)
    if len(ativos_base) != pesquisa.U67_EXPECTED_EFFECTIVE_COUNT:
        raise RuntimeError("O universo base efetivo nao tem 65 ativos.")

    configuracao = build_control_config(
        {ativo: barras[ativo] for ativo in ativos_base}, pesquisa.CONFIG,
    ).copiar_modelo(update={"analysis_end_date": contrato["analysis_end_date"]})

    (frames_base, dias_comuns, origem_calendario, simbolos_base,
     folds, dias_decisao, decisao_fold, metadados_decisao) = (
        pesquisa._construir_contexto_execucao(
            {ativo: barras[ativo] for ativo in ativos_base}, configuracao,
        )
    )
    if len(folds) != 3:
        raise RuntimeError(f"Esperados tres folds; observados {len(folds)}.")

    print("[TREINO] LightGBM base: modelos de calibracao e finais dos 3 folds",
          flush=True)
    estados, _, _ = pesquisa._train_baseline_states(
        frames_base, dias_comuns, simbolos_base, folds, configuracao,
    )
    quadros, continuidade, hashes = pesquisa._prepare_candidate_frames(
        list(selecionados), dias_comuns, dias_decisao, configuracao,
    )
    if tuple(sorted(quadros)) != tuple(sorted(selecionados)):
        raise RuntimeError("Nem todos os dez candidatos passaram na validacao.")
    if hashes != contrato["candidate_file_hashes"]:
        raise RuntimeError("Os hashes normalizados dos candidatos divergiram.")

    caches = {}
    for indice, simbolo in enumerate(selecionados, 1):
        print(f"[TREINO] candidato {indice}/10: {simbolo}", flush=True)
        caches[simbolo] = pesquisa._train_candidate_caches(
            simbolo, quadros[simbolo], estados, configuracao,
        )
    simbolos = sorted([*simbolos_base, *selecionados])
    if len(simbolos) != 75:
        raise RuntimeError(f"Universo efetivo difere de 75: {len(simbolos)}")
    print(f"[CONTEXTO] folds=3 ativos={len(simbolos)} "
          f"calendario={origem_calendario}", flush=True)

    return {
        "config": configuracao, "frames_base": frames_base,
        "simbolos_base": simbolos_base, "folds": folds,
        "dias_decisao": dias_decisao, "decisao_fold": decisao_fold,
        "metadados": metadados_decisao, "estados": estados,
        "quadros": quadros, "caches": caches,
        "simbolos": simbolos, "continuidade": continuidade,
    }


# %% 3. Controle congelado: mesma calibracao por fold em todos os cenarios.
def calibrar_controle(pesquisa: Any, contexto: dict,
                     selecionados: tuple[str, ...], contrato: dict):
    resultado, metricas, linhas = pesquisa._evaluate_scenario(
        "CONTROLE_CONGELADO_2_PREGOES",
        list(selecionados),
        contexto["frames_base"],
        contexto["quadros"],
        contexto["simbolos_base"],
        contexto["estados"], contexto["caches"], contexto["folds"],
        contexto["dias_decisao"], contexto["decisao_fold"],
        contexto["metadados"], contexto["config"],
    )
    observado = float(metricas["ending_capital"])
    esperado = float(contrato["expected_ending_capital"])
    if not math.isclose(observado, esperado, rel_tol=0, abs_tol=TOLERANCIA_CENTAVO):
        raise RuntimeError(
            f"Controle 2 dias NAO reproduzido: US$ {observado:,.8f}, "
            f"esperado US$ {esperado:,.8f}. Interrompido."
        )
    margens = {}
    for linha in linhas:
        if linha["selected"]:
            fold = int(linha["fold_id"])
            if fold in margens:
                raise RuntimeError(f"Mais de uma margem selecionada no fold {fold}")
            margens[fold] = float(linha["candidate_margin"])
    if len(margens) != 3 or not all(math.isfinite(x) for x in margens.values()):
        raise RuntimeError("Margens do Control nao foram verificadas nos 3 folds.")
    print(f"[PARIDADE] Controle=US$ {observado:,.2f}; "
          f"margens por fold={margens}", flush=True)
    return resultado, metricas, linhas, margens


# %% 4. Contrafactual: so a permanencia muda, usando EXACTAMENTE os mesmos scores.
def simular_contrafactual(pesquisa: Any, contexto: dict,
                          minimo: int, margens: dict[int, float]):
    # O contrato completo de 75 ativos precisa ser identico ao do executor
    # oficial; nao apenas a lista passada para a funcao de politica.
    config = pesquisa.construir_configuracao_controle(
        contexto["config"], assets=contexto["simbolos"],
    ).copiar_modelo(update={"rotation_min_holding_days": minimo})
    simbolos = contexto["simbolos"]
    frames = {**contexto["frames_base"], **contexto["quadros"]}
    politicas = {}
    diagnosticos = {}
    for estado in contexto["estados"]:
        extras = {
            ativo: contexto["caches"][ativo][estado.fold_id].decision_cache
            for ativo in contexto["quadros"]
        }
        cache_fold = pesquisa._merge_utility_caches(
            estado.decision_cache, contexto["simbolos_base"], extras, simbolos,
        )
        margem = margens[estado.fold_id]
        politicas[estado.fold_id] = pesquisa._politica_utilidade(
            {}, frames, simbolos, config,
            max(float(config.rotation_switch_margin), margem),
            decision_diagnostics=diagnosticos,
            fold_id=estado.fold_id,
            calibrated_switch_margin=margem,
            utility_cache=cache_fold,
        )
    resultado = pesquisa._simular_exato(
        "tcc_u67_asset_expansion",
        pesquisa._politica_agendada(politicas, contexto["decisao_fold"]),
        frames, simbolos, contexto["dias_decisao"], config,
        pesquisa.calcular_taxas_referencia, pesquisa.aplicar_deslizamento,
        decision_metadata=contexto["metadados"],
        policy_decision_diagnostics=diagnosticos,
        model_label="TCC Control | pesquisa permanencia minima",
        method_line="- Mesmos modelos e margens originais; apenas min_hold varia.",
    )
    metricas = pesquisa._scenario_metrics(
        resultado, contexto["folds"], float(config.initial_capital),
    )
    return resultado, metricas, diagnosticos


def _linha_resumo(minimo: int, metricas: dict, referencia: float) -> dict:
    capital = float(metricas["ending_capital"])
    return {
        "minimo_pregoes": minimo,
        "patrimonio_final_usd": capital,
        "diferenca_controle_usd": capital - referencia,
        "diferenca_controle_pct": 100.0 * (capital / referencia - 1),
        "cagr": metricas.get("cagr"),
        "sharpe": metricas.get("sharpe"),
        "max_drawdown": metricas.get("maximum_drawdown"),
        "rotacoes": metricas.get("capital_rotations"),
        "dias_em_caixa": metricas.get("cash_days"),
        "taxas_usd": metricas.get("total_transaction_fees"),
        "pior_retorno_fold": metricas.get("worst_fold_return"),
    }



# %% 5. Evidencias completas para analise, sem depender de arquivos externos.
def exportar_utilidades_oos(pesquisa: Any, contexto: dict, pasta: Path) -> None:
    """Cada score OOS original por ativo, dia e fold. Sem novo treino."""
    simbolos = contexto["simbolos"]
    caminho = pasta / "utilidades_oos_todos_ativos.csv"
    quantidade = 0
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(("data_decisao", "fold_id", "ativo", "utilidade",
                           "posicao_ranking", "disponivel"))
        for estado in contexto["estados"]:
            extras = {
                ativo: contexto["caches"][ativo][estado.fold_id].decision_cache
                for ativo in contexto["quadros"]
            }
            cache = pesquisa._merge_utility_caches(
                estado.decision_cache, contexto["simbolos_base"],
                extras, simbolos,
            )
            for data, vetor in sorted(cache.items()):
                if len(vetor) != len(simbolos) + 1:
                    raise RuntimeError("Vetor de utilidade fora do universo congelado.")
                validos = [
                    i for i in range(1, len(vetor))
                    if np.isfinite(float(vetor[i]))
                ]
                validos.sort(key=lambda i: (-float(vetor[i]), simbolos[i - 1]))
                ranking = {i: indice for indice, i in enumerate(validos, 1)}
                for indice, ativo in enumerate(simbolos, 1):
                    score = float(vetor[indice])
                    disponivel = math.isfinite(score)
                    escritor.writerow((
                        str(data), estado.fold_id, ativo,
                        score if disponivel else "",
                        ranking.get(indice, ""), int(disponivel),
                    ))
                    quantidade += 1
                caixa = float(vetor[0])
                escritor.writerow((
                    str(data), estado.fold_id, "CASH",
                    caixa if math.isfinite(caixa) else "",
                    0, int(math.isfinite(caixa)),
                ))
                quantidade += 1
    print(f"[EVIDENCIAS] utilidades OOS completas: {quantidade} registros", flush=True)


def exportar_cotacoes(contexto: dict, pasta: Path) -> None:
    """OHLCV normalizado de todos os 75 ativos, inclusive treino/calibracao."""
    destino = pasta / "cotacoes_ohlcv"
    destino.mkdir(parents=True, exist_ok=True)
    conjuntos = {**contexto["frames_base"], **contexto["quadros"]}
    if set(conjuntos) != set(contexto["simbolos"]) or len(conjuntos) != 75:
        raise RuntimeError("Faltam series de mercado para o pacote de analise.")
    for ativo in contexto["simbolos"]:
        frame = conjuntos[ativo]
        colunas = ("open", "high", "low", "close", "volume")
        if not all(coluna in frame.columns for coluna in colunas):
            raise RuntimeError(f"Cotacoes incompletas de {ativo}")
        frame.loc[:, list(colunas)].to_csv(
            destino / f"{ativo}.csv", index=True, float_format="%.17g",
        )
    print(f"[EVIDENCIAS] 75 series OHLCV no pacote", flush=True)


def empacotar_e_verificar(pasta: Path, destino: Path) -> tuple[int, str]:
    """ZIP atomico, lista SHA-256 por arquivo e verificacao integral de leitura."""
    arquivos = sorted(
        (item for item in pasta.rglob("*") if item.is_file()
         and item.name != "arquivos_sha256.json"),
        key=lambda item: item.relative_to(pasta).as_posix(),
    )
    if not arquivos:
        raise RuntimeError("Pacote vazio. Nenhum ZIP sera publicado.")
    checksums = {
        item.relative_to(pasta).as_posix(): {
            "sha256": _hash_arquivo(item), "bytes": item.stat().st_size,
        }
        for item in arquivos
    }
    _salvar_json(pasta / "arquivos_sha256.json", {
        "algoritmo": "SHA-256",
        "arquivos": checksums,
    })
    arquivos.append(pasta / "arquivos_sha256.json")
    temporario = destino.with_suffix(".zip.tmp")
    if temporario.exists():
        temporario.unlink()
    try:
        with ZipFile(temporario, "w", compression=ZIP_DEFLATED,
                     compresslevel=6, allowZip64=True) as zipado:
            for arquivo in arquivos:
                zipado.write(
                    arquivo,
                    arcname=f"pesquisa_permanencia_minima/"
                            f"{arquivo.relative_to(pasta).as_posix()}",
                )
        with ZipFile(temporario, "r") as zipado:
            if zipado.testzip() is not None:
                raise RuntimeError("Arquivo ZIP corrompido (CRC).")
            presentes = set(zipado.namelist())
            esperados = {
                f"pesquisa_permanencia_minima/{item.relative_to(pasta).as_posix()}"
                for item in arquivos
            }
            if presentes != esperados:
                raise RuntimeError("Arquivo ZIP com conteudo divergente.")
            for nome, registro in checksums.items():
                conteudo = zipado.read(f"pesquisa_permanencia_minima/{nome}")
                if hashlib.sha256(conteudo).hexdigest() != registro["sha256"]:
                    raise RuntimeError(f"Falha na integridade SHA-256: {nome}")
        return len(arquivos), _hash_arquivo(temporario)
    except Exception:
        temporario.unlink(missing_ok=True)
        raise


def publicar_resultado(pasta: Path) -> None:
    """So publica output e ZIP depois de validar completamente o pacote."""
    quantidade, digest = empacotar_e_verificar(pasta, ARQUIVO_ZIP)
    if SAIDA.exists():
        shutil.rmtree(SAIDA)
    pasta.replace(SAIDA)
    ARQUIVO_ZIP.with_suffix(".zip.tmp").replace(ARQUIVO_ZIP)
    print(f"[ZIP] {ARQUIVO_ZIP} | arquivos={quantidade} "
          f"sha256={digest}", flush=True)


# %% 6. Mesmo arquivo/branch para os proximos experimentos. Nao tocar em main.
def main() -> None:
    if MODO not in ("verificar", "comparar"):
        raise ValueError("MODO precisa ser 'verificar' ou 'comparar'.")
    inicio = time.perf_counter()
    pesquisa, contrato, selecionados = verificar_fontes()
    if MODO == "verificar":
        print("[OK] Cache historico conferido. Nenhum modelo foi treinado.")
        return

    # Nova execucao em pasta temporaria: um erro nao transforma arquivos
    # parciais em pacote de resultados concluido.
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    if SAIDA_TEMP.exists():
        shutil.rmtree(SAIDA_TEMP)
    SAIDA_TEMP.mkdir(parents=True)
    ARQUIVO_ZIP.unlink(missing_ok=True)
    ARQUIVO_ZIP.with_suffix(".zip.tmp").unlink(missing_ok=True)

    contexto = preparar_contexto(pesquisa, contrato, selecionados)
    controle, metricas_base, calibracoes, margens = calibrar_controle(
        pesquisa, contexto, selecionados, contrato,
    )
    pd.DataFrame(calibracoes).to_csv(
        SAIDA_TEMP / "margens_controle_por_fold.csv", index=False,
    )
    contexto["continuidade"].to_csv(
        SAIDA_TEMP / "auditoria_continuidade_candidatos.csv", index=False,
    )
    # Fonte de precos, vetor de pontuacoes e parametros para analise
    # independente. Nunca incluir .env, credenciais ou dados de contas Live.
    exportar_cotacoes(contexto, SAIDA_TEMP)
    exportar_utilidades_oos(pesquisa, contexto, SAIDA_TEMP)
    shutil.copyfile(
        MANIFESTO_CONGELADO,
        SAIDA_TEMP / "manifesto_universo_congelado_origem.json",
    )
    shutil.copyfile(
        ROOT / "dados" / "u67" / "manifest.json",
        SAIDA_TEMP / "manifesto_snapshot_u67_origem.json",
    )
    shutil.copyfile(
        Path(__file__), SAIDA_TEMP / "codigo_experimento_executado.py",
    )
    _salvar_json(SAIDA_TEMP / "parametros_motor_e_ambiente.json", {
        "configuracao_efetiva": asdict(contexto["config"]),
        "parametros_experimentais": {
            "variavel": "rotation_min_holding_days",
            "cenarios": list(PERMANENCIAS),
            "controle": PERMANENCIA_CONTROLE,
            "margem_congelada_por_fold": margens,
            "todos_os_folds_reusam_os_mesmos_modelos": True,
            "recalibra_modelos_por_cenario": False,
            "apenas_replay_altera_posicao_e_capital": True,
        },
        "ambiente": {
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": _versao_pacote("lightgbm"),
            "scikit_learn": _versao_pacote("scikit-learn"),
        },
        "fontes": {
            "script_sha256": _hash_arquivo(Path(__file__)),
            "motor_rotacao_sha256": _hash_arquivo(ROOT / "engine" / "rotacao.py"),
            "motor_modelo_sha256": _hash_arquivo(
                ROOT / "engine" / "modelo_lightgbm.py"),
            "expansao_sha256": _hash_arquivo(
                ROOT / "pesquisar_expansao_u67_spyder.py"),
            "snapshot_sha256": contrato["snapshot_sha256"],
            "hashes_candidatos": contrato["candidate_file_hashes"],
        },
    })

    referencia = float(metricas_base["ending_capital"])
    linhas = []
    auditoria = {}
    series = []
    por_fold = []
    for minimo in (2, 0, 1, 3):
        print(f"[REPLAY] minimo={minimo} pregoes", flush=True)
        result, metrics, diagnostics = simular_contrafactual(
            pesquisa, contexto, minimo, margens,
        )
        if minimo == 2:
            if not math.isclose(float(metrics["ending_capital"]), referencia,
                                rel_tol=0, abs_tol=TOLERANCIA_CENTAVO):
                raise RuntimeError(
                    "O replay isolado de 2 dias nao coincide com o motor oficial. "
                    "Nao comparar cenarios."
                )
        linhas.append(_linha_resumo(minimo, metrics, referencia))
        # Caminho financeiro INTEIRO recalculado para cada politica.
        result.predictions.to_csv(
            SAIDA_TEMP / f"patrimonio_diario_{minimo}_pregoes.csv",
            index=True, float_format="%.17g",
        )
        result.trades.to_csv(
            SAIDA_TEMP / f"operacoes_{minimo}_pregoes.csv",
            index=False, float_format="%.17g",
        )
        diagnosticos = pd.DataFrame([
            {"data": str(data), **row}
            for data, row in sorted(diagnostics.items())
        ])
        diagnosticos.to_csv(
            SAIDA_TEMP / f"decisoes_{minimo}_pregoes.csv", index=False,
        )
        if not {"selected_asset", "strategy_equity"}.issubset(
                result.predictions.columns):
            raise RuntimeError("Curva ou ativo ausente do replay.")
        serie = result.predictions[["strategy_equity", "selected_asset"]].rename(
            columns={
                "strategy_equity": f"patrimonio_{minimo}_pregoes",
                "selected_asset": f"ativo_{minimo}_pregoes",
            }
        )
        if serie.index.has_duplicates:
            raise RuntimeError("Datas duplicadas no replay diario.")
        series.append(serie)
        _salvar_json(
            SAIDA_TEMP / f"metricas_completas_{minimo}_pregoes.json", {
                "minimo_pregoes": minimo,
                "metricas_resumidas": metrics,
                "metricas_motor": result.metrics,
                "resumo_motor": result.summary,
                "motivos_decisao": (
                    diagnosticos["decision_reason"].value_counts(dropna=False).to_dict()
                    if "decision_reason" in diagnosticos else {}
                ),
            },
        )
        for fold in metrics.get("folds", []):
            por_fold.append({"minimo_pregoes": minimo, **fold})
        auditoria[str(minimo)] = {
            "folds": metrics.get("folds"),
            "patrimonio": float(metrics["ending_capital"]),
            "rotacoes": metrics.get("capital_rotations"),
            "dias_em_caixa": metrics.get("cash_days"),
            "arquivos_decisoes": f"decisoes_{minimo}_pregoes.csv",
            "arquivos_operacoes": f"operacoes_{minimo}_pregoes.csv",
            "arquivos_curva": f"patrimonio_diario_{minimo}_pregoes.csv",
        }

    resumo = pd.DataFrame(linhas).sort_values("minimo_pregoes")
    resumo.to_csv(
        SAIDA_TEMP / "comparacao_permanencia_0_1_2_3.csv", index=False,
    )
    pd.DataFrame(por_fold).to_csv(
        SAIDA_TEMP / "metricas_por_fold.csv", index=False,
    )
    comparacao_diaria = pd.concat(series, axis=1, join="inner")
    if len(comparacao_diaria) < 1000 or comparacao_diaria.isna().any().any():
        raise RuntimeError("Comparacao diaria com datas faltantes ou desalinhadas.")
    comparacao_diaria.index.name = "data"
    comparacao_diaria.to_csv(
        SAIDA_TEMP / "comparacao_diaria_ativos_e_patrimonio.csv",
        index=True, float_format="%.17g",
    )
    _salvar_json(SAIDA_TEMP / "manifesto_experimento.json", {
        "status": "completed",
        "concluido_em_utc": datetime.now(timezone.utc).isoformat(),
        "projeto": "tcc_mba_usp_data_science_analytics",
        "branch": "research/permanencia-minima-contrafactual-v1",
        "arquivo": Path(__file__).name,
        "modo": "contrafactual_exato_sem_retreino",
        "versao_motor": "TCC Control v1.22.1 | executor v1.23.0",
        "snapshot_base_sha256": contrato["snapshot_sha256"],
        "hashes_candidatos": contrato["candidate_file_hashes"],
        "ativos_solicitados": 77,
        "ativos_efetivos": 75,
        "folds": 3,
        "margens_fixadas_por_fold": margens,
        "capital_controle_verificado": referencia,
        "patrimonio_historico_esperado": contrato["expected_ending_capital"],
        "cenarios": auditoria,
        "segundos_execucao": time.perf_counter() - inicio,
        "limitacao": (
            "Avaliacao no mesmo conjunto historico usado para selecionar o universo; "
            "a comparacao nao valida, isoladamente, generalizacao futura."
        ),
    })
    (SAIDA_TEMP / "LEIA_ME_ANALISE.txt").write_text(
        "TCC - Pesquisa contrafactual de permanencia minima\n\n"
        "Arquivo unico: pesquisa_permanencia_minima.zip\n"
        "Cenarios: 0, 1, 2 e 3 pregoes. Baseline: 2 pregoes.\n"
        "Sem novos downloads; mesmos modelos e margens por fold.\n\n"
        "Para analisar:\n"
        "* comparacao_permanencia_0_1_2_3.csv: resultados gerais.\n"
        "* metricas_por_fold.csv: desempenho individual em cada fold.\n"
        "* comparacao_diaria_ativos_e_patrimonio.csv: carteira e riqueza diaria.\n"
        "* utilidades_oos_todos_ativos.csv: utilidade de TODOS os ativos por dia.\n"
        "* decisoes_N_pregoes.csv: motivo, margem e bloqueios da politica.\n"
        "* operacoes_N_pregoes.csv: compras, vendas e taxas.\n"
        "* patrimonio_diario_N_pregoes.csv: replay financeiro integral.\n"
        "* metricas_completas_N_pregoes.json: metricas do motor e resumo.\n"
        "* cotacoes_ohlcv/: series historicas usadas nos quatro replays.\n"
        "* margens_controle_por_fold.csv: calibracao original.\n"
        "* auditoria_continuidade_candidatos.csv: elegibilidade do universo.\n"
        "* parametros_motor_e_ambiente.json: parametros, versoes e hashes.\n"
        "* manifestos de origem, codigo executado e manifesto do experimento.\n"
        "* arquivos_sha256.json: verificacao SHA-256 de todos os arquivos.\n\n"
        "NAO ha credenciais, tokens, .env, conta pessoal ou acesso a corretora.\n"
        "Analise retrospectiva: nao prova generalizacao futura.\n",
        encoding="utf-8",
    )

    # So publicar depois de criar e verificar todos os arquivos.
    publicar_resultado(SAIDA_TEMP)
    print("\n" + resumo.to_string(index=False), flush=True)
    print(f"\n[FINALIZADO] ZIP pronto: {ARQUIVO_ZIP}", flush=True)


if __name__ == "__main__":
    main()
