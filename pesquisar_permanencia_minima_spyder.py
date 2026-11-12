"""TCC | Experimento contrafactual da permanencia minima: 0, 1, 2, 3 pregoes.

Este e o arquivo UNICO para todas as rodadas da pesquisa nesta branch:
    research/permanencia-minima-contrafactual-v1

Spyder: abra este arquivo e execute com F5.
Terminal: python pesquisar_permanencia_minima_spyder.py

Modos: MODO = "vantagem" (padrao; usa o ZIP historico),
"verificar" (hashes) ou "comparar" (reproduz os quatro cenarios).

A reproduçao utiliza o motor cientifico congelado do TCC (v1.23.0),
o mesmo conjunto de 77 ativos solicitados (75 efetivos), os mesmos
modelos, folds, margem calibrada por fold, custos e execucoes exatas.
No modo comparar, somente rotation_min_holding_days varia. No modo vantagem,
um estimador Ridge aprende a vantagem relativa de troca em folds anteriores.

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
MODO = "vantagem"  # "vantagem", "comparar" ou "verificar"
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
    if MODO == "vantagem":
        executar_pesquisa_vantagem()
        return
    if MODO not in ("verificar", "comparar"):
        raise ValueError("MODO precisa ser 'vantagem', 'verificar' ou 'comparar'.")
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



# %% 7. Nova pesquisa: vantagem relativa aprendida, sem bloqueio fixo de holding.
# Reusa as previsoes OOS da pesquisa anterior e o motor financeiro oficial.
# Fold 1 e controle; fold 2 treina no 1; fold 3 treina nos 1+2 (com purge).


def executar_pesquisa_vantagem() -> None:
    """Compara uma metapolitica causal contra o Control original, sem modificar o motor."""
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    from engine.configuracao import StandaloneBacktestConfig
    from engine.rotacao import _simular_exato
    from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia

    horizonte, alpha = 3, 20.0
    saida = ROOT / "output" / "pesquisa_vantagem_rotacao"
    pacote = ROOT / "output" / "pesquisa_permanencia_minima.zip"
    destino = ROOT / "output" / "pesquisa_vantagem_rotacao.zip"

    if not pacote.is_file():
        raise FileNotFoundError(
            f"Pacote historico ausente: {pacote}. Execute primeiro MODO='comparar'."
        )
    conteudo_original = pacote.read_bytes()
    prefixo = "pesquisa_permanencia_minima/"
    with ZipFile(BytesIO(conteudo_original)) as zipado:
        if zipado.testzip() is not None:
            raise RuntimeError("O ZIP historico esta corrompido.")

        def ler_csv(nome: str) -> pd.DataFrame:
            return pd.read_csv(BytesIO(zipado.read(prefixo + nome)), low_memory=False)

        manifesto = json.loads(zipado.read(prefixo + "manifesto_experimento.json"))
        if manifesto.get("status") != "completed" or manifesto.get("folds") != 3:
            raise ValueError("A pesquisa anterior nao concluiu os tres folds.")
        parametros = json.loads(zipado.read(
            prefixo + "parametros_motor_e_ambiente.json"
        ))["configuracao_efetiva"]
        if parametros["rotation_min_holding_days"] != 2:
            raise ValueError("O controle anterior precisa ter permanencia de 2 pregoes.")

        decisoes = ler_csv("decisoes_2_pregoes.csv")
        diario = ler_csv("patrimonio_diario_2_pregoes.csv")
        pontuacoes = ler_csv("utilidades_oos_todos_ativos.csv")
        datas = pd.DatetimeIndex(pd.to_datetime(decisoes["data"], utc=True))
        execucoes = pd.DatetimeIndex(pd.to_datetime(diario["timestamp"], utc=True))
        if len(datas) != len(execucoes) or len(datas) < 1000:
            raise ValueError("Quantidade de datas incorreta no pacote historico.")
        if not datas[1:].equals(execucoes[:-1]) or not datas.is_unique:
            raise ValueError("As execucoes devem ocorrer na proxima sessao de mercado.")

        calendario = datas.append(pd.DatetimeIndex([execucoes[-1]]))
        fold = decisoes["decision_fold_id"].astype(int).to_numpy()
        if sorted(set(fold)) != [1, 2, 3]:
            raise ValueError("Folds diferentes do contrato original.")
        if not np.array_equal(fold, diario["walk_forward_fold"].astype(int)):
            raise ValueError("Folds do controle e do patrimonio divergentes.")

        ativos = sorted(set(pontuacoes["ativo"].astype(str)).difference({"CASH"}))
        if len(ativos) != 75:
            raise ValueError("O universo precisa conter 75 ativos efetivos.")
        nomes = {ativo: indice for indice, ativo in enumerate(ativos)}
        # Caso um timestamp apareca em varios caches, usa apenas o fold dono da decisao.
        pertence_fold = dict(zip(datas, fold))
        pontuacoes["data_ts"] = pd.to_datetime(pontuacoes["data_decisao"], utc=True)
        pontuacoes = pontuacoes.loc[
            pontuacoes["fold_id"].astype(int)
            == pontuacoes["data_ts"].map(pertence_fold)
        ]
        if pontuacoes.duplicated(["data_ts", "ativo"]).any():
            raise ValueError("Utilidades duplicadas numa mesma data/ativo.")
        matriz_utilidade = pontuacoes.loc[
            pontuacoes["ativo"] != "CASH"
        ].pivot(index="data_ts", columns="ativo", values="utilidade").reindex(
            index=datas, columns=ativos,
        ).to_numpy(dtype=float)
        if not np.isfinite(matriz_utilidade).any(axis=1).all():
            raise ValueError("Algum pregao nao tem utilidades validas.")

        quadros = {}
        for ativo in ativos:
            frame = ler_csv(f"cotacoes_ohlcv/{ativo}.csv")
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
            if frame.index.has_duplicates:
                raise ValueError(f"Historico com duplicidades: {ativo}.")
            quadros[ativo] = frame[[
                "open", "high", "low", "close", "volume"
            ]].apply(pd.to_numeric, errors="coerce").sort_index()

        if any(
            quadros[ativo].reindex(calendario)[["open", "close"]].isna().any().any()
            for ativo in ativos
        ):
            raise ValueError("Faltam aberturas/fechamentos nas datas de execucao.")

    # Features calculadas somente a partir de informacoes ate o fechamento corrente.
    close = pd.DataFrame({a: quadros[a]["close"] for a in ativos}).reindex(calendario)
    opening = pd.DataFrame({a: quadros[a]["open"] for a in ativos}).reindex(calendario)
    historico_close = pd.DataFrame({a: quadros[a]["close"] for a in ativos})
    historico_volume = pd.DataFrame({a: quadros[a]["volume"] for a in ativos})
    retorno5 = historico_close.pct_change(
        5, fill_method=None
    ).reindex(datas).to_numpy(dtype=float)
    volatilidade20 = np.log(
        historico_close / historico_close.shift(1)
    ).rolling(20, min_periods=20).std().reindex(datas).to_numpy(dtype=float)
    liquidez20 = (historico_close * historico_volume).rolling(
        20, min_periods=20
    ).mean().reindex(datas).to_numpy(dtype=float)
    ohlc_open = opening.to_numpy(dtype=float)
    ohlc_close = close.to_numpy(dtype=float)
    n = len(datas)
    atual_control = np.array([
        nomes.get(a, -1) for a in decisoes["current_asset"]
    ], dtype=int)
    alvo_control = np.array([
        nomes.get(a, -1) for a in decisoes["final_action_asset"]
    ], dtype=int)
    holding_control = decisoes["holding_days_at_decision"].astype(int).to_numpy()
    resultados_base = diario["strategy_equity"].astype(float).to_numpy()

    atributos = (
        "vantagem_utilidade", "mudanca_relativa_1", "mudanca_relativa_3",
        "retorno_relativo_5", "volatilidade_relativa_20",
        "liquidez_log_relativa_20", "log_idade_posicao", "utilidade_candidato",
    )

    def features(i: int, current: int, candidatos: np.ndarray,
                 idade: int) -> np.ndarray:
        j = np.asarray(candidatos, dtype=int)
        u = matriz_utilidade
        base = u[i, j] - u[i, current]

        def tendencia(k: int) -> np.ndarray:
            # Mudancas de pontuacoes em fronteiras de folds nao sao comparaveis.
            if i < k or fold[i] != fold[i - k]:
                return np.zeros(len(j))
            return (u[i, j] - u[i - k, j]) - (
                u[i, current] - u[i - k, current]
            )

        return np.column_stack((
            base,
            tendencia(1),
            tendencia(3),
            retorno5[i, j] - retorno5[i, current],
            volatilidade20[i, j] - volatilidade20[i, current],
            np.log((np.maximum(0, liquidez20[i, j]) + 1) /
                   (max(0, liquidez20[i, current]) + 1)),
            np.full(len(j), math.log1p(idade)),
            u[i, j],
        ))

    def linha_rotulos(i: int, current: int,
                      candidatos: np.ndarray) -> np.ndarray:
        j = np.asarray(candidatos, dtype=int)
        # Decisao no fechamento de i, entrada na abertura i+1.
        # Label futuro: vantagem no fechamento de i+3, nao uma feature.
        return np.log(
            ohlc_close[i + horizonte, j] / ohlc_open[i + 1, j]
        ) - math.log(
            ohlc_close[i + horizonte, current] / ohlc_open[i + 1, current]
        )

    linhas_treino = {1: [], 2: []}
    for i in range(n - horizonte):
        fold_id = int(fold[i])
        current = int(atual_control[i])
        if fold_id not in linhas_treino or current < 0:
            continue
        js = np.flatnonzero(np.isfinite(matriz_utilidade[i]))
        js = js[js != current]
        if not len(js):
            continue
        x = features(i, current, js, int(holding_control[i]))
        y = linha_rotulos(i, current, js)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if np.any(ok):
            linhas_treino[fold_id].append((i, x[ok], y[ok]))

    modelos, auditoria_treino = {}, {}
    for destino_fold in (2, 3):
        inicio_teste = int(np.flatnonzero(fold == destino_fold)[0])
        treino = [
            amostra for treino_fold in (1, 2) if treino_fold < destino_fold
            for amostra in linhas_treino[treino_fold]
            if amostra[0] + horizonte < inicio_teste
        ]
        if not treino:
            raise RuntimeError(
                f"Treinamento sem amostras anteriores ao fold {destino_fold}."
            )
        X = np.vstack([item[1] for item in treino])
        y = np.concatenate([item[2] for item in treino])
        if len(y) < 300:
            raise RuntimeError(f"Treinamento insuficiente no fold {destino_fold}.")
        lim_inferior, lim_superior = np.percentile(y, [1, 99])
        modelo = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        modelo.fit(X, np.clip(y, lim_inferior, lim_superior))
        modelos[destino_fold] = modelo
        auditoria_treino[destino_fold] = {
            "n_amostras": int(len(y)),
            "fold_teste_inicia": str(datas[inicio_teste]),
            "ultimo_rotulo_termina_antes": str(datas[inicio_teste]),
            "caracteristicas": list(atributos),
            "coeficientes_padronizados":
                modelo.named_steps["ridge"].coef_.tolist(),
            "intercepto": float(modelo.named_steps["ridge"].intercept_),
            "percentis_winsorizacao":
                [float(lim_inferior), float(lim_superior)],
        }
    print(
        f"[VANTAGEM] treinamento: fold2={auditoria_treino[2]['n_amostras']} "
        f"fold3={auditoria_treino[3]['n_amostras']}", flush=True,
    )

    # Motor financeiro do proprio TCC. A paridade do Control e obrigatoria.
    campos = StandaloneBacktestConfig.__dataclass_fields__
    config = StandaloneBacktestConfig(**{
        chave: valor for chave, valor in parametros.items() if chave in campos
    }).copiar_modelo(update={"assets": tuple(ativos)})
    calendario_exec = datas.append(pd.DatetimeIndex([execucoes[-1]]))
    metadados = {
        pd.Timestamp(data): {"fold_id": int(fold[i])}
        for i, data in enumerate(datas)
    }
    scores_originais = pd.to_numeric(
        decisoes["final_action_score"], errors="coerce"
    ).to_numpy(dtype=float)
    indice_data = {
        pd.Timestamp(data): i for i, data in enumerate(datas)
    }

    def politica_controle(data, posicao, holding):
        i = indice_data[pd.Timestamp(data)]
        idx = int(alvo_control[i])
        return (idx + 1 if idx >= 0 else 0), float(scores_originais[i])

    def simular(politica, diagnostics, titulo):
        return _simular_exato(
            "tcc_research_switch_advantage", politica, quadros, ativos,
            calendario_exec, config, calcular_taxas_referencia,
            aplicar_deslizamento, decision_metadata=metadados,
            policy_decision_diagnostics=diagnostics, model_label=titulo,
            method_line="- Pesquisa offline; sem holding minimo nos folds 2 e 3.",
        )

    baseline = simular(
        politica_controle, {}, "Control 2 pregoes - replay auditorio",
    )
    erro = np.abs(
        baseline.predictions["strategy_equity"].to_numpy() - resultados_base
    )
    if (
        float(np.max(erro)) > 0.01 or
        abs(float(baseline.metrics["strategy_ending_capital"])
            - float(manifesto["capital_controle_verificado"])) > 0.01
    ):
        raise RuntimeError(
            "Replay do Control nao reproduziu os dados historicos: "
            f"erro maximo US$ {float(np.max(erro)):.6f}. "
            "Nenhum contrafactual pode ser aceito."
        )
    print("[CONTROLE] reproducao validada ate o centavo", flush=True)

    saidas_decisao = []
    rankings_completos = []
    diagnosticos_vantagem = {}

    def custo_rotacao(indice: int, posicao: int, candidato: int) -> float:
        # Aproximacao proporcional ex ante. O motor contabiliza taxas reais.
        # Usa apenas o fechamento conhecido no dia da decisao.
        pv = float(ohlc_close[indice, posicao])
        pc = float(ohlc_close[indice, candidato])
        deslizamento = max(0.0, float(config.slippage_bps)) / 10000
        return (
            2 * float(config.commission_rate)
            + float(config.sec_fee_rate)
            + float(config.taf_fee_per_share) / pv
            + float(config.cat_fee_per_share) / pv
            + float(config.cat_fee_per_share) / pc
            + 2 * deslizamento
        )

    def politica_vantagem(data, posicao, holding):
        i = indice_data[pd.Timestamp(data)]
        f = int(fold[i])
        if f == 1:
            return politica_controle(data, posicao, holding)
        if posicao <= 0:
            # Entrada em ativo a partir do caixa, ainda sem modelo de cash.
            u = matriz_utilidade[i]
            melhor = int(np.nanargmax(u))
            return melhor + 1, float(u[melhor])
        atual = posicao - 1
        candidatos = np.flatnonzero(np.isfinite(matriz_utilidade[i]))
        candidatos = candidatos[candidatos != atual]
        if not len(candidatos):
            return posicao, float(matriz_utilidade[i, atual])
        x = features(i, atual, candidatos, int(holding))
        validos = np.isfinite(x).all(axis=1)
        if not np.any(validos):
            return posicao, float(matriz_utilidade[i, atual])
        candidatos = candidatos[validos]
        valor_predito = modelos[f].predict(x[validos])
        custos = np.array([
            custo_rotacao(i, atual, int(j)) for j in candidatos
        ], dtype=float)
        liquidas = valor_predito - custos
        ordem = np.argsort(-liquidas)
        escolhido = int(ordem[0])
        escolhido_ativo = int(candidatos[escolhido])
        alvo = escolhido_ativo + 1 if liquidas[escolhido] > 0 else posicao

        # Ranking operacional tem score de vantagem relativa, nao utilidade bruta.
        for rank, k in enumerate(ordem, start=1):
            rankings_completos.append({
                "data_decisao": str(data),
                "fold": f,
                "rank_decisao": rank,
                "atual": ativos[atual],
                "candidato": ativos[int(candidatos[k])],
                "utilidade_bruta":
                    float(matriz_utilidade[i, int(candidatos[k])]),
                "vantagem_prevista": float(valor_predito[k]),
                "custo_estimado": float(custos[k]),
                "vantagem_liquida": float(liquidas[k]),
                "escolhido": int(alvo == int(candidatos[k]) + 1),
            })
        rankings_completos.append({
            "data_decisao": str(data), "fold": f, "rank_decisao": 0,
            "atual": ativos[atual], "candidato": ativos[atual],
            "utilidade_bruta": float(matriz_utilidade[i, atual]),
            "vantagem_prevista": 0.0,
            "custo_estimado": 0.0,
            "vantagem_liquida": 0.0,
            "escolhido": int(alvo == posicao),
        })
        ranking = [
            {"ativo": ativos[int(candidatos[k])],
             "vantagem_prevista": float(valor_predito[k]),
             "custo_estimado": float(custos[k]),
             "vantagem_liquida": float(liquidas[k])}
            for k in ordem[:10]
        ]
        diagnosticos_vantagem[pd.Timestamp(data)] = {
            "decision_fold_id": f,
            "current_asset": ativos[atual],
            "holding_days_at_decision": holding,
            "raw_best_asset":
                ativos[int(np.nanargmax(matriz_utilidade[i]))],
            "best_asset": ativos[escolhido_ativo],
            "final_action_asset": ativos[alvo - 1],
            "decision_reason":
                "LEARNED_SWITCH" if alvo != posicao else "LEARNED_HOLD",
            "predicted_switch_advantage": float(valor_predito[escolhido]),
            "estimated_switch_cost": float(custos[escolhido]),
            "decision_advantage_net": float(liquidas[escolhido]),
            "best_candidates_json": json.dumps(ranking, ensure_ascii=False),
            "min_hold_guard_applied": False,
        }
        saidas_decisao.append({
            "data": str(data), "fold": f,
            "posicao_atual": ativos[atual],
            "alvo": ativos[alvo - 1],
            "vantagem_liquida": float(liquidas[escolhido]),
            "ranking": json.dumps(ranking, ensure_ascii=False),
        })
        return alvo, float(matriz_utilidade[i, alvo - 1])

    novo = simular(
        politica_vantagem, diagnosticos_vantagem,
        "Vantagem estimada - sem holding minimo",
    )
    resultado_fold = []
    for f in (1, 2, 3):
        ids = np.flatnonzero(fold == f)
        inicial = (
            float(config.initial_capital) if ids[0] == 0 else
            float(novo.predictions["strategy_equity"].iloc[ids[0] - 1])
        )
        final = float(novo.predictions["strategy_equity"].iloc[ids[-1]])
        inicio_control = (
            float(config.initial_capital) if ids[0] == 0 else
            float(resultados_base[ids[0] - 1])
        )
        resultado_fold.append({
            "fold": f, "novo_fator": final / inicial,
            "control_fator": float(resultados_base[ids[-1]]) / inicio_control,
            "novo_final": final,
            "control_final": float(resultados_base[ids[-1]]),
        })

    # Nenhuma saida de um teste incompleto pode ser apresentada como concluida.
    destino.unlink(missing_ok=True)
    if saida.exists():
        shutil.rmtree(saida)
    saida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saidas_decisao).to_csv(
        saida / "decisoes_vantagem.csv", index=False,
    )
    pd.DataFrame(rankings_completos).to_csv(
        saida / "ranking_vantagem_por_ativo.csv", index=False,
    )
    pd.DataFrame(resultado_fold).to_csv(
        saida / "comparacao_por_fold.csv", index=False,
    )
    novo.predictions.to_csv(
        saida / "patrimonio_diario_vantagem.csv", index=True,
    )
    novo.trades.to_csv(
        saida / "operacoes_vantagem.csv", index=False,
    )
    pd.DataFrame({
        "data": execucoes, "control": resultados_base,
        "vantagem": novo.predictions["strategy_equity"].to_numpy(),
        "fold": fold,
    }).to_csv(saida / "comparacao_diaria.csv", index=False)
    _salvar_json(saida / "treinamento_meta_modelo.json", auditoria_treino)
    _salvar_json(saida / "resumo_experimento.json", {
        "status": "completed", "tipo": "comparacao_temporal_de_politicas",
        "capital_control": float(resultados_base[-1]),
        "capital_vantagem":
            float(novo.metrics["strategy_ending_capital"]),
        "rotacoes_control": int(baseline.metrics["capital_rotations"]),
        "rotacoes_vantagem": int(novo.metrics["capital_rotations"]),
        "erro_maximo_reproducao_control": float(np.max(erro)),
        "hiperparametros": {
            "horizonte": horizonte, "ridge_alpha": alpha,
            "features": atributos, "min_holding": 0,
        },
        "arquivo_origem_sha256":
            hashlib.sha256(conteudo_original).hexdigest(),
        "avaliacao":
            "fold 1 usa Control; folds 2 e 3 usam politica aprendida",
        "metodo": "Ridge pairwise treinado em folds anteriores",
        "limitacoes": [
            "Treino com posicoes Control: divergencia de distribuicao possivel",
            "Universo previamente selecionado no mesmo periodo historico",
            "Nao modela impacto de mercado nem limite de participacao no volume",
            "Custo de decisao proporcional e aproximado; motor apura taxas reais",
            "Fold 1 mantido como Control exclusivamente para treinamento",
        ],
    })
    hashes = {
        arquivo.name: _hash_arquivo(arquivo)
        for arquivo in sorted(saida.iterdir()) if arquivo.is_file()
    }
    _salvar_json(saida / "arquivos_sha256.json", hashes)

    tmp = destino.with_suffix(".zip.tmp")
    tmp.unlink(missing_ok=True)
    with ZipFile(
        tmp, "w", compression=ZIP_DEFLATED, compresslevel=6,
        allowZip64=True,
    ) as zip_final:
        for arquivo in sorted(saida.rglob("*")):
            if arquivo.is_file():
                zip_final.write(
                    arquivo, "pesquisa_vantagem_rotacao/" + arquivo.name,
                )
        zip_final.write(pacote, "dados_origem/" + pacote.name)
    with ZipFile(tmp) as zip_final:
        if zip_final.testzip() is not None:
            raise RuntimeError("ZIP do experimento falhou na verificacao de CRC.")
        for nome, esperado in hashes.items():
            dado = zip_final.read("pesquisa_vantagem_rotacao/" + nome)
            if hashlib.sha256(dado).hexdigest() != esperado:
                raise RuntimeError(f"SHA diferente para {nome}")
    tmp.replace(destino)
    print(
        f"[VANTAGEM] Control US$ {resultados_base[-1]:,.2f} | "
        f"aprendido US$ {novo.metrics['strategy_ending_capital']:,.2f}",
        flush=True,
    )
    print(f"[ZIP] {destino}", flush=True)

if __name__ == "__main__":
    main()
