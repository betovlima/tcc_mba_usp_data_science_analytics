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

# %% 7. MODO "vantagem": pesquisa pareada cronologica independente.
# O ZIP usado como entrada e aquele produzido pelo MODO "comparar" do TCC.
# Nunca importar a engine de producao nem alterar a politica original.
"""Pesquisa isolada de vantagem de rotacao para Winner (NUNCA opera Live).

Execucao:
    poetry run python scripts/pesquisar_permanencia_minima.py
    poetry run python scripts/pesquisar_permanencia_minima.py --pacote caminho/arquivo.zip

Fonte: ZIP gerado pelo TCC pesquisar_permanencia_minima_spyder.py.
As previsoes do modelo-base ja sao fora da amostra; o segundo estimador Ridge usa
somente folds ANTERIORES e purge de H sessoes. Fold 1: Control original;
folds 2 e 3: politica de vantagem aprendida. Sem permanencia minima,
sem otimizar parametros nos folds de teste, sem alteracoes de producao.

Rotacao = argmax de vantagem esperada liquida contra MANTER (valor 0).
Nao se usa a utilidade original como previsao percentual de retorno.
O custo estimado e calculado a partir das taxas e dos precos conhecidos
na decisao. O efeito do impacto de mercado continua NAO MODELADO.
Saida: output/pesquisa_vantagem_rotacao.zip (inclui ZIP-fonte completo).
Este modo compartilha ESTE MESMO arquivo e branch com a pesquisa de permanencia.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DESTINO = ROOT / "output" / "pesquisa_vantagem_rotacao"
H = 3
TOP_K = 6
RIDGE_ALPHA = 20.0
FEATURES = (
    "gap_utilidade", "diferenca_tendencia_1", "diferenca_tendencia_3",
    "diferenca_retorno_5", "diferenca_volatilidade_20",
    "log_razão_liquidez_20", "log_tempo_posicao", "utilidade_candidato",
)
NOME_ZIP = "pesquisa_permanencia_minima.zip"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n", encoding="utf-8",
    )


@dataclass
class Base:
    dates: pd.DatetimeIndex
    exec_dates: pd.DatetimeIndex
    fold: np.ndarray
    symbols: list[str]
    scores: np.ndarray
    op: np.ndarray
    cl: np.ndarray
    volume: np.ndarray
    vol20: np.ndarray
    ret5: np.ndarray
    adv20: np.ndarray
    base_actions: np.ndarray
    base_current: np.ndarray
    base_holding: np.ndarray
    base_equity: np.ndarray
    config: dict
    original: bytes
    original_sha: str
    margens: dict[int, float]


def _read(z: ZipFile, member: str) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(z.read("pesquisa_permanencia_minima/" + member)),
        low_memory=False,
    )


def carregar(pacote: Path) -> Base:
    origem = pacote.read_bytes()
    with ZipFile(io.BytesIO(origem)) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP de origem corrompido.")
        manifest = json.loads(z.read(
            "pesquisa_permanencia_minima/manifesto_experimento.json"
        ))
        if manifest.get("status") != "completed" or manifest.get("folds") != 3:
            raise ValueError("ZIP nao e um experimento completo de tres folds.")
        params = json.loads(z.read(
            "pesquisa_permanencia_minima/parametros_motor_e_ambiente.json"
        ))["configuracao_efetiva"]
        if int(params["rotation_min_holding_days"]) != 2:
            raise ValueError("O Control precisa ter permanencia de dois pregoes.")
        decisions = _read(z, "decisoes_2_pregoes.csv")
        daily = _read(z, "patrimonio_diario_2_pregoes.csv")
        util = _read(z, "utilidades_oos_todos_ativos.csv")
        margin = _read(z, "margens_controle_por_fold.csv")
        dates = pd.DatetimeIndex(pd.to_datetime(decisions["data"], utc=True))
        exec_dates = pd.DatetimeIndex(pd.to_datetime(daily["timestamp"], utc=True))
        if len(dates) != len(exec_dates) or len(dates) < 1000:
            raise ValueError("Calendario de decisoes/investimento nao corresponde.")
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError("Calendario de decisoes invalido.")
        if not (exec_dates[:-1].equals(dates[1:]) and exec_dates[-1] > dates[-1]):
            raise ValueError("Execucao precisa ocorrer no pregao seguinte.")
        timeline = dates.append(pd.DatetimeIndex([exec_dates[-1]]))
        fold = decisions["decision_fold_id"].astype(int).to_numpy()
        if sorted(set(fold)) != [1, 2, 3]:
            raise ValueError("Folds inesperados.")
        if not np.array_equal(fold, daily["walk_forward_fold"].astype(int)):
            raise ValueError("Folds nao correspondem entre os arquivos.")
        utility_symbols = sorted(set(util["ativo"].astype(str)).difference({"CASH"}))
        if len(utility_symbols) != 75:
            raise ValueError(f"Esperados 75 ativos, encontrados {len(utility_symbols)}.")
        lookup = {symbol: i for i, symbol in enumerate(utility_symbols)}
        # A primeira data do fold seguinte tambem aparece no cache do fold anterior.
        # Selecionar SOMENTE a previsao do fold ao qual a decisao pertence.
        expected_fold = dict(zip(dates, fold))
        expected_fold[exec_dates[-1]] = 3
        util["ts"] = pd.to_datetime(util["data_decisao"], utc=True)
        util = util.loc[
            util["fold_id"].astype(int)
            == util["ts"].map(expected_fold)
        ]
        if util.duplicated(["ts", "ativo"]).any():
            raise ValueError("Scores duplicados na mesma decisao/fold.")
        scores = util.loc[util["ativo"] != "CASH"].pivot(
            index="ts", columns="ativo", values="utilidade",
        ).reindex(index=timeline, columns=utility_symbols).to_numpy(dtype=float)
        n, m = len(timeline), len(utility_symbols)
        op = np.empty((n, m), dtype=float)
        cl = np.empty((n, m), dtype=float)
        volume = np.empty((n, m), dtype=float)
        for j, symbol in enumerate(utility_symbols):
            frame = _read(z, f"cotacoes_ohlcv/{symbol}.csv")
            frame.index = pd.to_datetime(frame["timestamp"], utc=True)
            if frame.index.has_duplicates:
                raise ValueError(f"Preco duplicado para {symbol}.")
            aligned = frame.reindex(timeline)
            for target, column in ((op, "open"), (cl, "close"), (volume, "volume")):
                target[:, j] = aligned[column].to_numpy(dtype=float)
        if not (np.isfinite(op).all() and np.isfinite(cl).all()
                and (op > 0).all() and (cl > 0).all()):
            raise ValueError("OHLCV incompleto; nao preencher precos artificialmente.")
        if not np.isfinite(scores[:-1]).any(axis=1).all():
            raise ValueError("Faltam utilidades para algumas decisoes.")
        action = np.array([
            lookup.get(str(a), -1) for a in decisions["final_action_asset"]
        ], dtype=int)
        current = np.array([
            lookup.get(str(a), -1) for a in decisions["current_asset"]
        ], dtype=int)
        holds = decisions["holding_days_at_decision"].astype(int).to_numpy()
        margins = margin[margin["selected"].astype(str).str.lower() == "true"]
        mapped = {
            int(r.fold_id): float(r.candidate_margin)
            for r in margins.itertuples()
        }
        if len(mapped) != 3:
            raise ValueError("Margens calibradas nao conferem.")
        logreturns = np.full_like(cl, np.nan)
        logreturns[1:] = np.log(cl[1:] / cl[:-1])
        vol20 = pd.DataFrame(logreturns).rolling(20, min_periods=20).std().to_numpy()
        ret5 = pd.DataFrame(cl).pct_change(5).to_numpy()
        adv20 = pd.DataFrame(cl * volume).rolling(
            20, min_periods=20,
        ).mean().to_numpy()
        return Base(
            dates, exec_dates, fold, utility_symbols, scores, op, cl, volume,
            vol20, ret5, adv20, action, current, holds,
            daily["strategy_equity"].to_numpy(dtype=float),
            params, origem, _hash_bytes(origem), mapped,
        )


def _features(b: Base, i: int, current: int, candidate: int,
              holding: int) -> np.ndarray:
    u = b.scores
    return np.array((
        u[i, candidate] - u[i, current],
        (u[i, candidate] - u[i - 1, candidate])
        - (u[i, current] - u[i - 1, current]),
        (u[i, candidate] - u[i - 3, candidate])
        - (u[i, current] - u[i - 3, current]),
        b.ret5[i, candidate] - b.ret5[i, current],
        b.vol20[i, candidate] - b.vol20[i, current],
        math.log((b.adv20[i, candidate] + 1) / (b.adv20[i, current] + 1)),
        math.log1p(holding),
        u[i, candidate],
    ), dtype=float)


def _train_data(b: Base, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, outcomes, indices = [], [], []
    for i in range(25, len(b.dates) - H):
        if b.fold[i] != fold or b.base_current[i] < 0:
            continue
        cur = int(b.base_current[i])
        for candidate in np.argsort(-b.scores[i])[:TOP_K]:
            if candidate == cur or not np.isfinite(b.scores[i, candidate]):
                continue
            features = _features(b, i, cur, int(candidate),
                                 int(b.base_holding[i]))
            if not np.isfinite(features).all():
                continue
            # Decisao no fechamento i; compra na abertura i+1.
            # Retorno comparado ate o fechamento i+H; nenhum futuro nas features.
            label = math.log(b.cl[i + H, candidate] / b.op[i + 1, candidate])
            label -= math.log(b.cl[i + H, cur] / b.op[i + 1, cur])
            if math.isfinite(label):
                rows.append(features)
                outcomes.append(label)
                indices.append(i)
    return np.asarray(rows), np.asarray(outcomes), np.asarray(indices, dtype=int)


def treinar(b: Base):
    datasets = {fold: _train_data(b, fold) for fold in (1, 2)}
    models, audit = {}, {}
    for target in (2, 3):
        start = int(np.flatnonzero(b.fold == target)[0])
        # Purge: outcome precisa estar INTEIRAMENTE conhecido no inicio
        # do fold de teste. Nunca reusar rotulos com data futura.
        parts = [
            (x[dateidx + H < start], y[dateidx + H < start])
            for fold, (x, y, dateidx) in datasets.items() if fold < target
        ]
        X = np.vstack([item[0] for item in parts])
        y = np.concatenate([item[1] for item in parts])
        if len(y) < 300:
            raise ValueError(f"Treinamento insuficiente para fold {target}.")
        bottom, top = np.percentile(y, [1, 99])
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(X, np.clip(y, bottom, top))
        models[target] = model
        ridge = model.named_steps["ridge"]
        scaler = model.named_steps["standardscaler"]
        audit[target] = {
            "linhas_treino": len(y), "inicio_fold_teste": str(b.dates[start]),
            "fim_maximo_rotulo_treino": str(b.dates[start - 1]),
            "variaveis": list(FEATURES),
            "coeficientes_padronizados": ridge.coef_.tolist(),
            "intercepto": float(ridge.intercept_),
            "media_features": scaler.mean_.tolist(),
            "escala_features": scaler.scale_.tolist(),
            "winsorizacao_treino": [float(bottom), float(top)],
        }
    return models, audit


def _taxa(b: Base, lado: str, q: float, preco: float) -> float:
    if q <= 0 or preco <= 0:
        return 0.0
    conf = b.config
    arred = lambda value: (
        math.ceil((value - 1e-12) * 100) / 100 if value > 0 else 0.0
    )
    cat = arred(q * float(conf["cat_fee_per_share"]))
    comissao = arred(q * preco * float(conf["commission_rate"]))
    if lado == "BUY":
        return cat + comissao
    sec = arred(q * preco * float(conf["sec_fee_rate"]))
    taf = arred(min(q * float(conf["taf_fee_per_share"]), float(conf["taf_fee_cap"])))
    return cat + comissao + sec + taf


def _preco(b: Base, preco: float, lado: str) -> float:
    slippage = float(b.config["slippage_bps"]) / 10000
    return preco * (1 + slippage if lado == "BUY" else 1 - slippage)


def _custo_estimado(b: Base, i: int, cur: int, j: int,
                    capital: float, q: float) -> float:
    # Cotacoes no fechamento i sao conhecidas; a abertura i+1 NAO e.
    if capital <= 0:
        return float("inf")
    sell = _preco(b, b.cl[i, cur], "SELL")
    buy = _preco(b, b.cl[i, j], "BUY")
    sell_fee = _taxa(b, "SELL", q, sell)
    buy_fee = _taxa(b, "BUY", capital / buy, buy)
    spread = max(0.0, float(b.config["slippage_bps"])) / 10000
    return (sell_fee + buy_fee) / capital + 2 * spread


def _score_test_pairs(b: Base, models) -> pd.DataFrame:
    records = []
    for target, model in models.items():
        for i in np.flatnonzero(b.fold == target):
            cur = int(b.base_current[i])
            if i < 25 or cur < 0 or i + H >= len(b.op):
                continue
            for j in np.argsort(-b.scores[i])[:TOP_K]:
                if j == cur:
                    continue
                x = _features(b, i, cur, int(j), int(b.base_holding[i]))
                if not np.isfinite(x).all():
                    continue
                observed = math.log(b.cl[i + H, j] / b.op[i + 1, j])
                observed -= math.log(b.cl[i + H, cur] / b.op[i + 1, cur])
                records.append({
                    "data": b.dates[i], "fold": target,
                    "atual": b.symbols[cur], "candidato": b.symbols[int(j)],
                    "vantagem_prevista": float(model.predict(x.reshape(1, -1))[0]),
                    "vantagem_realizada_3_pregoes_sem_trocas": observed,
                })
    return pd.DataFrame(records)


def simular(b: Base, models=None):
    """Contabilidade equivalente a Control: abre i+1, fecha i+1, final liquida."""
    cash = float(b.config["initial_capital"])
    pos, quantity, holding = -1, 0.0, 0
    equity, decisions, trades = [], [], []
    total_fees = 0.0
    rotations = 0
    for i, date in enumerate(b.dates):
        target = int(b.base_actions[i]) if models is None or b.fold[i] == 1 else pos
        ranking = []
        if models is not None and b.fold[i] >= 2:
            if pos < 0:
                target = int(np.nanargmax(b.scores[i]))
            else:
                capital = cash + quantity * b.cl[i, pos]
                model = models[int(b.fold[i])]
                for cand in np.argsort(-b.scores[i])[:TOP_K]:
                    if cand == pos or not np.isfinite(b.scores[i, cand]):
                        continue
                    features = _features(b, i, pos, int(cand), holding)
                    if not np.isfinite(features).all():
                        continue
                    predicted = float(model.predict(features.reshape(1, -1))[0])
                    cost = _custo_estimado(
                        b, i, pos, int(cand), capital, quantity,
                    )
                    ranking.append({
                        "candidato": b.symbols[int(cand)],
                        "vantagem_prevista": predicted,
                        "custo_estimado": cost,
                        "vantagem_liquida": predicted - cost,
                    })
                ranking.sort(key=lambda x: x["vantagem_liquida"], reverse=True)
                # Manter tem vantagem incremental zero; nenhuma regra por tempo.
                if ranking and ranking[0]["vantagem_liquida"] > 0:
                    target = b.symbols.index(ranking[0]["candidato"])
        old = pos
        actions = []
        if target != pos:
            if pos >= 0:
                price = _preco(b, b.op[i + 1, pos], "SELL")
                charge = _taxa(b, "SELL", quantity, price)
                value = quantity * price
                cash += value - charge
                total_fees += charge
                trades.append({
                    "data": b.exec_dates[i],
                    "data_decisao": date, "fold": int(b.fold[i]), "acao": "SELL",
                    "ativo": b.symbols[pos], "quantidade": quantity,
                    "preco": price, "valor": value, "taxas": charge,
                    "volume_dia": b.volume[i + 1, pos],
                    "participacao_volume": (
                        quantity / b.volume[i + 1, pos]
                        if b.volume[i + 1, pos] > 0 else None
                    ),
                })
                actions.append("SELL")
            pos = target
            quantity = 0.0
            holding = 0
            if pos >= 0:
                price = _preco(b, b.op[i + 1, pos], "BUY")
                quantity = cash / price
                for _ in range(25):
                    fee = _taxa(b, "BUY", quantity, price)
                    updated = max(0.0, (cash - fee) / price)
                    if abs(updated - quantity) < 1e-10:
                        quantity = updated
                        break
                    quantity = updated
                fee = _taxa(b, "BUY", quantity, price)
                value = quantity * price
                cash -= value + fee
                total_fees += fee
                trades.append({
                    "data": b.exec_dates[i],
                    "data_decisao": date, "fold": int(b.fold[i]), "acao": "BUY",
                    "ativo": b.symbols[pos], "quantidade": quantity,
                    "preco": price, "valor": value, "taxas": fee,
                    "volume_dia": b.volume[i + 1, pos],
                    "participacao_volume": (
                        quantity / b.volume[i + 1, pos]
                        if b.volume[i + 1, pos] > 0 else None
                    ),
                })
                actions.append("BUY")
                holding = 1
            if old >= 0 and target >= 0:
                rotations += 1
        elif pos >= 0:
            holding += 1
        amount = cash + (
            quantity * b.cl[i + 1, pos] if pos >= 0 else 0.0
        )
        equity.append(float(amount))
        decisions.append({
            "data_decisao": date, "fold": int(b.fold[i]),
            "posicao_antes": b.symbols[old] if old >= 0 else "CASH",
            "posicao_depois": b.symbols[pos] if pos >= 0 else "CASH",
            "holding_apos": holding,
            "acao": "+".join(actions) or "HOLD",
            "pontuacao_mais_alta_bruta": b.symbols[int(np.nanargmax(b.scores[i]))],
            "vantagem_melhor_candidato_liquida": (
                ranking[0]["vantagem_liquida"] if ranking else None
            ),
            "candidatos_avaliados": json.dumps(ranking, ensure_ascii=False),
            "patrimonio": float(amount),
        })
    if pos >= 0:
        price = _preco(b, b.cl[len(b.dates), pos], "SELL")
        fee = _taxa(b, "SELL", quantity, price)
        cash += quantity * price - fee
        total_fees += fee
        equity[-1] = cash
        decisions[-1]["patrimonio"] = cash
        trades.append({
            "data": b.exec_dates[-1], "data_decisao": b.dates[-1],
            "fold": int(b.fold[-1]), "acao": "FINAL_SELL",
            "ativo": b.symbols[pos], "quantidade": quantity, "preco": price,
            "valor": quantity * price, "taxas": fee,
            "volume_dia": b.volume[-1, pos],
            "participacao_volume": (
                quantity / b.volume[-1, pos] if b.volume[-1, pos] > 0 else None
            ),
        })
    return np.asarray(equity), pd.DataFrame(decisions), pd.DataFrame(trades), {
        "capital_final": float(equity[-1]),
        "rotacoes": int(rotations),
        "total_taxas": total_fees,
    }


def _performance_by_fold(b: Base, curve: np.ndarray) -> list[dict]:
    result = []
    for f in (1, 2, 3):
        positions = np.flatnonzero(b.fold == f)
        start = float(curve[positions[0] - 1]) if positions[0] else 10000.0
        end = float(curve[positions[-1]])
        daily = pd.Series(curve[positions]).pct_change().dropna()
        result.append({
            "fold": f, "inicio": str(b.dates[positions[0]]),
            "fim": str(b.dates[positions[-1]]),
            "fator_capital": end / start,
            "capital_inicio": start, "capital_final": end,
            "sharpe": (
                float(np.sqrt(252) * daily.mean() / daily.std())
                if daily.std() > 0 else None
            ),
        })
    return result


def _resumo_curva(curve: np.ndarray) -> dict:
    changes = pd.Series(curve).pct_change().dropna()
    maximum = np.maximum.accumulate(curve)
    return {
        "capital_final": float(curve[-1]),
        "drawdown_maximo": float(np.min(curve / maximum - 1)),
        "sharpe": (
            float(np.sqrt(252) * changes.mean() / changes.std())
            if changes.std() > 0 else None
        ),
    }


def _pacote(destino: Path, fonte: Path) -> Path:
    archive = destino.parent / (destino.name + ".zip")
    temp = archive.with_suffix(".zip.tmp")
    checksums = {
        file.relative_to(destino).as_posix(): _hash_bytes(file.read_bytes())
        for file in sorted(destino.rglob("*")) if file.is_file()
    }
    _write_json(destino / "arquivos_sha256.json", checksums)
    with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for file in sorted(destino.rglob("*")):
            if file.is_file():
                z.write(file, arcname=f"{destino.name}/{file.relative_to(destino)}")
        z.write(fonte, arcname="dados_origem/pesquisa_permanencia_minima.zip")
    with ZipFile(temp) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP falhou verificacao de CRC.")
        for file, expected in checksums.items():
            if _hash_bytes(z.read(f"{destino.name}/{file}")) != expected:
                raise RuntimeError(f"SHA incorreto: {file}")
    temp.replace(archive)
    return archive


def executar_vantagem() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote", type=Path, default=None)
    args, _ = parser.parse_known_args()
    pacote = args.pacote
    if pacote is None:
        options = [
            ROOT / "output" / NOME_ZIP,
            ROOT / "dados" / "experimentos" / "permanencia_minima" / "entradas" / NOME_ZIP,
            Path.home() / "Downloads" / NOME_ZIP,
            Path("/mnt/data") / NOME_ZIP,
        ]
        pacote = next((x for x in options if x.is_file()), None)
    if pacote is None or not pacote.is_file():
        raise FileNotFoundError(
            f"ZIP de origem nao encontrado: {NOME_ZIP}. "
            "Use --pacote caminho/do/ZIP."
        )
    b = carregar(pacote)
    print(f"[DADOS] {len(b.dates)} decisoes | 75 ativos | 3 folds")
    baseline, base_decisions, base_trades, base_stats = simular(b)
    error = float(np.max(np.abs(baseline - b.base_equity)))
    if error > 0.01 or abs(baseline[-1] - 587165941.1330212) > 0.01:
        raise RuntimeError(
            f"Control nao reproduzido: divergencia maxima US$ {error:.6f}. "
            "Nao comparar contrafactuais."
        )
    print(f"[PARIDADE] Control US$ {baseline[-1]:,.2f}; erro max {error:.8f}")
    models, training_audit = treinar(b)
    advantage_pairs = _score_test_pairs(b, models)
    learned, learned_decisions, learned_trades, learned_stats = simular(b, models)
    # Nunca empacotar artefatos de execucoes anteriores.
    import shutil
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "data_execucao": b.exec_dates,
        "fold": b.fold, "controle": baseline, "vantagem": learned,
    }).to_csv(DESTINO / "comparacao_patrimonio_diario.csv", index=False)
    base_decisions.to_csv(DESTINO / "decisoes_control.csv", index=False)
    learned_decisions.to_csv(DESTINO / "decisoes_vantagem.csv", index=False)
    base_trades.to_csv(DESTINO / "operacoes_control.csv", index=False)
    learned_trades.to_csv(DESTINO / "operacoes_vantagem.csv", index=False)
    advantage_pairs.to_csv(DESTINO / "vantagens_previstas_vs_realizadas.csv", index=False)
    _write_json(DESTINO / "modelos_treinados.json", training_audit)
    pairs_stats = {}
    for fold, frame in advantage_pairs.groupby("fold"):
        pairs_stats[int(fold)] = {
            "amostras": len(frame),
            "correlacao_spearman": float(frame["vantagem_prevista"].corr(
                frame["vantagem_realizada_3_pregoes_sem_trocas"], method="spearman",
            )),
            "positivos_previstos_pct": float(
                100 * (frame["vantagem_prevista"] > 0).mean()
            ),
        }
    _write_json(DESTINO / "relatorio.json", {
        "status": "completed_research_only",
        "criado_utc": datetime.now(timezone.utc),
        "modelo": "Ridge pairwise vantagem relativa de 3 pregoes",
        "hiperparametros": {
            "H": H, "top_k": TOP_K, "alpha": RIDGE_ALPHA,
            "winsorizar_treino_percentis": [1, 99],
            "treinamento_temporal": "Fold1->Fold2; Fold1+Fold2->Fold3",
            "fold1": "Control original, sem meta-modelo",
            "idade_posicao": "feature continua, nenhum minimo fixo",
            "rotacao": "predicao diferencial liquida > 0",
        },
        "paridade_control_max_erro_dolar": error,
        "origem_sha256": b.original_sha,
        "contabilizacao": "replay no fechamento/abertura com fees conforme Control",
        "controle": {**base_stats, **_resumo_curva(baseline)},
        "vantagem": {**learned_stats, **_resumo_curva(learned)},
        "folds_controle": _performance_by_fold(b, baseline),
        "folds_vantagem": _performance_by_fold(b, learned),
        "pares_oos": pairs_stats,
        "limitacoes": [
            "O universo foi selecionado a partir do historico; overfitting de selecao permanece.",
            "Sem impacto de mercado nem limite de participacao no volume; ganhos financeiros podem ser inexequiveis.",
            "Treino de pares condicionado a posicoes do Control, diferente da distribuicao da nova politica.",
            "O primeiro fold e identico ao Control e serve somente para treinar.",
            "A hipotese de retorno mantido por H pregoes ignora futuras novas rotacoes.",
            "Nenhuma autorizacao para uso na Winner Live.",
        ],
        "ambiente": {"python": sys.version.split()[0], "plataforma": platform.platform()},
    })
    archive = _pacote(DESTINO, pacote)
    print(f"[RESULTADO] Control=US$ {baseline[-1]:,.2f}; "
          f"vantagem=US$ {learned[-1]:,.2f}")
    print(f"[ZIP] {archive}")


if __name__ == "__main__":
    (executar_vantagem if MODO == "vantagem" else main)()
