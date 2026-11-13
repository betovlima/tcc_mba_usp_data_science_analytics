"""TCC | Experimento contrafactual da permanencia minima: 0, 1, 2, 3 pregoes.

Este e o arquivo UNICO para todas as rodadas da pesquisa nesta branch:
    research/permanencia-minima-contrafactual-v1

Spyder: abra este arquivo e execute com F5.
Terminal: python pesquisar_permanencia_minima_spyder.py

Modos: MODO = "permanencia_inteligente" (padrao; estudo da trava de dois pregoes),
"vantagem_economica" (comparacao economica), "analisar_permanencia",
"sensibilidade_margens", "deterioracao",
"vantagem_v3", "vantagem_v2", "vantagem" (historicos), "verificar" ou
"comparar" (reproduz quatro permanencias).

A reproduçao utiliza o motor cientifico congelado do TCC (v1.23.0),
o mesmo conjunto de 77 ativos solicitados (75 efetivos), os mesmos
modelos, folds, margem calibrada por fold, custos e execucoes exatas.
No modo comparar, somente rotation_min_holding_days varia.
No modo vantagem_v3, uma regressao pareada estima a vantagem de troca
considerando idade, retorno e risco da posicao efetiva. Um filtro causal
de Kalman compara o efeito de suavizar a pontuacao de cada ativo.
A calibracao e o treino usam apenas folds anteriores, com purge de 60 dias.
O primeiro fold preserva o Control; nao e usado para validar a V3.
O relatorio e sempre output/pesquisa_vantagem_rotacao.zip, substituido
somente apos a verificacao integral; nenhum sufixo vN e criado.
O modo deterioracao compara Control, saida para CASH e veto de entrada por
probabilidade de queda em 20 sessoes, treinado apenas em folds anteriores.
Os modos antigos de pesquisa permanecem disponiveis para reproducao.

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
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

# %% 0. Um arquivo, uma branch; somente este seletor muda entre modos.
MODO = "permanencia_inteligente"  # "vantagem_economica", "analisar_permanencia", "sensibilidade_margens", "deterioracao", "vantagem_v3", "vantagem_v2", "vantagem", "comparar", "verificar"
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
    saida = DESTINO_TEMP
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
    _publicar_pasta_vantagem(saida)
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
DESTINO_TEMP = ROOT / "output" / ".pesquisa_vantagem_rotacao_em_execucao"
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


def simular(b: Base, models=None, decision_callback=None, *, permitir_cash=False, aplicar_todos_folds=False):
    """Contabilidade equivalente a Control: abre i+1, fecha i+1, final liquida."""
    cash = float(b.config["initial_capital"])
    pos, quantity, holding = -1, 0.0, 0
    equity, decisions, trades = [], [], []
    total_fees = 0.0
    rotations = 0
    for i, date in enumerate(b.dates):
        target = int(b.base_actions[i]) if models is None or b.fold[i] == 1 else pos
        ranking = []
        if decision_callback is not None and (b.fold[i] >= 2 or aplicar_todos_folds):
            if pos < 0 and not permitir_cash:
                target = int(np.nanargmax(b.scores[i]))
            else:
                capital = cash + (quantity * b.cl[i, pos] if pos >= 0 else 0.0)
                target, ranking = decision_callback(
                    b, i, pos, holding, capital, quantity,
                )
        elif models is not None and b.fold[i] >= 2:
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
    """Empacota o relatorio em nome fixo, usando staging para impedir ZIP parcial."""
    archive = DESTINO.with_suffix(".zip")
    temp = archive.with_suffix(".zip.tmp")
    checksums = {
        file.relative_to(destino).as_posix(): _hash_bytes(file.read_bytes())
        for file in sorted(destino.rglob("*")) if file.is_file()
    }
    _write_json(destino / "arquivos_sha256.json", checksums)
    with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for file in sorted(destino.rglob("*")):
            if file.is_file():
                z.write(file, arcname=f"{DESTINO.name}/{file.relative_to(destino)}")
        z.write(fonte, arcname="dados_origem/pesquisa_permanencia_minima.zip")
    with ZipFile(temp) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP falhou verificacao de CRC.")
        for file, expected in checksums.items():
            if _hash_bytes(z.read(f"{DESTINO.name}/{file}")) != expected:
                raise RuntimeError(f"SHA incorreto: {file}")
    temp.replace(archive)
    return archive


def _publicar_pasta_vantagem(temporaria: Path) -> None:
    """Substitui apenas a pasta de relatorios depois do ZIP verificado."""
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    temporaria.replace(DESTINO)


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




# %% 8. Pesquisa V2: valor de acao multihorizonte e impacto de liquidez.
V2_H = (5, 10, 20, 40, 60)
V2_W = (0.10, 0.15, 0.20, 0.30, 0.25)
V2_ALPHA = 100.0
V2_FEATURES = (
    "utilidade", "variacao_1", "variacao_3", "retorno_passado_5",
    "volatilidade_20", "log_adv20", "zscore", "percentil_ranking",
)


def _v2_features_labels(b: Base):
    """Variaveis apenas em t; alvo futuro usado apenas em treinamento/auditoria."""
    n, m = len(b.dates), len(b.symbols)
    x = np.full((n, m, len(V2_FEATURES)), np.nan)
    y = np.full((n, m), np.nan)
    for i in range(25, n):
        u = b.scores[i]
        ok = np.isfinite(u)
        if not ok.any():
            continue
        rank = np.full(m, np.nan)
        ids = np.flatnonzero(ok)
        ids = ids[np.argsort(-u[ids], kind="stable")]
        rank[ids] = (np.arange(len(ids)) + 1) / len(ids)
        mu, sd = float(np.mean(u[ok])), float(np.std(u[ok]))
        d1 = u - b.scores[i - 1] if b.fold[i] == b.fold[i - 1] else np.zeros(m)
        d3 = u - b.scores[i - 3] if b.fold[i] == b.fold[i - 3] else np.zeros(m)
        x[i] = np.column_stack((
            u, d1, d3, b.ret5[i], b.vol20[i],
            np.log1p(np.maximum(b.adv20[i], 0)),
            (u-mu)/max(sd, 1e-8), rank,
        ))
        if i + 60 >= len(b.cl):
            continue
        price = b.op[i + 1]
        outcome = np.zeros(m)
        for h, w in zip(V2_H, V2_W):
            ret = np.log(b.cl[i + h] / price)
            downside = np.maximum(-ret, 0)
            trough = np.min(b.cl[i + 1:i + h + 1], axis=0)
            dd = np.maximum(np.log(price / np.maximum(trough, 1e-10)), 0)
            outcome += w*(ret - 0.20*downside - 0.35*dd)
        y[i] = outcome
    return x, y


def _v2_train(b: Base):
    """Modelos aprendem todos os ativos de folds anteriores; purge de 60 sessoes."""
    x, y = _v2_features_labels(b)
    q = np.full(y.shape, np.nan)
    reports = {}
    for fold in (2, 3):
        start = int(np.flatnonzero(b.fold == fold)[0])
        train = np.arange(25, start - 60)
        train = train[b.fold[train] < fold]
        X, Y = x[train], y[train]
        mask = np.isfinite(X).all(axis=2) & np.isfinite(Y)
        X, Y = X[mask], Y[mask]
        if len(Y) < 1000:
            raise RuntimeError("Fold sem amostras suficientes: "+str(fold))
        lo, hi = np.percentile(Y, [1, 99])
        model = make_pipeline(StandardScaler(), Ridge(alpha=V2_ALPHA))
        model.fit(X, np.clip(Y, lo, hi))
        ids = np.flatnonzero(b.fold == fold)
        xt = x[ids]
        good = np.isfinite(xt).all(axis=2)
        forecast = np.full((len(ids),len(b.symbols)), np.nan)
        forecast[good] = model.predict(xt[good])
        q[ids] = forecast
        actual = y[ids]
        valid = good & np.isfinite(actual)
        rho = float(pd.Series(forecast[valid]).corr(
            pd.Series(actual[valid]), method="spearman",
        ))
        reports[fold] = {
            "fold":fold,"n_treino":int(len(Y)),
            "data_inicio_teste":str(b.dates[start]),
            "ultimo_rotulo_treino":str(b.dates[train[-1]+60]),
            "purge_sessoes":60,"spearman_oos":rho,
            "coeficientes":model.named_steps["ridge"].coef_.tolist(),
            "intercepto":float(model.named_steps["ridge"].intercept_),
            "winsorizacao":[float(lo),float(hi)],
        }
        print("[V2] fold",fold,"treino",len(Y),"rho",round(rho,5),flush=True)
    return q, y, reports


def _v2_decider(b: Base, q: np.ndarray, liquidity: float, audit: list):
    """MANTER equivale a ganho relativo zero; nenhuma restricao por dias."""
    def decide(_b, i, current, age, capital, quantity):
        current_q = float(q[i,current])
        if not np.isfinite(current_q):
            audit.append({"data":str(b.dates[i]),"motivo":"SEM_SCORE_ATUAL"})
            return current, []
        ranked = []
        for j, symbol in enumerate(b.symbols):
            if j == current or not np.isfinite(q[i,j]):
                continue
            gain = float(q[i,j]-current_q)
            cost = float(_custo_estimado(b,i,current,j,capital,quantity))
            impact = 0.0
            if liquidity:
                sale, buy = b.adv20[i,current], b.adv20[i,j]
                if (not np.isfinite(sale) or sale<=0 or
                        not np.isfinite(buy) or buy<=0):
                    continue
                impact = float(liquidity*(
                    b.vol20[i,current]*math.sqrt(
                        max(0,quantity*b.cl[i,current])/sale,
                    ) + b.vol20[i,j]*math.sqrt(capital/buy)
                ))
            net = float(gain-cost-impact)
            if math.isfinite(net):
                ranked.append({
                    "candidato":symbol,"vantagem_prevista":gain,
                    "custo_estimado":cost,"impacto_proxy":impact,
                    "vantagem_liquida":net,
                    "utilidade_bruta":float(b.scores[i,j]),
                    "valor_acao_estimado":float(q[i,j]),
                })
        ranked.sort(key=lambda r:r["vantagem_liquida"],reverse=True)
        best = ranked[0] if ranked else None
        target = (b.symbols.index(best["candidato"])
                  if best and best["vantagem_liquida"]>0 else current)
        audit.append({
            "data":str(b.dates[i]),"fold":int(b.fold[i]),
            "posicao_antes":b.symbols[current],
            "posicao_depois":b.symbols[target],
            "idade_posicao":int(age),"pontuacao_atual":current_q,
            "melhor_vantagem_liquida":best["vantagem_liquida"] if best else None,
            "motivo":"TROCAR" if target!=current else "MANTER",
            "ranking_top10":json.dumps(ranked[:10],ensure_ascii=False),
        })
        return target, ranked
    return decide


def executar_vantagem_v2():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote",type=Path,default=None)
    args,_ = parser.parse_known_args()
    source = args.pacote or ROOT/"output"/"pesquisa_permanencia_minima.zip"
    if not source.is_file():
        raise FileNotFoundError(
            f"ZIP historico ausente: {source}. Rode primeiro MODO='comparar'."
        )
    b = carregar(source)
    baseline,_,_,_ = simular(b)
    error = float(np.max(np.abs(baseline-b.base_equity)))
    if error > 0.01 or abs(float(baseline[-1])-587165941.1330212)>0.01:
        raise RuntimeError("Controle original nao reproduzido: "+str(error))
    q, realized, training = _v2_train(b)
    import shutil
    destination = DESTINO_TEMP
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    rows = []
    for i in range(len(b.dates)):
        if b.fold[i]<2: continue
        for j,symbol in enumerate(b.symbols):
            if np.isfinite(q[i,j]):
                rows.append({
                    "data":str(b.dates[i]),"fold":int(b.fold[i]),
                    "ativo":symbol,"utilidade_original":float(b.scores[i,j]),
                    "valor_previsto":float(q[i,j]),
                    "valor_realizado":(
                        float(realized[i,j]) if np.isfinite(realized[i,j]) else None
                    ),"volatilidade_20":float(b.vol20[i,j]),
                    "liquidez_media_20_usd":float(b.adv20[i,j]),
                })
    pd.DataFrame(rows).to_csv(
        destination/"previsoes_oos_75_ativos.csv",index=False,
    )
    curves = pd.DataFrame({
        "data_execucao":b.exec_dates,"fold":b.fold,"controle":baseline,
    })
    summary = []
    for impact in (0.0,1.0):
        audit = []
        curve,decisions,trades,metrics = simular(
            b,decision_callback=_v2_decider(b,q,impact,audit),
        )
        name = "sem_impacto" if impact==0 else "com_impacto_proxy"
        curves[name] = curve
        pd.DataFrame(audit).to_csv(
            destination/("ranking_"+name+".csv"),index=False,
        )
        decisions.to_csv(destination/("decisoes_"+name+".csv"),index=False)
        trades.to_csv(destination/("operacoes_"+name+".csv"),index=False)
        risk = _resumo_curva(curve)
        _write_json(destination/("metricas_"+name+".json"),{
            "metricas":metrics,"risco":risk,
            "folds":_performance_by_fold(b,curve),
            "coeficiente_impacto_proxy":impact,
        })
        summary.append({
            "cenario":name,"capital_final":float(curve[-1]),
            "rotacoes":int(metrics["rotacoes"]),
            "sharpe":risk["sharpe"],
            "drawdown_maximo":risk["drawdown_maximo"],
        })
        print("[V2]",name,"capital",float(curve[-1]),flush=True)
    curves.to_csv(destination/"comparacao_diaria.csv",index=False)
    pd.DataFrame(summary).to_csv(
        destination/"comparacao_resultados.csv",index=False,
    )
    _write_json(destination/"treinamento_multihorizonte.json",training)
    _write_json(destination/"manifesto_experimento.json",{
        "modo": MODO,
        "status":"completed_research_only",
        "modelo":"Ridge multihorizonte por ativo, sem restricao temporal",
        "horizontes":V2_H,"pesos":V2_W,"features":V2_FEATURES,
        "ridge_alpha":V2_ALPHA,
        "penalidades_alvo":{"downside":0.20,"drawdown":0.35},
        "treino":"Fold1 para Fold2; Fold1+Fold2 para Fold3, purge 60",
        "fold1":"Control inalterado",
        "criterio":"argmax vantagem liquida com MANTER=0",
        "paridade_controle_erro_maximo_dolar":error,
        "fonte_sha256":b.original_sha,
        "resultados":summary,
        "limitacoes":[
            "Alvo de retorno multihorizonte nao equivale ao valor de uma politica sequencial.",
            "O universo historico pode ter vies de selecao.",
            "Proxy de impacto nao calibrado com ordens reais.",
            "Impacto afeta decisao, nao altera preco executado no replay.",
            "Sem limite de volume ou capacidade de execucao realista.",
            "Nenhuma mudanca em operacoes Live.",
        ],
    })
    archive = _pacote(destination,source)
    _publicar_pasta_vantagem(destination)
    print("[ZIP]",archive,flush=True)



# %% 9. V3: vantagem CONDICIONAL + estabilidade estatistica das utilidades.
#
# Nenhum treino usa as observacoes do fold avaliado. Os rotulos do fold
# anterior precisam estar conhecidos ANTES da primeira data do fold de teste.
# A vantagem financeira alvo e uma aproximacao com posicao estatica por H
# sessoes. Nao representa uma Q-function de politica sequencial otimizada.
V3_H = (5, 10, 20, 40, 60)
V3_W = (0.10, 0.15, 0.20, 0.30, 0.25)
V3_Q_R_GRID = (0.005, 0.02, 0.10, 0.5, 2.0)
V3_TOP = 12
V3_RIDGE_ALPHA = 50.0
V3_STATE_RANKS = (0, 1, 2, 4, 9, 19)
V3_FEATURES = (
    "gap_utilidade_bruta", "gap_utilidade_filtrada",
    "gap_variacao_score_1", "gap_variacao_score_3",
    "gap_retorno_passado_5", "gap_volatilidade_20",
    "log_razao_liquidez_20", "log_dias_na_posicao",
    "retorno_desde_entrada", "drawdown_desde_entrada",
    "rank_percentil_atual", "rank_percentil_candidato",
    "utilidade_bruta_atual", "utilidade_bruta_candidato",
    "diferenca_volatilidade_absoluta",
)


def _v3_kalman(b: Base, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Filtro local-level causal 1D por ativo; estado reinicia em cada fold.

    R=1 e Q=ratio sao escalas relativas; este filtro nao gera probabilidade
    de ganho ou intervalo de confianca calibrado.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("A razao Q/R do Kalman deve ser positiva.")
    out = np.full_like(b.scores, np.nan, dtype=float)
    uncert = np.full_like(b.scores, np.nan, dtype=float)
    previous_fold = None
    state = np.full(len(b.symbols), np.nan)
    variance = np.ones(len(b.symbols))
    for i in range(len(b.dates)):
        if b.fold[i] != previous_fold:
            state[:] = np.nan
            variance[:] = 1.0
            previous_fold = b.fold[i]
        observed = b.scores[i]
        good = np.isfinite(observed)
        initial = good & ~np.isfinite(state)
        state[initial] = observed[initial]
        variance[initial] = 1.0
        known = good & ~initial
        variance = variance + float(ratio)
        if known.any():
            gain = variance[known] / (variance[known] + 1.0)
            state[known] += gain * (observed[known] - state[known])
            variance[known] *= (1.0 - gain)
        out[i] = state
        uncert[i] = variance
    return out, uncert


def _v3_ic_diario(estimativa: np.ndarray, realizada: np.ndarray,
                  indices: np.ndarray) -> tuple[float | None, int]:
    """Media dos Spearman cross-sectionais: uma sessao, uma observacao."""
    ics = []
    for i in indices:
        mask = np.isfinite(estimativa[i]) & np.isfinite(realizada[i])
        if int(mask.sum()) < 15:
            continue
        x, y = estimativa[i, mask], realizada[i, mask]
        if np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
            continue
        rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        if math.isfinite(rho):
            ics.append(rho)
    return (float(np.mean(ics)) if ics else None), len(ics)


def _v3_selecionar_filtro(b: Base, labels: np.ndarray,
                         fold: int) -> tuple[float, list[dict]]:
    """Escolha de Q/R apenas com dados anteriores, purgados em 60 sessoes."""
    first = int(np.flatnonzero(b.fold == fold)[0])
    eligible = np.flatnonzero(
        (b.fold < fold) & (np.arange(len(b.dates)) + 60 < first)
    )
    if len(eligible) < 100:
        raise RuntimeError("Sem dados maduros para calibrar o filtro.")
    rows = []
    for ratio in V3_Q_R_GRID:
        filtered, _ = _v3_kalman(b, ratio)
        ic, count = _v3_ic_diario(filtered, labels, eligible)
        rows.append({
            "fold_teste": fold, "q_sobre_r": ratio,
            "ic_spearman_medio_treino": ic, "sessoes_treino_ic": count,
        })
    valid = [r for r in rows if r["ic_spearman_medio_treino"] is not None]
    if not valid:
        raise RuntimeError("Nao foi possivel calibrar o filtro em treino.")
    best = max(valid, key=lambda r: (
        r["ic_spearman_medio_treino"], -r["q_sobre_r"],
    ))
    return float(best["q_sobre_r"]), rows


def _v3_state_features(b: Base, filt: np.ndarray, i: int,
                       current: int, candidate: int, age: int) -> np.ndarray:
    """Estatisticas causais no fechamento, referentes a posicao especifica."""
    if i < 22 or age < 1 or current < 0 or candidate == current:
        return np.full(len(V3_FEATURES), np.nan)
    u = b.scores
    previous1 = (u[i] - u[i - 1]
                 if b.fold[i] == b.fold[i-1] else np.zeros(len(b.symbols)))
    previous3 = (u[i] - u[i - 3]
                 if b.fold[i] == b.fold[i-3] else np.zeros(len(b.symbols)))
    start = max(0, i - age + 1)
    entry = float(b.op[start, current])
    current_close = float(b.cl[i, current])
    if entry <= 0 or current_close <= 0:
        return np.full(len(V3_FEATURES), np.nan)
    pnl = math.log(current_close / entry)
    trough = float(np.min(b.cl[start:i + 1, current]))
    dd = math.log(max(trough, 1e-10) / entry)
    ranked = np.argsort(-u[i], kind="stable")
    ranks = np.empty(len(b.symbols), dtype=float)
    ranks[ranked] = (np.arange(len(ranked)) + 1) / len(ranked)
    vol_cur = b.vol20[i, current]
    vol_cand = b.vol20[i, candidate]
    return np.asarray((
        u[i, candidate] - u[i, current],
        filt[i, candidate] - filt[i, current],
        previous1[candidate] - previous1[current],
        previous3[candidate] - previous3[current],
        b.ret5[i, candidate] - b.ret5[i, current],
        vol_cand - vol_cur,
        math.log((max(b.adv20[i, candidate], 0) + 1) /
                 (max(b.adv20[i, current], 0) + 1)),
        math.log1p(age),
        pnl, dd, ranks[current], ranks[candidate],
        u[i, current], u[i, candidate],
        abs(vol_cand) - abs(vol_cur),
    ), dtype=float)


def _v3_pairs_treino(b: Base, filt: np.ndarray, labels: np.ndarray,
                     fold: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Estados sinteticos apenas do passado, diversos ativos e idades.

    Sao contrafactuais de posicao sem dependencias de modelo-base:
    top ranks 1,2,3,5,10,20 e posicao Control quando existente.
    Entradas historicas usam datas e precos passados, jamais futuros.
    A idade e o PnL entram como features, nao como bloqueio de operacao.
    """
    first = int(np.flatnonzero(b.fold == fold)[0])
    eligible = np.flatnonzero(
        (b.fold < fold) & (np.arange(len(b.dates)) + 60 < first)
    )
    x, y, provenance = [], [], []
    for i in eligible:
        if i < 25:
            continue
        utilities = b.scores[i]
        top = [
            int(j) for j in np.argsort(-utilities, kind="stable")
            if np.isfinite(utilities[j])
        ]
        current_indices = [
            top[k] for k in V3_STATE_RANKS if k < len(top)
        ]
        held = int(b.base_current[i])
        if held >= 0 and held not in current_indices:
            current_indices.append(held)
        for current in current_indices:
            ages = [1, (2, 3, 5, 10, 20)[i % 5]]
            if held == current and int(b.base_holding[i]) > 0:
                ages.append(int(b.base_holding[i]))
            for age in sorted(set(ages)):
                for cand in top[:V3_TOP]:
                    if current == cand:
                        continue
                    outcome = labels[i, cand] - labels[i, current]
                    if not np.isfinite(outcome):
                        continue
                    row = _v3_state_features(
                        b, filt, int(i), current, cand, int(age),
                    )
                    if not np.isfinite(row).all():
                        continue
                    x.append(row)
                    y.append(float(outcome))
                    provenance.append({
                        "data":str(b.dates[i]), "fold_origem":int(b.fold[i]),
                        "atual":b.symbols[current], "candidato":b.symbols[cand],
                        "idade":int(age), "vantagem_realizada":float(outcome),
                    })
    if not x:
        raise RuntimeError("Nenhum par causal disponivel para treino.")
    return np.stack(x), np.asarray(y, dtype=float), provenance


def _v3_fit_conditional(b: Base, filt: np.ndarray, labels: np.ndarray,
                        fold: int, name: str):
    x, y, provenance = _v3_pairs_treino(b, filt, labels, fold)
    if len(y) < 500:
        raise RuntimeError(f"Treinamento de vantagem insuficiente fold={fold}.")
    low, high = np.percentile(y, [1, 99])
    model = make_pipeline(StandardScaler(), Ridge(alpha=V3_RIDGE_ALPHA))
    model.fit(x, np.clip(y, low, high))
    return model, {
        "modelo":name,"fold_teste":fold,"n_pares_treino":len(y),
        "features":V3_FEATURES, "alpha":V3_RIDGE_ALPHA,
        "winsorizacao":[float(low),float(high)],
        "coeficientes":model.named_steps["ridge"].coef_.tolist(),
        "intercepto":float(model.named_steps["ridge"].intercept_),
        "primeiro_estado":provenance[0],
        "ultimo_estado":provenance[-1],
        "proporcao_estados_control":sum(
            r["atual"] == b.symbols[int(b.base_current[
                b.dates.get_loc(pd.Timestamp(r["data"]))
            ])]
            for r in provenance
            if int(b.base_current[
                b.dates.get_loc(pd.Timestamp(r["data"]))
            ]) >= 0
        ) / len(provenance),
    }


def _v3_action_policy(
    b: Base, q: np.ndarray, filt: np.ndarray, model_by_fold: dict,
    method: str, liquidity_factor: float, log: list[dict],
):
    """Decisao baseada na posicao EFETIVA, independentemente da sua idade."""
    def decide(_b, i, current, age, capital, quantity):
        fold = int(b.fold[i])
        floor = float(b.config["rotation_cash_threshold"])
        raw_best = float(np.nanmax(b.scores[i]))
        if raw_best <= floor:
            log.append({
                "data":str(b.dates[i]),"fold":fold,
                "posicao_atual":b.symbols[current],
                "posicao_escolhida":"CASH","motivo":"CASH_THRESHOLD",
            })
            return -1, []
        if method == "kalman":
            # Score ausente hoje nao pode ser recuperado de dias anteriores.
            estimates = np.where(np.isfinite(b.scores[i]), filt[i], np.nan)
            best = int(np.nanargmax(estimates))
            gap = float(estimates[best] - estimates[current])
            required = max(
                float(b.config["rotation_switch_margin"]),
                float(b.margens[fold]),
            )
            target = best if gap > required else current
            ranked = [{
                "candidato":b.symbols[best],
                "vantagem_filtrada":gap,"margem_base":required,
                "vantagem_liquida":gap-required,
            }]
        else:
            candidates = [
                int(j) for j in np.argsort(-(
                    filt[i] if method.startswith("hibrida") else b.scores[i]
                ),kind="stable")[:V3_TOP]
                if j != current
            ]
            rows, kept = [], []
            for j in candidates:
                features = _v3_state_features(
                    b, filt, i, current, j, age,
                )
                if not np.isfinite(features).all():
                    continue
                rows.append(features)
                kept.append(j)
            ranked = []
            if rows:
                forecasts = model_by_fold[fold].predict(np.stack(rows))
                for j, pred in zip(kept, forecasts):
                    cost = _custo_estimado(
                        b, i, current, j, capital, quantity,
                    )
                    impact = 0.0
                    if liquidity_factor:
                        sell_adv, buy_adv = (
                            b.adv20[i,current], b.adv20[i,j]
                        )
                        if not(np.isfinite(sell_adv) and sell_adv > 0 and
                               np.isfinite(buy_adv) and buy_adv > 0):
                            continue
                        impact = float(liquidity_factor*(
                            b.vol20[i,current]*math.sqrt(
                                max(0,quantity*b.cl[i,current]) / sell_adv,
                            ) + b.vol20[i,j]*math.sqrt(
                                capital / buy_adv,
                            )
                        ))
                    net = float(pred - cost - impact)
                    if math.isfinite(net):
                        ranked.append({
                            "candidato":b.symbols[j],
                            "vantagem_prevista":float(pred),
                            "taxas_previstas":float(cost),
                            "impacto_proxy":float(impact),
                            "vantagem_liquida":net,
                            "utilidade_original":float(b.scores[i,j]),
                            "utilidade_filtrada":float(filt[i,j]),
                        })
            ranked.sort(key=lambda r:r["vantagem_liquida"],reverse=True)
            best = (
                b.symbols.index(ranked[0]["candidato"])
                if ranked and ranked[0]["vantagem_liquida"] > 0 else current
            )
            target = best
        log.append({
            "data":str(b.dates[i]),"fold":fold,"metodo":method,
            "indice_decisao":int(i),
            "posicao_atual":b.symbols[current],
            "posicao_escolhida":b.symbols[target] if target >= 0 else "CASH",
            "idade_posicao":int(age),
            "retorno_desde_entrada":float(
                b.cl[i,current] / b.op[max(0,i-age+1),current] - 1
            ),
            "utilidade_original_atual":float(b.scores[i,current]),
            "utilidade_filtrada_atual":float(filt[i,current]),
            "motivo":"ROTACIONAR" if target != current else "MANTER",
            "ranking_top12":json.dumps(ranked[:V3_TOP],ensure_ascii=False),
            "vantagem_maxima_liquida":(
                ranked[0]["vantagem_liquida"] if ranked else None
            ),
        })
        return target, ranked
    return decide


def _v3_liquidez_ordens(trades: pd.DataFrame) -> dict:
    if trades.empty or "participacao_volume" not in trades:
        return {"ordens":0}
    ratio = pd.to_numeric(trades["participacao_volume"],errors="coerce")
    valid = trades["acao"].isin(["BUY","SELL"])
    volume = pd.to_numeric(trades["volume_dia"],errors="coerce")
    return {
        "ordens":int(valid.sum()),
        "acima_10pct_volume":int(((ratio > 0.10)&valid).sum()),
        "acima_100pct_volume":int(((ratio > 1.0)&valid).sum()),
        "volume_igual_zero":int(((volume <= 0)&valid).sum()),
    }


def _v3_avaliar_vantagens(
    b: Base, labels: np.ndarray, logs: list[dict], cenario: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Avalia previsoes contra contrafactuais observados, APOS cada replay.

    Valores realizados nunca entram na decisao ou no treinamento do fold.
    Retorno futuro representa manter estaticamente cada ativo em 5-60 sessoes,
    e nao ganho realizavel com trocas subsequentes.
    """
    observations = []
    for row in logs:
        idx = row.get("indice_decisao")
        if idx is None:
            continue
        i = int(idx)
        current_name = row["posicao_atual"]
        current = b.symbols.index(current_name)
        if not np.isfinite(labels[i, current]):
            continue
        for candidate in json.loads(row.get("ranking_top12") or "[]"):
            symbol = candidate["candidato"]
            j = b.symbols.index(symbol)
            if not np.isfinite(labels[i, j]):
                continue
            predicted = candidate.get(
                "vantagem_prevista", candidate.get("vantagem_filtrada")
            )
            if predicted is None or not math.isfinite(float(predicted)):
                continue
            actual = float(labels[i, j]-labels[i, current])
            observations.append({
                "cenario":cenario,"data_decisao":str(b.dates[i]),
                "fold":int(b.fold[i]),"atual":current_name,
                "candidato":symbol,
                "idade_posicao":int(row.get("idade_posicao") or 0),
                "vantagem_prevista_bruta":float(predicted),
                "vantagem_realizada_estatica":actual,
                "custo_previsto":float(candidate.get(
                    "taxas_previstas",0.0
                )),
                "impacto_proxy_previsto":float(candidate.get(
                    "impacto_proxy",0.0
                )),
            })
    frame = pd.DataFrame(observations)
    metrics = []
    if frame.empty:
        return frame, metrics
    for fold, group in frame.groupby("fold"):
        daily = []
        for _, day in group.groupby("data_decisao"):
            if len(day) >= 4:
                rho = day["vantagem_prevista_bruta"].corr(
                    day["vantagem_realizada_estatica"],
                    method="spearman",
                )
                if pd.notna(rho):
                    daily.append(float(rho))
        positive = group["vantagem_prevista_bruta"] > 0
        precision = (
            float((group.loc[positive,
                "vantagem_realizada_estatica"] > 0).mean())
            if positive.any() else None
        )
        metrics.append({
            "cenario":cenario,"fold":int(fold),"pares":int(len(group)),
            "sessoes_distintas":int(group["data_decisao"].nunique()),
            "correlacao_spearman_pares":float(
                group["vantagem_prevista_bruta"].corr(
                    group["vantagem_realizada_estatica"],
                    method="spearman",
                )
            ),
            "media_spearman_por_dia":(
                float(np.mean(daily)) if daily else None
            ),
            "dias_com_4_ou_mais_pares":int(len(daily)),
            "precisao_quando_predicao_positiva":precision,
            "proporcao_predicoes_positivas":float(positive.mean()),
        })
    return frame, metrics


def executar_vantagem_v3() -> None:
    """V3 no TCC. Salva um unico ZIP com auditoria dos quatro cenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote",type=Path,default=None)
    args,_ = parser.parse_known_args()
    source = args.pacote or ROOT/"output"/"pesquisa_permanencia_minima.zip"
    if not source.is_file():
        raise FileNotFoundError(
            f"ZIP historico ausente: {source}. "
            "Execute MODO='comparar' ou informe --pacote."
        )
    b = carregar(source)
    baseline, base_decisions, base_trades, base_metrics = simular(b)
    parity = float(np.max(np.abs(baseline-b.base_equity)))
    if parity > 0.01 or abs(float(baseline[-1])-587165941.1330212)>0.01:
        raise RuntimeError(
            f"Controle nao reproduzido; maior diferenca USD {parity:.9f}."
        )
    # Alvos futuros usados APENAS em treino e avaliacao retrospectiva.
    _, labels = _v2_features_labels(b)
    configurations, training, filters = {}, {}, {}
    prediction_rows = []
    for fold in (2,3):
        ratio, grid = _v3_selecionar_filtro(b, labels, fold)
        filt, uncertainty = _v3_kalman(b, ratio)
        configurations[fold] = {"q_over_r":ratio,"calibracao":grid}
        filters[fold] = filt
        print(f"[V3] Fold {fold} Q/R escolhido {ratio}",flush=True)
        for mode in ("condicional", "hibrida"):
            use = b.scores if mode == "condicional" else filt
            model, audit = _v3_fit_conditional(b,use,labels,fold,mode)
            training[(mode,fold)] = model
            configurations[fold][mode] = audit
        ids = np.flatnonzero(b.fold == fold)
        daily_ic, sessions = _v3_ic_diario(filt,labels,ids)
        configurations[fold]["ic_kalman_oos_medio"] = daily_ic
        configurations[fold]["sessoes_ic_oos"] = sessions
        for i in ids:
            for j, symbol in enumerate(b.symbols):
                if np.isfinite(b.scores[i,j]):
                    prediction_rows.append({
                        "data":str(b.dates[i]),"fold":fold,
                        "ativo":symbol,"utilidade_bruta":float(b.scores[i,j]),
                        "utilidade_kalman":(
                            float(filt[i,j]) if np.isfinite(filt[i,j]) else None
                        ),
                        "variancia_filtro":(
                            float(uncertainty[i,j]) if np.isfinite(
                                uncertainty[i,j]) else None
                        ),
                        "valor_realizado_estatico_5_60":(
                            float(labels[i,j]) if np.isfinite(labels[i,j]) else None
                        ),
                    })
    # O ultimo registro em b.scores pertence a data final de EXECUCAO.
    # Nao existe decisao nem fold correspondente a essa linha.
    # Selecionar por indices das 1560 decisoes, nunca aplicar uma mascara
    # de 1560 elementos diretamente na matriz com 1561 linhas.
    score_kalman = np.full_like(b.scores, np.nan)
    if len(b.fold) != len(b.dates) or len(b.scores) != len(b.dates) + 1:
        raise RuntimeError(
            "Calendario do replay inconsistente: esperadas N decisoes "
            "e N+1 linhas de precos/utilidades."
        )
    for fold in (2, 3):
        indices = np.flatnonzero(b.fold == fold)
        filtrado = filters[fold]
        if filtrado.shape != b.scores.shape:
            raise RuntimeError(
                f"Fold {fold}: utilidades filtradas {filtrado.shape} "
                f"divergem do original {b.scores.shape}."
            )
        score_kalman[indices, :] = filtrado[indices, :]
    dest = DESTINO_TEMP
    import shutil
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    pd.DataFrame(prediction_rows).to_csv(
        dest/"utilidades_brutas_filtradas_75_ativos.csv",index=False,
    )
    curves = pd.DataFrame({
        "data_execucao":b.exec_dates,"fold":b.fold,"control_2_pregoes":baseline,
    })
    base_decisions.to_csv(dest/"decisoes_controle.csv",index=False)
    base_trades.to_csv(dest/"operacoes_controle.csv",index=False)
    pair_metrics = []
    summary = [{
        "cenario":"control_2_pregoes",
        **_resumo_curva(baseline),**base_metrics,
        **_v3_liquidez_ordens(base_trades),
    }]
    scenarios = (
        ("kalman","kalman",0.0),
        ("vantagem_condicional","condicional",0.0),
        ("hibrida","hibrida",0.0),
        ("hibrida_liquidez","hibrida",1.0),
    )
    for name,mode,impact in scenarios:
        track = []
        def callback(_b,i,current,age,capital,quantity,
                     _mode=mode,_impact=impact,_track=track):
            fold=int(b.fold[i])
            selected_filter = (
                b.scores if _mode=="condicional"
                else filters[fold]
            )
            model = (
                {} if _mode=="kalman"
                else {fold:training[(_mode,fold)]}
            )
            return _v3_action_policy(
                b,score_kalman,selected_filter,
                model,_mode,_impact,_track,
            )(_b,i,current,age,capital,quantity)
        curve,decisions,trades,metrics = simular(
            b,decision_callback=callback,
        )
        curves[name]=curve
        pd.DataFrame(track).to_csv(dest/f"ranking_{name}.csv",index=False)
        comparisons, diagnostic = _v3_avaliar_vantagens(
            b, labels, track, name,
        )
        comparisons.to_csv(
            dest/f"vantagem_prevista_realizada_{name}.csv",index=False,
        )
        pair_metrics.extend(diagnostic)
        decisions.to_csv(dest/f"decisoes_{name}.csv",index=False)
        trades.to_csv(dest/f"operacoes_{name}.csv",index=False)
        risk = _resumo_curva(curve)
        folds = _performance_by_fold(b,curve)
        liquidez = _v3_liquidez_ordens(trades)
        _write_json(dest/f"metricas_{name}.json",{
            "risco":risk,"metricas":metrics,
            "folds":folds,"liquidez":liquidez,
        })
        result = {"cenario":name,**risk,**metrics,**liquidez}
        summary.append(result)
        print(f"[V3] {name}: USD {curve[-1]:,.2f}; "
              f"rotacoes {metrics['rotacoes']}",flush=True)
    curves.to_csv(dest/"comparacao_patrimonio_diario.csv",index=False)
    pd.DataFrame(pair_metrics).to_csv(
        dest/"qualidade_previsoes_por_fold.csv",index=False,
    )
    pd.DataFrame(summary).to_csv(
        dest/"comparacao_resultados.csv",index=False,
    )
    _write_json(dest/"calibracao_filtros_e_modelos.json",configurations)
    _write_json(dest/"manifesto_experimento.json",{
        "modo": MODO,
        "status":"completed_research_only",
        "branch":"research/permanencia-minima-contrafactual-v1",
        "arquivo":"pesquisar_permanencia_minima_spyder.py",
        "fonte_sha256":b.original_sha,
        "paridade_control_diferenca_max_usd":parity,
        "universo":75,"folds":3,"horizontes":V3_H,
        "pesos_horizontes":V3_W,"caracteristicas":V3_FEATURES,
        "fold1":"Control, sem treinamento anterior para metamodelo",
        "selecao_kalman":"Q/R por IC medio no passado com purge de 60",
        "modelos":"Ridge de vantagem relativa condicionada a posicao atual",
        "estado":"idade, PnL, drawdown, utilidades, volatilidade, liquidez",
        "sem_regra_tempo_minimo":True,
        "validacao_temporal":"Fold1->Fold2, Folds1+2->Fold3; purge de 60",
        "resultados":summary,
        "qualidade_previsao_por_fold":pair_metrics,
        "alvo_realizado":"retorno logaritmico multihorizonte 5-60 com penalidade downside/drawdown para posicoes estaticas",
        "alertas":[
            "Rotulos de 5-60 pregoes pressupoe permanencia estatica; nao valor Q sequencial.",
            "Estados sinteticos da carteira podem ter distribuicao distinta da politica.",
            "Estimativas fora da amostra em folds 2 e 3; fold 1 e Control.",
            "O universo historico pode conter vies de selecao.",
            "Sem limite real de participacao no volume negociado.",
            "Impacto de liquidez e apenas proxy na decisao, nao no preco executado.",
            "Nao foram feitas ordens e a Winner Live permanece inalterada.",
        ],
    })
    shutil.copyfile(Path(__file__),dest/"codigo_executado.py")
    archive=_pacote(dest,source)
    _publicar_pasta_vantagem(dest)
    print(f"[V3] ZIP pronto: {archive}",flush=True)


# %% 10. Pesquisa da deterioracao de posicao e protecao para CASH.
# Nenhum codigo da Winner Live ou motor oficial e modificado.
D_HORIZONTE = 20
D_QUEDA_ALVO = -0.15
D_PURGE = 60
D_ATRIBUTOS = (
    "utilidade", "variacao_utilidade_1", "variacao_utilidade_3",
    "zscore_utilidade", "rank_percentil", "distancia_topo",
    "retorno_5", "retorno_20", "volatilidade_20",
    "drawdown_20", "liquidez_log_20", "fracao_ativos_negativos_5",
    "fracao_utilidades_positivas",
)


def _d_features_labels(b: Base) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variaveis causais no fechamento i. Rotulos futuros so para treino/auditoria."""
    n, m = len(b.dates), len(b.symbols)
    x = np.full((n, m, len(D_ATRIBUTOS)), np.nan)
    y = np.full((n, m), np.nan)
    pior_retorno = np.full((n, m), np.nan)
    for i in range(25, n):
        u = b.scores[i]
        valid = np.isfinite(u)
        if not valid.any():
            continue
        ids = np.flatnonzero(valid)
        order = ids[np.argsort(-u[ids], kind="stable")]
        rank = np.full(m, np.nan)
        rank[order] = (np.arange(len(order)) + 1) / len(order)
        mean, stdev = float(np.mean(u[ids])), float(np.std(u[ids]))
        d1 = u - b.scores[i - 1] if b.fold[i] == b.fold[i - 1] else np.zeros(m)
        d3 = u - b.scores[i - 3] if b.fold[i] == b.fold[i - 3] else np.zeros(m)
        peak = np.max(b.cl[i - 19:i + 1], axis=0)
        ret5 = b.ret5[i]
        breadth = float(np.mean(ret5[np.isfinite(ret5)] < 0))
        x[i] = np.column_stack((
            u, d1, d3, (u - mean) / max(stdev, 1e-8), rank,
            np.max(u[ids]) - u, b.ret5[i], b.cl[i] / b.cl[i - 20] - 1,
            b.vol20[i], b.cl[i] / peak - 1,
            np.log1p(np.maximum(b.adv20[i], 0)),
            np.full(m, breadth), np.full(m, float(np.mean(u[ids] > 0))),
        ))
        # Decisao ao fechamento i, ordem na proxima abertura; minimos nos
        # fechamentos de 1 a 20 sessoes seguintes. Nunca entram nas features.
        if i + D_HORIZONTE >= len(b.cl):
            continue
        downside = np.min(b.cl[i + 1:i + D_HORIZONTE + 1], axis=0)
        worst = downside / b.op[i + 1] - 1
        pior_retorno[i] = worst
        y[i] = (worst <= D_QUEDA_ALVO).astype(float)
    return x, y, pior_retorno


def _d_control(b: Base, i: int, pos: int, holding: int) -> int:
    """Recalcula a regra Control usando posicao e permanencia efetivas."""
    u = b.scores[i]
    if not np.isfinite(u).any():
        return -1
    best = int(np.nanargmax(u))
    best_score = float(u[best])
    current_score = float(u[pos]) if pos >= 0 else 0.0
    minimum = float(b.config["rotation_cash_threshold"])
    if (pos >= 0 and np.isfinite(current_score)
            and holding < int(b.config["rotation_min_holding_days"])):
        return pos
    if best_score <= minimum:
        return -1
    if pos < 0:
        threshold = minimum + float(b.config["rotation_min_expected_edge"])
        return best if best_score >= threshold else -1
    if best == pos:
        return pos
    required = max(
        float(b.config["rotation_switch_margin"]),
        float(b.margens[int(b.fold[i])]),
    )
    return best if best_score >= current_score + required else pos


def _d_modelos(b: Base, x: np.ndarray, y: np.ndarray):
    """Classificador por fold: so folds anteriores com purge de 60 sessoes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    predicted = np.full(y.shape, np.nan)
    audit = {}
    for fold in (2, 3):
        first = int(np.flatnonzero(b.fold == fold)[0])
        indices = np.flatnonzero(
            (b.fold < fold) & (np.arange(len(b.dates)) + D_PURGE < first)
        )
        train_x, train_y = x[indices], y[indices]
        mask = np.isfinite(train_x).all(axis=2) & np.isfinite(train_y)
        train_x, train_y = train_x[mask], train_y[mask].astype(int)
        if len(train_y) < 1000 or len(np.unique(train_y)) < 2:
            raise RuntimeError(f"Treino de risco insuficiente no fold {fold}.")
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.3, max_iter=300),
        )
        model.fit(train_x, train_y)
        idx_test = np.flatnonzero(b.fold == fold)
        test_x = x[idx_test]
        good = np.isfinite(test_x).all(axis=2)
        fold_predictions = np.full((len(idx_test), len(b.symbols)), np.nan)
        fold_predictions[good] = model.predict_proba(test_x[good])[:, 1]
        predicted[idx_test] = fold_predictions
        actual = y[idx_test]
        valid = good & np.isfinite(actual)
        target, pred = actual[valid], fold_predictions[valid]
        audit[fold] = {
            "fold": fold,
            "inicio_teste": str(b.dates[first]),
            "fim_maximo_rotulo_treino": str(b.dates[indices[-1] + D_HORIZONTE]),
            "ultimo_indice_treino": int(indices[-1]),
            "purge": D_PURGE,
            "quantidade_treino": int(len(train_y)),
            "evento_treino": float(np.mean(train_y)),
            "quantidade_teste": int(len(target)),
            "evento_teste": float(np.mean(target)),
            "auc_oos": (
                float(roc_auc_score(target, pred))
                if len(np.unique(target)) == 2 else None
            ),
            "precisao_media_oos": float(average_precision_score(target, pred)),
            "brier_oos": float(brier_score_loss(target, pred)),
            "predicao_media_oos": float(np.mean(pred)),
            "fracao_risco_alto_060": float(np.mean(pred >= 0.60)),
            "fracao_risco_alto_075": float(np.mean(pred >= 0.75)),
            "coeficientes": model.named_steps["logisticregression"].coef_[0].tolist(),
        }
        print(
            f"[DETERIORACAO] Fold {fold}: AUC={audit[fold]['auc_oos']:.3f} "
            f"risco={audit[fold]['evento_teste']:.2%}", flush=True,
        )
    return predicted, audit


def _d_politica(
    b: Base, risks: np.ndarray, *, exit_at: float,
    reentry_at: float, veto_rotacao: bool, audit: list[dict],
):
    """Control escolhe ativos, estimador indica risco e pode bloquear CASH/entrada."""
    def decide(_b, i, pos, holding, capital, quantity):
        original = _d_control(b, i, pos, holding)
        target = original
        raw_current = float(risks[i, pos]) if pos >= 0 else None
        raw_proposed = float(risks[i, original]) if original >= 0 else None
        risk_current = (
            raw_current
            if raw_current is not None and np.isfinite(raw_current) else None
        )
        risk_proposed = (
            raw_proposed
            if raw_proposed is not None and np.isfinite(raw_proposed) else None
        )
        reason = "CONTROL"
        if (original >= 0 and (pos < 0 or veto_rotacao)
                and (risk_proposed is None or risk_proposed > reentry_at)
                and original != pos):
            target = pos
            reason = "VETO_ENTRADA_RISCO"
        if pos >= 0 and risk_current is not None and risk_current >= exit_at:
            if holding >= int(b.config["rotation_min_holding_days"]):
                if target == pos:
                    target = -1
                    reason = "DETERIORACAO_SAIR_CASH"
                elif target >= 0:
                    reason = "DETERIORACAO_ROTACIONAR_CONTROL"
                else:
                    reason = "DETERIORACAO_SAIR_CASH"
        elif pos < 0 and target >= 0 and risk_proposed is not None:
            reason = "REENTRADA_SEGURA"
        if target < 0 and pos < 0 and original >= 0:
            reason = "PERMANECER_CASH_RISCO"
        audit.append({
            "data_decisao": str(b.dates[i]), "fold": int(b.fold[i]),
            "atual": b.symbols[pos] if pos >= 0 else "CASH",
            "original_control": (
                b.symbols[original] if original >= 0 else "CASH"
            ),
            "decisao": b.symbols[target] if target >= 0 else "CASH",
            "holding": int(holding), "risco_atual": risk_current,
            "risco_proposto": risk_proposed, "limiar_saida": exit_at,
            "limiar_entrada": reentry_at, "motivo": reason,
            "score_atual": float(b.scores[i, pos]) if pos >= 0 else None,
            "score_candidato": (
                float(b.scores[i, original]) if original >= 0 else None
            ),
        })
        return target, []
    return decide


def executar_deterioracao() -> None:
    """Pesquisa em dados congelados, com um so ZIP e sem ordens reais."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote", type=Path, default=None)
    args, _ = parser.parse_known_args()
    source = args.pacote or ROOT / "output" / "pesquisa_permanencia_minima.zip"
    if not source.is_file():
        raise FileNotFoundError(f"Pacote historico de permanencia ausente: {source}")
    b = carregar(source)
    base, base_decisions, base_trades, base_metrics = simular(b)
    parity = float(np.max(np.abs(base - b.base_equity)))
    if parity > 0.01 or abs(float(base[-1]) - 587165941.1330212) > 0.01:
        raise RuntimeError(f"Control nao reproduzido. Erro USD {parity:.9f}")
    mismatches = []
    for i in range(len(b.dates)):
        proposed = _d_control(
            b, i, int(b.base_current[i]), int(b.base_holding[i]),
        )
        if proposed != int(b.base_actions[i]):
            mismatches.append(int(i))
    if mismatches:
        raise RuntimeError(
            f"Regra Control divergente em {len(mismatches)} decisoes: "
            f"{mismatches[:8]}"
        )
    features, labels, realized = _d_features_labels(b)
    risks, training = _d_modelos(b, features, labels)
    dest = DESTINO_TEMP
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    curves = pd.DataFrame({
        "data_execucao": b.exec_dates, "fold": b.fold, "control": base,
    })
    base_decisions.to_csv(dest / "decisoes_control.csv", index=False)
    base_trades.to_csv(dest / "operacoes_control.csv", index=False)
    scenario_results = [{"cenario": "control", **_resumo_curva(base), **base_metrics}]
    for name, exit_at, reentry_at, veto_rotacao in (
        ("permanencia_conservadora", 0.75, 0.45, False),
        ("permanencia_moderada", 0.60, 0.40, False),
        ("permanencia_e_veto_entrada", 0.75, 0.45, True),
    ):
        log = []
        callback = _d_politica(
            b, risks, exit_at=exit_at, reentry_at=reentry_at,
            veto_rotacao=veto_rotacao, audit=log,
        )
        curve, decisions, trades, metrics = simular(
            b, decision_callback=callback, permitir_cash=True,
        )
        pd.DataFrame(log).to_csv(dest / f"auditoria_{name}.csv", index=False)
        decisions.to_csv(dest / f"decisoes_{name}.csv", index=False)
        trades.to_csv(dest / f"operacoes_{name}.csv", index=False)
        folds = _performance_by_fold(b, curve)
        risk = _resumo_curva(curve)
        counts = pd.DataFrame(log)["motivo"].value_counts().to_dict()
        _write_json(dest / f"metricas_{name}.json", {
            "metricas": metrics, "risco": risk, "folds": folds,
            "contagem_decisoes": counts,
        })
        scenario_results.append({"cenario": name, **risk, **metrics, **counts})
        curves[name] = curve
        print(
            f"[DETERIORACAO] {name}: USD {curve[-1]:,.2f}; "
            f"MDD {risk['drawdown_maximo']:.2%}; eventos {counts}", flush=True,
        )
    curves.to_csv(dest / "comparacao_patrimonio_diario.csv", index=False)
    pd.DataFrame(scenario_results).to_csv(
        dest / "comparacao_resultados.csv", index=False,
    )
    pd.DataFrame(list(training.values())).to_csv(
        dest / "qualidade_previsoes_por_fold.csv", index=False,
    )
    rows = []
    for i in range(len(b.dates)):
        if int(b.fold[i]) < 2:
            continue
        for j, symbol in enumerate(b.symbols):
            if np.isfinite(risks[i, j]):
                rows.append({
                    "data": str(b.dates[i]), "fold": int(b.fold[i]),
                    "ativo": symbol,
                    "probabilidade_queda_20d": float(risks[i, j]),
                    "queda_realizada_20d": (
                        float(realized[i, j]) if np.isfinite(realized[i, j])
                        else None
                    ),
                    "evento_realizado_15pct": (
                        float(labels[i, j]) if np.isfinite(labels[i, j])
                        else None
                    ),
                })
    pd.DataFrame(rows).to_csv(
        dest / "previsoes_risco_oos.csv", index=False,
    )
    _write_json(dest / "manifesto_experimento.json", {
        "status": "completed_research_only",
        "modo": "deterioracao",
        "arquivo": "pesquisar_permanencia_minima_spyder.py",
        "branch": "research/permanencia-minima-contrafactual-v1",
        "paridade_control_max_usd": parity,
        "paridade_control_acoes_divergentes": len(mismatches),
        "fonte_sha256": b.original_sha,
        "folds": 3, "universo": len(b.symbols),
        "alvo": (
            f"queda de {abs(D_QUEDA_ALVO):.0%} em algum fechamento dentre "
            f"{D_HORIZONTE} sessoes apos a abertura seguinte"
        ),
        "modelo": "LogisticRegression de risco de queda, regularizacao C=0.3",
        "features": D_ATRIBUTOS, "purge": D_PURGE,
        "validacao": training, "cenarios": scenario_results,
        "thresholds_predefinidos": {
            "permanencia_conservadora": {
                "saida": 0.75, "entrada": 0.45, "veto_rotacao": False,
            },
            "permanencia_moderada": {
                "saida": 0.60, "entrada": 0.40, "veto_rotacao": False,
            },
            "permanencia_e_veto_entrada": {
                "saida": 0.75, "entrada": 0.45, "veto_rotacao": True,
            },
        },
        "ressalvas": [
            "Fold 1 Control; politica de risco apenas nos folds 2 e 3.",
            "Precos de abertura/fechamento, custos e slippage referencia.",
            "O risco previsto nao e retorno esperado nem garantia de protecao.",
            "A execucao contrafactual pressupoe ausencia de impacto da ordem.",
            "Nao ha limite realista de capacidade para ordens acima do volume.",
            "Folds 2/3 ja inspecionados na pesquisa anterior: exploratorio.",
            "Nenhuma alteracao da Winner Live ou de producao.",
        ],
    })
    shutil.copyfile(Path(__file__), dest / "codigo_executado.py")
    archive = _pacote(dest, source)
    _publicar_pasta_vantagem(dest)
    print(f"[DETERIORACAO] ZIP pronto: {archive}", flush=True)


# %% 12. Sensibilidade da margem de rotacao: somente base cientifica congelada.
# Nenhuma conexao operacional, aquisicao de dados ou retreino de modelos.
MARGENS_ALTERNATIVAS = {1: 0.01, 2: 0.0, 3: 0.01}


def executar_sensibilidade_margens() -> None:
    """Compara apenas a calibracao por fold, mantendo todo o resto identico."""
    fonte = ROOT / "output" / "pesquisa_permanencia_minima.zip"
    if not fonte.is_file():
        raise FileNotFoundError(
            f"Pacote cientifico congelado ausente: {fonte}. "
            "Execute o modo 'comparar' apenas com a base congelada valida."
        )
    base = carregar(fonte)
    original = dict(base.margens)
    if set(original) != {1, 2, 3}:
        raise RuntimeError(
            f"Esperados tres folds na base historica; encontrado: {original}"
        )
    control, control_decisions, control_trades, control_metrics = simular(base)
    parity = float(np.max(np.abs(control - base.base_equity)))
    if parity > 0.01 or abs(float(control[-1]) - 587165941.1330212) > 0.01:
        raise RuntimeError(
            f"Control congelado nao reproduzido; erro maximo US$ {parity:.9f}."
        )

    def decidir(_base, i, pos, holding, capital, quantity):
        return _d_control(_base, i, pos, holding), []

    # Prova de equivalencia: reexecutar com a mesma regra e margens originais,
    # agora recalculando a posicao real em TODOS os folds.
    refeito, _, _, _ = simular(
        base, decision_callback=decidir,
        permitir_cash=True, aplicar_todos_folds=True,
    )
    erro_reexecucao = float(np.max(np.abs(refeito - control)))
    if erro_reexecucao > 0.01:
        raise RuntimeError(
            "Motor da sensibilidade nao reproduz o Control da mesma base: "
            f"US$ {erro_reexecucao:.9f}."
        )
    base.margens = dict(MARGENS_ALTERNATIVAS)
    alternativo, alt_decisions, alt_trades, alt_metrics = simular(
        base, decision_callback=decidir,
        permitir_cash=True, aplicar_todos_folds=True,
    )
    base.margens = original

    saida = DESTINO_TEMP
    if saida.exists():
        shutil.rmtree(saida)
    saida.mkdir(parents=True)
    curves = pd.DataFrame({
        "data_execucao": base.exec_dates,
        "fold": base.fold,
        "control": control,
        "margens_alternativas": alternativo,
    })
    curves.to_csv(saida / "comparacao_patrimonio_diario.csv", index=False)
    control_decisions.to_csv(saida / "decisoes_control.csv", index=False)
    alt_decisions.to_csv(saida / "decisoes_margens_alternativas.csv", index=False)
    control_trades.to_csv(saida / "operacoes_control.csv", index=False)
    alt_trades.to_csv(saida / "operacoes_margens_alternativas.csv", index=False)
    folds_control = _performance_by_fold(base, control)
    folds_alternativo = _performance_by_fold(base, alternativo)
    por_fold = []
    for c, a in zip(folds_control, folds_alternativo):
        por_fold.append({
            "fold": int(c["fold"]),
            "control": c,
            "alternativo": a,
        })
    _write_json(saida / "comparacao_por_fold.json", por_fold)
    resultado = []
    for nome, curve, metrics in (
        ("control", control, control_metrics),
        ("margens_alternativas", alternativo, alt_metrics),
    ):
        resultado.append({
            "cenario": nome, **_resumo_curva(curve), **metrics,
        })
    pd.DataFrame(resultado).to_csv(
        saida / "comparacao_resultados.csv", index=False,
    )
    _write_json(saida / "manifesto_experimento.json", {
        "status": "completed_research_only",
        "modo": "sensibilidade_margens",
        "arquivo": Path(__file__).name,
        "fonte_sha256": base.original_sha,
        "paridade_controle_usd": parity,
        "paridade_motor_contrafactual_usd": erro_reexecucao,
        "margens_por_fold_control": original,
        "margens_por_fold_alternativas": MARGENS_ALTERNATIVAS,
        "escopo": (
            "Somente margens de troca variam. Precos, sinais, folds, "
            "custos, universos, calendario e capital inicial congelados."
        ),
        "criterio": "analise de sensibilidade ex post, nao politica aprovada",
        "resultados": resultado,
        "limitacoes": [
            "As margens alternativas foram fixadas apos observacao historica.",
            "Nao corresponde a treinamento ou calibracao nova.",
            "Nao demonstra a identidade de bases externas.",
            "Sem impacto realista de mercado para ordens grandes.",
            "Nenhuma alteracao na estrategia operacional.",
        ],
    })
    shutil.copyfile(Path(__file__), saida / "codigo_executado.py")
    archive = _pacote(saida, fonte)
    _publicar_pasta_vantagem(saida)
    print(
        f"[MARGENS] Control US$ {control[-1]:,.2f} | "
        f"Alternativo US$ {alternativo[-1]:,.2f} | "
        f"erro de paridade US$ {erro_reexecucao:.9f}",
        flush=True,
    )
    print(f"[ZIP] {archive}", flush=True)


# %% 13. Auditoria observacional de permanencia em uma exportacao de replay.
# Utiliza SOMENTE o ZIP fornecido e seus campos de decisao. Nao refaz treino,
# nao usa contas externas, nao presume OHLCV ou execucoes contrafactuais.


def _a_localizar_replay() -> Path:
    """Descobre o replay completo nas pastas habituais, sem parametros externos."""
    dirs = [ROOT / "output", ROOT, Path.home() / "Downloads"]
    encontrados = []
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.zip"):
            if not path.is_file() or path.name.startswith("pesquisa_"):
                continue
            try:
                with ZipFile(path) as z:
                    names = z.namelist()
                    valid = (
                        sum(n.endswith("_predictions.csv") for n in names) == 1
                        and sum(n.endswith("_metrics.json") for n in names) == 1
                        and "experiment_manifest.json" in names
                    )
                if valid:
                    encontrados.append(path)
            except (OSError, ValueError, BadZipFile):
                continue
    if not encontrados:
        raise FileNotFoundError(
            "Nao foi encontrado um ZIP de replay com predictions e metrics. "
            "Coloque o ZIP exportado na pasta output/ ou Downloads."
        )
    return max(encontrados, key=lambda p: (p.stat().st_mtime, p.name))


def _a_calcular(
    df: pd.DataFrame, initial_capital: float = 10000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sinais de fechamento e resultados posteriores com censura por rotacao."""
    required = (
        "timestamp", "decision_date", "current_asset", "selected_asset",
        "strategy_equity", "walk_forward_fold", "holding_days_at_decision",
        "position_drawdown_from_peak", "position_return_since_entry",
        "score_change_from_entry", "current_score", "raw_best_score",
        "decision_reason", "universe_breadth_5",
        "consecutive_days_current_not_top1",
    )
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError("Exportacao sem os campos necessarios: " + str(absent))
    df = df.copy().reset_index(drop=True)
    df["data_execucao"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="raise",
    ).dt.strftime("%Y-%m-%d")
    df["data_decisao"] = pd.to_datetime(
        df["decision_date"], utc=True, errors="raise",
    ).dt.strftime("%Y-%m-%d")
    if df["data_execucao"].duplicated().any():
        raise ValueError("Replay com datas de execucao duplicadas.")
    df["episodio_id"] = df["selected_asset"].ne(
        df["selected_asset"].shift(),
    ).cumsum()
    df["patrimonio_anterior"] = df["strategy_equity"].shift()
    held = (
        df["selected_asset"].eq(df["current_asset"])
        & df["current_asset"].ne("CASH")
        & df["holding_days_at_decision"].ge(2)
    )
    df["posicao_mantida"] = held
    n = len(df)
    for horizon in (1, 3, 5):
        future = np.full(n, np.nan)
        for i in range(1, n - horizon + 1):
            if not held.iloc[i]:
                continue
            asset = df["current_asset"].iloc[i]
            if not df["selected_asset"].iloc[i - 1:i + horizon].eq(asset).all():
                continue
            before = float(df["strategy_equity"].iloc[i - 1])
            after = float(df["strategy_equity"].iloc[i + horizon - 1])
            if before > 0:
                future[i] = after / before - 1
        df[f"retorno_observado_{horizon}d"] = future

    episodes = []
    capital_start = float(initial_capital)
    for episode_id, group in df.groupby("episodio_id", sort=True):
        asset = str(group["selected_asset"].iloc[0])
        if asset == "CASH":
            continue
        start, end = int(group.index[0]), int(group.index[-1])
        before = (
            float(df["strategy_equity"].iloc[start - 1])
            if start else capital_start
        )
        after = float(df["strategy_equity"].iloc[end])
        equity = np.r_[before, group["strategy_equity"].to_numpy(float)]
        peaks = np.maximum.accumulate(equity)
        row = {
            "episodio_id": int(episode_id), "ativo": asset,
            "fold": int(group["walk_forward_fold"].iloc[0]),
            "data_inicio": str(group["data_execucao"].iloc[0]),
            "data_fim": str(group["data_execucao"].iloc[-1]),
            "prego_es": int(len(group)),
            "capital_inicio": before, "capital_final": after,
            "retorno_capital": after / before - 1,
            "drawdown_maximo": float((equity / peaks - 1).min()),
        }
        for limit in (10, 15, 20, 30):
            hits = group.loc[
                group["posicao_mantida"]
                & group["position_drawdown_from_peak"].le(-limit / 100),
            ]
            row[f"primeiro_alerta_dd_{limit}"] = (
                str(hits["data_decisao"].iloc[0]) if len(hits) else None
            )
        episodes.append(row)
    episodes = pd.DataFrame(episodes)
    flags = {
        "drawdown_10": df["position_drawdown_from_peak"].le(-0.10),
        "drawdown_20": df["position_drawdown_from_peak"].le(-0.20),
        "drawdown_30": df["position_drawdown_from_peak"].le(-0.30),
        "pnl_negativo_10": df["position_return_since_entry"].le(-0.10),
        "queda_utilidade_010": df["score_change_from_entry"].le(-0.10),
        "drawdown_10_score_subiu": (
            df["position_drawdown_from_peak"].le(-0.10)
            & df["score_change_from_entry"].gt(0)
        ),
        "drawdown_10_score_caiu": (
            df["position_drawdown_from_peak"].le(-0.10)
            & df["score_change_from_entry"].le(0)
        ),
        "fora_topo_2": df["consecutive_days_current_not_top1"].ge(2),
        "mercado_fraco": df["universe_breadth_5"].lt(0.30),
    }
    statistics = []
    for horizon in (1, 3, 5):
        outcomes = df[f"retorno_observado_{horizon}d"]
        for fold in ("todos", 1, 2, 3):
            eligible = held & outcomes.notna()
            if fold != "todos":
                eligible &= df["walk_forward_fold"].eq(fold)
            for name, flag in flags.items():
                cases = outcomes.loc[eligible & flag]
                others = outcomes.loc[eligible & ~flag]
                statistics.append({
                    "fold": fold, "horizonte": horizon, "sinal": name,
                    "observacoes": int(eligible.sum()),
                    "com_sinal": int(len(cases)),
                    "sem_sinal": int(len(others)),
                    "retorno_medio_sinal": (
                        float(cases.mean()) if len(cases) else None
                    ),
                    "retorno_mediano_sinal": (
                        float(cases.median()) if len(cases) else None
                    ),
                    "fracao_negativa_sinal": (
                        float(cases.lt(0).mean()) if len(cases) else None
                    ),
                    "fracao_perda_10_sinal": (
                        float(cases.le(-0.10).mean()) if len(cases) else None
                    ),
                    "retorno_medio_sem_sinal": (
                        float(others.mean()) if len(others) else None
                    ),
                    "fracao_negativa_sem_sinal": (
                        float(others.lt(0).mean()) if len(others) else None
                    ),
                })
    return df, episodes, pd.DataFrame(statistics)



def _a_auditar_cobertura_de_perdas(
    diario: pd.DataFrame, episodios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia alertas em perdas observadas; NAO simula qualquer venda.

    'Cobertura' mede quanto do risco *ja ocorrido* um sinal teria marcado
    no fechamento anterior. Dias do mesmo episodio nao sao independentes.
    """
    held = diario.loc[
        diario["posicao_mantida"]
        & diario["retorno_observado_1d"].notna()
    ].copy()
    signals = {
        "drawdown_10": held["position_drawdown_from_peak"].le(-0.10),
        "drawdown_20": held["position_drawdown_from_peak"].le(-0.20),
        "drawdown_10_score_caiu": (
            held["position_drawdown_from_peak"].le(-0.10)
            & held["score_change_from_entry"].le(0)
        ),
        "drawdown_10_score_subiu": (
            held["position_drawdown_from_peak"].le(-0.10)
            & held["score_change_from_entry"].gt(0)
        ),
        "fora_topo_2": held["consecutive_days_current_not_top1"].ge(2),
        "mercado_fraco": held["universe_breadth_5"].lt(0.30),
        "queda_utilidade_010": held["score_change_from_entry"].le(-0.10),
        "pnl_negativo_10": held["position_return_since_entry"].le(-0.10),
    }
    detail = []
    summary = []
    for name, flagged in signals.items():
        flagged = flagged.fillna(False)
        first_by_episode = (
            held.loc[flagged]
            .sort_values("data_decisao")
            .groupby("episodio_id", sort=True).first()
        )
        for episode_id, first in first_by_episode.iterrows():
            episode = episodios.loc[
                episodios["episodio_id"].eq(episode_id)
            ]
            if len(episode) != 1:
                raise ValueError(
                    f"Episodio {episode_id} ausente ou repetido na auditoria."
                )
            e = episode.iloc[0]
            detail.append({
                "sinal": name, "episodio_id": int(episode_id),
                "ativo": str(first["selected_asset"]),
                "fold": int(first["walk_forward_fold"]),
                "primeira_data_decisao": str(first["data_decisao"]),
                "retorno_observado_1d_primeiro_alerta": float(
                    first["retorno_observado_1d"]
                ),
                "retorno_episodio_completo": float(e["retorno_capital"]),
                "drawdown_episodio_completo": float(e["drawdown_maximo"]),
                "resultado_episodio_futuro_nao_e_feature": True,
            })
        for fold in ("todos", 1, 2, 3):
            population = pd.Series(True, index=held.index)
            if fold != "todos":
                population &= held["walk_forward_fold"].eq(fold)
            for loss in (0.05, 0.10):
                target = held["retorno_observado_1d"].le(-loss)
                fired = population & flagged
                events = population & target
                detected = fired & events
                total_events = int(events.sum())
                total_fired = int(fired.sum())
                n_detected = int(detected.sum())
                summary.append({
                    "fold": fold, "sinal": name,
                    "limite_perda_diaria": -loss,
                    "observacoes_validas": int(population.sum()),
                    "quedas_observadas": total_events,
                    "alertas_emitidos": total_fired,
                    "quedas_antecipadas_por_alerta": n_detected,
                    "quedas_nao_sinalizadas": total_events - n_detected,
                    "alertas_sem_queda_do_limite": total_fired - n_detected,
                    "precisao_alerta": (
                        n_detected / total_fired if total_fired else None
                    ),
                    "cobertura_quedas": (
                        n_detected / total_events if total_events else None
                    ),
                    "taxa_base_quedas": (
                        total_events / int(population.sum())
                        if population.any() else None
                    ),
                    "episodios_distintos_com_alerta": int(
                        held.loc[fired, "episodio_id"].nunique()
                    ),
                    "episodios_distintos_com_queda_antecipada": int(
                        held.loc[detected, "episodio_id"].nunique()
                    ),
                    "hipotese_nao_validada_fora_amostra": True,
                })
    return pd.DataFrame(summary), pd.DataFrame(detail)



# Sinais conhecidos no fechamento da decisao. A resposta de treino e o
# rendimento que a carteira REALMENTE teve ao continuar com a mesma posicao.
A_FEATURES_CAUSAS = (
    "holding_days_at_decision",
    "score_change_from_entry",
    "position_drawdown_from_peak",
    "position_return_since_entry",
    "best_vs_second_gap",
    "spy_return_5",
    "universe_breadth_5",
)



def _a_ic_auc_por_episodio(
    targets: np.ndarray, scores: np.ndarray, episodio_ids: np.ndarray,
    *, n_amostras: int = 1200, seed: int = 42,
) -> tuple[float | None, float | None, int, int]:
    """Bootstrap de episodios (nao dias); intervalo exploratorio, nao prospectivo."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(targets, dtype=int)
    p = np.asarray(scores, dtype=float)
    g = np.asarray(episodio_ids)
    if not (len(y) == len(p) == len(g)):
        raise ValueError("Comprimentos de evento, score e episodio divergentes.")
    groups = pd.Series(np.arange(len(g))).groupby(g, sort=True)
    slots = [v.to_numpy(dtype=int) for _, v in groups]
    positive_groups = int(len(np.unique(g[y == 1])))
    if len(slots) < 2 or positive_groups < 2:
        return None, None, int(len(slots)), positive_groups
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_amostras):
        chosen = rng.integers(0, len(slots), len(slots))
        idx = np.concatenate([slots[k] for k in chosen])
        if np.unique(y[idx]).size == 2:
            estimates.append(float(roc_auc_score(y[idx], p[idx])))
    if len(estimates) < 100:
        return None, None, int(len(slots)), positive_groups
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), int(len(slots)), positive_groups


def _a_sensibilidade_primeiro_alerta(
    alertas: pd.DataFrame, capital_control: float,
) -> tuple[pd.DataFrame, dict]:
    """Influencias leave-one-episode-out da ablacao teórica.

    Nao retira alertas para tomar decisoes: revela quanto cada observacao
    JA OCORRIDA altera a conclusao. Nao testa uma politica operacional.
    """
    needed = {
        "episodio_id", "data_decisao", "selected_asset",
        "contribuicao_log_alerta", "retorno_control", "walk_forward_fold",
    }
    if not needed.issubset(alertas):
        raise ValueError("Dados incompletos para sensibilidade de episodios.")
    if alertas["episodio_id"].duplicated().any():
        raise ValueError("Mais de um primeiro alerta para mesmo episodio.")
    if capital_control <= 0:
        raise ValueError("Capital final do Control deve ser positivo.")
    total = float(alertas["contribuicao_log_alerta"].sum())
    shadow = float(capital_control * np.exp(-total))
    res = alertas.loc[:, [
        "episodio_id", "data_decisao", "selected_asset", "walk_forward_fold",
        "retorno_control", "contribuicao_log_alerta",
    ]].copy()
    res["capital_apos_remover_um_alerta"] = (
        float(capital_control) *
        np.exp(-total + res["contribuicao_log_alerta"].to_numpy(float))
    )
    res["variacao_sobre_control_apos_remocao"] = (
        res["capital_apos_remover_um_alerta"] / capital_control - 1
    )
    res["mudou_conclusao_para_superior_ao_control"] = (
        res["capital_apos_remover_um_alerta"] > capital_control
    )
    res = res.sort_values("capital_apos_remover_um_alerta", ascending=False)
    return res, {
        "capital_control_final": float(capital_control),
        "capital_shadow_primeiros_alertas": shadow,
        "episodios_distintos": int(len(res)),
        "capital_shadow_menor_apos_remocao": (
            float(res["capital_apos_remover_um_alerta"].min())
            if len(res) else float(capital_control)
        ),
        "capital_shadow_maior_apos_remocao": (
            float(res["capital_apos_remover_um_alerta"].max())
            if len(res) else float(capital_control)
        ),
        "num_remocoes_que_invertem_conclusao": int(
            res["mudou_conclusao_para_superior_ao_control"].sum()
        ),
        "restricao": (
            "Teste ex post, usando o retorno conhecido so para auditoria. "
            "Nao permite selecionar que alertas ignorar nas operacoes."
        ),
    }


def _a_experimento_risco_cauda_oos(
    diario: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classificacao de perda >=5% no dia seguinte, so treino anterior.

    O threshold e fixado no TREINO, nao no fold de teste. Os folds usados
    ja foram vistos em analises anteriores; isto e exploratorio, nao
    certificacao prospectiva nem modelo liberado para uso financeiro.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    held = diario.loc[
        diario["posicao_mantida"]
        & diario["retorno_observado_1d"].notna()
    ].copy()
    held["evento_perda_5pct_dia_seguinte"] = (
        held["retorno_observado_1d"].le(-0.05)
    ).astype(int)
    held["probabilidade_modelo_nao_calibrada"] = np.nan
    held["alerta_limiar_treino"] = False
    metrics = []
    for fold in (2, 3):
        test = held.loc[held["walk_forward_fold"].eq(fold)]
        if test.empty:
            raise RuntimeError(f"Sem amostra de teste do fold {fold}")
        start = pd.to_datetime(test["data_decisao"], utc=True).min()
        train = held.loc[
            held["walk_forward_fold"].lt(fold)
            & pd.to_datetime(held["data_execucao"], utc=True).lt(
                start - pd.Timedelta(days=30)
            )
        ]
        y = train["evento_perda_5pct_dia_seguinte"].to_numpy(int)
        observed = test["evento_perda_5pct_dia_seguinte"].to_numpy(int)
        if len(y) < 100 or y.sum() < 8 or observed.sum() < 2:
            raise RuntimeError(
                f"Fold {fold} sem eventos suficientes para diagnostico de cauda."
            )
        tr = train.loc[:, list(A_FEATURES_CAUSAS)].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        te = test.loc[:, list(A_FEATURES_CAUSAS)].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.2, class_weight="balanced",
                max_iter=2000, random_state=42,
            ),
        )
        model.fit(tr, y)
        scores_train = model.predict_proba(tr)[:, 1]
        threshold = float(np.quantile(scores_train, 0.90))
        scores_test = model.predict_proba(te)[:, 1]
        alerts = scores_test >= threshold
        held.loc[test.index, "probabilidade_modelo_nao_calibrada"] = scores_test
        held.loc[test.index, "alerta_limiar_treino"] = alerts
        detected = int((alerts & (observed == 1)).sum())
        n_alert = int(alerts.sum())
        n_events = int(observed.sum())
        ic_low, ic_high, episode_count, event_episode_count = (
            _a_ic_auc_por_episodio(
                observed, scores_test, test["episodio_id"].to_numpy(),
                seed=42 + fold,
            )
        )
        auc_train = float(roc_auc_score(y, scores_train))
        auc_test = float(roc_auc_score(observed, scores_test))
        metrics.append({
            "auc_roc_treino": auc_train,
            "auc_roc_oos_gap_treino_menos_teste": auc_train - auc_test,
            "precisao_media_treino": float(
                average_precision_score(y, scores_train)
            ),
            "taxa_base_treino": float(np.mean(y)),
            "brier_treino_nao_calibrado": float(
                brier_score_loss(y, scores_train)
            ),
            "brier_oos_nao_calibrado": float(
                brier_score_loss(observed, scores_test)
            ),
            "brier_baseline_constante_teste": float(
                np.mean((observed - np.mean(y)) ** 2)
            ),
            "score_medio_previsto_teste": float(np.mean(scores_test)),
            "episodios_teste": episode_count,
            "episodios_com_perda_teste": event_episode_count,
            "ic95_auc_bootstrap_episodios_inferior": ic_low,
            "ic95_auc_bootstrap_episodios_superior": ic_high,
            "ic_auc_amostras_episodios": 1200,
            "fold_teste": fold,
            "primeira_decisao_teste": str(start.date()),
            "ultima_execucao_treino": str(
                pd.to_datetime(train["data_execucao"], utc=True).max().date()
            ),
            "embargo_dias_calendario": 30,
            "observacoes_treino": int(len(train)),
            "perdas_5pct_treino": int(y.sum()),
            "observacoes_teste": int(len(test)),
            "perdas_5pct_teste": n_events,
            "taxa_base_teste": n_events / len(test),
            "auc_roc_oos": auc_test,
            "precisao_media_oos": float(
                average_precision_score(observed, scores_test)
            ),
            "limiar_alerta_percentil_90_treino": threshold,
            "alertas": n_alert, "perdas_antecipadas": detected,
            "perdas_nao_sinalizadas": n_events - detected,
            "precisao_alertas": (
                detected / n_alert if n_alert else None
            ),
            "cobertura_quedas": detected / n_events,
            "observacao": (
                "Amostra de Control censurada por rotacoes; "
                "treino por folds anteriores, mas features e escolhas "
                "experimentais ja foram examinadas neste periodo."
            ),
        })
    return held.loc[held["walk_forward_fold"].ge(2), [
        "data_decisao", "data_execucao", "walk_forward_fold",
        "episodio_id", "selected_asset",
        "evento_perda_5pct_dia_seguinte",
        "probabilidade_modelo_nao_calibrada",
        "alerta_limiar_treino",
        *A_FEATURES_CAUSAS,
    ]].copy(), pd.DataFrame(metrics)



def _a_impacto_alertas_exposicao(
    diario: pd.DataFrame, previsoes: pd.DataFrame,
    capital_inicial: float, *, primeiro_por_episodio: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ablacao economica CLOSE-TO-CLOSE, sem simular execucao/ordens.

    Hipotese restrita: substituir por retorno zero cada sessao de execucao
    alertada; nas outras, manter o retorno historico do Control. Ignora
    abertura, reentrada, custos adicionais, dependencia da politica e
    impacto de mercado. NAO e backtest de operacao em CASH.
    """
    if capital_inicial <= 0:
        raise ValueError("Capital inicial deve ser positivo.")
    required_daily = {
        "data_decisao", "data_execucao", "strategy_equity",
        "walk_forward_fold",
    }
    required_pred = {
        "data_decisao", "data_execucao", "alerta_limiar_treino",
        "episodio_id", "selected_asset",
    }
    if not required_daily.issubset(diario):
        raise ValueError("Curva nao tem datas ou patrimonio completos.")
    if not required_pred.issubset(previsoes):
        raise ValueError("Previsoes sem identificacao unica de alertas.")
    ordered = diario.sort_values("data_execucao").copy().reset_index(drop=True)
    if ordered["data_execucao"].duplicated().any():
        raise ValueError("Curva com datas de execucao duplicadas.")
    if previsoes["data_execucao"].duplicated().any():
        raise ValueError("Previsoes com datas de execucao duplicadas.")
    capital = pd.to_numeric(ordered["strategy_equity"], errors="raise")
    if not np.isfinite(capital).all() or (capital <= 0).any():
        raise ValueError("Capital invalido na curva.")
    is_alert = previsoes["alerta_limiar_treino"].eq(True)
    fired = previsoes.loc[is_alert].copy()
    if primeiro_por_episodio:
        # Criterio cronologico: reage so ao PRIMEIRO alerta da posicao,
        # sem consultar o retorno de dias futuros.
        fired = (
            fired.sort_values(["data_execucao", "data_decisao"])
            .drop_duplicates(subset=["episodio_id"], keep="first")
        )
    if not fired["data_execucao"].isin(ordered["data_execucao"]).all():
        raise ValueError("Data alertada nao existe na curva do Control.")
    matched = ordered[["data_execucao", "data_decisao"]].merge(
        fired[["data_execucao", "data_decisao"]],
        on=["data_execucao", "data_decisao"],
        validate="one_to_one",
        how="inner",
    )
    if len(matched) != len(fired):
        raise ValueError("Alerta possui data de decisao incongruente.")
    ordered["alerta_gerado_no_fechamento_anterior"] = (
        ordered["data_execucao"].isin(fired["data_execucao"])
    )
    prior = capital.shift(1).fillna(float(capital_inicial))
    daily_gross = capital / prior
    ordered["retorno_control"] = daily_gross - 1.0
    ordered["retorno_exposicao_shadow"] = np.where(
        ordered["alerta_gerado_no_fechamento_anterior"],
        0.0, daily_gross - 1.0,
    )
    ordered["capital_control_recalculado"] = (
        float(capital_inicial) * daily_gross.cumprod()
    )
    ordered["capital_exposicao_shadow"] = (
        float(capital_inicial)
        * (1.0 + ordered["retorno_exposicao_shadow"]).cumprod()
    )
    dif = float(np.max(np.abs(
        ordered["capital_control_recalculado"].to_numpy() -
        capital.to_numpy()
    )))
    if dif > 0.01:
        raise ValueError(f"Curva original nao reproduzida: USD {dif:.9f}")
    ordered["contribuicao_log_alerta"] = np.where(
        ordered["alerta_gerado_no_fechamento_anterior"],
        np.log(daily_gross), 0.0,
    )
    alert_rows = ordered.loc[
        ordered["alerta_gerado_no_fechamento_anterior"],
        [
            "data_decisao", "data_execucao", "walk_forward_fold",
            "retorno_control", "contribuicao_log_alerta",
            "capital_control_recalculado", "capital_exposicao_shadow",
        ],
    ].merge(
        fired[[
            "data_decisao", "data_execucao", "selected_asset",
            "episodio_id", "probabilidade_modelo_nao_calibrada",
        ]],
        on=["data_decisao", "data_execucao"],
        how="inner", validate="one_to_one",
    )
    stats = []
    for fold, block in [("todos", ordered)] + [
        (int(fold), sub)
        for fold, sub in ordered.groupby("walk_forward_fold", sort=True)
    ]:
        alerts = block.loc[block["alerta_gerado_no_fechamento_anterior"]]
        logsum = float(alerts["contribuicao_log_alerta"].sum())
        ratio = float(np.exp(-logsum))
        ctrl = block["capital_control_recalculado"].to_numpy(float)
        shadow = block["capital_exposicao_shadow"].to_numpy(float)
        stats.append({
            "fold": fold, "sessoes": int(len(block)),
            "sessoes_com_alerta": int(len(alerts)),
            "sessoes_positivas_alertadas": int(
                alerts["retorno_control"].gt(0).sum()
            ),
            "sessoes_negativas_alertadas": int(
                alerts["retorno_control"].lt(0).sum()
            ),
            "log_retorno_original_dos_alertas": logsum,
            "fator_capital_shadow_vs_control_no_fold": ratio,
            "capital_control_fim_periodo": float(ctrl[-1]),
            "capital_shadow_fim_periodo": float(shadow[-1]),
            "drawdown_control_na_curva_completa": float(
                (ctrl / np.maximum.accumulate(ctrl) - 1.0).min()
            ),
            "drawdown_shadow_na_curva_completa": float(
                (shadow / np.maximum.accumulate(shadow) - 1.0).min()
            ),
            "limite": (
                "Ablacao de exposicao diaria; NAO contabiliza operacoes "
                "alternativas, entradas em abertura ou custos adicionais."
            ),
        })
    if len(alert_rows) != len(fired):
        raise ValueError("Nem todos os alertas apareceram no impacto.")
    return ordered, alert_rows, pd.DataFrame(stats)


def _a_publicar_sem_fonte(working: Path) -> Path:
    """Nome unico, CRC e SHA antes da publicacao. O ZIP fonte nao e copiado."""
    archive = DESTINO.with_suffix(".zip")
    tmp = archive.with_suffix(".zip.tmp")
    checksums = {
        f.relative_to(working).as_posix(): _hash_arquivo(f)
        for f in sorted(working.rglob("*")) if f.is_file()
    }
    _write_json(working / "arquivos_sha256.json", checksums)
    tmp.unlink(missing_ok=True)
    try:
        with ZipFile(tmp, "w", ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(working.rglob("*")):
                if file.is_file():
                    z.write(
                        file, f"{DESTINO.name}/{file.relative_to(working)}",
                    )
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise RuntimeError("ZIP de auditoria corrompido.")
            for name, expected in checksums.items():
                if hashlib.sha256(
                    z.read(f"{DESTINO.name}/{name}")
                ).hexdigest() != expected:
                    raise RuntimeError(f"SHA divergente em {name}")
        tmp.replace(archive)
        _publicar_pasta_vantagem(working)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return archive


def executar_analise_permanencia() -> None:
    """Analisa decisoes registradas; nao simula venda ou troca inexistente."""
    from io import BytesIO

    source = _a_localizar_replay()
    with ZipFile(source) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP de origem corrompido.")
        predictions = next(
            n for n in z.namelist() if n.endswith("_predictions.csv")
        )
        metric_name = next(
            n for n in z.namelist() if n.endswith("_metrics.json")
        )
        raw = pd.read_csv(BytesIO(z.read(predictions)), low_memory=False)
        metrics = json.loads(z.read(metric_name))
    df, episodes, associations = _a_calcular(
        raw, initial_capital=float(metrics.get("initial_capital") or 10000.0),
    )
    if len(df) < 100:
        raise RuntimeError("Historico de decisoes insuficiente.")
    dest = DESTINO_TEMP
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    df.to_csv(dest / "sinais_por_decisao.csv", index=False)
    episodes.to_csv(dest / "episodios_posicao.csv", index=False)
    associations.to_csv(
        dest / "associacoes_sinais_e_retornos_observados.csv", index=False,
    )
    cobertura, alertas_episodio = _a_auditar_cobertura_de_perdas(df, episodes)
    cobertura.to_csv(dest / "cobertura_de_perdas_por_fold.csv", index=False)
    alertas_episodio.to_csv(dest / "primeiros_alertas_por_episodio.csv", index=False)
    previstos, metrica_oos = _a_experimento_risco_cauda_oos(df)
    previstos.to_csv(dest / "risco_cauda_previsoes_oos.csv", index=False)
    metrica_oos.to_csv(dest / "risco_cauda_qualidade_por_fold.csv", index=False)
    impacto_curva, impacto_alertas, impacto_resumo = _a_impacto_alertas_exposicao(
        df, previstos, capital_inicial=float(metrics.get("initial_capital") or 10000.0),
    )
    impacto_curva.to_csv(dest / "impacto_alertas_curva_shadow.csv", index=False)
    impacto_alertas.to_csv(dest / "impacto_alertas_por_decisao.csv", index=False)
    impacto_resumo.to_csv(dest / "impacto_alertas_resumo_por_fold.csv", index=False)
    primeiro_curva, primeiro_alertas, primeiro_resumo = (
        _a_impacto_alertas_exposicao(
            df, previstos,
            capital_inicial=float(metrics.get("initial_capital") or 10000.0),
            primeiro_por_episodio=True,
        )
    )
    primeiro_curva.to_csv(
        dest / "impacto_primeiro_alerta_curva_shadow.csv", index=False,
    )
    primeiro_alertas.to_csv(
        dest / "impacto_primeiro_alerta_por_episodio.csv", index=False,
    )
    primeiro_resumo.to_csv(
        dest / "impacto_primeiro_alerta_resumo_por_fold.csv", index=False,
    )
    sensibilidades, sensibilidade_resumo = _a_sensibilidade_primeiro_alerta(
        primeiro_alertas,
        capital_control=float(impacto_curva["capital_control_recalculado"].iloc[-1]),
    )
    sensibilidades.to_csv(
        dest / "sensibilidade_primeiro_alerta_por_episodio.csv", index=False,
    )
    metrica_oos.to_csv(
        dest / "generalizacao_risco_cauda_por_fold.csv", index=False,
    )
    concentracao = (
        impacto_alertas.groupby(["episodio_id", "selected_asset"], as_index=False)
        .agg(
            alertas=("retorno_control", "size"),
            alertas_antes_de_alta=("retorno_control", lambda v: int((v > 0).sum())),
            alertas_antes_de_queda=("retorno_control", lambda v: int((v < 0).sum())),
            soma_log_retorno_dos_alertas=("contribuicao_log_alerta", "sum"),
        )
        .sort_values("soma_log_retorno_dos_alertas", ascending=False)
    )
    concentracao.to_csv(dest / "concentracao_alertas_por_episodio.csv", index=False)
    case = df.loc[
        df["data_execucao"].between("2026-07-20", "2026-08-21")
    ]
    if not case.empty:
        case.to_csv(
            dest / "caso_capr_julho_agosto_2026.csv", index=False,
        )
    _write_json(dest / "manifesto_experimento.json", {
        "status": "completed_descriptive_audit",
        "modo": "analisar_permanencia",
        "arquivo": Path(__file__).name,
        "fonte_sha256": _hash_arquivo(source),
        "capital_final_replay": metrics.get("strategy_ending_capital"),
        "api_version": metrics.get("api_version"),
        "sessoes": len(df),
        "episodios": len(episodes),
        "observacoes_posicao_mantida": int(df["posicao_mantida"].sum()),
        "observacoes_1d": int(df["retorno_observado_1d"].notna().sum()),
        "observacoes_3d": int(df["retorno_observado_3d"].notna().sum()),
        "observacoes_5d": int(df["retorno_observado_5d"].notna().sum()),
        "indicadores_cauda": ["queda_diaria_5pct", "queda_diaria_10pct"],
        "auditado_por_episodio": True,
        "impacto_exposicao_teorico": {
            "modo": "substituir por 0% o retorno close-to-close dos dias alertados",
            "nao_e_backtest_executavel": True,
            "resultados": impacto_resumo.to_dict(orient="records"),
            "primeiro_alerta_por_episodio": {
                "sensibilidade_leave_one_episode_out": sensibilidade_resumo,
                "regra": (
                    "Neutralizar apenas o dia seguinte ao primeiro alerta "
                    "cronologico por posicao historica; dias seguintes "
                    "seguem os retornos do Control original."
                ),
                "nao_e_backtest_executavel": True,
                "resultados": primeiro_resumo.to_dict(orient="records"),
            },
        },
        "experimento_risco_cauda": {
            "treinamento": "folds anteriores com embargo de 30 dias",
            "features": A_FEATURES_CAUSAS,
            "alvo": "queda de ao menos 5pct no proximo fechamento",
            "limiar": "percentil 90 das pontuacoes de treino",
            "limites": (
                "apenas dias mantidos pelo Control; dados futuros apenas "
                "nos rotulos; class_weight balanced sem calibracao de probabilidade; "
                "teste ja examinado, nao confirma generalizacao independente"
            ),
            "metricas": metrica_oos.to_dict(orient="records"),
        },
        "limitacoes": [
            "Features correspondem ao fechamento da data de decisao.",
            "Patrimonio corresponde ao fechamento posterior a ordem.",
            "Retornos somente em periodos sem rotacao; censura seletiva.",
            "Dias vizinhos e horizontes nao sao amostras independentes.",
            "Cobertura de perdas tem censura por rotacao e depende das decisoes Control.",
            "Resultados finais dos episodios aparecem apenas como auditoria futura.",
            "Nao e backtest contrafactual de venda, rotacao ou caixa.",
            "O impacto close-to-close nao tem reentradas, novas ordens ou slippage.",
            "O primeiro alerta por episodio e hipotese idealizada: nao testa recompra real.",
            "Treino e teste do classificador tem eventos raros por episodio.",
            "AUC por bootstrap de episodios e estimativa exploratoria incerta.",
            "O score balanced nao representa probabilidade calibrada de perda.",
            "Usa apenas eventos futuros como rotulos e auditoria ex post.",
            "Nao treina modelos nem utiliza dados futuros nas features.",
            "Nenhuma conta real ou ordem externa e acessada.",
        ],
    })
    (dest / "LEIA_ME.txt").write_text(
        "Diagnostico observacional de permanencia em replay historico. "
        "Nao e simulacao executavel de sair para CASH.\n"
        "Sinais no fechamento de decisao; patrimonio no fechamento seguinte.\n"
        "Para cada horizonte futuro, observacoes sao censuradas em rotacoes.\n"
        "Os arquivos impacto_alertas_* sao ablacao de exposicao, nao ordens.\n"
        "impacto_primeiro_alerta_* neutraliza so 1 dia por episodio historico.\n"
        "concentracao_alertas_por_episodio.csv explica a dependencia dos alertas.\n"
        "generalizacao_risco_cauda_por_fold.csv mede treino versus teste.\n"
        "sensibilidade_primeiro_alerta_por_episodio.csv e teste leave-one-out.\n",
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__), dest / "codigo_executado.py")
    archive = _a_publicar_sem_fonte(dest)
    print(
        f"[PERMANENCIA] {len(df)} sessoes, {len(episodes)} episodios; "
        f"Control exportado US$ {metrics.get('strategy_ending_capital'):,.2f}.",
        flush=True,
    )
    print("[ZIP]", archive, flush=True)


# %% 14. Vantagem economica: avaliacoes pareadas e CASH shadow.
# Testa decisoes registradas; nao muda a sequencia original de posicoes.
ECO_FEATURES = (
    "holding_days_at_decision", "score_change_from_entry",
    "position_drawdown_from_peak", "position_return_since_entry",
    "best_vs_second_gap", "best_vs_current_gap",
    "spy_return_5", "universe_breadth_5",
    "current_score", "raw_best_score", "cash_score",
    "effective_switch_margin",
)
ECO_CUSTO_CASH = 0.002  # custo hipotetico: 10bps por operacao (ida/volta)
ECO_EMBARGO_DIAS = 30


def _eco_modelo():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=80.0),
    )


def _eco_x(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(ECO_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Variaveis de decisao ausentes: {sorted(missing)}")
    return df.loc[:, list(ECO_FEATURES)].apply(
        pd.to_numeric, errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)


def _eco_cash_diario(
    frame: pd.DataFrame, initial: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Um dia de CASH contra a acao real, sem gap nem ordens simuladas."""
    d = frame.copy().reset_index(drop=True)
    execution = pd.to_datetime(d["timestamp"], utc=True)
    decision = pd.to_datetime(d["decision_date"], utc=True)
    if execution.duplicated().any() or not execution.is_monotonic_increasing:
        raise ValueError("Datas nao unicas ou fora de ordem.")
    equity = d["strategy_equity"].to_numpy(float)
    if initial <= 0 or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Capital inicial ou curva invalidos.")
    logs = np.log(equity / np.r_[float(initial), equity[:-1]])
    d["log_retorno_control_observado"] = logs
    d["log_retorno_esperado_control"] = np.nan
    d["retorno_esperado_vs_cash_apos_custo"] = np.nan
    d["acao_original"] = np.where(
        d["selected_asset"].eq(d["current_asset"]), "HOLD", "ROTATE",
    )
    d["acao_shadow"] = d["acao_original"]
    details = []
    for fold in (2, 3):
        is_test = d["walk_forward_fold"].eq(fold)
        if not is_test.any():
            raise ValueError(f"Fold de teste ausente: {fold}")
        start = decision.loc[is_test].min()
        is_train = (
            d["walk_forward_fold"].lt(fold)
            & execution.lt(start - pd.Timedelta(days=ECO_EMBARGO_DIAS))
        )
        if int(is_train.sum()) < 100:
            raise ValueError(f"Treino insuficiente para fold {fold}")
        model = _eco_modelo()
        model.fit(_eco_x(d.loc[is_train]), logs[is_train.to_numpy()])
        predicted = model.predict(_eco_x(d.loc[is_test]))
        choose_cash = predicted < -ECO_CUSTO_CASH
        d.loc[is_test, "log_retorno_esperado_control"] = predicted
        d.loc[is_test, "retorno_esperado_vs_cash_apos_custo"] = (
            predicted + ECO_CUSTO_CASH
        )
        d.loc[is_test, "acao_shadow"] = np.where(
            choose_cash, "CASH", d.loc[is_test, "acao_original"],
        )
        details.append({
            "fold": fold, "treino": int(is_train.sum()),
            "inicio_teste": str(start.date()),
            "ultima_execucao_treino": str(execution.loc[is_train].max().date()),
            "decisoes_cash": int(choose_cash.sum()),
            "decisoes_testadas": int(is_test.sum()),
        })
    shadow_logs = np.where(d["acao_shadow"].eq("CASH"), -ECO_CUSTO_CASH, logs)
    d["log_retorno_shadow"] = shadow_logs
    d["capital_control_reproduzido"] = initial * np.exp(np.cumsum(logs))
    d["capital_shadow_cash"] = initial * np.exp(np.cumsum(shadow_logs))
    d["capital_shadow_cash_sem_custo"] = initial * np.exp(np.cumsum(
        np.where(d["acao_shadow"].eq("CASH"), 0.0, logs)
    ))
    parity = np.max(np.abs(
        d["capital_control_reproduzido"].to_numpy(float) - equity
    ))
    if parity > 0.01:
        raise RuntimeError(f"Control nao reproduzido: USD {parity}")
    summary = []
    for info in details:
        fold = info["fold"]
        subset = d.loc[d["walk_forward_fold"].eq(fold)]
        fired = subset.loc[subset["acao_shadow"].eq("CASH")]
        first = int(subset.index.min())
        prior_shadow = (
            float(d["capital_shadow_cash"].iloc[first - 1]) if first
            else initial
        )
        prior_ctrl = (
            float(d["capital_control_reproduzido"].iloc[first - 1])
            if first else initial
        )
        delta = float(
            (fired["log_retorno_shadow"]
             - fired["log_retorno_control_observado"]).sum()
        )
        info.update({
            "cash_dias_com_alta_original": int(
                fired["log_retorno_control_observado"].gt(0).sum()
            ),
            "cash_dias_com_queda_original": int(
                fired["log_retorno_control_observado"].lt(0).sum()
            ),
            "capital_control_fim": float(
                subset["capital_control_reproduzido"].iloc[-1]
            ),
            "capital_shadow_fim": float(
                subset["capital_shadow_cash"].iloc[-1]
            ),
            "fator_shadow_vs_control_no_fold": float(np.exp(delta)),
            "retorno_control_dentro_fold": (
                float(subset["capital_control_reproduzido"].iloc[-1])
                / prior_ctrl - 1
            ),
            "retorno_shadow_dentro_fold": (
                float(subset["capital_shadow_cash"].iloc[-1])
                / prior_shadow - 1
            ),
            "impacto_log_cash_vs_control_no_fold": delta,
        })
        summary.append(info)
    return d, pd.DataFrame(summary)


def _eco_rotacoes_pareadas(
    trade: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retornos ROTATE/HOLD/CASH no MESMO intervalo originalmente negociado.

    Nao encadear as decisoes locais: a proxima posicao dependeria da
    escolha alternativa, e seu historico de precos nao esta no export.
    """
    buys = trade.loc[
        trade["action"].eq("BUY")
        & trade["counterfactual_previous_asset_return"].notna()
    ].copy().reset_index(drop=True)
    if len(buys) < 100:
        raise ValueError("Pares de rotacoes insuficientes.")
    bought = pd.to_datetime(buys["timestamp"], utc=True)
    decided = pd.to_datetime(buys["decision_timestamp"], utc=True)
    sells = trade.loc[
        trade["action"].isin(("SELL", "FINAL_SELL")),
        ["entry_timestamp", "timestamp", "asset"],
    ].copy()
    sells["_entry"] = pd.to_datetime(sells["entry_timestamp"], utc=True)
    sells["_end"] = pd.to_datetime(sells["timestamp"], utc=True)
    buys["_bought"] = bought
    buys["_decided"] = decided
    buys = buys.merge(
        sells[["_entry", "_end", "asset"]],
        left_on=["_bought", "asset"],
        right_on=["_entry", "asset"],
        validate="one_to_one", how="left",
    )
    if buys["_end"].isna().any():
        raise ValueError("Intervalo sem venda final para rotulo.")
    rotate = buys["chosen_market_return"].to_numpy(float)
    hold = buys["counterfactual_previous_asset_return"].to_numpy(float)
    if np.min(np.r_[rotate, hold]) <= -1:
        raise ValueError("Retorno de perda total ou inferior invalido.")
    if np.max(np.abs(
        buys["rotation_value_added"].to_numpy(float) - (rotate - hold)
    )) > 1e-8:
        raise ValueError("Vantagem de rotacao nao coincide com os retornos.")
    observed = np.column_stack((
        np.log1p(rotate), np.log1p(hold),
        np.full(len(buys), -ECO_CUSTO_CASH),
    ))
    buys["acao_sugerida"] = "NAO_AVALIADO"
    buys["log_retorno_esperado_rotate"] = np.nan
    buys["log_retorno_esperado_hold"] = np.nan
    buys["log_retorno_observado_rotate"] = observed[:, 0]
    buys["log_retorno_observado_hold"] = observed[:, 1]
    buys["log_retorno_observado_cash_teorico"] = observed[:, 2]
    buys["log_vantagem_acao_sugerida_vs_rotate"] = np.nan
    summary = []
    for fold in (2, 3):
        is_test = buys["walk_forward_fold"].eq(fold)
        start = buys.loc[is_test, "_decided"].min()
        is_train = (
            buys["walk_forward_fold"].lt(fold)
            & buys["_end"].lt(start - pd.Timedelta(days=ECO_EMBARGO_DIAS))
        )
        if int(is_train.sum()) < 40:
            raise ValueError(f"Treino pareado insuficiente no fold {fold}")
        model = _eco_modelo()
        model.fit(
            _eco_x(buys.loc[is_train]), observed[is_train.to_numpy(), :2],
        )
        forecast = model.predict(_eco_x(buys.loc[is_test]))
        chosen = np.argmax(np.column_stack((
            forecast,
            np.full(int(is_test.sum()), -ECO_CUSTO_CASH),
        )), axis=1)
        actual = observed[is_test.to_numpy()]
        advantages = actual[np.arange(len(chosen)), chosen] - actual[:, 0]
        buys.loc[is_test, "acao_sugerida"] = (
            np.array(["ROTATE", "HOLD", "CASH"])[chosen]
        )
        buys.loc[is_test, "log_retorno_esperado_rotate"] = forecast[:, 0]
        buys.loc[is_test, "log_retorno_esperado_hold"] = forecast[:, 1]
        buys.loc[is_test, "log_vantagem_acao_sugerida_vs_rotate"] = advantages
        summary.append({
            "fold": fold, "treino": int(is_train.sum()),
            "intervalos_avaliados": int(is_test.sum()),
            "ultima_saida_com_rotulo_treino": str(
                buys.loc[is_train, "_end"].max().date()
            ),
            "inicio_decisoes_teste": str(start.date()),
            "sugeriu_rotate": int((chosen == 0).sum()),
            "sugeriu_hold": int((chosen == 1).sum()),
            "sugeriu_cash": int((chosen == 2).sum()),
            "vantagem_media_original_rotate_vs_hold_log": float(
                (actual[:, 0] - actual[:, 1]).mean()
            ),
            "taxa_original_rotate_superior_hold": float(
                (actual[:, 0] > actual[:, 1]).mean()
            ),
            "vantagem_media_sugerida_vs_rotate_log_por_intervalo": float(
                advantages.mean()
            ),
            "soma_das_vantagens_log_locais_nao_capitalizavel": float(
                advantages.sum()
            ),
            "decisoes_sugeridas_melhores_que_original": int(
                (advantages > 0).sum()
            ),
            "decisoes_sugeridas_piores_que_original": int(
                (advantages < 0).sum()
            ),
        })
    cols = [
        "timestamp", "decision_timestamp", "walk_forward_fold", "asset",
        "rotation_from_asset", "rotation_to_asset",
        "subsequent_holding_days", "chosen_market_return",
        "counterfactual_previous_asset_return", "rotation_value_added",
        "acao_sugerida", "log_retorno_esperado_rotate",
        "log_retorno_esperado_hold", "log_retorno_observado_rotate",
        "log_retorno_observado_hold", "log_retorno_observado_cash_teorico",
        "log_vantagem_acao_sugerida_vs_rotate",
    ]
    return buys.loc[:, cols], pd.DataFrame(summary)



def _eco_diagnostico_previsao_diaria(
    cash: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refaz ajuste exato para medir treino, previsao temporal e deriva.

    Reajuste somente para auditoria: a politica CASH nao e alterada,
    e as previsoes reproduzidas devem coincidir com as ja exportadas.
    """
    dated = cash.copy().reset_index(drop=True)
    executions = pd.to_datetime(dated["timestamp"], utc=True)
    decisions = pd.to_datetime(dated["decision_date"], utc=True)
    stats, annual = [], []
    for fold in (2, 3):
        test_mask = dated["walk_forward_fold"].eq(fold)
        if not test_mask.any():
            raise ValueError(f"Fold sem decisoes: {fold}")
        start = decisions.loc[test_mask].min()
        train_mask = (
            dated["walk_forward_fold"].lt(fold)
            & executions.lt(start - pd.Timedelta(days=ECO_EMBARGO_DIAS))
        )
        train = dated.loc[train_mask]
        test = dated.loc[test_mask].copy()
        if len(train) < 100 or len(test) < 100:
            raise ValueError("Amostras insuficientes para diagnostico.")
        model = _eco_modelo()
        y_train = train["log_retorno_control_observado"].to_numpy(float)
        y_test = test["log_retorno_control_observado"].to_numpy(float)
        model.fit(_eco_x(train), y_train)
        estimate_train = model.predict(_eco_x(train))
        estimate_test = model.predict(_eco_x(test))
        known = test["log_retorno_esperado_control"].to_numpy(float)
        parity = float(np.max(np.abs(estimate_test - known)))
        if parity > 1e-10:
            raise ValueError(
                f"Previsao da auditoria diverge da avaliacao: {parity}"
            )
        constant = float(np.mean(y_train))
        train_mse = float(np.mean((y_train - estimate_train) ** 2))
        train_base_mse = float(np.mean((y_train - constant) ** 2))
        test_mse = float(np.mean((y_test - estimate_test) ** 2))
        test_base_mse = float(np.mean((y_test - constant) ** 2))
        if train_base_mse <= 0 or test_base_mse <= 0:
            raise ValueError("Variancia do retorno insuficiente.")
        correlation = float(np.corrcoef(y_test, estimate_test)[0, 1])
        marked = test["acao_shadow"].eq("CASH")
        errors = test.loc[marked, "log_retorno_control_observado"]
        stats.append({
            "fold": fold,
            "data_inicio_avaliacao": str(start.date()),
            "treino_observacoes": int(len(train)),
            "avaliacao_observacoes": int(len(test)),
            "media_retorno_log_treino": constant,
            "media_retorno_log_previsto_avaliacao": float(
                np.mean(estimate_test)
            ),
            "media_retorno_log_real_avaliacao": float(np.mean(y_test)),
            "vies_medio_previsao_menos_real": float(
                np.mean(estimate_test - y_test)
            ),
            "r2_treino": 1.0 - train_mse / train_base_mse,
            "r2_avaliacao_vs_media_do_treino": (
                1.0 - test_mse / test_base_mse
            ),
            "correlacao_previsao_observado_avaliacao": correlation,
            "erro_quadratico_medio_previsao": test_mse,
            "erro_quadratico_medio_media_do_treino": test_base_mse,
            "fracao_previsoes_negativas": float(
                np.mean(estimate_test < 0)
            ),
            "frac_recomendacoes_cash": float(marked.mean()),
            "dias_cash": int(marked.sum()),
            "dias_cash_que_eram_positivos": int(errors.gt(0).sum()),
            "soma_log_retorno_perdido_em_cash": float(errors.sum()),
            "soma_log_altas_nao_capturadas": float(
                errors.loc[errors.gt(0)].sum()
            ),
            "soma_log_perdas_evitadas": float(
                -errors.loc[errors.lt(0)].sum()
            ),
            "paridade_maxima_das_previsoes": parity,
            "analise": (
                "Avalia o retorno da carteira Control sob as decisoes "
                "originais; nao e trajetoria de caixa negociavel."
            ),
        })
        test["ano"] = pd.to_datetime(
            test["timestamp"], utc=True
        ).dt.year
        for year, block in test.groupby("ano", sort=True):
            hits = block.loc[block["acao_shadow"].eq("CASH")]
            annual.append({
                "fold": fold, "ano": int(year),
                "pregoes": int(len(block)),
                "dias_cash": int(len(hits)),
                "fracao_cash": float(len(hits) / len(block)),
                "retorno_log_medio_observado": float(
                    block["log_retorno_control_observado"].mean()
                ),
                "retorno_log_medio_previsto": float(
                    block["log_retorno_esperado_control"].mean()
                ),
                "soma_log_retorno_original_dias_cash": float(
                    hits["log_retorno_control_observado"].sum()
                ),
            })
    fired = dated.loc[
        dated["acao_shadow"].eq("CASH")
        & dated["walk_forward_fold"].isin((2, 3))
    ].copy()
    # Zero alertas e um resultado valido de uma politica conservadora.
    fired["log_altas_perdidas"] = fired[
        "log_retorno_control_observado"
    ].clip(lower=0)
    fired["log_quedas_protegidas"] = -fired[
        "log_retorno_control_observado"
    ].clip(upper=0)
    concentration = fired.groupby(
        ["walk_forward_fold", "selected_asset"], as_index=False,
    ).agg(
        dias_cash=("timestamp", "size"),
        dias_de_alta_perdidos=("log_retorno_control_observado",
                              lambda v: int(v.gt(0).sum())),
        soma_log_retorno_original=("log_retorno_control_observado", "sum"),
        soma_log_altas_perdidas=("log_altas_perdidas", "sum"),
        soma_log_quedas_protegidas=("log_quedas_protegidas", "sum"),
    ).sort_values(
        ["walk_forward_fold", "soma_log_retorno_original"],
        ascending=[True, False],
    )
    return pd.DataFrame(stats), pd.DataFrame(annual), concentration


def executar_vantagem_economica() -> None:
    """Uma unica saida: auditoria existente e comparacao economica."""
    from io import BytesIO

    executar_analise_permanencia()
    source = _a_localizar_replay()
    with ZipFile(source) as z:
        pred = next(n for n in z.namelist() if n.endswith("_predictions.csv"))
        trades = next(n for n in z.namelist() if n.endswith("_trades.csv"))
        metric = next(n for n in z.namelist() if n.endswith("_metrics.json"))
        original = pd.read_csv(BytesIO(z.read(pred)), low_memory=False)
        operations = pd.read_csv(BytesIO(z.read(trades)), low_memory=False)
        metrics = json.loads(z.read(metric))
    initial = float(metrics["initial_capital"])
    cash, cash_folds = _eco_cash_diario(original, initial)
    rotation, rot_folds = _eco_rotacoes_pareadas(operations)
    baseline = float(metrics["strategy_ending_capital"])
    if abs(float(cash["capital_control_reproduzido"].iloc[-1])-baseline) > .01:
        raise RuntimeError("Falhou a paridade do Control atualizado.")
    # O modo descritivo publicou em DESTINO e moveu DESTINO_TEMP; agora
    # adicionamos ao pacote anterior e republicamos atomicamente.
    dest = DESTINO
    cash.to_csv(dest/"vantagem_economica_cash_diario_shadow.csv", index=False)
    cash_folds.to_csv(dest/"vantagem_economica_cash_por_fold.csv", index=False)
    rotation.to_csv(dest/"vantagem_economica_rotacoes_pareadas.csv", index=False)
    rot_folds.to_csv(dest/"vantagem_economica_rotacoes_por_fold.csv", index=False)
    diagnostico, diagnostico_ano, impacto_ativos = (
        _eco_diagnostico_previsao_diaria(cash)
    )
    diagnostico.to_csv(
        dest/"vantagem_economica_generalizacao_diaria.csv", index=False,
    )
    diagnostico_ano.to_csv(
        dest/"vantagem_economica_cash_por_ano.csv", index=False,
    )
    impacto_ativos.to_csv(
        dest/"vantagem_economica_cash_por_ativo.csv", index=False,
    )
    errors_rotate = rotation.loc[
        rotation["walk_forward_fold"].isin((2, 3))
        & rotation["acao_sugerida"].ne("ROTATE")
    ].copy().sort_values("log_vantagem_acao_sugerida_vs_rotate")
    errors_rotate.to_csv(
        dest/"vantagem_economica_influencia_rotacoes.csv", index=False,
    )
    manifest_file = dest/"manifesto_experimento.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["estudo_vantagem_economica"] = {
        "capital_control": baseline,
        "capital_cash_shadow": float(cash["capital_shadow_cash"].iloc[-1]),
        "capital_cash_shadow_sem_custo": float(
            cash["capital_shadow_cash_sem_custo"].iloc[-1]
        ),
        "custo_cash_teorico": ECO_CUSTO_CASH,
        "cash_por_fold": cash_folds.to_dict(orient="records"),
        "generalizacao_da_previsao_diaria": diagnostico.to_dict(orient="records"),
        "medidas_de_deriva_por_ano": diagnostico_ano.to_dict(orient="records"),
        "rotacoes_por_fold": rot_folds.to_dict(orient="records"),
        "decisoes_contrafactuais_executaveis": False,
        "limitacoes": [
            "CASH de um dia e shadow close-to-close, nao operacao executada.",
            "0.2% por dia de CASH e custo hipotetico, sem gap de abertura.",
            "HOLD/ROTATE/CASH pareados sobre os intervalos originais.",
            "Nao sao possiveis trajetorias alternativas com dados atuais.",
            "Treino e rotulos anteriores ao embargo cronologico de 30 dias.",
            "Folds de avaliacao ja examinados na pesquisa, sem teste novo.",
        ],
    }
    _write_json(manifest_file, manifest)
    with (dest/"LEIA_ME.txt").open("a", encoding="utf-8") as stream:
        stream.write(
            "\nVANTAGEM ECONOMICA: CSVs vantagem_economica_*.\n"
            "CASH diario = overlay close-to-close com custo hipotetico.\n"
            "HOLD/ROTATE/CASH em rotacoes = comparacao por intervalo, "
            "NAO carteira contrafactual executavel.\n"
            "Diagnosticos de generalizacao, previsao por ano, ativo e rotacao.\n"
        )
    shutil.copyfile(Path(__file__), dest/"codigo_executado.py")
    (dest/"arquivos_sha256.json").unlink(missing_ok=True)
    if DESTINO_TEMP.exists():
        shutil.rmtree(DESTINO_TEMP)
    dest.replace(DESTINO_TEMP)
    archive = _a_publicar_sem_fonte(DESTINO_TEMP)
    print(
        "[ECONOMIA] Control USD", round(baseline, 2),
        "| CASH shadow USD", round(float(cash["capital_shadow_cash"].iloc[-1]), 2),
        "| rotacoes pareadas avaliadas", int(rot_folds["intervalos_avaliados"].sum()),
        "| ZIP", archive,
    )



# %% 15. Permanencia inteligente: variantes causais sobre a trava de 2 pregoes.
# Nao altera o classificador principal: compara apenas uso dinamico da trava.
# O backtest completo desta etapa usa os precos CONGELADOS do pacote
# de reproducao, e nao se confunde com o replay de producao atualizado.
PI_GAP_URGENTE = 0.045
PI_QUEDA_SCORE_URGENTE = -0.04
PI_GAP_URGENTE_MULT_MARGEM = 3.0
PI_CONFIRMACAO_GAP_FORTE = 0.05
PI_CONFIRMACAO_MULT_MARGEM = 3.0
PI_EXTENSAO_SCORE_MELHORA = 0.03
PI_EXTENSAO_MULT_MARGEM = 2.0


def _pi_decisao(
    base: Base, i: int, pos: int, holding: int, *,
    politica: str,
) -> tuple[int, str, dict]:
    """Regra por fechamento, sem cotacao de abertura futura no decisor."""
    original = _d_control(base, i, pos, holding)
    info = {
        "data_decisao": str(base.dates[i]),
        "fold": int(base.fold[i]),
        "holding": int(holding),
        "ativo_atual": base.symbols[pos] if pos >= 0 else "CASH",
        "controle": base.symbols[original] if original >= 0 else "CASH",
        "politica": politica,
        "motivo": "CONTROL",
    }
    if (
        pos < 0 or holding < 1 or i < holding
        or int(base.fold[i]) != int(base.fold[i - holding])
        or not np.isfinite(base.scores[i, pos])
    ):
        return original, "CONTROL", info
    util = base.scores[i]
    candidate = int(np.nanargmax(util))
    current_score = float(util[pos])
    best_score = float(util[candidate])
    gap = best_score - current_score
    margin = max(
        float(base.config["rotation_switch_margin"]),
        float(base.margens[int(base.fold[i])]),
    )
    change = current_score - float(base.scores[i - holding, pos])
    info.update({
        "melhor_ativo": base.symbols[candidate],
        "gap_utilidade": gap,
        "mudanca_score_desde_entrada": change,
        "margem_efetiva": margin,
    })
    if politica not in {
        "saida_antecipada", "confirmacao", "confirmacao_mais_saida",
        "extensao_um_dia",
    }:
        raise ValueError(f"Politica desconhecida: {politica}")
    target = original
    if politica in {"saida_antecipada", "confirmacao_mais_saida"}:
        if (
            holding == 1 and candidate != pos
            and np.isfinite(change)
            and gap >= max(
                PI_GAP_URGENTE,
                PI_GAP_URGENTE_MULT_MARGEM * margin,
            )
            and change <= PI_QUEDA_SCORE_URGENTE
            and best_score > float(base.config["rotation_cash_threshold"])
        ):
            target = candidate
            info["motivo"] = "SAIDA_ANTE_DOS_DOIS_PREGOES"
    if politica in {"confirmacao", "confirmacao_mais_saida"}:
        if (
            holding >= 2 and original == candidate
            and candidate != pos
            and gap < max(
                PI_CONFIRMACAO_GAP_FORTE,
                PI_CONFIRMACAO_MULT_MARGEM * margin,
            )
        ):
            # A vantagem precisa existir TAMBEM no fechamento anterior,
            # ou o decisor espera sem prazo fixo pelo proximo sinal.
            yesterday = float(
                base.scores[i - 1, candidate] - base.scores[i - 1, pos]
            )
            if not np.isfinite(yesterday) or yesterday <= 0:
                target = pos
                info["motivo"] = "ESPERAR_CONFIRMACAO_DA_VANTAGEM"
    if politica == "extensao_um_dia":
        if (
            holding == 2 and original == candidate and candidate != pos
            and gap < max(0.03, PI_EXTENSAO_MULT_MARGEM * margin)
            and change > PI_EXTENSAO_SCORE_MELHORA
            and current_score > float(base.config["rotation_cash_threshold"])
        ):
            target = pos
            info["motivo"] = "MAIS_UM_PREGAO_POR_FORCA_RELATIVA"
    info["decisao"] = base.symbols[target] if target >= 0 else "CASH"
    return target, info["motivo"], info


def _pi_replay(base: Base, politica: str):
    rows = []

    def decide(b, i, pos, holding, capital, quantity):
        target, motivo, info = _pi_decisao(
            b, i, pos, holding, politica=politica,
        )
        if motivo != "CONTROL":
            rows.append(info)
        return target, []

    curve, decisions, trades, summary = simular(
        base, decision_callback=decide, permitir_cash=False,
    )
    return curve, decisions, trades, summary, pd.DataFrame(rows)


def _pi_validar_zip_base(arquivo: Path) -> bool:
    """Evita confundir um ZIP de relatorio com a fonte de precos e utilidades."""
    required = (
        "pesquisa_permanencia_minima/manifesto_experimento.json",
        "pesquisa_permanencia_minima/decisoes_2_pregoes.csv",
        "pesquisa_permanencia_minima/patrimonio_diario_2_pregoes.csv",
        "pesquisa_permanencia_minima/utilidades_oos_todos_ativos.csv",
        "pesquisa_permanencia_minima/margens_controle_por_fold.csv",
    )
    try:
        with ZipFile(arquivo) as z:
            files = set(z.namelist())
            return (
                all(name in files for name in required)
                and sum(
                    name.startswith("pesquisa_permanencia_minima/cotacoes_ohlcv/")
                    and name.endswith(".csv")
                    for name in files
                ) == 75
            )
    except (OSError, BadZipFile, ValueError):
        return False


def _pi_pastas_de_busca() -> list[Path]:
    """Raiz do projeto, saidas anteriores e downloads, independente do Spyder."""
    dirs = [
        ROOT / "output", ROOT, ROOT.parent / "output", ROOT.parent,
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path.home() / "Desktop",
    ]
    # Evita varrer todo o disco ou confundir ZIPs de projetos diferentes.
    return list(dict.fromkeys(p for p in dirs if p.is_dir()))


def _pi_recuperar_da_pasta_completa(target: Path) -> Path | None:
    """Reempacota o resultado concluido, SEM recalcular preco ou treinar.

    O proprio executor original ja salva os CSVs em
    output/pesquisa_permanencia_minima e seus hashes. Caso so o ZIP
    tenha sido apagado, esse diretorio e fonte suficiente e verificavel.
    """
    required = (
        "manifesto_experimento.json", "decisoes_2_pregoes.csv",
        "patrimonio_diario_2_pregoes.csv",
        "utilidades_oos_todos_ativos.csv", "margens_controle_por_fold.csv",
        "arquivos_sha256.json",
    )
    for parent in _pi_pastas_de_busca():
        folder = parent / "pesquisa_permanencia_minima"
        if not folder.is_dir() or not all(
            (folder / name).is_file() for name in required
        ):
            continue
        bars = sorted((folder / "cotacoes_ohlcv").glob("*.csv"))
        if len(bars) != 75:
            continue
        try:
            metadata = json.loads(
                (folder / "manifesto_experimento.json").read_text(encoding="utf-8")
            )
            if metadata.get("status") != "completed" or metadata.get("folds") != 3:
                continue
            recorded = json.loads(
                (folder / "arquivos_sha256.json").read_text(encoding="utf-8")
            )["arquivos"]
            if not recorded or len(recorded) < 80:
                continue
            if any(
                not (folder / name).is_file()
                or (folder / name).stat().st_size != record["bytes"]
                or _hash_arquivo(folder / name) != record["sha256"]
                for name, record in recorded.items()
            ):
                continue
            # Nao incluir arquivos extras de pesquisas posteriores no baseline.
            members = [folder / name for name in sorted(recorded)]
            members.append(folder / "arquivos_sha256.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = target.with_name(".pesquisa_permanencia_minima_recuperacao.tmp")
            try:
                with ZipFile(stage, "w", ZIP_DEFLATED, compresslevel=6) as archive:
                    for member in members:
                        archive.write(
                            member,
                            arcname="pesquisa_permanencia_minima/"
                                    + member.relative_to(folder).as_posix(),
                        )
                if not _pi_validar_zip_base(stage):
                    continue
                with ZipFile(stage) as archive:
                    if archive.testzip() is not None:
                        continue
                stage.replace(target)
                print(
                    "[DADOS] Base completa recuperada da pasta de saida",
                    folder, "| sha256", _hash_arquivo(target), flush=True,
                )
                return target
            finally:
                stage.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, TypeError, BadZipFile):
            continue
    return None


def _pi_localizar_base() -> Path | None:
    """Recupera a fonte ORIGINAL de ZIPs anteriores sem novos downloads.

    Relatorios antigos podem carregar a base completa em
    dados_origem/pesquisa_permanencia_minima.zip. Nesse caso, extrai-se
    somente esse membro, validando CRC, SHA e contrato antes de publicar.
    """
    target = ARQUIVO_ZIP
    direct = []
    for folder in _pi_pastas_de_busca():
        direct.extend(folder.glob("pesquisa_permanencia_minima*.zip"))
    for candidate in sorted(
        set(direct), key=lambda x: (x != target, -x.stat().st_mtime)
    ):
        if _pi_validar_zip_base(candidate):
            print(f"[DADOS] Base congelada localizada: {candidate}", flush=True)
            return candidate

    from_folder = _pi_recuperar_da_pasta_completa(target)
    if from_folder is not None:
        return from_folder

    containers = []
    for folder in _pi_pastas_de_busca():
        containers.extend(folder.glob("pesquisa_vantagem_rotacao*.zip"))
    # O ZIP atual e preferido, mas qualquer historico da mesma pesquisa
    # que contenha a fonte valida serve como recuperacao.
    embedded_member = "dados_origem/pesquisa_permanencia_minima.zip"
    for archive in sorted(
        set(containers), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            with ZipFile(archive) as z:
                if embedded_member not in z.namelist():
                    continue
                data = z.read(embedded_member)  # valida CRC do membro
        except (OSError, ValueError, BadZipFile, RuntimeError):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.with_name(".pesquisa_permanencia_minima_recuperacao.tmp")
        try:
            stage.write_bytes(data)
            if not _pi_validar_zip_base(stage):
                continue
            # A validacao completa do ZIP interno ocorre antes de
            # publicar a recuperacao com o nome canonico.
            with ZipFile(stage) as inner:
                if inner.testzip() is not None:
                    continue
                manifest = json.loads(inner.read(
                    "pesquisa_permanencia_minima/manifesto_experimento.json"
                ))
                if manifest.get("status") != "completed" or manifest.get("folds") != 3:
                    continue
            digest = _hash_arquivo(stage)
            stage.replace(target)
            print(
                "[DADOS] Base historica recuperada de",
                archive.name, "| sha256", digest,
                flush=True,
            )
            return target
        except (OSError, BadZipFile, ValueError, KeyError):
            continue
        finally:
            stage.unlink(missing_ok=True)
    return None


def _pi_resumo_retorno_observado(first: pd.DataFrame) -> pd.DataFrame:
    """Retorno realizado do proprio Control; NAO retorno de ativo concorrente.

    A trava e avaliada com sinais do fechamento anterior. O retorno
    observado no pregão de execucao so aparece depois, no resultado.
    A comparacao e descritiva, sem otimizacao dos limiares.
    """
    required = {
        "walk_forward_fold", "outro_ativo_melhor",
        "candidato_saida_antecipada", "retorno_control_no_pregao",
    }
    missing = required.difference(first.columns)
    if missing:
        raise ValueError(f"Campos de auditoria ausentes: {sorted(missing)}")
    groups = {
        "todas_as_travas": first.index == first.index,
        "outro_ativo_no_topo": first["outro_ativo_melhor"].eq(True),
        "excecao_proposta": first["candidato_saida_antecipada"].eq(True),
        "outro_no_topo_sem_excecao": (
            first["outro_ativo_melhor"].eq(True)
            & first["candidato_saida_antecipada"].eq(False)
        ),
        "sem_excecao": first["candidato_saida_antecipada"].eq(False),
    }
    rows = []
    for fold in ("todos", 1, 2, 3):
        fold_mask = (
            pd.Series(True, index=first.index) if fold == "todos"
            else first["walk_forward_fold"].eq(fold)
        )
        for label, mask in groups.items():
            subset = first.loc[fold_mask & mask]
            returns = subset["retorno_control_no_pregao"].dropna()
            if returns.empty:
                continue
            rows.append({
                "fold": fold, "grupo": label,
                "observacoes": int(len(returns)),
                "pregoes_positivos": int(returns.gt(0).sum()),
                "pregoes_negativos": int(returns.lt(0).sum()),
                "pregoes_com_perda_5pct": int(returns.le(-0.05).sum()),
                "pregoes_com_alta_10pct": int(returns.ge(0.10).sum()),
                "media_retorno_control": float(returns.mean()),
                "mediana_retorno_control": float(returns.median()),
                "menor_retorno_control": float(returns.min()),
                "maior_retorno_control": float(returns.max()),
                "limitacao": (
                    "Retorno realizado da posicao mantida no Control; "
                    "nao compara retorno do candidato nem a carteira "
                    "que resultaria de uma rotacao antecipada."
                ),
            })
    return pd.DataFrame(rows)


def _pi_diagnostico_replay_atual(source: Path) -> tuple[pd.DataFrame, dict]:
    """Identifica candidatos a liberar a trava, SEM criar ordens ficticias."""
    from io import BytesIO

    with ZipFile(source) as z:
        if z.testzip() is not None:
            raise ValueError("Replay atualizado corrompido.")
        names = [n for n in z.namelist() if n.endswith("_predictions.csv")]
        if len(names) != 1:
            raise ValueError("Replay com arquivos de decisoes ambiguos.")
        frame = pd.read_csv(BytesIO(z.read(names[0])), low_memory=False)
    needed = {
        "decision_date", "timestamp", "walk_forward_fold",
        "holding_days_at_decision", "min_hold_guard_applied",
        "raw_best_asset", "current_asset", "raw_best_score",
        "best_vs_current_gap", "effective_switch_margin",
        "score_change_from_entry", "rotation_cash_threshold",
        "strategy_equity", "decision_is_rotation",
    }
    if not needed.issubset(frame):
        raise ValueError("Replay atualizado sem diagnosticos da trava.")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
        raise ValueError("Replay sem calendario de execucao ordenado e unico.")
    equity = pd.to_numeric(frame["strategy_equity"], errors="raise")
    if not np.isfinite(equity).all() or equity.le(0).any():
        raise ValueError("Patrimonio invalido para diagnostico da trava.")
    frame["retorno_control_no_pregao"] = equity.pct_change()
    first = frame.loc[frame["min_hold_guard_applied"].eq(True)].copy()
    if not first["holding_days_at_decision"].eq(1).all():
        raise ValueError("A trava aplicada nao corresponde ao primeiro pregao.")
    margin = first["effective_switch_margin"].astype(float)
    first["outro_ativo_melhor"] = (
        first["raw_best_asset"].ne(first["current_asset"])
    )
    first["candidato_saida_antecipada"] = (
        first["outro_ativo_melhor"]
        & first["best_vs_current_gap"].ge(np.maximum(
            PI_GAP_URGENTE, PI_GAP_URGENTE_MULT_MARGEM * margin,
        ))
        & first["score_change_from_entry"].le(PI_QUEDA_SCORE_URGENTE)
        & first["raw_best_score"].gt(first["rotation_cash_threshold"])
    )
    first["analise_sem_execucao_de_ordem"] = True
    comparative = _pi_resumo_retorno_observado(first)
    summary = {
        "fonte_sha256": _hash_arquivo(source),
        "referencia_capital_final_observado": float(
            frame["strategy_equity"].iloc[-1]
        ),
        "sessoes": int(len(frame)),
        "travas_primeiro_pregao": int(len(first)),
        "travas_com_outro_ativo_no_topo": int(
            first["outro_ativo_melhor"].sum()
        ),
        "possiveis_excecoes_primeiro_pregao": int(
            first["candidato_saida_antecipada"].sum()
        ),
        "retornos_observados_da_trava": comparative.to_dict(orient="records"),
        "possiveis_excecoes_por_fold": {
            str(fold): int(group["candidato_saida_antecipada"].sum())
            for fold, group in first.groupby("walk_forward_fold")
        },
        "alerta": (
            "Sinais de possivel intervencao somente. Sem cenario "
            "alternativo de patrimonio: faltam todas as series de precos "
            "e as posicoes futuras mudariam apos a primeira troca."
        ),
    }
    return first, summary



def _pi_somente_diagnostico_atual() -> None:
    """Nao interrompe F5 quando so existe o replay completo atualizado.

    Deixa registrado o que e possivel avaliar sem reconstruir ordens
    alternativas. Nunca finge que o patrimonio alternativo foi simulado.
    """
    try:
        replay = _a_localizar_replay()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Nenhuma fonte historica completa foi encontrada. Foram "
            "verificados os ZIPs de pesquisa ja existentes no projeto e "
            "em Downloads, inclusive a base dentro de relatorios anteriores. "
            "Tambem nao foi encontrado um backtest atualizado para diagnostico. "
            "Nao e possivel reconstruir as rotacoes sem os precos, utilidades "
            "e calendario originais; nenhum resultado foi substituido."
        ) from exc
    first, summary = _pi_diagnostico_replay_atual(replay)
    dest = DESTINO_TEMP
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    first.to_csv(dest / "trava_primeiro_pregao_replay_atual.csv", index=False)
    _pi_resumo_retorno_observado(first).to_csv(
        dest / "retornos_observados_trava_por_fold.csv", index=False,
    )
    manifest = {
        "modo": "permanencia_inteligente",
        "status": "completed_observational_diagnostic_only",
        "simulacao_contrafactual_executada": False,
        "motivo": (
            "A serie historica completa com precos e utilidades nao foi "
            "encontrada localmente nem em ZIPs de saida anteriores."
        ),
        "replay_atual_somente_diagnostico": summary,
        "dados_ausentes": (
            "pesquisa_permanencia_minima.zip original ou relatorio "
            "anterior com dados_origem/pesquisa_permanencia_minima.zip"
        ),
        "limitacoes": [
            "Nenhuma operacao alternativa foi reconstruida.",
            "Nenhum valor de capital contrafactual foi estimado.",
            "A estrategia original permanece inalterada.",
            "Retornos observados sao do Control, nao dos concorrentes.",
        ],
    }
    _write_json(dest / "manifesto_experimento.json", manifest)
    (dest / "LEIA_ME.txt").write_text(
        "Auditoria da trava de dois pregoes com o replay atualizado.\\n"
        "A base completa nao foi localizada, portanto NAO se simulou "
        "qualquer carteira alternativa.\\n"
        "Retornos observados da trava NAO sao lucros de rotacoes alternativas.\\n",
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__), dest / "codigo_executado.py")
    archive = _a_publicar_sem_fonte(dest)
    print(
        "[PERMANENCIA] Diagnostico do replay atual gerado, SEM "
        "simulacao de novas operacoes. | ZIP", archive, flush=True,
    )


def executar_permanencia_inteligente() -> None:
    """Compara politica estatica e adaptativas sem alterar operacao real."""
    source = _pi_localizar_base()
    if source is None:
        _pi_somente_diagnostico_atual()
        return
    base = carregar(source)
    reference, _, _, initial = simular(base)
    error = float(np.max(np.abs(reference - base.base_equity)))
    if error > TOLERANCIA_CENTAVO:
        raise RuntimeError(
            f"Paridade do replay historico falhou: USD {error:.8f}."
        )
    dest = DESTINO_TEMP
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    curves = pd.DataFrame({
        "data_execucao": base.exec_dates,
        "fold": base.fold,
        "control_2_pregoes": reference,
    })
    summary = [{
        "politica": "control_2_pregoes",
        "capital_final": float(reference[-1]),
        "capital_relativo_control_pct": 0.0,
        "rotacoes": int(initial["rotacoes"]),
        "taxas": float(initial["total_taxas"]),
        **_resumo_curva(reference),
    }]
    per_fold = [
        {"politica": "control_2_pregoes", **r}
        for r in _performance_by_fold(base, reference)
    ]
    intervention_details = []
    for policy in (
        "saida_antecipada", "confirmacao", "confirmacao_mais_saida",
        "extensao_um_dia",
    ):
        curve, decisions, trades, metrics, events = _pi_replay(base, policy)
        curves[policy] = curve
        fold_metrics = _performance_by_fold(base, curve)
        per_fold.extend(
            [{"politica": policy, **r} for r in fold_metrics]
        )
        summary.append({
            "politica": policy,
            "capital_final": float(curve[-1]),
            "capital_relativo_control_pct": float(
                100.0 * (curve[-1] / reference[-1] - 1.0)
            ),
            "rotacoes": int(metrics["rotacoes"]),
            "taxas": float(metrics["total_taxas"]),
            "intervencoes_total": int(len(events)),
            **_resumo_curva(curve),
        })
        if not events.empty:
            intervention_details.append(events)
        decisions.to_csv(
            dest / f"decisoes_{policy}.csv", index=False,
        )
        trades.to_csv(
            dest / f"operacoes_{policy}.csv", index=False,
        )
    curves.to_csv(dest / "comparacao_patrimonio_diario.csv", index=False)
    pd.DataFrame(summary).to_csv(
        dest / "comparacao_resultados.csv", index=False,
    )
    pd.DataFrame(per_fold).to_csv(
        dest / "comparacao_por_fold.csv", index=False,
    )
    if intervention_details:
        pd.concat(intervention_details, ignore_index=True).to_csv(
            dest / "intervencoes_adaptativas.csv", index=False,
        )
    current_summary = None
    try:
        replay = _a_localizar_replay()
    except FileNotFoundError:
        replay = None
    if replay is not None:
        first, current_summary = _pi_diagnostico_replay_atual(replay)
        first.to_csv(
            dest / "trava_primeiro_pregao_replay_atual.csv", index=False,
        )
        _pi_resumo_retorno_observado(first).to_csv(
            dest / "retornos_observados_trava_por_fold.csv", index=False,
        )
    manifest = {
        "modo": "permanencia_inteligente",
        "status": "completed_exploratory_research_only",
        "fonte_historica_sha256": _hash_arquivo(source),
        "capital_control_historico": float(reference[-1]),
        "paridade_historica_maxima_usd": error,
        "replay_atual_somente_diagnostico": current_summary,
        "periodos_validos_politica_adaptativa": [2, 3],
        "fold_1": "Control preservado; nenhum efeito de politica alternativa",
        "regra": {
            "primeiro_pregao": (
                "Aceita alternativa somente com gap muito expressivo "
                "e utilidade da posicao em deterioracao."
            ),
            "apos_dois_pregoes": (
                "Se o gap nao e forte, aguarda confirmacao da vantagem "
                "no fechamento anterior."
            ),
            "terceiro_pregao": (
                "Teste isolado de mais um dia quando score da posicao "
                "melhora e concorrente tem vantagem pequena."
            ),
            "gap_urgente": PI_GAP_URGENTE,
            "queda_score_urgente": PI_QUEDA_SCORE_URGENTE,
            "multiplicador_margem_urgente": PI_GAP_URGENTE_MULT_MARGEM,
            "confirmacao_gap_forte": PI_CONFIRMACAO_GAP_FORTE,
            "confirmacao_multiplicador_margem": PI_CONFIRMACAO_MULT_MARGEM,
            "extensao_melhoria_score": PI_EXTENSAO_SCORE_MELHORA,
        },
        "cenarios": summary,
        "limitacoes": [
            "Control historico de cotacoes congeladas NAO e o replay atualizado.",
            "Resultados testados em periodos ja examinados na pesquisa.",
            "Limiar fixado antes deste replay, sem calibracao prospectiva.",
            "OHLCV historico permite contabilizar ordens hipoteticas; "
            "nao reproduz negociacao real/impacto de mercado.",
            "No replay atualizado so existe diagnostico da trava, "
            "nao patrimonio contrafactual com troca de ativo.",
            "Nenhuma mudanca no motor ou na operacao de contas reais.",
        ],
    }
    _write_json(dest / "manifesto_experimento.json", manifest)
    (dest / "LEIA_ME.txt").write_text(
        "Permanencia minima adaptativa: analise sobre sinais de utilidade.\n"
        "O replay completo das politicas usa base historica CONGELADA.\n"
        "O replay atualizado e auditado, NAO reexecutado com novas ordens.\n"
        "Fold1 permanece Control. Folds2-3 permitem intervencoes.\n"
        "Nenhum resultado valida implementacao na operacao real.\n",
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__), dest / "codigo_executado.py")
    archive = _pacote(dest, source)
    _publicar_pasta_vantagem(dest)
    print(
        f"[PERMANENCIA] Control historico USD {float(reference[-1]):,.2f}",
        "| experimentos", len(summary)-1,
        "| ZIP", archive,
        flush=True,
    )
    if current_summary:
        print(
            "[REPLAY ATUAL] Travas", current_summary["travas_primeiro_pregao"],
            "| sinais de excecao sem execucao",
            current_summary["possiveis_excecoes_primeiro_pregao"],
            flush=True,
        )


if __name__ == "__main__":
    if MODO == "permanencia_inteligente":
        executar_permanencia_inteligente()
    elif MODO == "vantagem_economica":
        executar_vantagem_economica()
    elif MODO == "analisar_permanencia":
        executar_analise_permanencia()
    elif MODO == "sensibilidade_margens":
        executar_sensibilidade_margens()
    elif MODO == "deterioracao":
        executar_deterioracao()
    elif MODO == "vantagem_v3":
        executar_vantagem_v3()
    elif MODO == "vantagem_v2":
        executar_vantagem_v2()
    elif MODO == "vantagem":
        executar_vantagem()
    else:
        main()
