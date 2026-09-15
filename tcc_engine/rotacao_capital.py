"""Orquestração do experimento de rotação de capital do TCC."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .caracteristicas import preparar_painel_rotacao
from .modelo import treinar_modelos_lightgbm
from .politica_rotacao import combinar_politicas_por_janela, criar_politica_rotacao
from .simulacao import ResultadoRotacao, avaliar_politica_calibracao, desempenho_por_janela, simular_rotacao
from .validacao_temporal import construir_janelas_walk_forward, datas_decisao_analise, mapear_datas_para_janelas


def executar_modelos_rotacao(
    barras_por_ativo: dict[str, pd.DataFrame],
    configuracao: Any,
    calcular_taxas: Callable,
    aplicar_deslizamento: Callable,
    progresso: Callable[[float, str, int], None] | None = None,
) -> list[ResultadoRotacao]:
    """Executa LightGBM, validação walk-forward, calibração e simulação."""
    quadros, datas_comuns = preparar_painel_rotacao(barras_por_ativo, configuracao)
    ativos = sorted(quadros)
    janelas = construir_janelas_walk_forward(datas_comuns, configuracao)
    datas_analise = datas_decisao_analise(datas_comuns, janelas)
    mapa_data_janela, metadados = mapear_datas_para_janelas(janelas)

    resultados: list[ResultadoRotacao] = []
    repeticoes = int(configuracao.repeticoes)
    for repeticao in range(repeticoes):
        semente = int(configuracao.semente_aleatoria) + repeticao * int(configuracao.passo_semente)
        config_repeticao = configuracao.copiar(semente_aleatoria=semente)
        politicas: dict[int, Callable] = {}
        diagnosticos: dict[pd.Timestamp, dict[str, Any]] = {}
        detalhes_margem: list[dict[str, Any]] = []

        for posicao_janela, janela in enumerate(janelas, start=1):
            janela_id = int(janela["janela_id"])
            datas_treinamento = datas_comuns[: int(janela["fim_treinamento_indice"])]
            datas_calibracao = datas_comuns[
                int(janela["inicio_calibracao_indice"]): int(janela["fim_calibracao_indice"])
            ]
            datas_ajuste_final = datas_comuns[: int(janela["fim_ajuste_final_indice"])]

            if progresso is not None:
                fracao = (posicao_janela - 1) / max(1, len(janelas))
                progresso(
                    20.0 + 60.0 * fracao,
                    f"Janela {posicao_janela}/{len(janelas)} — treinamento de calibração",
                    repeticao,
                )

            modelos_calibracao = treinar_modelos_lightgbm(
                quadros,
                ativos,
                datas_treinamento,
                config_repeticao,
            )

            melhor_candidata = float(config_repeticao.candidatas_margem_troca[0])
            melhor_pontuacao = float("-inf")
            for candidata in config_repeticao.candidatas_margem_troca:
                politica_calibracao = criar_politica_rotacao(
                    modelos_calibracao,
                    quadros,
                    ativos,
                    config_repeticao,
                    float(candidata),
                )
                pontuacao = avaliar_politica_calibracao(
                    politica_calibracao,
                    quadros,
                    ativos,
                    datas_calibracao,
                    config_repeticao,
                )
                if pontuacao > melhor_pontuacao:
                    melhor_pontuacao = float(pontuacao)
                    melhor_candidata = float(candidata)

            modelos_finais = treinar_modelos_lightgbm(
                quadros,
                ativos,
                datas_ajuste_final,
                config_repeticao,
            )
            margem_efetiva = max(float(config_repeticao.margem_troca), melhor_candidata)
            politicas[janela_id] = criar_politica_rotacao(
                modelos_finais,
                quadros,
                ativos,
                config_repeticao,
                margem_efetiva,
                diagnosticos=diagnosticos,
                janela_id=janela_id,
            )
            detalhes_margem.append(
                {
                    "fold_id": janela_id,
                    "calibrated_candidate_margin": melhor_candidata,
                    "effective_switch_margin": margem_efetiva,
                    "calibration_risk_adjusted_score": melhor_pontuacao,
                }
            )

        politica_programada = combinar_politicas_por_janela(politicas, mapa_data_janela)
        resultado = simular_rotacao(
            politica_programada,
            quadros,
            ativos,
            datas_analise,
            config_repeticao,
            calcular_taxas,
            aplicar_deslizamento,
            metadados_decisao=metadados,
            diagnosticos_decisao=diagnosticos,
        )

        resultado.metrics.update(
            {
                "random_seed": semente,
                "repetition_index": repeticao + 1,
                "repetition_count": repeticoes,
                "walk_forward_fold_count": len(janelas),
                "walk_forward_folds": desempenho_por_janela(
                    resultado.predictions,
                    janelas,
                    float(config_repeticao.capital_inicial),
                ),
                "effective_switch_margin": float(
                    np.mean([d["effective_switch_margin"] for d in detalhes_margem])
                ),
                "calibrated_switch_margin": float(
                    np.mean([d["calibrated_candidate_margin"] for d in detalhes_margem])
                ),
                "deterministic_execution": bool(config_repeticao.execucao_deterministica),
                "numeric_thread_limit": int(config_repeticao.limite_threads_numericas),
                "decision_diagnostics_rows": len(diagnosticos),
                "model_family": "lightgbm_utility",
            }
        )
        detalhes_por_janela = {d["fold_id"]: d for d in detalhes_margem}
        for linha in resultado.metrics["walk_forward_folds"]:
            linha.update(detalhes_por_janela.get(linha["fold_id"], {}))
        resultados.append(resultado)

    resultados.sort(key=lambda item: int(item.metrics.get("repetition_index", 1)))
    return resultados
