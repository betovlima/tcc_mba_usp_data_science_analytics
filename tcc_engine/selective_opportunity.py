from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

MODO_ROTACAO_SELETIVA = "COMPOUND_ROTATION_SWING_SELECTIVE"
MODO_FILTRO_CAIXA_OPORTUNIDADE = "COMPOUND_ROTATION_SWING_OPPORTUNITY_CASH_GATE"
MODO_ALOCACAO_OTIMIZADA = "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION"
MODO_ALOCACAO_CONCENTRADA = "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION"

SESSOES_ATUALIZACAO_FILTRO_CAIXA_V2 = 21
JANELA_AMOSTRAS_FILTRO_CAIXA_V2 = 252
MINIMO_LINHAS_VALIDACAO_FILTRO_CAIXA_V2 = 60
TAXA_MINIMA_EXPOSICAO_MERCADO_FILTRO_CAIXA_V2 = 0.60

CARACTERISTICAS_OPORTUNIDADE = (
    "best_score",
    "second_score",
    "best_vs_second_gap",
    "universe_score_mean",
    "universe_score_std",
    "best_score_zscore",
    "positive_score_fraction",
    "best_return_5",
    "best_return_20",
    "best_return_60",
    "best_vol_20",
    "best_vol_60",
    "best_trend_efficiency_20",
    "best_trend_efficiency_60",
    "universe_breadth_5",
    "universe_breadth_20",
)


class AvaliacaoOportunidade(NamedTuple):
    probability: float
    confidence: float
    accepted: bool
    features: dict[str, float]
    best_position: int


@dataclass(frozen=True)
class FiltroOportunidadeSeletiva:
    model: Any | None
    threshold: float
    constant_probability: float | None
    training_rows: int
    positive_rate: float
    threshold_validation_rows: int
    threshold_validation_score: float
    reference_probabilities: tuple[float, ...] = ()
    threshold_validation_accepted: int = 0
    calibration_method: str = "prequential_relative_confidence_v2"
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    threshold_validation_transitions: int = 0
    threshold_basis: str = "relative_confidence"
    target_basis: str = "weighted_forward_net_log_return"
    target_horizon_sessions: int | None = None
    regularized_to_base_policy: bool = False
    threshold_validation_alpha: float | None = None
    threshold_validation_exposure_ratio: float | None = None

    @property
    def limite(self) -> float:
        return float(self.threshold)

    @property
    def limite_entrada(self) -> float | None:
        return self.entry_threshold

    @property
    def limite_saida(self) -> float | None:
        return self.exit_threshold

    @property
    def base_limite(self) -> str:
        return str(self.threshold_basis)

    @property
    def metodo_calibracao(self) -> str:
        return str(self.calibration_method)

    @property
    def linhas_treinamento(self) -> int:
        return int(self.training_rows)

    @property
    def taxa_positiva(self) -> float:
        return float(self.positive_rate)

    @property
    def linhas_validacao_limite(self) -> int:
        return int(self.threshold_validation_rows)

    @property
    def pontuacao_validacao_limite(self) -> float:
        return float(self.threshold_validation_score)

    @property
    def aceitos_validacao_limite(self) -> int:
        return int(self.threshold_validation_accepted)

    @property
    def transicoes_validacao_limite(self) -> int:
        return int(self.threshold_validation_transitions)

    @property
    def base_alvo(self) -> str:
        return str(self.target_basis)

    @property
    def horizonte_alvo_sessoes(self) -> int | None:
        return self.target_horizon_sessions

    @property
    def regularizado_politica_base(self) -> bool:
        return bool(self.regularized_to_base_policy)

    @property
    def alpha_validacao_limite(self) -> float | None:
        return self.threshold_validation_alpha

    @property
    def taxa_exposicao_validacao_limite(self) -> float | None:
        return self.threshold_validation_exposure_ratio

    def limite_ativo(self, posicao_atual: int | None = None) -> float:
        if (
            self.entry_threshold is None
            or self.exit_threshold is None
            or posicao_atual is None
        ):
            return float(self.threshold)
        return float(
            self.exit_threshold if int(posicao_atual) > 0 else self.entry_threshold
        )

    def valor_decisao(self, probabilidade: float, confianca: float) -> float:
        if self.threshold_basis == "absolute_probability":
            return min(1.0, max(0.0, float(probabilidade)))
        return min(1.0, max(0.0, float(confianca)))

    def aceita(
        self,
        probabilidade: float,
        confianca: float,
        posicao_atual: int | None = None,
    ) -> bool:
        return bool(
            self.valor_decisao(probabilidade, confianca)
            >= self.limite_ativo(posicao_atual)
        )

    def probabilidade(self, valores: dict[str, float]) -> float:
        vetor = pd.DataFrame(
            [[float(valores[nome]) for nome in CARACTERISTICAS_OPORTUNIDADE]],
            columns=list(CARACTERISTICAS_OPORTUNIDADE),
            dtype=float,
        )
        if not np.isfinite(vetor.to_numpy(dtype=float)).all():
            return 0.0
        if self.model is None:
            return float(self.constant_probability or 0.0)
        probabilidade = float(self.model.predict_proba(vetor)[0, 1])
        return min(1.0, max(0.0, probabilidade))

    def confianca_da_probabilidade(self, probabilidade: float) -> float:
        valor = min(1.0, max(0.0, float(probabilidade)))
        if self.model is None or not self.reference_probabilities:
            return valor
        referencia = np.asarray(self.reference_probabilities, dtype=float)
        posto = int(np.searchsorted(referencia, valor, side="right"))
        return min(1.0, max(0.0, float(posto / len(referencia))))

    def confianca(self, valores: dict[str, float]) -> float:
        return self.confianca_da_probabilidade(self.probabilidade(valores))


@dataclass
class FiltroCaixaOportunidadeAdaptativo:
    """Filtro CAIXA adaptativo sem antecipação para a política base protegida.

    O modelo de utilidade/ranking permanece congelado em cada janela walk-forward.
    Apenas este pequeno classificador logístico é atualizado quando os resultados
    de uma sessão da política base ficam disponíveis. O histórico fora da amostra
    compartilhado permite que janelas posteriores aprendam somente com decisões
    já realizadas, sem consultar resultados futuros.
    """

    initial_samples: pd.DataFrame
    shared_history: list[dict[str, Any]]
    random_state: int
    fold_id: int | None = None
    refresh_interval: int = SESSOES_ATUALIZACAO_FILTRO_CAIXA_V2
    rolling_window: int = JANELA_AMOSTRAS_FILTRO_CAIXA_V2
    gate: FiltroOportunidadeSeletiva | None = None
    last_refit_history_count: int = 0
    refresh_count: int = 0

    def __post_init__(self) -> None:
        self.gate = _ajustar_filtro_caixa_v2_por_amostras(
            self.initial_samples,
            self.random_state,
        )
        self.last_refit_history_count = len(self.shared_history)

    def _amostras_combinadas(self) -> pd.DataFrame:
        historico = pd.DataFrame(self.shared_history)
        partes = [
            quadro
            for quadro in (self.initial_samples, historico)
            if not quadro.empty
        ]
        if not partes:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    *CARACTERISTICAS_OPORTUNIDADE,
                    "realized_net_log_return",
                    "label",
                ]
            )
        combinado = pd.concat(partes, ignore_index=True)
        combinado = (
            combinado.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
        )
        if len(combinado) > int(self.rolling_window):
            combinado = combinado.iloc[-int(self.rolling_window) :]
        return combinado.reset_index(drop=True)

    def registrar_amostra_madura(self, amostra: dict[str, Any]) -> None:
        self.shared_history.append(dict(amostra))

    def atualizar_se_necessario(self, *, forcar: bool = False) -> bool:
        quantidade_historico = len(self.shared_history)
        if (
            not forcar
            and quantidade_historico - self.last_refit_history_count
            < int(self.refresh_interval)
        ):
            return False
        combinado = self._amostras_combinadas()
        if len(combinado) < 30:
            return False
        self.gate = _ajustar_filtro_caixa_v2_por_amostras(
            combinado,
            self.random_state,
        )
        self.last_refit_history_count = quantidade_historico
        self.refresh_count += 1
        return True

    @property
    def atual(self) -> FiltroOportunidadeSeletiva:
        if self.gate is None:
            raise RuntimeError(
                "O filtro adaptativo de oportunidade ainda não foi ajustado."
            )
        return self.gate

    @property
    def limite(self) -> float:
        return float(self.atual.threshold)

    @property
    def limite_entrada(self) -> float | None:
        return self.atual.entry_threshold

    @property
    def limite_saida(self) -> float | None:
        return self.atual.exit_threshold

    @property
    def base_limite(self) -> str:
        return self.atual.threshold_basis

    @property
    def metodo_calibracao(self) -> str:
        return self.atual.calibration_method

    @property
    def linhas_treinamento(self) -> int:
        return int(self.atual.training_rows)

    @property
    def taxa_positiva(self) -> float:
        return float(self.atual.positive_rate)

    @property
    def linhas_validacao_limite(self) -> int:
        return int(self.atual.threshold_validation_rows)

    @property
    def pontuacao_validacao_limite(self) -> float:
        return float(self.atual.threshold_validation_score)

    @property
    def aceitos_validacao_limite(self) -> int:
        return int(self.atual.threshold_validation_accepted)

    @property
    def transicoes_validacao_limite(self) -> int:
        return int(self.atual.threshold_validation_transitions)

    @property
    def base_alvo(self) -> str:
        return self.atual.target_basis

    @property
    def horizonte_alvo_sessoes(self) -> int | None:
        return self.atual.target_horizon_sessions

    @property
    def regularizado_politica_base(self) -> bool:
        return bool(self.atual.regularized_to_base_policy)

    @property
    def alpha_validacao_limite(self) -> float | None:
        return self.atual.threshold_validation_alpha

    @property
    def taxa_exposicao_validacao_limite(self) -> float | None:
        return self.atual.threshold_validation_exposure_ratio

    def limite_ativo(self, posicao_atual: int | None = None) -> float:
        return self.atual.limite_ativo(posicao_atual)

    def valor_decisao(self, probabilidade: float, confianca: float) -> float:
        return self.atual.valor_decisao(probabilidade, confianca)

    def aceita(
        self,
        probabilidade: float,
        confianca: float,
        posicao_atual: int | None = None,
    ) -> bool:
        return self.atual.aceita(probabilidade, confianca, posicao_atual)

    def probabilidade(self, valores: dict[str, float]) -> float:
        return self.atual.probabilidade(valores)

    def confianca_da_probabilidade(self, probabilidade: float) -> float:
        return self.atual.confianca_da_probabilidade(probabilidade)

    def confianca(self, valores: dict[str, float]) -> float:
        return self.atual.confianca(valores)


def filtro_caixa_oportunidade_ativado(configuracao: Any) -> bool:
    return (
        str(getattr(configuracao, "strategy_mode", ""))
        == MODO_FILTRO_CAIXA_OPORTUNIDADE
    )


def oportunidade_seletiva_ativada(configuracao: Any) -> bool:
    return str(getattr(configuracao, "strategy_mode", "")) in {
        MODO_ROTACAO_SELETIVA,
        MODO_FILTRO_CAIXA_OPORTUNIDADE,
        MODO_ALOCACAO_OTIMIZADA,
        MODO_ALOCACAO_CONCENTRADA,
    }


def caracteristicas_oportunidade(
    utilidades: np.ndarray,
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    instante: pd.Timestamp,
) -> tuple[dict[str, float], int] | None:
    ordenadas = sorted(
        (
            posicao
            for posicao in range(1, len(utilidades))
            if np.isfinite(utilidades[posicao])
        ),
        key=lambda posicao: (
            -float(utilidades[posicao]),
            simbolos[posicao - 1],
        ),
    )
    if not ordenadas:
        return None

    melhor_posicao = ordenadas[0]
    melhor_pontuacao = float(utilidades[melhor_posicao])
    segunda_pontuacao = (
        float(utilidades[ordenadas[1]])
        if len(ordenadas) > 1
        else melhor_pontuacao
    )
    pontuacoes_finitas = np.asarray(
        [float(utilidades[posicao]) for posicao in ordenadas],
        dtype=float,
    )
    media_pontuacao = float(np.mean(pontuacoes_finitas))
    desvio_pontuacao = float(np.std(pontuacoes_finitas))
    melhor_z = (
        float((melhor_pontuacao - media_pontuacao) / desvio_pontuacao)
        if desvio_pontuacao > 1e-12
        else 0.0
    )
    fracao_positiva = float(np.mean(pontuacoes_finitas > 0.0))

    melhor_simbolo = simbolos[melhor_posicao - 1]
    melhor_quadro = quadros.get(melhor_simbolo)
    if melhor_quadro is None or instante not in melhor_quadro.index:
        return None
    linha = melhor_quadro.loc[instante]

    amplitude_5: list[float] = []
    amplitude_20: list[float] = []
    for simbolo in simbolos:
        quadro = quadros.get(simbolo)
        if quadro is None or instante not in quadro.index:
            continue
        atual = quadro.loc[instante]
        valor_5 = float(atual.get("return_5", float("nan")))
        valor_20 = float(atual.get("return_20", float("nan")))
        if np.isfinite(valor_5):
            amplitude_5.append(valor_5)
        if np.isfinite(valor_20):
            amplitude_20.append(valor_20)

    valores = {
        "best_score": melhor_pontuacao,
        "second_score": segunda_pontuacao,
        "best_vs_second_gap": float(melhor_pontuacao - segunda_pontuacao),
        "universe_score_mean": media_pontuacao,
        "universe_score_std": desvio_pontuacao,
        "best_score_zscore": melhor_z,
        "positive_score_fraction": fracao_positiva,
        "best_return_5": float(linha.get("return_5", float("nan"))),
        "best_return_20": float(linha.get("return_20", float("nan"))),
        "best_return_60": float(linha.get("return_60", float("nan"))),
        "best_vol_20": float(linha.get("vol_20", float("nan"))),
        "best_vol_60": float(linha.get("vol_60", float("nan"))),
        "best_trend_efficiency_20": float(
            linha.get("trend_efficiency_20", float("nan"))
        ),
        "best_trend_efficiency_60": float(
            linha.get("trend_efficiency_60", float("nan"))
        ),
        "universe_breadth_5": (
            float(np.mean(np.asarray(amplitude_5) > 0.0))
            if amplitude_5
            else float("nan")
        ),
        "universe_breadth_20": (
            float(np.mean(np.asarray(amplitude_20) > 0.0))
            if amplitude_20
            else float("nan")
        ),
    }
    if not all(
        np.isfinite(valores[nome])
        for nome in CARACTERISTICAS_OPORTUNIDADE
    ):
        return None
    return valores, melhor_posicao


def montar_amostras_oportunidade(
    modelos: dict[str, Any],
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    datas: pd.DatetimeIndex,
    utilidades_no_instante: Callable[
        [dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp],
        np.ndarray,
    ],
) -> pd.DataFrame:
    linhas: list[dict[str, Any]] = []
    for instante in datas:
        ts = pd.Timestamp(instante)
        utilidades = utilidades_no_instante(modelos, quadros, simbolos, ts)
        construida = caracteristicas_oportunidade(
            utilidades,
            quadros,
            simbolos,
            ts,
        )
        if construida is None:
            continue
        caracteristicas, melhor_posicao = construida
        simbolo = simbolos[melhor_posicao - 1]
        alvo = float(
            quadros[simbolo]
            .loc[ts]
            .get("forward_net_log_return", float("nan"))
        )
        if not np.isfinite(alvo):
            continue
        linhas.append(
            {
                "timestamp": ts,
                **caracteristicas,
                "realized_net_log_return": alvo,
                "label": int(alvo > 0.0),
            }
        )
    if not linhas:
        return pd.DataFrame(
            columns=[
                "timestamp",
                *CARACTERISTICAS_OPORTUNIDADE,
                "realized_net_log_return",
                "label",
            ]
        )
    return pd.DataFrame(linhas).sort_values("timestamp").reset_index(drop=True)


def montar_amostras_oportunidade_politica_base(
    modelos: dict[str, Any],
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    datas: pd.DatetimeIndex,
    utilidades_no_instante: Callable[
        [dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp],
        np.ndarray,
    ],
    politica_base: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    retorno_acao_realizada: Callable[[pd.Timestamp, pd.Timestamp, int, int], float],
) -> pd.DataFrame:
    """Monta rótulos de uma sessão para a ação da política base protegida.

    A decisão no fechamento de t controla apenas a exposição depois da abertura
    seguinte. Por isso, o alvo usa o crescimento líquido da sessão de execução,
    em vez do antigo rótulo de utilidade de 5 a 60 sessões.
    """
    linhas: list[dict[str, Any]] = []
    posicao = 0
    dias_manutencao = 0
    ordenadas = pd.DatetimeIndex(datas)
    for indice in range(max(0, len(ordenadas) - 1)):
        instante = pd.Timestamp(ordenadas[indice])
        proximo_instante = pd.Timestamp(ordenadas[indice + 1])
        utilidades = utilidades_no_instante(
            modelos,
            quadros,
            simbolos,
            instante,
        )
        construida = caracteristicas_oportunidade(
            utilidades,
            quadros,
            simbolos,
            instante,
        )
        posicao_anterior = int(posicao)
        acao, _ = politica_base(instante, posicao_anterior, dias_manutencao)
        acao = int(acao)
        if construida is not None and acao > 0:
            caracteristicas, _ = construida
            alvo = float(
                retorno_acao_realizada(
                    instante,
                    proximo_instante,
                    posicao_anterior,
                    acao,
                )
            )
            if np.isfinite(alvo):
                linhas.append(
                    {
                        "timestamp": instante,
                        **caracteristicas,
                        "realized_net_log_return": alvo,
                        "label": int(alvo > 0.0),
                    }
                )
        if acao == posicao_anterior:
            dias_manutencao = dias_manutencao + 1 if acao > 0 else 0
        else:
            posicao = acao
            dias_manutencao = 1 if acao > 0 else 0
    if not linhas:
        return pd.DataFrame(
            columns=[
                "timestamp",
                *CARACTERISTICAS_OPORTUNIDADE,
                "realized_net_log_return",
                "label",
            ]
        )
    return pd.DataFrame(linhas).sort_values("timestamp").reset_index(drop=True)


def _ajustar_classificador(amostras: pd.DataFrame, estado_aleatorio: int) -> Any | None:
    rotulos = amostras["label"].astype(int)
    if rotulos.nunique() < 2:
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    modelo = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    random_state=int(estado_aleatorio),
                ),
            ),
        ]
    )
    modelo.fit(amostras[list(CARACTERISTICAS_OPORTUNIDADE)], rotulos)
    return modelo


def _probabilidades(
    modelo: Any | None,
    constante: float | None,
    amostras: pd.DataFrame,
) -> np.ndarray:
    if amostras.empty:
        return np.asarray([], dtype=float)
    if modelo is None:
        return np.full(len(amostras), float(constante or 0.0), dtype=float)
    return np.asarray(
        modelo.predict_proba(amostras[list(CARACTERISTICAS_OPORTUNIDADE)])[:, 1],
        dtype=float,
    )


def _confianca_relativa(
    probabilidade: float,
    probabilidades_referencia: np.ndarray,
    *,
    modelo_constante: bool = False,
) -> float:
    valor = min(1.0, max(0.0, float(probabilidade)))
    finitas = np.sort(
        probabilidades_referencia[np.isfinite(probabilidades_referencia)]
    )
    if modelo_constante or not len(finitas):
        return valor
    posto = int(np.searchsorted(finitas, valor, side="right"))
    return min(1.0, max(0.0, float(posto / len(finitas))))


def _validacao_prequencial(
    amostras: pd.DataFrame,
    estado_aleatorio: int,
    horizonte_rotulo: int,
) -> pd.DataFrame:
    intervalo = max(1, int(horizonte_rotulo))
    minimo_treinamento = max(24, len(CARACTERISTICAS_OPORTUNIDADE) * 2)
    primeiro_indice_validacao = intervalo + minimo_treinamento
    if primeiro_indice_validacao >= len(amostras):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "probability",
                "confidence",
                "realized_net_log_return",
                "label",
            ]
        )

    linhas: list[dict[str, Any]] = []
    for indice in range(primeiro_indice_validacao, len(amostras)):
        fim_treinamento = indice - intervalo
        treinamento = amostras.iloc[:fim_treinamento]
        atual = amostras.iloc[[indice]]
        modelo = _ajustar_classificador(treinamento, int(estado_aleatorio))
        constante = (
            float(treinamento["label"].mean()) if modelo is None else None
        )
        probabilidade = float(_probabilidades(modelo, constante, atual)[0])
        probabilidades_referencia = _probabilidades(
            modelo,
            constante,
            treinamento,
        )
        confianca = _confianca_relativa(
            probabilidade,
            probabilidades_referencia,
            modelo_constante=modelo is None,
        )
        linhas.append(
            {
                "timestamp": atual.iloc[0]["timestamp"],
                "probability": probabilidade,
                "confidence": confianca,
                "realized_net_log_return": float(
                    atual.iloc[0]["realized_net_log_return"]
                ),
                "label": int(atual.iloc[0]["label"]),
            }
        )
    return pd.DataFrame(linhas)


def _candidatos_limite_confianca() -> tuple[float, ...]:
    return tuple(float(valor) for valor in np.linspace(0.0, 0.90, 19))


def _calibrar_limite_confianca(
    validacao: pd.DataFrame,
) -> tuple[float, float, int]:
    if validacao.empty:
        return 0.5, float("nan"), 0

    minimo_aceitos = max(8, int(np.ceil(np.sqrt(len(validacao)))))
    realizados = validacao["realized_net_log_return"].to_numpy(dtype=float)
    confiancas = validacao["confidence"].to_numpy(dtype=float)
    melhor_limite = 0.0
    melhor_pontuacao = float("-inf")
    melhor_quantidade = len(validacao)

    for limite in _candidatos_limite_confianca():
        aceitos = confiancas >= float(limite)
        quantidade = int(np.sum(aceitos))
        if quantidade < minimo_aceitos:
            continue
        retornos_aceitos = realizados[aceitos]
        total = float(np.sum(retornos_aceitos))
        incerteza = (
            float(np.std(retornos_aceitos, ddof=1) * np.sqrt(quantidade))
            if quantidade > 1
            else 0.0
        )
        pontuacao = total - incerteza
        if (
            pontuacao > melhor_pontuacao + 1e-12
            or (
                abs(pontuacao - melhor_pontuacao) <= 1e-12
                and float(limite) < melhor_limite
            )
        ):
            melhor_limite = float(limite)
            melhor_pontuacao = float(pontuacao)
            melhor_quantidade = quantidade

    if not np.isfinite(melhor_pontuacao):
        return 0.0, float(np.sum(realizados)), len(validacao)
    return melhor_limite, melhor_pontuacao, melhor_quantidade


def _calibrar_limites_histerese(
    validacao: pd.DataFrame,
    *,
    coluna_sinal: str = "probability",
) -> tuple[float, float, float, int, int]:
    """Calibra um filtro CAIXA↔MERCADO com estado usando apenas validação.

    Os limites de entrada e saída são escolhidos somente pelo sinal prequencial.
    A restrição saída <= entrada cria a histerese. A quantidade de transições é
    apenas critério determinístico de desempate, para não premiar giro excessivo.
    """
    if validacao.empty:
        return 0.5, 0.5, float("nan"), 0, 0
    if coluna_sinal not in validacao.columns:
        raise ValueError(
            f"A coluna do sinal para histerese está ausente: {coluna_sinal}"
        )

    realizados = validacao["realized_net_log_return"].to_numpy(dtype=float)
    sinal = validacao[coluna_sinal].to_numpy(dtype=float)
    minimo_exposto = max(8, int(np.ceil(np.sqrt(len(validacao)))))

    melhor_entrada = 0.0
    melhor_saida = 0.0
    melhor_pontuacao = float("-inf")
    melhor_exposto = len(validacao)
    melhores_transicoes = 0

    candidatos = _candidatos_limite_confianca()
    for limite_entrada in candidatos:
        for limite_saida in candidatos:
            if float(limite_saida) > float(limite_entrada) + 1e-12:
                continue
            investido = False
            exposto = np.zeros(len(validacao), dtype=bool)
            transicoes = 0
            for indice, valor in enumerate(sinal):
                anterior = investido
                if investido:
                    if float(valor) < float(limite_saida):
                        investido = False
                elif float(valor) >= float(limite_entrada):
                    investido = True
                if investido != anterior:
                    transicoes += 1
                exposto[indice] = investido

            quantidade_exposta = int(np.sum(exposto))
            if quantidade_exposta < minimo_exposto:
                continue
            retornos_aceitos = realizados[exposto]
            total = float(np.sum(retornos_aceitos))
            incerteza = (
                float(
                    np.std(retornos_aceitos, ddof=1)
                    * np.sqrt(quantidade_exposta)
                )
                if quantidade_exposta > 1
                else 0.0
            )
            pontuacao = total - incerteza
            melhor = pontuacao > melhor_pontuacao + 1e-12
            empatado = abs(pontuacao - melhor_pontuacao) <= 1e-12
            if melhor or (
                empatado
                and (
                    transicoes < melhores_transicoes
                    or (
                        transicoes == melhores_transicoes
                        and float(limite_entrada) < melhor_entrada
                    )
                    or (
                        transicoes == melhores_transicoes
                        and abs(float(limite_entrada) - melhor_entrada) <= 1e-12
                        and float(limite_saida) > melhor_saida
                    )
                )
            ):
                melhor_entrada = float(limite_entrada)
                melhor_saida = float(limite_saida)
                melhor_pontuacao = float(pontuacao)
                melhor_exposto = quantidade_exposta
                melhores_transicoes = int(transicoes)

    if not np.isfinite(melhor_pontuacao):
        total = float(np.sum(realizados))
        return 0.0, 0.0, total, len(validacao), 0
    return (
        melhor_entrada,
        melhor_saida,
        melhor_pontuacao,
        melhor_exposto,
        melhores_transicoes,
    )


def _candidatos_limite_filtro_caixa_v2() -> tuple[float, ...]:
    # O valor zero preserva a política base e mantém a exposição liberada.
    return (
        0.0,
        *tuple(float(valor) for valor in np.arange(0.30, 0.751, 0.05)),
    )


def _calibrar_limites_filtro_caixa_v2(
    validacao: pd.DataFrame,
) -> tuple[float, float, float, int, int, float, float, bool]:
    """Permite intervenção em CAIXA apenas quando supera a política base.

    A pontuação mede o crescimento logarítmico incremental do filtro contra a
    política base e desconta incerteza. Pelo menos 60% das sessões de validação
    devem continuar expostas. Sem evidência conservadora positiva, os limites
    voltam para zero e a política base permanece integralmente no controle.
    """
    if len(validacao) < MINIMO_LINHAS_VALIDACAO_FILTRO_CAIXA_V2:
        realizados = validacao.get(
            "realized_net_log_return",
            pd.Series(dtype=float),
        ).to_numpy(dtype=float)
        return (
            0.0,
            0.0,
            0.0,
            len(validacao),
            0,
            0.0,
            1.0,
            True,
        )

    realizados = validacao["realized_net_log_return"].to_numpy(dtype=float)
    probabilidades = validacao["probability"].to_numpy(dtype=float)
    quantidade = len(validacao)
    minimo_exposto = max(
        24,
        int(
            np.ceil(
                TAXA_MINIMA_EXPOSICAO_MERCADO_FILTRO_CAIXA_V2
                * quantidade
            )
        ),
    )

    melhor: tuple[float, float, float, int, int, float, float] | None = None
    candidatos = _candidatos_limite_filtro_caixa_v2()
    for limite_entrada in candidatos:
        for limite_saida in candidatos:
            if float(limite_saida) > float(limite_entrada) + 1e-12:
                continue
            investido = False
            exposto = np.zeros(quantidade, dtype=bool)
            transicoes = 0
            for indice, probabilidade in enumerate(probabilidades):
                anterior = investido
                if investido:
                    if float(probabilidade) < float(limite_saida):
                        investido = False
                elif float(probabilidade) >= float(limite_entrada):
                    investido = True
                if investido != anterior:
                    transicoes += 1
                exposto[indice] = investido

            quantidade_exposta = int(np.sum(exposto))
            if quantidade_exposta < minimo_exposto:
                continue
            intervencao = np.where(exposto, 0.0, -realizados)
            alpha = float(np.sum(intervencao))
            incerteza = (
                float(np.std(intervencao, ddof=1) * np.sqrt(quantidade))
                if quantidade > 1
                else 0.0
            )
            pontuacao_conservadora = float(alpha - 0.50 * incerteza)
            taxa_exposicao = float(quantidade_exposta / quantidade)
            candidato = (
                float(limite_entrada),
                float(limite_saida),
                pontuacao_conservadora,
                quantidade_exposta,
                int(transicoes),
                alpha,
                taxa_exposicao,
            )
            if melhor is None:
                melhor = candidato
                continue
            (
                _,
                _,
                melhor_pontuacao,
                melhor_exposto,
                melhores_transicoes,
                melhor_alpha,
                _,
            ) = melhor
            superior = pontuacao_conservadora > melhor_pontuacao + 1e-12
            empatado = abs(pontuacao_conservadora - melhor_pontuacao) <= 1e-12
            if superior or (
                empatado
                and (
                    alpha > melhor_alpha + 1e-12
                    or (
                        abs(alpha - melhor_alpha) <= 1e-12
                        and transicoes < melhores_transicoes
                    )
                    or (
                        abs(alpha - melhor_alpha) <= 1e-12
                        and transicoes == melhores_transicoes
                        and quantidade_exposta > melhor_exposto
                    )
                )
            ):
                melhor = candidato

    if melhor is None:
        return 0.0, 0.0, 0.0, quantidade, 0, 0.0, 1.0, True
    entrada, saida, pontuacao, exposto, transicoes, alpha, taxa_exposicao = melhor
    if pontuacao <= 0.0 or alpha <= 0.0:
        return 0.0, 0.0, 0.0, quantidade, 0, 0.0, 1.0, True
    return (
        entrada,
        saida,
        pontuacao,
        exposto,
        transicoes,
        alpha,
        taxa_exposicao,
        False,
    )


def _ajustar_filtro_caixa_v2_por_amostras(
    amostras: pd.DataFrame,
    estado_aleatorio: int,
) -> FiltroOportunidadeSeletiva:
    if len(amostras) < 30:
        # Sem evidência suficiente, preserva a política base e não força CAIXA.
        return FiltroOportunidadeSeletiva(
            model=None,
            threshold=0.0,
            constant_probability=1.0,
            training_rows=int(len(amostras)),
            positive_rate=(
                float(amostras["label"].mean())
                if len(amostras)
                else float("nan")
            ),
            threshold_validation_rows=0,
            threshold_validation_score=0.0,
            entry_threshold=0.0,
            exit_threshold=0.0,
            calibration_method="adaptive_base_policy_one_step_hysteresis_v2",
            threshold_basis="absolute_probability",
            target_basis=(
                "protected_base_policy_next_session_open_to_close_net_log_return"
            ),
            target_horizon_sessions=1,
            regularized_to_base_policy=True,
            threshold_validation_alpha=0.0,
            threshold_validation_exposure_ratio=1.0,
        )

    validacao = _validacao_prequencial(amostras, int(estado_aleatorio), 1)
    (
        entrada,
        saida,
        pontuacao,
        exposto,
        transicoes,
        alpha,
        taxa_exposicao,
        regularizado,
    ) = _calibrar_limites_filtro_caixa_v2(validacao)
    modelo_final = _ajustar_classificador(amostras, int(estado_aleatorio))
    probabilidade_constante = (
        float(amostras["label"].mean()) if modelo_final is None else None
    )
    probabilidades_referencia = (
        tuple(
            float(valor)
            for valor in np.sort(
                _probabilidades(
                    modelo_final,
                    probabilidade_constante,
                    amostras,
                )
            )
        )
        if modelo_final is not None
        else ()
    )
    # Um classificador constante não contém informação de regime ou separação.
    if modelo_final is None:
        entrada = saida = 0.0
        regularizado = True
        alpha = 0.0
        taxa_exposicao = 1.0

    return FiltroOportunidadeSeletiva(
        model=modelo_final,
        threshold=float(entrada),
        constant_probability=probabilidade_constante,
        training_rows=int(len(amostras)),
        positive_rate=float(amostras["label"].mean()),
        threshold_validation_rows=int(len(validacao)),
        threshold_validation_score=float(pontuacao),
        reference_probabilities=probabilidades_referencia,
        threshold_validation_accepted=int(exposto),
        entry_threshold=float(entrada),
        exit_threshold=float(saida),
        threshold_validation_transitions=int(transicoes),
        calibration_method="adaptive_base_policy_one_step_hysteresis_v2",
        threshold_basis="absolute_probability",
        target_basis=(
            "protected_base_policy_next_session_open_to_close_net_log_return"
        ),
        target_horizon_sessions=1,
        regularized_to_base_policy=bool(regularizado),
        threshold_validation_alpha=float(alpha),
        threshold_validation_exposure_ratio=float(taxa_exposicao),
    )


def ajustar_filtro_caixa_oportunidade_adaptativo(
    amostras_iniciais: pd.DataFrame,
    *,
    estado_aleatorio: int,
    historico_compartilhado: list[dict[str, Any]],
    identificador_janela: int | None = None,
    intervalo_atualizacao: int = SESSOES_ATUALIZACAO_FILTRO_CAIXA_V2,
    janela_rolante: int = JANELA_AMOSTRAS_FILTRO_CAIXA_V2,
) -> FiltroCaixaOportunidadeAdaptativo:
    return FiltroCaixaOportunidadeAdaptativo(
        initial_samples=amostras_iniciais.copy(),
        shared_history=historico_compartilhado,
        random_state=int(estado_aleatorio),
        fold_id=identificador_janela,
        refresh_interval=int(intervalo_atualizacao),
        rolling_window=int(janela_rolante),
    )


def ajustar_filtro_oportunidade_seletiva(
    modelos: dict[str, Any],
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    datas_calibracao: pd.DatetimeIndex,
    utilidades_no_instante: Callable[
        [dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp],
        np.ndarray,
    ],
    *,
    estado_aleatorio: int,
    horizonte_rotulo: int,
    histerese: bool = False,
) -> FiltroOportunidadeSeletiva:
    amostras = montar_amostras_oportunidade(
        modelos,
        quadros,
        simbolos,
        datas_calibracao,
        utilidades_no_instante,
    )
    if len(amostras) < 30:
        raise ValueError(
            "A oportunidade seletiva exige pelo menos 30 decisões válidas "
            f"de calibração; apenas {len(amostras)} estão disponíveis."
        )

    validacao = _validacao_prequencial(
        amostras,
        int(estado_aleatorio),
        int(horizonte_rotulo),
    )
    if histerese:
        (
            limite_entrada,
            limite_saida,
            melhor_pontuacao,
            melhor_quantidade,
            quantidade_transicoes,
        ) = _calibrar_limites_histerese(
            validacao,
            coluna_sinal="probability",
        )
        melhor_limite = float(limite_entrada)
    else:
        (
            melhor_limite,
            melhor_pontuacao,
            melhor_quantidade,
        ) = _calibrar_limite_confianca(validacao)
        limite_entrada = None
        limite_saida = None
        quantidade_transicoes = 0

    modelo_final = _ajustar_classificador(amostras, int(estado_aleatorio))
    probabilidade_constante = (
        float(amostras["label"].mean()) if modelo_final is None else None
    )
    probabilidades_referencia = (
        tuple(
            float(valor)
            for valor in np.sort(
                _probabilidades(
                    modelo_final,
                    probabilidade_constante,
                    amostras,
                )
            )
        )
        if modelo_final is not None
        else ()
    )
    if modelo_final is None:
        melhor_limite = 0.5
        if histerese:
            limite_entrada = 0.5
            limite_saida = 0.5

    return FiltroOportunidadeSeletiva(
        model=modelo_final,
        threshold=float(melhor_limite),
        constant_probability=probabilidade_constante,
        training_rows=int(len(amostras)),
        positive_rate=float(amostras["label"].mean()),
        threshold_validation_rows=int(len(validacao)),
        threshold_validation_score=float(melhor_pontuacao),
        reference_probabilities=probabilidades_referencia,
        threshold_validation_accepted=int(melhor_quantidade),
        calibration_method=(
            "prequential_absolute_probability_hysteresis_v1"
            if histerese
            else "prequential_relative_confidence_v2"
        ),
        entry_threshold=(
            float(limite_entrada) if limite_entrada is not None else None
        ),
        exit_threshold=(
            float(limite_saida) if limite_saida is not None else None
        ),
        threshold_validation_transitions=int(quantidade_transicoes),
        threshold_basis=(
            "absolute_probability" if histerese else "relative_confidence"
        ),
        target_basis="weighted_forward_net_log_return",
        target_horizon_sessions=int(horizonte_rotulo),
    )


def avaliar_oportunidade(
    filtro: FiltroOportunidadeSeletiva | FiltroCaixaOportunidadeAdaptativo,
    utilidades: np.ndarray,
    quadros: dict[str, pd.DataFrame],
    simbolos: list[str],
    instante: pd.Timestamp,
    *,
    posicao_atual: int | None = None,
) -> AvaliacaoOportunidade | None:
    construida = caracteristicas_oportunidade(
        utilidades,
        quadros,
        simbolos,
        instante,
    )
    if construida is None:
        return None
    caracteristicas, melhor_posicao = construida
    probabilidade = filtro.probabilidade(caracteristicas)
    confianca = filtro.confianca_da_probabilidade(probabilidade)
    return AvaliacaoOportunidade(
        probability=float(probabilidade),
        confidence=float(confianca),
        accepted=filtro.aceita(probabilidade, confianca, posicao_atual),
        features=caracteristicas,
        best_position=int(melhor_posicao),
    )
