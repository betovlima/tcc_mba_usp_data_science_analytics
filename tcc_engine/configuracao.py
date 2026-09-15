"""Configuração declarativa do experimento acadêmico.

Os nomes do domínio do TCC ficam em português. Apenas os hiperparâmetros que
são enviados diretamente ao LightGBM mantêm os nomes definidos pela biblioteca.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

DATA_INICIO = "2016-01-01"
DATA_FIM = "2026-09-04"

ATIVOS = (
    "NVDA", "MSFT", "META", "TSLA", "AMD", "JPM", "SPY", "AVGO", "NFLX",
    "ORCL", "COST", "LLY", "XOM", "CAT", "WMT", "V", "HD", "ADC", "ADEA",
    "ADI", "ADM", "GKOS", "VNCE", "CORT", "UNFI", "DNN", "MKSI", "APD",
    "DDS", "RACE", "UNF", "TX", "CEF", "YANG", "KKR", "BXMT", "SCSC",
)


def configuracoes_lightgbm() -> dict[str, Any]:
    """Retorna os hiperparâmetros usados diretamente pelo LightGBM."""
    return {
        "n_estimators": 329,
        "learning_rate": 0.020731,
        "max_depth": 3,
        "num_leaves": 6,
        "min_child_samples": 18,
        "min_child_weight": 5.0,
        "subsample": 0.85,
        "subsample_freq": 0,
        "colsample_bytree": 0.88067,
        "reg_alpha": 0.050837,
        "reg_lambda": 3.596305,
        "max_bin": 255,
        "n_jobs": -1,
    }


@dataclass(frozen=True)
class ConfiguracaoBacktest:
    """Parâmetros congelados do experimento."""

    ativos: tuple[str, ...] = ATIVOS
    data_inicio: str = DATA_INICIO
    data_fim: str = DATA_FIM

    horizontes_alvo: tuple[int, ...] = (5, 10, 20, 40, 60)
    pesos_horizontes: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.25)
    peso_captura_movimento: float = 0.35
    peso_persistencia_tendencia: float = 0.20

    linhas_minimas_treinamento: int = 700
    dias_calibracao: int = 126
    dias_teste: int = 504
    dias_minimos_teste: int = 126
    dias_separacao: int = 60

    penalidade_baixa: float = 0.20
    penalidade_drawdown: float = 0.35
    dias_minimos_posicao: int = 2
    vantagem_minima_entrada: float = 0.001
    limiar_caixa: float = 0.0
    margem_troca: float = 0.0005
    candidatas_margem_troca: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01)

    capital_inicial: float = 10_000.0
    acoes_inteiras: bool = False
    deslizamento_bps: float = 0.0
    comissao: float = 0.0
    taxa_sec: float = 2.06e-5
    taxa_taf_por_acao: float = 0.000195
    limite_taxa_taf: float = 9.79
    taxa_cat_por_acao: float = 3e-6

    semente_aleatoria: int = 42
    execucao_deterministica: bool = False
    limite_threads_numericas: int = 1
    repeticoes: int = 1
    passo_semente: int = 1000
    ativos_ancora_calendario: tuple[str, ...] = ATIVOS
    numero_janelas_forcado: int | None = None
    hiperparametros_lightgbm: dict[str, Any] = field(default_factory=configuracoes_lightgbm)

    @property
    def acoes_fracionarias(self) -> bool:
        """Indica se a simulação permite quantidades fracionárias."""
        return not self.acoes_inteiras

    def copiar(self, **alteracoes: Any) -> "ConfiguracaoBacktest":
        """Cria uma nova configuração com os campos informados alterados."""
        return replace(self, **alteracoes)


CONFIGURACAO = ConfiguracaoBacktest()
