from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
DIRETORIO_SAIDA_PADRAO = RAIZ_PROJETO / "output"
DESTINO_PADRAO = RAIZ_PROJETO / "analysis" / "tcc_backtest_output_analysis.xlsx"

COR_AZUL = "2F75B5"
COR_AZUL_ESCURO = "17365D"
COR_BRANCA = "FFFFFF"


def ler_argumentos():
    """Lê os caminhos opcionais informados pela linha de comando."""
    analisador = argparse.ArgumentParser(
        description="Gera a planilha de auditoria do experimento a partir de output/."
    )
    analisador.add_argument(
        "--output-dir",
        type=Path,
        default=DIRETORIO_SAIDA_PADRAO,
        help="Diretório que contém os artefatos gerados pelo backtest.",
    )
    analisador.add_argument(
        "--destination",
        type=Path,
        default=DESTINO_PADRAO,
        help="Arquivo Excel que será gerado.",
    )
    return analisador.parse_args()


def carregar_dados(diretorio_saida: Path):
    """Carrega os artefatos necessários para construir a auditoria."""
    arquivos_necessarios = [
        "backtest_result.json",
        "equity_curve.csv",
        "folds.csv",
        "trades.csv",
        "summary.txt",
        "market_data.csv",
    ]
    arquivos_ausentes = [
        nome
        for nome in arquivos_necessarios
        if not (diretorio_saida / nome).exists()
    ]
    if arquivos_ausentes:
        raise FileNotFoundError(
            "Arquivos ausentes: " + ", ".join(arquivos_ausentes)
        )

    resultado = json.loads(
        (diretorio_saida / "backtest_result.json").read_text(encoding="utf-8")
    )
    curva_capital = pd.read_csv(diretorio_saida / "equity_curve.csv")
    janelas = pd.read_csv(diretorio_saida / "folds.csv")
    operacoes = pd.read_csv(diretorio_saida / "trades.csv")
    dados_mercado = pd.read_csv(diretorio_saida / "market_data.csv")
    resumo = (diretorio_saida / "summary.txt").read_text(encoding="utf-8")

    dados_mercado["timestamp"] = pd.to_datetime(
        dados_mercado["timestamp"],
        utc=True,
    )
    return resultado, curva_capital, janelas, operacoes, dados_mercado, resumo


def montar_ciclos(operacoes: pd.DataFrame) -> pd.DataFrame:
    """Combina compras e vendas para formar os ciclos completos de posição."""
    colunas_compra = [
        "asset",
        "timestamp",
        "best_alternative_asset",
        "best_alternative_return",
        "opportunity_cost",
    ]
    colunas_venda = [
        "entry_timestamp",
        "timestamp",
        "asset",
        "entry_price",
        "execution_price",
        "quantity",
        "realized_pnl",
        "position_return",
        "holding_bars",
        "total_fee",
        "walk_forward_fold",
        "rotation_id",
        "position_entry_score",
        "current_score",
        "best_score",
        "best_vs_current_gap",
        "effective_switch_margin",
        "current_asset_rank",
        "maximum_favorable_excursion",
        "maximum_adverse_excursion",
        "profit_capture_ratio",
    ]

    colunas_compra = [
        coluna for coluna in colunas_compra if coluna in operacoes.columns
    ]
    colunas_venda = [
        coluna for coluna in colunas_venda if coluna in operacoes.columns
    ]

    compras = operacoes.loc[
        operacoes["action"].eq("BUY"),
        colunas_compra,
    ].copy()
    if "timestamp" in compras.columns:
        compras = compras.rename(columns={"timestamp": "entry_timestamp"})

    vendas = operacoes.loc[
        operacoes["action"].isin(["SELL", "FINAL_SELL"]),
        colunas_venda,
    ].copy()

    ciclos = vendas.merge(
        compras,
        on=["asset", "entry_timestamp"],
        how="left",
    )
    ciclos = ciclos.rename(
        columns={
            "timestamp": "exit_timestamp",
            "execution_price": "exit_price",
            "total_fee": "exit_fee",
            "walk_forward_fold": "fold",
            "position_entry_score": "entry_score",
            "maximum_favorable_excursion": "mfe",
            "maximum_adverse_excursion": "mae",
        }
    )

    if "position_return" in ciclos.columns:
        ciclos["resultado"] = ciclos["position_return"].map(
            lambda valor: (
                "GANHO"
                if valor > 0
                else "PERDA"
                if valor < 0
                else "NEUTRO"
            )
        )
    return ciclos


def montar_mensal(curva_capital: pd.DataFrame) -> pd.DataFrame:
    """Calcula os retornos mensais da estratégia e da referência."""
    tabela = curva_capital.copy()
    coluna_data = tabela.columns[0]
    tabela[coluna_data] = (
        pd.to_datetime(tabela[coluna_data], utc=True)
        .dt.tz_localize(None)
    )
    tabela["mes"] = tabela[coluna_data].dt.to_period("M").astype(str)

    mensal = tabela.groupby("mes", as_index=False).agg(
        capital_estrategia=("strategy_equity", "last"),
        capital_referencia=("buy_hold_equity", "last"),
    )
    mensal["retorno_estrategia"] = mensal["capital_estrategia"].pct_change()
    mensal["retorno_referencia"] = mensal["capital_referencia"].pct_change()
    mensal["retorno_excedente"] = (
        mensal["retorno_estrategia"] - mensal["retorno_referencia"]
    )
    return mensal


def montar_resumo_ativos(dados_mercado: pd.DataFrame) -> pd.DataFrame:
    """Resume o intervalo e a variação do fechamento de cada ativo."""
    linhas = []
    for ativo, tabela in dados_mercado.groupby("symbol", sort=False):
        tabela = tabela.sort_values("timestamp").copy()
        datas = pd.to_datetime(
            tabela["timestamp"],
            utc=True,
        ).dt.tz_localize(None)

        primeiro_fechamento = float(tabela.iloc[0]["close"])
        ultimo_fechamento = float(tabela.iloc[-1]["close"])

        linhas.append(
            {
                "Ativo": ativo,
                "Linhas": len(tabela),
                "Início": datas.iloc[0],
                "Fim": datas.iloc[-1],
                "Primeiro fechamento": primeiro_fechamento,
                "Último fechamento": ultimo_fechamento,
                "Retorno do fechamento": (
                    ultimo_fechamento / primeiro_fechamento - 1.0
                ),
            }
        )
    return pd.DataFrame(linhas)


def achatar_estrutura(valor, prefixo=""):
    """Converte uma estrutura aninhada em linhas caminho/tipo/valor."""
    linhas = []
    if isinstance(valor, dict):
        for chave, filho in valor.items():
            caminho = f"{prefixo}.{chave}" if prefixo else str(chave)
            linhas.extend(achatar_estrutura(filho, caminho))
    elif isinstance(valor, list):
        linhas.append(
            (
                prefixo,
                "lista",
                json.dumps(valor, ensure_ascii=False, default=str),
            )
        )
    else:
        linhas.append((prefixo, type(valor).__name__, valor))
    return linhas


def preparar_serie_ativo(
    dados_mercado: pd.DataFrame,
    ativo: str,
) -> pd.DataFrame:
    """Prepara a série temporal de um ativo para exibição no Excel."""
    serie = (
        dados_mercado.loc[dados_mercado["symbol"].eq(ativo)]
        .sort_values("timestamp")
        .copy()
    )
    serie["timestamp"] = (
        pd.to_datetime(serie["timestamp"], utc=True)
        .dt.tz_localize(None)
    )
    serie = serie[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    serie.columns = [
        "Data",
        "Abertura",
        "Máxima",
        "Mínima",
        "Fechamento",
        "Volume",
    ]
    return serie


def estilizar_cabecalho(aba):
    """Aplica o padrão visual do cabeçalho de uma aba."""
    aba.sheet_view.showGridLines = False
    aba.freeze_panes = "A2"

    for celula in aba[1]:
        celula.fill = PatternFill("solid", fgColor=COR_AZUL)
        celula.font = Font(color=COR_BRANCA, bold=True)
        celula.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    for indice_coluna in range(1, min(aba.max_column, 20) + 1):
        maior = max(
            len(str(aba.cell(linha, indice_coluna).value or ""))
            for linha in range(1, min(aba.max_row, 40) + 1)
        )
        aba.column_dimensions[
            get_column_letter(indice_coluna)
        ].width = max(11, min(30, maior + 2))


def estilizar_aba_ativo(aba):
    """Acrescenta cálculos derivados e formatação à série de um ativo."""
    ultima_linha = aba.max_row
    aba["G1"] = "Retorno diário"
    aba["H1"] = "Pico acumulado"
    aba["I1"] = "Queda desde o pico"

    for linha in range(2, ultima_linha + 1):
        if linha == 2:
            aba[f"H{linha}"] = f"=E{linha}"
            aba[f"I{linha}"] = 0
        else:
            aba[f"G{linha}"] = f"=E{linha}/E{linha-1}-1"
            aba[f"H{linha}"] = f"=MAX(H{linha-1},E{linha})"
            aba[f"I{linha}"] = f"=E{linha}/H{linha}-1"

        aba[f"A{linha}"].number_format = "yyyy-mm-dd"
        for coluna in "BCDEH":
            aba[f"{coluna}{linha}"].number_format = "0.0000"
        aba[f"F{linha}"].number_format = "#,##0"
        aba[f"G{linha}"].number_format = "0.0000%"
        aba[f"I{linha}"].number_format = "0.0000%"

    if ultima_linha >= 2:
        aba.conditional_formatting.add(
            f"I2:I{ultima_linha}",
            ColorScaleRule(
                start_type="min",
                start_color="F8696B",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFEB84",
                end_type="max",
                end_color="63BE7B",
            ),
        )


def preencher_indicadores(pasta, resultado: dict):
    """Preenche a aba de indicadores e adiciona o gráfico de capital."""
    aba = pasta["01_Indicadores"]
    metricas = resultado["metrics"]

    ultima_curva = pasta["02_Curva"].max_row
    ultima_operacao = pasta["04_Operacoes"].max_row
    ultima_janela = pasta["03_Janelas"].max_row

    janelas = list(metricas.get("walk_forward_folds") or [])
    retorno_pior_janela = (
        min(float(item["strategy_return"]) for item in janelas)
        if janelas
        else None
    )

    linhas = [
        (
            "Capital inicial",
            metricas.get("initial_capital"),
            "='03_Janelas'!K2",
            "Capital inicial da simulação.",
        ),
        (
            "Capital final",
            metricas.get("strategy_ending_capital"),
            f"='02_Curva'!B{ultima_curva}",
            "Último capital da curva.",
        ),
        (
            "Retorno da estratégia",
            metricas.get("strategy_return"),
            "=C3/C2-1",
            "Capital final dividido pelo inicial menos um.",
        ),
        (
            "CAGR",
            metricas.get("strategy_cagr"),
            None,
            "Taxa composta anual de crescimento.",
        ),
        (
            "Índice de Sharpe",
            metricas.get("strategy_sharpe"),
            None,
            "Retorno ajustado pela volatilidade.",
        ),
        (
            "Queda máxima",
            metricas.get("strategy_maximum_drawdown"),
            None,
            "Maior queda entre um pico e o vale seguinte.",
        ),
        (
            "Compras",
            metricas.get("simulated_buys"),
            f'=COUNTIF(\'04_Operacoes\'!B2:B{ultima_operacao},"BUY")',
            "Quantidade de ordens de compra.",
        ),
        (
            "Vendas",
            metricas.get("simulated_sells"),
            (
                f'=COUNTIF(\'04_Operacoes\'!B2:B{ultima_operacao},"SELL")'
                f'+COUNTIF(\'04_Operacoes\'!B2:B{ultima_operacao},"FINAL_SELL")'
            ),
            "Quantidade de encerramentos de posição.",
        ),
        (
            "Rotações",
            metricas.get("capital_rotations"),
            f'=COUNTIF(\'04_Operacoes\'!B2:B{ultima_operacao},"SELL")',
            "Trocas efetivas de ativo.",
        ),
        (
            "Permanência média",
            metricas.get("average_holding_bars"),
            None,
            "Quantidade média de sessões por posição.",
        ),
        (
            "Taxas totais",
            metricas.get("total_transaction_fees"),
            None,
            "Custos totais contabilizados pelo motor.",
        ),
        (
            "Retorno da pior janela",
            retorno_pior_janela,
            (
                f"=MIN('03_Janelas'!M2:M{ultima_janela})"
                if ultima_janela >= 2
                else None
            ),
            "Menor retorno entre as janelas temporais.",
        ),
    ]

    cabecalhos = [
        "Métrica",
        "Motor / JSON",
        "Excel",
        "Diferença",
        "Situação",
        "Reconstrução",
    ]
    for coluna, valor in enumerate(cabecalhos, start=1):
        aba.cell(1, coluna, valor)

    for indice, (nome, motor, formula, explicacao) in enumerate(
        linhas,
        start=2,
    ):
        aba.cell(indice, 1, nome)
        aba.cell(indice, 2, motor)
        if formula:
            aba.cell(indice, 3, formula)
            aba.cell(indice, 4, f"=C{indice}-B{indice}")
            aba.cell(
                indice,
                5,
                f'=IF(ABS(D{indice})<1E-8,"OK","VERIFICAR")',
            )
        else:
            aba.cell(indice, 3, motor)
            aba.cell(indice, 4, 0)
            aba.cell(indice, 5, "REFERÊNCIA")
        aba.cell(indice, 6, explicacao)

    aba_curva = pasta["02_Curva"]
    grafico = LineChart()
    grafico.title = "Capital: estratégia x referência"
    grafico.add_data(
        Reference(
            aba_curva,
            min_col=2,
            max_col=3,
            min_row=1,
            max_row=ultima_curva,
        ),
        titles_from_data=True,
    )
    grafico.height = 8
    grafico.width = 16
    aba.add_chart(grafico, "H2")


def acrescentar_calculos_curva(pasta):
    """Acrescenta retornos e queda desde o pico à curva de capital."""
    aba = pasta["02_Curva"]
    ultima_linha = aba.max_row
    primeira_coluna_nova = aba.max_column + 1

    titulos = [
        "Retorno diário da estratégia",
        "Retorno diário da referência",
        "Pico da estratégia",
        "Queda desde o pico",
    ]
    for deslocamento, titulo in enumerate(titulos):
        aba.cell(1, primeira_coluna_nova + deslocamento, titulo)

    for linha in range(2, ultima_linha + 1):
        if linha > 2:
            aba.cell(
                linha,
                primeira_coluna_nova,
                f"=B{linha}/B{linha-1}-1",
            )
            aba.cell(
                linha,
                primeira_coluna_nova + 1,
                f"=C{linha}/C{linha-1}-1",
            )

        coluna_pico = get_column_letter(primeira_coluna_nova + 2)
        aba.cell(
            linha,
            primeira_coluna_nova + 2,
            (
                f"=MAX(10000,B{linha})"
                if linha == 2
                else f"=MAX({coluna_pico}{linha-1},B{linha})"
            ),
        )
        aba.cell(
            linha,
            primeira_coluna_nova + 3,
            f"=B{linha}/{coluna_pico}{linha}-1",
        )


def preencher_reconciliacao(pasta, resultado: dict):
    """Monta uma conferência contábil simples do capital final."""
    aba = pasta["10_Reconciliacao"]
    metricas = resultado["metrics"]

    aba.append(["Item", "Valor"])
    aba.append(["Capital inicial", metricas.get("initial_capital")])
    aba.append(
        [
            "Capital final informado pelo motor",
            metricas.get("strategy_ending_capital"),
        ]
    )
    aba.append(["Taxas totais", metricas.get("total_transaction_fees")])
    aba.append(["Rotações", metricas.get("capital_rotations")])


def estilizar_planilha(
    caminho: Path,
    resultado: dict,
    resumo: str,
    nomes_ativos: list[str],
):
    """Aplica formatação final, fórmulas e elementos visuais."""
    pasta = load_workbook(caminho)

    for aba in pasta.worksheets:
        estilizar_cabecalho(aba)

    for ativo in nomes_ativos:
        if ativo in pasta.sheetnames:
            estilizar_aba_ativo(pasta[ativo])
            estilizar_cabecalho(pasta[ativo])

    guia = pasta["00_Guia"]
    guia.insert_rows(1, 1)
    guia.merge_cells("A1:F1")
    guia["A1"] = "Auditoria do backtest no Excel"
    guia["A1"].fill = PatternFill("solid", fgColor=COR_AZUL_ESCURO)
    guia["A1"].font = Font(color=COR_BRANCA, bold=True, size=15)

    acrescentar_calculos_curva(pasta)
    preencher_indicadores(pasta, resultado)
    preencher_reconciliacao(pasta, resultado)

    aba_json = pasta["09_JSON"]
    aba_json.cell(aba_json.max_row + 2, 1, "Conteúdo de summary.txt")
    aba_json.cell(aba_json.max_row + 1, 1, resumo)

    if hasattr(pasta, "calculation"):
        pasta.calculation.fullCalcOnLoad = True
        pasta.calculation.forceFullCalc = True
        pasta.calculation.calcMode = "auto"

    pasta.save(caminho)


def gerar_planilha():
    """Gera a planilha completa de auditoria do experimento."""
    argumentos = ler_argumentos()
    (
        resultado,
        curva_capital,
        janelas,
        operacoes,
        dados_mercado,
        resumo,
    ) = carregar_dados(argumentos.output_dir)

    ciclos = montar_ciclos(operacoes)
    mensal = montar_mensal(curva_capital)
    resumo_ativos = montar_resumo_ativos(dados_mercado)

    guia = pd.DataFrame(
        [
            [
                "market_data.csv",
                "Snapshot do Yahoo",
                "OHLCV efetivamente usado pelo motor",
                "Sim",
            ],
            [
                "backtest_result.json",
                "Resultado canônico",
                "Métricas e metadados",
                "Parcial",
            ],
            [
                "equity_curve.csv",
                "Curva de capital",
                "Crescimento, Sharpe e queda máxima",
                "Sim",
            ],
            [
                "folds.csv",
                "Janelas temporais",
                "Retorno e capital por janela",
                "Sim",
            ],
            [
                "trades.csv",
                "Livro de operações",
                "Resultado, custos e diagnósticos",
                "Sim",
            ],
            [
                "summary.txt",
                "Resumo textual",
                "Conferência",
                "Não necessário",
            ],
        ],
        columns=["Arquivo", "Papel", "Auditoria", "Usado no Excel"],
    )

    dicionario = pd.DataFrame(
        [
            ["strategy_equity", "Capital da estratégia ao fim da sessão"],
            ["buy_hold_equity", "Capital da referência ao fim da sessão"],
            ["selected_asset", "Ativo selecionado para a sessão"],
            ["effective_switch_margin", "Margem mínima efetiva para rotação"],
            ["current_score", "Pontuação da posição atual"],
            ["best_score", "Pontuação do melhor candidato"],
            ["best_vs_current_gap", "Diferença entre candidato e posição"],
            ["current_asset_rank", "Posição do ativo atual no ranqueamento"],
            ["maximum_favorable_excursion", "Maior excursão favorável"],
            ["maximum_adverse_excursion", "Maior excursão adversa"],
            ["profit_capture_ratio", "Fração do movimento favorável capturada"],
            ["opportunity_cost", "Custo de oportunidade medido após a decisão"],
        ],
        columns=["Campo técnico", "Significado"],
    )

    estrutura_json = pd.DataFrame(
        achatar_estrutura(resultado),
        columns=["Caminho JSON", "Tipo", "Valor"],
    )

    nomes_ativos = list(dict.fromkeys(dados_mercado["symbol"].astype(str)))

    argumentos.destination.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        argumentos.destination,
        engine="openpyxl",
    ) as escritor:
        guia.to_excel(escritor, sheet_name="00_Guia", index=False)
        pd.DataFrame().to_excel(
            escritor,
            sheet_name="01_Indicadores",
            index=False,
        )
        curva_capital.to_excel(escritor, sheet_name="02_Curva", index=False)
        janelas.to_excel(escritor, sheet_name="03_Janelas", index=False)
        operacoes.to_excel(escritor, sheet_name="04_Operacoes", index=False)
        ciclos.to_excel(escritor, sheet_name="05_Ciclos", index=False)
        resumo_ativos.to_excel(escritor, sheet_name="06_Ativos", index=False)
        mensal.to_excel(escritor, sheet_name="07_Mensal", index=False)
        dicionario.to_excel(escritor, sheet_name="08_Dicionario", index=False)
        estrutura_json.to_excel(escritor, sheet_name="09_JSON", index=False)
        pd.DataFrame().to_excel(
            escritor,
            sheet_name="10_Reconciliacao",
            index=False,
        )
        pd.DataFrame({"Ativo": nomes_ativos}).to_excel(
            escritor,
            sheet_name="11_Lista_Ativos",
            index=False,
        )

        for ativo in nomes_ativos:
            preparar_serie_ativo(
                dados_mercado,
                ativo,
            ).to_excel(
                escritor,
                sheet_name=ativo[:31],
                index=False,
            )

    estilizar_planilha(
        argumentos.destination,
        resultado,
        resumo,
        nomes_ativos,
    )

    print(f"Excel gerado: {argumentos.destination}")


if __name__ == "__main__":
    gerar_planilha()
