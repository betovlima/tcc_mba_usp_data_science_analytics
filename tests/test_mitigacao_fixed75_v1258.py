"""Guarda estrutural para impedir reintrodução de treino 65+10 na pesquisa.

Teste AST estático, não reproduz o backtest e não acessa dados ou rede.
"""
from __future__ import annotations

import ast
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "pesquisar_mitigacao_overfitting_spyder.py"


def _source_and_tree():
    source = RUNNER.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(RUNNER))


def _methods(function: ast.AST) -> list[str]:
    return [
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def test_painel_unico_de_75_antes_dos_folds():
    source, tree = _source_and_tree()
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert 'VERSION = "1.25.8"' in source
    assert 'raw_75 = {}' in source
    assert 'universe_role="fixed75"' in source
    assert 'len(symbols) != 75' in source
    methods = _methods(functions["prepare_once"])
    assert "_construir_contexto_execucao" in methods
    assert "_train_baseline_states" not in methods


def test_uma_unica_rotina_de_treino_calibracao_e_replay_para_todos():
    source, tree = _source_and_tree()
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "main")
    methods = _methods(main)
    assert methods.count("_train_baseline_states") == 1
    assert methods.count("_simular_exato") == 1
    assert "_train_candidate_caches" not in source
    assert "_evaluate_scenario" not in source
    assert "legacy_65_plus_10_training" in source


def test_o_controle_ainda_usa_parada_se_checkpoint_nao_coincidir():
    source, _ = _source_and_tree()
    assert "EXPECTED_OLD_CAPITAL" in source
    assert "math.isclose(capital, EXPECTED_OLD_CAPITAL" in source
    assert "raise RuntimeError" in source
    assert "output/mitigacao_overfitting_v1258" in source
