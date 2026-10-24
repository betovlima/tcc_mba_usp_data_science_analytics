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
    assert 'VERSION = "1.25.9"' in source
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


def test_legacy_reference_is_not_forced_on_new_fixed75_design():
    source, _ = _source_and_tree()
    assert "EXPECTED_OLD_CAPITAL" in source
    assert "legacy_reference_is_fixed75_checkpoint" in source
    assert "math.isclose(capital, EXPECTED_OLD_CAPITAL" not in source
    assert '"mitigacao_overfitting_v1259"' in source


def _load_identity_predicate():
    """Executa somente a função pura de identidade, sem treinar modelos."""
    _, tree = _source_and_tree()
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_verified_unrelated_meta_etf_action"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[func], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace[func.name]


def _meta_real_actions():
    import csv
    path = RUNNER.parent / "dados" / "u67" / "corporate_actions" / "META.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_evento_etf_homonimo_e_comprovado_por_cusip_e_data_no_snapshot():
    import csv
    validate = _load_identity_predicate()
    actions = _meta_real_actions()
    etf = next(
        a for a in actions
        if a["action_type"] == "name_change"
        and a["old_symbol"] == "META" and a["new_symbol"] == "METV"
    )
    issuer = next(
        a for a in actions
        if a["action_type"] == "name_change"
        and a["old_symbol"] == "FB" and a["new_symbol"] == "META"
    )
    assert etf["old_cusip"] == etf["new_cusip"] == "53656F417"
    assert issuer["old_cusip"] == issuer["new_cusip"] == "30303M102"
    assert validate("META", etf, actions) is True

    # Evidência de que as barras congeladas são FB/Meta Platforms, não ETF.
    prices = RUNNER.parent / "dados" / "u67" / "raw_bars" / "META.csv"
    with prices.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    end_january = next(row for row in rows
                       if row["timestamp"].startswith("2022-01-31"))
    assert abs(float(end_january["close"]) - 313.26) < 1e-8


def test_outros_eventos_estruturais_sao_rejeitados():
    validate = _load_identity_predicate()
    actions = _meta_real_actions()
    etf = next(a for a in actions
               if a["old_symbol"] == "META" and a["new_symbol"] == "METV")
    assert validate("OTHER", etf, actions) is False
    assert validate("META", {**etf, "new_cusip": "OTHER"}, actions) is False
    assert validate("META", {**etf, "new_symbol": "XYZ"}, actions) is False
    assert validate("META", {**etf, "process_date": "2022-02-01"}, actions) is False
    assert validate("META", etf, [etf]) is False
    assert validate("CLMT", etf, actions) is False


def test_identity_audit_not_silently_ignored_or_rewritten():
    source, tree = _source_and_tree()
    assert "identity_audit.json" in source
    assert "Ativo estruturalmente inconsistente" in source
    assert "price_series_modified" in source
    assert '"data_files_modified": False' in source
    assert "_train_candidate_caches" not in source
