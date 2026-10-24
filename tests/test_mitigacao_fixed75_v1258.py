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
    assert 'VERSION = "1.26.0"' in source
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
    assert "auditar_identidade_cusip" in source
    assert "p._generic_identity_issue" not in source
    assert "_verified_unrelated_meta_etf_action" not in source


def test_legacy_reference_is_not_forced_on_new_fixed75_design():
    source, _ = _source_and_tree()
    assert "EXPECTED_OLD_CAPITAL" in source
    assert "legacy_reference_is_fixed75_checkpoint" in source
    assert "math.isclose(capital, EXPECTED_OLD_CAPITAL" not in source
    assert '"mitigacao_overfitting_v1260"' in source



def _real_actions(symbol: str):
    import csv
    path = RUNNER.parent / "dados" / "u67" / "corporate_actions" / f"{symbol}.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_meta_prior_etf_is_foreign_and_fb_rename_is_same_cusip():
    from reproducao.preparacao import auditar_identidade_cusip
    audit = auditar_identidade_cusip("META", _real_actions("META"))
    assert audit["issue"] is None
    assert audit["reference_cusips"] == ["30303M102"]
    assert len(audit["ignored_events"]) == 1
    assert audit["ignored_events"][0]["cusips"] == ["53656F417"]
    assert audit["ignored_events"][0]["reason"] == "other_issuer_cusip"
    assert audit["ticker_changes"][0]["from"] == "FB"
    assert audit["ticker_changes"][0]["to"] == "META"
    assert audit["ticker_changes"][0]["cusip"] == "30303M102"

    # A série FB/META preservada de 2016 não corresponde ao ETF homônimo.
    import csv
    path = RUNNER.parent / "dados" / "u67" / "raw_bars" / "META.csv"
    with path.open(encoding="utf-8", newline="") as file:
        price = next(row for row in csv.DictReader(file)
                     if row["timestamp"].startswith("2022-01-31"))
    assert abs(float(price["close"]) - 313.26) < 1e-8


def test_renomeacao_generica_com_cusip_constante_nunca_bloqueia():
    from reproducao.preparacao import auditar_identidade_cusip
    actions = [
        {"action_type": "cash_dividend", "symbol": "XYZ", "cusip": "123456789"},
        {"action_type": "name_change", "old_symbol": "ABC", "new_symbol": "XYZ",
         "old_cusip": "123456789", "new_cusip": "123456789"},
    ]
    result = auditar_identidade_cusip("XYZ", actions)
    assert result["issue"] is None
    assert result["ignored_events"] == []
    assert result["ticker_changes"][0]["from"] == "ABC"

    # Ausência de dividendos não invalida o identificador de evento de ingresso.
    assert auditar_identidade_cusip("XYZ", actions[1:])["issue"] is None


def test_multiplas_corporate_actions_nao_sao_misturadas_por_ticker():
    from reproducao.preparacao import auditar_identidade_cusip
    actions = [
        {"action_type": "cash_dividend", "symbol": "XYZ", "cusip": "123456789"},
        {"action_type": "name_change", "old_symbol": "XYZ", "new_symbol": "OLD",
         "old_cusip": "987654321", "new_cusip": "987654321"},
        {"action_type": "name_change", "old_symbol": "AAA", "new_symbol": "XYZ",
         "old_cusip": "123456789", "new_cusip": "123456789"},
    ]
    audit = auditar_identidade_cusip("XYZ", actions)
    assert audit["issue"] is None
    assert len(audit["ignored_events"]) == 1
    assert len(audit["ticker_changes"]) == 1
    assert len(audit["relevant_actions"]) == 2

    # Mudança de CUSIP sem evento econômico corroborado continua bloqueada.
    wrong = [{**actions[2], "new_cusip": "111111111"}]
    assert auditar_identidade_cusip("XYZ", actions[:1] + wrong)["issue"] is not None


def test_split_reverso_com_conversao_cusip_documentada_eh_permitido():
    from reproducao.preparacao import auditar_identidade_cusip
    for symbol in ("MUX", "PXLW", "YANG"):
        audit = auditar_identidade_cusip(symbol, _real_actions(symbol))
        assert audit["issue"] is None, (symbol, audit["issue"])
        assert audit["split_cusip_transitions"], symbol


def test_exclusoes_estruturais_antigas_sao_preservadas():
    from reproducao.preparacao import auditar_identidade_cusip
    assert auditar_identidade_cusip("CLMT", _real_actions("CLMT"))["issue"]
    assert auditar_identidade_cusip("DOC", _real_actions("DOC"))["issue"]


def test_reorganizacao_de_xom_nao_eh_confundida_com_mudanca_de_nome():
    from reproducao.preparacao import auditar_identidade_cusip
    result = auditar_identidade_cusip("XOM", _real_actions("XOM"))
    assert result["issue"] is not None
    assert result["issue"]["reason"] in {
        "multiple_unlinked_cusip_identifiers", "non_split_cusip_transition",
    }
    assert set(result["reference_cusips"]) == {"30231G102", "30233Q108"}


def test_todos_os_outros_ativos_base_ficam_livres_de_falsos_bloqueios():
    from engine.configuracao import ASSETS
    from reproducao.preparacao import auditar_identidade_cusip
    for symbol in ASSETS:
        if symbol in {"CLMT", "DOC", "XOM"}:
            continue
        audit = auditar_identidade_cusip(symbol, _real_actions(symbol))
        assert audit["issue"] is None, (symbol, audit["issue"])


def test_auditoria_de_cusip_e_preservacao_de_dados_no_executor():
    source, tree = _source_and_tree()
    assert "identity_audit.json" in source
    assert "auditar_identidade_cusip(symbol, actions)" in source
    assert "rows_used_in_model" in source
    assert "identity_issues" in source
    assert "source_files_modified" in source
    assert "relevant_actions" in source
    assert "_train_candidate_caches" not in source


def test_yang_runtime_loader_matches_real_identity_audit():
    """Previous test used csv.DictReader, unlike the Spyder's pandas path."""
    from reproducao.preparacao import auditar_identidade_cusip, load_actions_file
    path = RUNNER.parent / "dados" / "u67" / "corporate_actions" / "YANG.csv"
    rows = load_actions_file(path)
    audit = auditar_identidade_cusip("YANG", rows)
    assert audit["issue"] is None, audit["issue"]
    assert set(audit["reference_cusips"]) == {"25460E521", "25461A460"}
    assert len(audit["split_cusip_transitions"]) == 1
