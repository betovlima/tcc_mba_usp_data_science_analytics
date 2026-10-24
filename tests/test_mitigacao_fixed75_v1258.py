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
    assert 'VERSION = "1.26.2"' in source
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
    assert '"mitigacao_overfitting_v1262"' in source



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


def test_xom_conversao_um_para_um_documentada_na_sec():
    from reproducao.preparacao import auditar_identidade_cusip
    result = auditar_identidade_cusip("XOM", _real_actions("XOM"))
    assert result["issue"] is None
    assert set(result["reference_cusips"]) == {"30231G102", "30233Q108"}
    assert len(result["successor_cusip_transitions"]) == 1
    conversion = result["successor_cusip_transitions"][0]
    assert conversion["exchange_ratio"] == 1.0
    assert conversion["process_date"] == "2026-07-02"
    assert conversion["effective_date"] == "2026-07-01"
    assert conversion["price_series_modified"] is False


def test_todos_os_outros_ativos_base_ficam_livres_de_falsos_bloqueios():
    from engine.configuracao import ASSETS
    from reproducao.preparacao import auditar_identidade_cusip
    for symbol in ASSETS:
        if symbol in {"CLMT", "DOC"}:
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

def test_identifiers_are_read_as_strings_not_infinite_floats(tmp_path):
    from reproducao.preparacao import load_actions_file
    path = tmp_path / "corporate_actions.csv"
    path.write_text(
        "action_type,cusip,old_cusip,new_cusip,symbol,old_rate,new_rate\n"
        "reverse_split,,25460E521,25461A460,YANG,20,1\n"
        "cash_dividend,000123456,,,YANG,,\n",
        encoding="utf-8",
    )
    rows = load_actions_file(path)
    assert rows[0]["old_cusip"] == "25460E521"
    assert rows[0]["new_cusip"] == "25461A460"
    assert rows[1]["cusip"] == "000123456"
    assert rows[0]["old_rate"] == 20.0


def test_xom_successor_eh_estritamente_documentado():
    from reproducao.preparacao import auditar_identidade_cusip
    actions = _real_actions("XOM")
    good = auditar_identidade_cusip("XOM", actions)
    assert good["issue"] is None
    name_change = next(x for x in actions if x["action_type"] == "name_change")
    mutated = [{**x, "new_cusip": "999999999"}
               if x is name_change else x for x in actions]
    bad = auditar_identidade_cusip("XOM", mutated)
    assert bad["issue"] is not None


def test_mchb_simbolo_reutilizado_nao_eh_unificado_por_ticker():
    from reproducao.preparacao import auditar_identidade_cusip
    # MCHB antigo e HMST->MCHB novo são emissores diferentes.
    # A Nasdaq DN2025-25 proíbe carregar o histórico do ticker antigo.
    rows = [
        {"action_type": "cash_dividend", "symbol": "MCHB", "cusip": "111111111"},
        {"action_type": "name_change", "old_symbol": "HMST",
         "new_symbol": "MCHB", "old_cusip": "43785V102",
         "new_cusip": "43785V102", "process_date": "2025-09-02"},
    ]
    audit = auditar_identidade_cusip("MCHB", rows)
    assert audit["issue"] is not None
    assert audit["issue"]["reason"] == "multiple_unlinked_cusip_identifiers"

def test_mchb_identity_from_frozen_cusip_events_is_hmst_not_old_otc():
    """Regression for the actual 15-event frozen cache without uploading it."""
    from reproducao.preparacao import auditar_identidade_cusip
    events = [
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "583614102", "rate": 2569.0,
         "ex_date": "2021-07-30"},
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "583614102", "rate": 467.0,
         "ex_date": "2024-09-09"},
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "43785V102", "rate": 0.21,
         "ex_date": "2025-12-08"},
        {"action_type": "name_change", "old_symbol": "HMST",
         "new_symbol": "MCHB", "old_cusip": "43785V102",
         "new_cusip": "43785V102", "process_date": "2025-09-02"},
        {"action_type": "stock_merger", "acquiree_symbol": "MCHB",
         "acquirer_symbol": "MCHB", "acquiree_cusip": "583614102",
         "acquirer_cusip": "43785V102", "process_date": "2025-09-02",
         "acquiree_rate": 1, "acquirer_rate": 3301.092},
    ]
    audit = auditar_identidade_cusip("MCHB", events)
    assert audit["issue"] is None
    assert audit["reference_cusips"] == ["43785V102"]
    assert audit["identity_anchor_cusips"] == ["43785V102"]
    assert audit["excluded_foreign_cash_dividends"] == 2
    assert len(audit["ignored_events"]) == 2
    assert all(event["cusips"] == ["583614102"]
               for event in audit["ignored_events"])
    assert len(audit["relevant_actions"]) == 3
    assert audit["ticker_changes"][0]["from"] == "HMST"
    assert audit["ticker_changes"][0]["to"] == "MCHB"
    assert not audit["split_cusip_transitions"]
    assert not audit["successor_cusip_transitions"]


def test_mchb_ticker_only_dividends_cannot_prove_continuity():
    from reproducao.preparacao import auditar_identidade_cusip
    # Without an issuer-anchoring event, old/new dividends must not be
    # assumed to describe the same security merely because ticker matches.
    events = [
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "583614102"},
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "43785V102"},
    ]
    result = auditar_identidade_cusip("MCHB", events)
    assert result["issue"] is not None
    assert result["issue"]["reason"] == "multiple_unlinked_cusip_identifiers"


def test_mchb_wrong_receiver_does_not_anchor_old_dividends_to_new_issuer():
    from reproducao.preparacao import auditar_identidade_cusip
    events = [
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "583614102"},
        {"action_type": "cash_dividend", "symbol": "MCHB",
         "cusip": "43785V102"},
        {"action_type": "name_change", "old_symbol": "HMST",
         "new_symbol": "OTHER", "old_cusip": "43785V102",
         "new_cusip": "43785V102"},
    ]
    result = auditar_identidade_cusip("MCHB", events)
    assert result["issue"] is not None


def test_mchb_audit_fields_surface_on_spyder_path():
    source, _ = _source_and_tree()
    assert '"identity_anchor_cusips"' in source
    assert '"excluded_foreign_cash_dividends"' in source
    assert '"blocking_event_candidates"' in source
    assert 'split_normalize(raw, audit["relevant_actions"])' in source
