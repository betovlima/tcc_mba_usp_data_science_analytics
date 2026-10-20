"""Contrato da estrutura limpa: somente os dois executáveis aprovados."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_somente_dois_runners_ativos_na_raiz() -> None:
    found = {p.name for p in ROOT.glob("*.py")}
    assert found == {"buscar_ativos.py", "reproduzir_experimento.py"}


def test_reproducao_e_busca_nao_compartilham_ciclo_de_download() -> None:
    runner = (ROOT / "reproduzir_experimento.py").read_text(
        encoding="utf-8"
    )
    discovery = (ROOT / "buscar_ativos.py").read_text(
        encoding="utf-8"
    )
    assert "executar_control" in runner
    implementation = (ROOT / "reproducao" / "control.py").read_text(
        encoding="utf-8"
    )
    assert "SnapshotPaths.referencia" in implementation
    assert "load_alpaca_credentials" not in implementation
    assert "download_raw_bars" not in implementation
    assert "load_alpaca_credentials" in discovery
    assert "read_snapshot_manifest(REFERENCE_DATA)" in discovery
    assert "EXPECTED_BRANCH" not in discovery
    assert "if __name__ == \"__main__\":" in discovery


def test_regras_aprovadas_ficam_no_motor_congelado() -> None:
    config = (ROOT / "engine" / "configuracao.py").read_text(
        encoding="utf-8"
    )
    rotation = (ROOT / "engine" / "rotacao.py").read_text(
        encoding="utf-8"
    )
    assert "rotation_min_holding_days: int = 2" in config
    assert '"MIN_HOLD_GUARD"' in rotation
    assert '"SWITCH_MARGIN_GUARD"' in rotation


def test_snapshot_e_evidencias_sao_preservados() -> None:
    assert (ROOT / "dados" / "referencia" / "manifest.json").is_file()
    assert (ROOT / "evidencias" / "eligibilidade_v1.22.0-dev.10.json").is_file()
    assert (ROOT / "figuras" / "procedencia_figuras.json").is_file()


def test_busca_avisos_sonoros_diferenciam_sucesso_de_erro(monkeypatch) -> None:
    import importlib.util
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(
        sys.modules, "winsound",
        SimpleNamespace(Beep=lambda frequency, duration: calls.append(
            (frequency, duration)
        )),
    )
    spec = importlib.util.spec_from_file_location(
        "buscar_ativos", ROOT / "buscar_ativos.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.sinal_sonoro_busca(sucesso=True)
    assert calls == [(880, 160), (1046, 220)]
    calls.clear()
    module.sinal_sonoro_busca(sucesso=False)
    assert calls == [(440, 420), (330, 470)]


def test_selecao_independente_de_cardinalidade(tmp_path, monkeypatch) -> None:
    import json
    import reproducao.control as control

    names = ["AAA", "BBB", "CCC"]
    selection = tmp_path / "selecao.json"
    selection.write_text(
        json.dumps({"selected_candidates": names, "reference_capital_usd": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(control, "SELECTION_FILE", selection)
    selected, reference_capital = control.carregar_selecao()
    assert selected == names
    assert reference_capital is None


def test_codigos_ativos_nao_exigem_identificador_de_cardinalidade() -> None:
    for relative in (
        "engine/configuracao.py",
        "reproducao/control.py",
        "reproducao/preparar_snapshot.py",
        "reproducao/dados.py",
        "buscar_ativos.py",
        "reproduzir_experimento.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8").lower()
        assert "u67" not in source, relative
        assert "u77" not in source, relative
