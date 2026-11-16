"""Reproducao do universo configurado em ASSETS, em um unico replay.

Quando a atualizacao RAW esta ativa, usa somente o universo atual da configuracao,
coleta historico completo SIP/1Day e eventos corporativos e executa os tres
folds juntos. Nao existem ativos ou modelos complementares, excecoes de
cardinalidade nem novas estrategias. O checkpoint original permanece intacto.

Spyder: abra este arquivo e execute com F5.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import warnings
from zipfile import ZipFile

import pandas as pd

from reproducao.experimento import reproduzir_universo_congelado

_PANDAS4_WARNING = getattr(pd.errors, "Pandas4Warning", None)
if _PANDAS4_WARNING is not None:
    warnings.filterwarnings("ignore", category=_PANDAS4_WARNING)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "reproducao_universo"
REPRODUCTION_VERSION = "1.23.0+dynamic-universe.test"
EXECUTION_SCHEMA = "universe-reproduction-v4"

# Uma unica entrada, mesmo arquivo e mesmo ZIP da reproducao anterior.
ATUALIZAR_SERIES_RAW = True
DATA_FINAL = None  # None = ultima sessao XNYS segura; ou "AAAA-MM-DD".


def _publicar_diretorio(novo: Path, atual: Path, anterior: Path) -> None:
    """Troca diretorio preparado; preserva anterior para eventual rollback."""
    if anterior.exists():
        raise RuntimeError(
            f"Transicao anterior pendente: {anterior}. "
            "Resolva antes de executar novamente."
        )
    if atual.exists():
        atual.replace(anterior)
    try:
        novo.replace(atual)
    except Exception:
        if anterior.exists() and not atual.exists():
            anterior.replace(atual)
        raise


def main() -> dict:
    """Executa os ativos de ASSETS em conjunto, sem etapas complementares."""
    if not ATUALIZAR_SERIES_RAW:
        return reproduzir_universo_congelado(
            root=ROOT,
            output=OUT,
            version=REPRODUCTION_VERSION,
            execution_schema=EXECUTION_SCHEMA,
        )

    from engine.configuracao import ASSETS
    from reproducao.artefatos import criar_pacote_analise
    from reproducao.dados import (
        SnapshotPaths, auditar_revisoes_raw,
        preparar_snapshot_raw_atualizado, ultimo_pregao_sip_completo,
    )

    cutoff_safe = ultimo_pregao_sip_completo()
    cutoff = str(DATA_FINAL or cutoff_safe)
    if DATA_FINAL is not None:
        import exchange_calendars as xcals

        try:
            session = xcals.get_calendar("XNYS").date_to_session(
                pd.Timestamp(cutoff), direction="previous"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"DATA_FINAL invalida: {cutoff}") from exc
        if session.date().isoformat() != cutoff:
            raise ValueError(f"DATA_FINAL={cutoff} nao e um pregao XNYS.")
    if cutoff > cutoff_safe:
        raise ValueError(
            f"Data {cutoff} sem candle SIP completo; ultima segura={cutoff_safe}"
        )

    dados_atuais = SnapshotPaths.from_root(ROOT / "dados" / "universo_77")
    dados_temporarios = SnapshotPaths.from_root(
        ROOT / "dados" / ".universo_atualizacao_em_execucao"
    )
    saida_temporaria = OUT.with_name(".reproducao_universo_em_execucao")
    backup_dados = ROOT / "dados" / ".universo_atualizacao_anterior_transicao"
    backup_saida = OUT.with_name(".reproducao_universo_anterior_transicao")
    if backup_dados.exists() or backup_saida.exists():
        raise RuntimeError(
            "Uma transicao anterior ainda existe. Preserve seus dados e "
            "resolva a transicao antes de executar novamente."
        )
    for caminho in (dados_temporarios.root, saida_temporaria):
        if caminho.exists():
            shutil.rmtree(caminho)

    try:
        print(
            f"[RAW/SIP] ativos={len(ASSETS)} cutoff={cutoff} "
            "coleta=integral execucao=unica folds=3", flush=True,
        )
        preparar_snapshot_raw_atualizado(
            ROOT, dados_temporarios, tuple(ASSETS), cutoff,
        )
        resultado = reproduzir_universo_congelado(
            root=ROOT,
            output=saida_temporaria,
            version=REPRODUCTION_VERSION,
            execution_schema=EXECUTION_SCHEMA,
            analysis_end_date=cutoff,
            full_refresh_performed=True,
            updated_snapshot_path=dados_temporarios.root,
        )
        universo = resultado.get("universe") or {}
        fit_rows = (resultado.get("calibration") or {}).get("fit_counts") or []
        if (
            tuple(universo.get("requested_assets") or ()) != tuple(ASSETS)
            or set(universo.get("effective_assets") or ()) != set(ASSETS)
            or set(universo.get("trained_assets") or ()) != set(ASSETS)
            or universo.get("excluded_assets")
            or len(fit_rows) != 6
            or any(row.get("model_count") != len(ASSETS) for row in fit_rows)
        ):
            raise RuntimeError(
                "Replay incompleto ou com treinamento fora do unico universo ASSETS."
            )

        fontes = (
            dados_atuais,
            SnapshotPaths.archived_source(ROOT),
            SnapshotPaths.archived_extension(ROOT),
        )
        drift = auditar_revisoes_raw(
            dados_temporarios, fontes, tuple(ASSETS),
        )
        drift.to_csv(
            saida_temporaria / "auditoria_revisoes_raw.csv",
            index=False, float_format="%.17g",
        )
        archive = criar_pacote_analise(
            saida_temporaria,
            comparison_file="reproducao_universo.json",
            execution_schema=EXECUTION_SCHEMA,
            archive_name="pacote_reproducao_universo.zip",
        )
        with ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError("Falha de integridade do ZIP da reproducao.")
        # Somente o replay e o snapshot inteiros podem substituir os anteriores.
        _publicar_diretorio(dados_temporarios.root, dados_atuais.root, backup_dados)
        try:
            _publicar_diretorio(saida_temporaria, OUT, backup_saida)
        except Exception:
            if dados_atuais.root.exists():
                shutil.rmtree(dados_atuais.root)
            if backup_dados.exists():
                backup_dados.replace(dados_atuais.root)
            raise

        for caminho in (backup_dados, backup_saida):
            if caminho.exists():
                shutil.rmtree(caminho)
        print(
            f"[CONCLUIDO] universo={len(ASSETS)} "
            f"capital=US$ {resultado['metrics']['ending_capital']:,.2f} "
            f"zip={OUT / 'pacote_reproducao_universo.zip'}", flush=True,
        )
        return resultado
    finally:
        for caminho in (dados_temporarios.root, saida_temporaria):
            if caminho.exists():
                shutil.rmtree(caminho)


if __name__ == "__main__":
    main()
