"""Uma instalação nova precisa do snapshot U67 completo e estruturalmente válido."""
from pathlib import Path

from reproducao.dados import SnapshotPaths, validate_snapshot


def test_snapshot_oficial_versionado_preserva_contrato_e_arquivos():
    root = Path(__file__).resolve().parents[1]
    manifest = validate_snapshot(SnapshotPaths.u67(root))

    assets = tuple(manifest["assets"])
    assert manifest["schema_version"] == 3
    assert len(assets) == 67
    assert set(manifest["row_counts"]) == set(assets)
    assert set(manifest["corporate_action_counts"]) == set(assets)
