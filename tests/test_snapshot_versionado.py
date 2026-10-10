"""Uma instalacao nova precisa dos mesmos dados usados na medicao cientifica."""
from pathlib import Path

from reproducao.dados import SnapshotPaths, validate_snapshot


def test_snapshot_oficial_versionado_preserva_identidade_e_134_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = validate_snapshot(SnapshotPaths.u67(root))
    assert manifest["snapshot_sha256"] == (
        "e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575"
    )
    assert len(manifest["file_hashes"]) == 134
