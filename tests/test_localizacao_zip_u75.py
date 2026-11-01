"""Testa seleção do ZIP MCT sem dados de mercado e sem download."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pesquisa.localizacao_zip_u75 import locate_mct_zip


class LocalizacaoZipU75Test(unittest.TestCase):
    def test_quatro_zips_em_output_nao_geram_ambiguidade(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            nested = root / 'output' / 'backtests'
            nested.mkdir(parents=True)
            for n in range(3):
                (root / 'output' / f'market_cycle_trader_2026102{n}T000000-outro.zip').write_bytes(b'outro' + bytes([n]))
            name = 'market_cycle_trader_20261101T021336-83e49685.zip'
            ref = nested / name
            ref.write_bytes(b'backtest congelado')
            chosen = locate_mct_zip(root, reference_name=name,
                                    reference_sha256=sha256(ref.read_bytes()).hexdigest())
            self.assertEqual(chosen, ref)

    def test_nao_escolhe_mais_recente_quando_referencia_ausente(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / 'output').mkdir()
            (root / 'output' / 'market_cycle_trader_novo.zip').write_bytes(b'novo')
            with self.assertRaisesRegex(FileNotFoundError, 'referência'):
                locate_mct_zip(root)

    def test_rejeita_mesmo_nome_com_hash_diferente(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / 'output').mkdir()
            (root / 'output' / 'market_cycle_trader_20261101T021336-83e49685.zip').write_bytes(b'troca indevida')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 divergente'):
                locate_mct_zip(root)

    def test_override_explicito_e_suportado(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            expected = root / 'output' / 'nova' / 'outra_execucao.zip'
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b'execucao nova')
            chosen = locate_mct_zip(root, explicit='output/nova/outra_execucao.zip')
            self.assertEqual(chosen, expected)

    def test_override_ausente_falha_em_vez_de_selecionar_outro(self):
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, 'MCT_BACKTEST_ZIP'):
                locate_mct_zip(Path(d), explicit='output/inexistente.zip')


    def test_copia_zip_validado_da_raiz_para_output(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            source = root / "market_cycle_trader_referencia_da_raiz.zip"
            source.write_bytes(b"conteudo de backtest verificado")
            original_digest = sha256(source.read_bytes()).hexdigest()
            (root / "output").mkdir()
            selected = locate_mct_zip(
                root, reference_name=source.name, reference_sha256=original_digest
            )
            self.assertEqual(
                selected,
                root / "output" / "busca_externa_lider_u75" / "referencia_mct" / source.name,
            )
            self.assertEqual(sha256(selected.read_bytes()).hexdigest(), original_digest)
            self.assertEqual(source.read_bytes(), b"conteudo de backtest verificado")
            self.assertEqual(
                locate_mct_zip(
                    root, reference_name=source.name,
                    reference_sha256=original_digest
                ),
                selected,
            )

    def test_copia_zip_de_evidencias_sem_mover_original(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            evidence = root / "evidencias"
            evidence.mkdir()
            source = evidence / "market_cycle_trader_referencia_evidencias.zip"
            source.write_bytes(b"arquivo validado em evidencias")
            selected = locate_mct_zip(
                root, reference_name=source.name,
                reference_sha256=sha256(source.read_bytes()).hexdigest(),
            )
            self.assertTrue(selected.is_file())
            self.assertTrue(source.is_file())
            self.assertEqual(selected.read_bytes(), source.read_bytes())

    def test_nao_importa_zip_com_mesmo_nome_sem_hash_identico(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            root.mkdir(exist_ok=True)
            (root / "output").mkdir()
            source = root / "market_cycle_trader_referencia_invalida.zip"
            source.write_bytes(b"zip adulterado")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 divergente"):
                locate_mct_zip(
                    root, reference_name=source.name,
                    reference_sha256=sha256(b"conteudo original").hexdigest(),
                )



if __name__ == '__main__':
    unittest.main()
