import unittest
import pandas as pd
from reproducao.validacao_preditiva import avaliar_previsoes_oos

class DiagnosticosOOSTest(unittest.TestCase):
    def setUp(self):
        self.dados = pd.DataFrame([
            {"data_decisao":"2026-01-01T00:00:00Z", "ativo":s, "fold":1,
             "previsao":p, "observado":y, "label_maturado":True,
             "split_validado":True}
            for s, p, y in [("AAA",0.3,0.4),("BBB",0.1,0.2),("CCC",-0.1,-0.2)]
        ])
    def test_medidas_e_ranking(self):
        report, folds, ranks, assets=avaliar_previsoes_oos(self.dados)
        self.assertEqual(report["status"],"completed")
        self.assertEqual(report["valid_rows"],3)
        self.assertAlmostEqual(report["overall"]["mae"], 0.1)
        self.assertAlmostEqual(ranks.iloc[0]["spearman"],1.0)
        self.assertAlmostEqual(ranks.iloc[0]["ndcg_5"],1.0)
    def test_sem_auditoria_split_nao_fabrica_metrica(self):
        self.dados["split_validado"] = False
        report, *_=avaliar_previsoes_oos(self.dados)
        self.assertEqual(report["status"],"insufficient_evidence")
        self.assertEqual(report["valid_rows"],0)
    def test_sem_maturidade_nao_fabrica_metrica(self):
        self.dados["label_maturado"] = False
        report, *_=avaliar_previsoes_oos(self.dados)
        self.assertEqual(report["valid_rows"],0)
    def test_duplicacao_aborta(self):
        with self.assertRaises(ValueError):
            avaliar_previsoes_oos(pd.concat([self.dados,self.dados]))
    def test_colunas_ausentes_aborta(self):
        with self.assertRaises(ValueError):
            avaliar_previsoes_oos(self.dados.drop(columns="observado"))

if __name__ == "__main__":
    unittest.main()
