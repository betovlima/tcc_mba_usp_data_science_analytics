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


class NormalizacaoSplitTest(unittest.TestCase):
    def test_compara_ohlcv_normalizado_e_rejeita_desvio(self):
        import sys
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch
        from reproducao.validacao_preditiva import auditar_normalizacao_splits
        preparo = ModuleType("reproducao.preparacao")
        def carregar_barras(path, *, float_precision=None):
            x = pd.read_csv(path)
            x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
            return x.set_index("timestamp")
        def carregar_eventos(path):
            return pd.read_csv(path).to_dict("records")
        def normalizar_barras(raw, actions):
            normalized = raw.copy()
            for a in actions:
                before = normalized.index < pd.Timestamp(a["ex_date"], tz="UTC")
                factor = float(a["old_rate"])/float(a["new_rate"])
                normalized.loc[before,["open","high","low","close"]] *= factor
                normalized.loc[before,"volume"] /= factor
            return normalized, actions
        preparo.load_raw_bar_file = carregar_barras
        preparo.load_actions_file = carregar_eventos
        preparo.split_normalize = normalizar_barras
        with patch.dict(sys.modules, {"reproducao.preparacao": preparo}), tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"raw_bars").mkdir()
            (root/"corporate_actions").mkdir()
            dates = pd.date_range("2026-01-02",periods=6,tz="UTC")
            raw = pd.DataFrame({"timestamp":dates,"open":[10,12,14,8,9,10],
                "high":[11,13,15,9,10,11],"low":[9,11,13,7,8,9],
                "close":[10,12,14,8,9,10],"volume":[100]*6})
            raw.to_csv(root/"raw_bars"/"AAA.csv",index=False)
            pd.DataFrame([{"action_type":"reverse_split",
                "ex_date":"2026-01-05","old_rate":2,"new_rate":1}]).to_csv(
                root/"corporate_actions"/"AAA.csv",index=False)
            frame = raw.set_index("timestamp")
            frame.loc[dates[:3],["open","high","low","close"]] *= 2
            frame.loc[dates[:3],"volume"] /= 2
            paths = SimpleNamespace(raw_bars=root/"raw_bars",
                corporate_actions=root/"corporate_actions")
            flags, _ = auditar_normalizacao_splits(frames={"AAA":frame},
                origem_base=paths,origem_candidatos=paths,candidatos=set())
            self.assertTrue(flags["AAA"])
            frame.loc[dates[0],"close"] += 1
            flags, _ = auditar_normalizacao_splits(frames={"AAA":frame},
                origem_base=paths,origem_candidatos=paths,candidatos=set())
            self.assertFalse(flags["AAA"])

class ExportadorOOSTest(unittest.TestCase):
    def test_exporta_caches_e_marca_alvos_imaturos(self):
        from types import SimpleNamespace
        from reproducao.validacao_preditiva import exportar_previsoes_oos
        dates = pd.date_range("2026-01-01", periods=10, tz="UTC", freq="D")
        state = SimpleNamespace(fold_id=1, decision_dates=dates[3:8],
            decision_cache={t: [0.0, 0.2] for t in dates[3:8]})
        frame_a = pd.DataFrame({"forward_risk_adjusted_utility": 0.3}, index=dates)
        frame_b = pd.DataFrame({"forward_risk_adjusted_utility": 0.1}, index=dates)
        cache_b = SimpleNamespace(decision_cache={t: [0.0, 0.1] for t in dates[3:8]})
        rows = exportar_previsoes_oos(
            states=[state], baseline_frames={"A":frame_a},baseline_symbols=["A"],
            common_dates=dates,horizon_maximo=3,data_corte="2026-01-09",
            candidate_frames={"B":frame_b}, candidate_caches={"B":{1:cache_b}},
            splits_auditados={"A": True, "B": False})
        self.assertEqual(len(rows),8)
        self.assertTrue(rows[rows.ativo=="A"].split_validado.all())
        self.assertFalse(rows[rows.ativo=="B"].split_validado.any())
        self.assertEqual(int(rows.label_maturado.sum()),6)


class CutoffSemNanossegundosTest(unittest.TestCase):
    def test_cutoff_inclusivo_com_calendario_em_microssegundos(self):
        from types import SimpleNamespace
        from reproducao.validacao_preditiva import exportar_previsoes_oos
        dates = pd.date_range("2026-01-05 04:00:00+00:00", periods=5, freq="D").as_unit("us")
        state = SimpleNamespace(
            fold_id=1,
            decision_dates=dates[:4],
            decision_cache={t: [0.0, 0.2] for t in dates[:4]},
        )
        frame = pd.DataFrame(
            {"forward_risk_adjusted_utility": [0.2] * len(dates)},
            index=dates,
        )
        result = exportar_previsoes_oos(
            states=[state],
            baseline_frames={"AAA": frame},
            baseline_symbols=["AAA"],
            common_dates=dates,
            horizon_maximo=1,
            data_corte="2026-01-07",
            splits_auditados=True,
        )
        self.assertEqual(result["label_maturado"].tolist(), [True, True, False])
        self.assertTrue(result["split_validado"].all())

if __name__ == "__main__":
    unittest.main()


class AuditoriaTemporalTest(unittest.TestCase):
    def test_detecta_alvo_que_ultrapassa_a_divisao(self):
        from types import SimpleNamespace
        from reproducao.validacao_preditiva import auditar_limites_temporais
        calendar = pd.date_range("2026-01-01", periods=12, tz="UTC", freq="D")
        state = SimpleNamespace(
            fold_id=1, train_dates=calendar[:5],
            calibration_dates=calendar[5:7], final_fit_dates=calendar[:7],
            decision_dates=calendar[7:],
        )
        result = auditar_limites_temporais(states=[state], common_dates=calendar,
                                          horizon_maximo=3)
        self.assertEqual(list(result["treinos_com_alvo_ainda_futuro"]), [3,3])
        self.assertTrue((result["status"] == "requer_verificacao_purge").all())

if __name__ == "__main__":
    unittest.main()
