"""[TCC-DFL:FIX-005] Auditoria descritiva da cobertura de estados.

SEM treino, sem acesso a precos futuros para decidir, sem tuning em OOS.
Os estados sao pares PARCIAIS (ativo incumbente, holding em sessoes), nao
o estado financeiro completo da carteira.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DiagnosticoCoberturaEstado:
    folds: pd.DataFrame
    ativos: pd.DataFrame
    sessoes: pd.DataFrame
    checks: dict[str, object]


def _datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("Datas vazias em auditoria de estados")
    return parsed


def _fold_rows(df: pd.DataFrame, key: str, fold: int) -> pd.DataFrame:
    return df.loc[pd.to_numeric(df[key], errors="raise") == fold].copy()


def diagnosticar_cobertura_estados(
    labels: pd.DataFrame,
    calibration: pd.DataFrame,
    control_predictions: pd.DataFrame,
    regression_decisions: pd.DataFrame,
) -> DiagnosticoCoberturaEstado:
    """Verifica cobertura parcial de estados usando apenas OOF de TREINO.

    Os rótulos de calibração e OOS são EXCLUÍDOS do conjunto de suporte.
    Control e Regressão OOS são usados exclusivamente como observações
    descritivas; não afetam modelos, guards ou hiperparâmetros.
    """
    required = {
        "fold_id", "state_source", "decision_date", "outcome_end",
        "incumbent", "holding_days", "candidate_position",
        "control_position", "log_advantage",
    }
    if not required.issubset(labels):
        raise ValueError("Colunas ausentes nos rotulos OOF: " + str(sorted(required - set(labels))))
    for frame, required_cols, name in (
        (calibration, {"fold_id", "validation_start"}, "calibracao"),
        (
            control_predictions,
            {"walk_forward_fold", "decision_date", "current_asset",
             "holding_days_at_decision"},
            "Control",
        ),
        (
            regression_decisions,
            {"decision_fold_id", "decision_date", "current_asset",
             "research_holding_days_at_decision"},
            "Regressao",
        ),
    ):
        if not required_cols.issubset(frame):
            raise ValueError(f"Colunas ausentes em {name}: {sorted(required_cols - set(frame))}")

    original = labels.copy()
    original["decision_date"] = _datetime(original["decision_date"])
    original["outcome_end"] = _datetime(original["outcome_end"])
    original["holding_days"] = pd.to_numeric(original["holding_days"], errors="raise")
    calibration = calibration.copy()
    calibration["validation_start"] = _datetime(calibration["validation_start"])
    if calibration["fold_id"].duplicated().any():
        raise ValueError("Calibracao possui folds duplicados")
    fold_ids = sorted(int(x) for x in calibration["fold_id"].unique())
    if set(fold_ids) != set(int(x) for x in original["fold_id"].unique()):
        raise ValueError("Folds de rotulos e calibracao diferentes")
    if not set(original["state_source"].unique()).issubset(
        {"OOF_CONTROL", "OOF_LEARNED_STATE"}
    ):
        raise ValueError("Fonte de estado desconhecida")
    baseline_rows = original.loc[
        original["candidate_position"] == original["control_position"],
        "log_advantage",
    ]
    if baseline_rows.empty or pd.to_numeric(baseline_rows, errors="raise").abs().max() > 1e-9:
        raise AssertionError("Label de acao-base nao e vantagem zero")

    fold_records, asset_records, session_records = [], [], []
    for fold in fold_ids:
        subset = _fold_rows(original, "fold_id", fold)
        cut = calibration.loc[
            calibration["fold_id"] == fold, "validation_start"
        ].iloc[0]
        train = subset.loc[
            (subset["decision_date"] < cut)
            & (subset["outcome_end"] < cut)
        ].copy()
        if train.empty:
            raise ValueError(f"Fold {fold} sem labels seguros de treinamento")
        if not train["outcome_end"].lt(cut).all():
            raise AssertionError("Leakage: desfecho alcanca calibracao")
        if train["holding_days"].isna().any():
            raise ValueError("Holding ausente nos labels OOF")
        # Multiplicidade de candidatos NAO deve ser confundida com
        # novas datas/estados estatisticamente independentes.
        unique_states = train[
            ["decision_date", "state_source", "incumbent", "holding_days"]
        ].drop_duplicates()
        known_assets = set(train["incumbent"].astype(str))
        known_pairs = set(zip(
            train["incumbent"].astype(str),
            train["holding_days"].astype(int),
        ))
        learned_pairs = set(zip(
            train.loc[train["state_source"] == "OOF_LEARNED_STATE", "incumbent"].astype(str),
            train.loc[train["state_source"] == "OOF_LEARNED_STATE", "holding_days"].astype(int),
        ))
        initial_summary = {
            "fold_id": fold,
            "validation_start": cut,
            "training_rows": len(train),
            "training_unique_dates": int(train["decision_date"].nunique()),
            "training_unique_state_groups": len(unique_states),
            "training_unique_incumbents": len(known_assets),
            "training_unique_asset_hold_pairs": len(known_pairs),
            "training_learned_rows": int((train["state_source"] == "OOF_LEARNED_STATE").sum()),
            "training_learned_unique_dates": int(
                train.loc[train["state_source"] == "OOF_LEARNED_STATE", "decision_date"].nunique()
            ),
            "training_max_outcome": train["outcome_end"].max(),
        }

        for policy, data, fold_col, hold_col in (
            (
                "CONTROL", control_predictions, "walk_forward_fold",
                "holding_days_at_decision",
            ),
            (
                "REGRESSION", regression_decisions, "decision_fold_id",
                "research_holding_days_at_decision",
            ),
        ):
            obs = _fold_rows(data, fold_col, fold)
            if obs.empty:
                raise ValueError(f"Fold {fold} {policy} sem sessoes")
            obs["decision_date"] = _datetime(obs["decision_date"])
            if obs["decision_date"].duplicated().any():
                raise AssertionError(f"Fold {fold} {policy} com decisoes duplicadas")
            if obs["current_asset"].isna().any() or obs[hold_col].isna().any():
                raise ValueError(f"Fold {fold} {policy} sem estado atual")
            holdings = pd.to_numeric(obs[hold_col], errors="raise")
            if (holdings < 0).any() or (holdings != holdings.astype(int)).any():
                raise ValueError("Holding OOS invalido")
            obs["holding_days"] = holdings.astype(int)
            obs["current_asset"] = obs["current_asset"].astype(str)
            obs = obs.sort_values("decision_date").reset_index(drop=True)
            obs["seen_incumbent_in_train"] = obs["current_asset"].isin(known_assets)
            obs["seen_pair_in_train"] = [
                (asset, holding) in known_pairs
                for asset, holding in zip(obs["current_asset"], obs["holding_days"])
            ]
            obs["seen_pair_in_learned_train"] = [
                (asset, holding) in learned_pairs
                for asset, holding in zip(obs["current_asset"], obs["holding_days"])
            ]
            obs["unseen_incumbent_in_train"] = ~obs["seen_incumbent_in_train"]
            obs["seen_asset_unseen_hold"] = (
                obs["seen_incumbent_in_train"] & ~obs["seen_pair_in_train"]
            )
            fold_records.append({
                **initial_summary,
                "policy": policy,
                "oos_decisions": len(obs),
                "oos_unique_incumbents": int(obs["current_asset"].nunique()),
                "oos_unseen_incumbent_decisions": int(obs["unseen_incumbent_in_train"].sum()),
                "oos_unseen_incumbent_fraction": float(obs["unseen_incumbent_in_train"].mean()),
                "oos_seen_asset_unseen_hold_decisions": int(obs["seen_asset_unseen_hold"].sum()),
                "oos_unseen_pair_decisions": int((~obs["seen_pair_in_train"]).sum()),
                "oos_unseen_pair_fraction": float((~obs["seen_pair_in_train"]).mean()),
                "oos_seen_learned_pair_decisions": int(obs["seen_pair_in_learned_train"].sum()),
                "training_max_hold": int(train["holding_days"].max()),
                "oos_max_hold": int(obs["holding_days"].max()),
            })
            for asset, part in obs.groupby("current_asset", sort=True):
                asset_records.append({
                    "fold_id": fold,
                    "policy": policy,
                    "asset": asset,
                    "oos_decisions": len(part),
                    "train_has_incumbent": bool(asset in known_assets),
                    "oos_seen_pair_decisions": int(part["seen_pair_in_train"].sum()),
                    "oos_hold_min": int(part["holding_days"].min()),
                    "oos_hold_max": int(part["holding_days"].max()),
                    "train_state_rows": int((train["incumbent"] == asset).sum()),
                    "train_unique_state_dates": int(
                        train.loc[train["incumbent"] == asset, "decision_date"].nunique()
                    ),
                })
            obs.insert(0, "fold_id", fold)
            obs.insert(1, "policy", policy)
            session_records.append(obs[[
                "fold_id", "policy", "decision_date", "current_asset",
                "holding_days", "seen_incumbent_in_train", "seen_pair_in_train",
                "seen_pair_in_learned_train", "unseen_incumbent_in_train",
                "seen_asset_unseen_hold",
            ]])

    sessions = pd.concat(session_records, ignore_index=True)
    # A mesma janela OOS e exigida; jamais comparar dois conjuntos de datas.
    for fold in fold_ids:
        a = sessions.loc[
            (sessions["fold_id"] == fold) & (sessions["policy"] == "CONTROL"),
            "decision_date",
        ].reset_index(drop=True)
        b = sessions.loc[
            (sessions["fold_id"] == fold) & (sessions["policy"] == "REGRESSION"),
            "decision_date",
        ].reset_index(drop=True)
        if not a.equals(b):
            raise AssertionError(f"Fold {fold}: calendarios OOS nao coincidem")

    counts = pd.DataFrame(fold_records)
    return DiagnosticoCoberturaEstado(
        folds=counts,
        ativos=pd.DataFrame(asset_records),
        sessoes=sessions,
        checks={
            "schema_version": 1,
            "method": "train-only OOF partial state support; descriptive OOS coverage",
            "no_model_training": True,
            "control_regression_dates_identical": True,
            "training_labels_total": int(counts.loc[counts.policy == "CONTROL", "training_rows"].sum()),
            "oos_sessions_per_policy": int(counts.loc[counts.policy == "CONTROL", "oos_decisions"].sum()),
            "fold_count": len(fold_ids),
            "state_definition": "(incumbent_asset, holding_sessions), NOT complete portfolio state",
            "no_confirmatory_claim": True,
        },
    )
