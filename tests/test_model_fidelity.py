"""The deployed model must be the Module 8 model: the same risk scores and the same test results."""
import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import predictor
from conftest import REPO, inputs_of

# Quality gate: the lowest test results a model may have and still be deployed. They sit just
# below the Module 8 results (ROC-AUC .789, F1 .555), so a clearly worse replacement model fails.
MIN_ROC_AUC = 0.78
MIN_F1_AT_THRESHOLD = 0.54


def outcomes(test_clients):
    return np.array([int(row["DEFAULT"]) for row in test_clients])


def test_risk_scores_match_module8(test_clients, deployed_risk):
    """predictor.py (cleaning + exported model) gives the Module 8 risk score for all 6,000 clients."""
    module8 = np.array([float(row["MODULE8_RISK"]) for row in test_clients])
    assert len(module8) == 6000
    assert np.abs(np.array(deployed_risk) - module8).max() < 1e-12


def test_exported_code_matches_catboost_file(test_clients, deployed_risk):
    """The plain-Python export gives the same scores as the original .cbm file run by CatBoost."""
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostClassifier()
    model.load_model(str(REPO / "model_source" / "catboost_tuned.cbm"))
    X = pd.DataFrame([predictor.check(inputs_of(row))[0] for row in test_clients])[predictor.NAMES]
    X[predictor.CATEGORY_NAMES] = X[predictor.CATEGORY_NAMES].astype(int)
    assert list(model.feature_names_) == predictor.NAMES
    assert model.tree_count_ == predictor.INFO["model"]["trees"]
    assert np.abs(model.predict_proba(X)[:, 1] - np.array(deployed_risk)).max() < 1e-12


@pytest.mark.parametrize("which", ["at_threshold", "at_0.5"])
def test_test_results_match_module8_report(test_clients, deployed_risk, which):
    """Confusion counts and scores on the test set equal the Module 8 results in model_info.json."""
    saved = predictor.INFO["test_results"][which]
    y, flagged = outcomes(test_clients), np.array(deployed_risk) >= saved["threshold"]
    counts = {"TP": int((flagged & (y == 1)).sum()), "FP": int((flagged & (y == 0)).sum()),
              "FN": int((~flagged & (y == 1)).sum()), "TN": int((~flagged & (y == 0)).sum())}
    assert counts == {k: saved[k] for k in counts}
    assert counts["TP"] + counts["FP"] == saved["flagged"]


def test_ranking_scores_match_module8(test_clients, deployed_risk):
    saved = predictor.INFO["test_results"]
    y = outcomes(test_clients)
    assert roc_auc_score(y, deployed_risk) == pytest.approx(saved["roc_auc"], abs=1e-12)
    assert average_precision_score(y, deployed_risk) == pytest.approx(saved["average_precision"], abs=1e-12)
    assert round(saved["roc_auc"], 3) == 0.789 and round(saved["average_precision"], 3) == 0.568


def test_quality_gate(test_clients, deployed_risk):
    """Deployment is blocked if the model falls below the minimum test results."""
    y, flagged = outcomes(test_clients), np.array(deployed_risk) >= predictor.THRESHOLD
    tp, fp, fn = (flagged & (y == 1)).sum(), (flagged & (y == 0)).sum(), (~flagged & (y == 1)).sum()
    assert roc_auc_score(y, deployed_risk) >= MIN_ROC_AUC
    assert 2 * tp / (2 * tp + fp + fn) >= MIN_F1_AT_THRESHOLD


def test_model_files_are_the_exported_pair():
    """model_info.json records the hashes of the source model and of its plain-Python export."""
    info = predictor.INFO["model"]
    for key, path in (("source_sha256", info["source_file"]), ("exported_sha256", info["exported_file"])):
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == info[key], path


def test_examples_give_their_module8_risk():
    for example in predictor.INFO["examples"]:
        assert predictor.predict(example["values"])["risk"] == pytest.approx(example["module8_risk"], abs=1e-12)
