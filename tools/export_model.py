"""Export the tuned CatBoost model from Module 8 for deployment.

Run once, locally, with the Python that trained the model (Anaconda base, CatBoost 1.2.10):

    python tools/export_model.py ..

The argument is the ml2 project folder. Its files are only read, never changed:
    data/default of credit card clients.xls, data/split_ids.csv     (the data and the fixed split)
    tuning/models/catboost_tuned.cbm, tuning/models/training_summary.json
    tuning/results/test_risk.csv, tuning/results/tuned_vs_original.csv

It writes, inside this repository:
    model_source/catboost_tuned.cbm   an exact copy of the Module 8 model file
    space/model/catboost_model.py     the same model, exported by CatBoost as plain Python code
    space/model/model_info.json       inputs, valid values, training ranges, threshold, test results
    tests/data/test_clients.csv       the 6,000 test clients with their Module 8 risk scores
"""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[1]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
MONTHS = ["September", "August", "July", "June", "May", "April"]  # column 1 = September 2005 ... 6 = April 2005

# Labels follow the UCI data dictionary. Repayment status -1 means paid on time and 1-8 months of
# delay; -2 and 0 occur in the data but are not defined there (as noted in the Module 2 report).
STATUS_CODES = {"-2": "−2 · undocumented code", "-1": "−1 · paid on time", "0": "0 · undocumented code",
                **{str(m): f"{m} · {m} month{'s' if m > 1 else ''} late" for m in range(1, 9)}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_specs(X_train):
    """The 23 inputs in the model's column order: label, group, valid values and training range."""
    def seen(name):
        return [float(X_train[name].min()), float(X_train[name].max())]

    specs = [
        {"name": "LIMIT_BAL", "label": "Credit limit", "unit": "NT$", "group": "client", "kind": "number",
         "valid": {"min": 1}, "training_range": seen("LIMIT_BAL")},
        {"name": "SEX", "label": "Sex", "group": "client", "kind": "category",
         "codes": {"1": "Male", "2": "Female"}},
        # DATA CLEANING (as in training): undocumented codes are merged into "others".
        {"name": "EDUCATION", "label": "Education", "group": "client", "kind": "category",
         "codes": {"1": "Graduate school", "2": "University", "3": "High school", "4": "Others"},
         "merged_codes": {"0": 4, "5": 4, "6": 4}},
        {"name": "MARRIAGE", "label": "Marital status", "group": "client", "kind": "category",
         "codes": {"1": "Married", "2": "Single", "3": "Others"}, "merged_codes": {"0": 3}},
        {"name": "AGE", "label": "Age", "unit": "years", "group": "client", "kind": "number", "whole": True,
         "valid": {"min": 18, "max": 100}, "training_range": seen("AGE")},
    ]
    for i, month in enumerate(MONTHS, start=1):
        name = f"PAY_{i}"
        specs.append({"name": name, "label": f"Repayment status, {month} 2005", "group": "status",
                      "kind": "status", "whole": True, "codes": STATUS_CODES,
                      "training_codes": sorted(int(v) for v in X_train[name].unique())})
    for i, month in enumerate(MONTHS, start=1):
        specs.append({"name": f"BILL_AMT{i}", "label": f"Bill amount, {month} 2005", "unit": "NT$",
                      "group": "bills", "kind": "number", "valid": {}, "training_range": seen(f"BILL_AMT{i}")})
    for i, month in enumerate(MONTHS, start=1):
        specs.append({"name": f"PAY_AMT{i}", "label": f"Amount paid, {month} 2005", "unit": "NT$",
                      "group": "payments", "kind": "number", "valid": {"min": 0},
                      "training_range": seen(f"PAY_AMT{i}")})
    for spec in specs:
        if spec["kind"] in ("category", "status"):
            spec["whole"] = True
            spec.setdefault("valid", {})
    return specs


def results(y, risk, threshold):
    """Counts and scores when a risk of `threshold` or more is flagged (as in the Module 8 notebook)."""
    y, flag = np.asarray(y), np.asarray(risk) >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    return {"threshold": float(threshold), "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ml2", type=Path, help="path to the ml2 project folder")
    ml2 = parser.parse_args().ml2.resolve()

    # Raw data with the Module 8 column names; the cleaned copy is what the model was trained on.
    raw = pd.read_excel(ml2 / "data" / "default of credit card clients.xls", header=1)
    raw = raw.rename(columns={"PAY_0": "PAY_1", "default payment next month": "DEFAULT"}).set_index("ID")
    clean = raw.copy()
    clean["EDUCATION"] = clean["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    clean["MARRIAGE"] = clean["MARRIAGE"].replace({0: 3})
    X, y = clean.drop(columns="DEFAULT"), clean["DEFAULT"]
    split = pd.read_csv(ml2 / "data" / "split_ids.csv", index_col="ID")["split"]
    train_ids, test_ids = split.index[split == "train"], split.index[split == "test"]
    X_train, X_test, y_train, y_test = X.loc[train_ids], X.loc[test_ids], y.loc[train_ids], y.loc[test_ids]

    # 1. The Module 8 model must still give the saved Module 8 test risk scores.
    cbm = ml2 / "tuning" / "models" / "catboost_tuned.cbm"
    model = CatBoostClassifier()
    model.load_model(str(cbm))
    assert list(model.feature_names_) == list(X.columns), "column order differs from the model"
    cbm_risk = model.predict_proba(X_test)[:, 1]
    saved = pd.read_csv(ml2 / "tuning" / "results" / "test_risk.csv", index_col="ID").loc[test_ids, "CB_tuned"]
    gap = float(np.abs(cbm_risk - saved.to_numpy()).max())
    assert gap < 1e-12, gap
    print(f"Module 8 model reproduces the saved test risk scores (largest gap {gap:.1e})")

    # 2. Export it as plain Python. The training pool supplies the category values for the hash table.
    out_py = REPO / "space" / "model" / "catboost_model.py"
    out_py.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_py), format="python", pool=Pool(X_train, y_train, cat_features=CATEGORICAL))
    # Unix line endings, so the file (and its hash in model_info.json) is byte-identical on
    # Windows, in GitHub Actions and on the Space (.gitattributes keeps every text file as LF).
    out_py.write_bytes(out_py.read_bytes().replace(b"\r\n", b"\n"))
    spec = importlib.util.spec_from_file_location("catboost_model", out_py)
    exported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported)
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    py_risk = np.array([1 / (1 + math.exp(-exported.apply_catboost_model(
        [float(v) for v in row[numeric]], [str(int(v)) for v in row[CATEGORICAL]])))
        for _, row in X_test.iterrows()])
    gap = float(np.abs(py_risk - cbm_risk).max())
    assert gap < 1e-12, gap
    print(f"Exported Python model matches the .cbm file on {len(X_test):,} test clients (largest gap {gap:.1e})")

    # 3. Copy the source model, and write the model card used by the web page and the tests.
    (REPO / "model_source").mkdir(exist_ok=True)
    shutil.copy2(cbm, REPO / "model_source" / "catboost_tuned.cbm")
    summary = json.loads((ml2 / "tuning" / "models" / "training_summary.json").read_text())
    table = pd.read_csv(ml2 / "tuning" / "results" / "tuned_vs_original.csv", index_col=0)
    threshold = float(table.loc["CatBoost (tuned), F1 threshold", "threshold"])
    assert round(threshold, 4) == summary["f1_thresholds"]["tuned_catboost"]
    params = model.get_all_params()
    # The Module 8 setting as chosen (CatBoost stores the learning rate as float32: 0.0299999993).
    setting = summary["best_setting"]
    for key in ("learning_rate", "depth", "l2_leaf_reg"):
        assert abs(float(params[key]) - setting[key]) < 1e-6, key
    at_threshold, at_half = results(y_test, cbm_risk, threshold), results(y_test, cbm_risk, 0.5)
    for row, mine in (("CatBoost (tuned), F1 threshold", at_threshold), ("CatBoost (tuned), threshold 0.5", at_half)):
        for key in ("flagged", "TP", "FP", "FN", "TN", "precision", "recall", "f1"):
            assert abs(float(table.loc[row, key]) - mine[key]) < 1e-12, (row, key)
    roc_auc = float(table.loc["CatBoost (tuned), F1 threshold", "roc_auc"])
    average_precision = float(table.loc["CatBoost (tuned), F1 threshold", "average_precision"])

    # Three test clients as worked examples: the ones closest to a risk of 8%, 28% and 70%,
    # among clients whose category codes are all documented (so the form can show them).
    documented = X_test["EDUCATION"].isin([1, 2, 3, 4]) & X_test["MARRIAGE"].isin([1, 2, 3]) \
        & (raw.loc[test_ids, "EDUCATION"] == X_test["EDUCATION"]) & (raw.loc[test_ids, "MARRIAGE"] == X_test["MARRIAGE"])
    risk_s = pd.Series(cbm_risk, index=test_ids)[documented]
    examples = []
    for label, target in (("Lower risk", 0.08), ("Just above the threshold", 0.28), ("Higher risk", 0.70)):
        cid = int((risk_s - target).abs().idxmin())
        examples.append({"label": label, "test_client_id": cid,
                         "values": {c: int(v) for c, v in raw.loc[cid, list(X.columns)].items()},
                         "module8_risk": float(risk_s.loc[cid]), "defaulted": bool(y_test.loc[cid])})

    learn = int(round(summary["train_size"] * (1 - summary["validation_share"])))
    info = {
        "title": "Credit card default risk",
        "threshold": threshold,
        "features": feature_specs(X_train),
        "groups": {"client": "Client", "status": "Repayment status, April–September 2005",
                   "bills": "Bill statement amounts (NT$)", "payments": "Amounts paid (NT$)"},
        "model": {
            "name": "CatBoost, tuned in Module 8", "trees": int(model.tree_count_),
            "trees_built": int(summary["final_trees_built"]), "depth": setting["depth"],
            "learning_rate": setting["learning_rate"], "l2_leaf_reg": setting["l2_leaf_reg"],
            "boosting_type": params["boosting_type"], "grow_policy": params["grow_policy"],
            "learned_from_clients": learn, "early_stopping_clients": summary["train_size"] - learn,
            "early_stopping_patience": summary["patience"], "catboost_version": catboost.__version__,
            "source_file": "model_source/catboost_tuned.cbm", "source_sha256": sha256(cbm),
            "exported_file": "space/model/catboost_model.py", "exported_sha256": sha256(out_py),
        },
        "test_results": {"clients": int(len(y_test)), "defaulters": int(y_test.sum()),
                         "roc_auc": roc_auc, "average_precision": average_precision,
                         "at_threshold": at_threshold, "at_0.5": at_half},
        "data": {"name": "Default of Credit Card Clients", "source": "UCI Machine Learning Repository",
                 "doi": "https://doi.org/10.24432/C55S3H", "clients": int(len(X)),
                 "period": "April–September 2005", "train_clients": int(len(X_train)),
                 "test_clients": int(len(X_test)), "split": "random 80/20, seed 20 (saved client IDs)"},
        "examples": examples,
    }
    (out_py.parent / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8", newline="\n")

    # 4. The test clients with raw values (so the tests also exercise the cleaning) and Module 8 risk.
    fixture = raw.loc[test_ids, list(X.columns) + ["DEFAULT"]].copy()
    fixture["MODULE8_RISK"] = saved.to_numpy()
    (REPO / "tests" / "data").mkdir(parents=True, exist_ok=True)
    fixture.to_csv(REPO / "tests" / "data" / "test_clients.csv", float_format="%.17g", lineterminator="\n")

    print(f"Threshold {threshold:.4f} | test ROC-AUC {roc_auc:.3f} | flagged {at_threshold['flagged']:,} "
          f"(TP {at_threshold['TP']}, FP {at_threshold['FP']}) | examples "
          + ", ".join(f"{e['test_client_id']} ({e['module8_risk']:.1%})" for e in examples))
    print("Training codes seen for repayment status:",
          {f["name"]: f["training_codes"] for f in info["features"] if f["kind"] == "status"})
    for path in (REPO / "model_source" / "catboost_tuned.cbm", out_py, out_py.parent / "model_info.json",
                 REPO / "tests" / "data" / "test_clients.csv"):
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
