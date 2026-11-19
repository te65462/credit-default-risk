# Credit card default risk: deployment and CI/CD

[![CI/CD](https://github.com/te65462/credit-default-risk/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/te65462/credit-default-risk/actions/workflows/ci-cd.yml)

**Live demo:** https://huggingface.co/spaces/Te687780/credit-default-risk

This repository deploys the CatBoost model tuned in Module 8 of a machine learning course
(AAI6610) as a free Hugging Face Space, with a GitHub Actions pipeline that tests every change
before it goes live. The model estimates the chance that a credit card client defaults next
month from 23 inputs: credit limit, sex, education, marital status, age, and six months of
repayment status, bill amounts and payments.

- **Model:** CatBoost with 468 trees of depth 8, learning rate 0.03 and L2 penalty 10. A client is
  flagged when the estimated risk is 26.6% or more (the threshold with the best F1 score on
  out-of-fold predictions of the training set).
- **Test results (6,000 clients not used in training):** ROC-AUC .789; at the threshold,
  precision .524, recall .591 and F1 .555.

## How it works

```
GitHub repository ──push──> GitHub Actions ──────────────────────────────> Hugging Face Space
                            1. test:   pytest checks and a quality gate    (static, free)
                            2. deploy: upload space/ (hub-sync action)          │
                            3. verify: live files == this commit                │ page + model
                                                                                ▼
                                                     visitor's browser: Pyodide runs predictor.py
```

- **Free hosting.** As of November 2026, Gradio and Docker Spaces need a paid Hugging Face plan,
  while static Spaces are free for every account. So the page is static: CatBoost exported the trained
  model as plain Python code (`space/model/catboost_model.py`), and [Pyodide](https://pyodide.org)
  runs it, together with `space/predictor.py`, in the visitor's browser. No server receives the
  inputs.
- **Same model as Module 8.** The tests check that the deployed code gives the Module 8 risk score
  for every one of the 6,000 test clients (to within 10⁻¹²), that it matches the original
  `.cbm` file run by CatBoost, and that the test results equal the Module 8 report. A quality
  gate blocks any replacement model with ROC-AUC below .78 or F1 below .54.
- **Checked after deployment.** The last job downloads the files the live Space serves and
  compares them byte for byte with the commit. The page itself also re-checks three reference
  clients every time it loads.

## Repository layout

| Path | Contents |
|---|---|
| `space/` | Everything that is deployed: `index.html`, `style.css`, `app.js`, `predictor.py`, `model/`, and `README.md` (the Space settings). |
| `model_source/catboost_tuned.cbm` | The original Module 8 model file. |
| `tests/` | pytest tests and `data/test_clients.csv` (the 6,000 test clients with their Module 8 risk scores). |
| `tools/export_model.py` | Re-exports the model from the Module 8 files (run locally). |
| `tools/check_live_space.py` | The post-deployment check. |
| `.github/workflows/ci-cd.yml` | The pipeline: test, then deploy, then verify. |

## Run the tests locally

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

To try the page, serve the `space/` folder, for example with `python -m http.server --directory space`,
and open http://localhost:8000.

## One-time setup

1. Create a static Space named `credit-default-risk` on Hugging Face.
2. Create a fine-grained Hugging Face access token with write access to that Space only.
3. Add it to this repository as an Actions secret named `HF_TOKEN`.

## Data and licence

Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H. Licensed under CC BY 4.0. `tests/data/test_clients.csv` is a
subset of this dataset.

A coursework demonstration, not a tool for real credit decisions.
