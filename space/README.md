---
title: Credit Card Default Risk
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: CatBoost credit default risk that runs in your browser
---

# Credit card default risk

Estimates the chance that a credit card client defaults next month (misses the next payment),
using the CatBoost model tuned in Module 8 of a machine learning course. The model runs
entirely in the visitor's browser (Python through [Pyodide](https://pyodide.org)), so no server
receives the inputs.

- **Model:** CatBoost with 468 trees of depth 8, learning rate 0.03 and L2 penalty 10. A client is
  flagged when the estimated risk is 26.6% or more.
- **Test results (6,000 clients not used in training):** ROC-AUC .789; at the threshold,
  precision .524, recall .591 and F1 .555.
- **Deployment:** GitHub Actions updates this Space automatically after the tests pass.
  Source code and pipeline: https://github.com/te65462/credit-default-risk
- **Data:** Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0).

A coursework demonstration, not a tool for real credit decisions.
