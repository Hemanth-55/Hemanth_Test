# What changed the results

All main experiments used identical five-fold stratified validation (shuffle=True, random_state=835). These are local ROC-AUC results, not Kaggle results or percent accuracy.

| Experiment | Mean local AUC | Fold SD | Runtime (seconds) |
|---|---:|---:|---:|
| 01_provisional_lr_numeric_ids | 0.890032 | 0.002123 | 7.7 |
| 02_lr_categorical_ids | 0.907403 | 0.001840 | 9.3 |
| 03_lr_features | 0.907966 | 0.001981 | 10.6 |
| 04_lr_features_C01 | 0.907139 | 0.002257 | 7.5 |
| 05_rf_leaf5 | 0.944152 | 0.001230 | 250.6 |
| 06_hgb_leaf15 | 0.945110 | 0.001321 | 29.3 |
| 07_hgb_leaf31 | 0.949233 | 0.001322 | 27.9 |
| 08_hgb_conservative | 0.921987 | 0.001812 | 18.0 |
| 09_hgb_no_country | 0.907966 | 0.002306 | 17.1 |
| 10_hgb_no_deposit | 0.943411 | 0.001305 | 21.0 |


- Treating agent/company as categories instead of numeric amounts changed logistic mean AUC by +0.017370.
- Adding four simple row features changed logistic mean AUC by +0.000564. This is a small difference relative to fold variation; it is not proof of a reliable improvement in future data.
- Changing logistic C from 1 to 0.1 changed mean AUC by -0.000827. A smaller C means stronger regularization.
- Increasing boosting leaves from 15 to 31 changed mean AUC by +0.004123.
- Dropping the four timing-uncertain fields from the 15-leaf boosting reference changed mean AUC by -0.023124.
- Removing country from the same boosting reference changed mean AUC by -0.037145.
- Removing deposit type from the same reference changed mean AUC by -0.001699.

Selected candidate: **07_hgb_leaf31**, mean AUC **0.949233**, fold SD **0.001322**. Selection used the first seven predefined candidate experiments; the three feature ablations are diagnostics, not extra winner candidates. Settings: `{"engineered": true, "kind": "hgb", "max_leaf_nodes": 31}`.

The selected model's duplicate-profile grouped diagnostic scored **0.941616** mean AUC, compared with **0.949233** on random stratified folds. This difference reflects a different validation design, not a paired isolated model comparison. Grouping exact profiles still cannot identify the same guest or hotel booking across imperfectly matching records.

Pooled out-of-fold AUC is 0.949209; it differs from the average of fold AUCs because it pools scores across separately trained models. Brier score is 0.087170; smaller is better. The ROC and calibration charts use held-out predictions. Descriptive country/deposit charts use labeled training rows and do not claim causality.

Model selection reused validation folds and introduces optimism; no nested CV or untouched holdout score is claimed. The official starter, prediction-time definitions and course-method permissions remain unverified. No public or private Kaggle AUC is available.
