# Team HH — Hotel Booking Cancellation

An ISOM 835 competition project at Suffolk University, prepared as a beginner-friendly portfolio draft. Predict a cancellation probability for each booking using only competition-provided data.

**Status:** runnable provisional workflow with executed local results. Official starter, professor instructions, course notebooks, project guide, sample submission and memo rubric are still needed. This is not a claim that every assignment requirement is complete.

## Verified local results

Selected model: **07_hgb_leaf31**. Five-fold stratified local ROC-AUC: **0.949233 ± 0.001322** (fold standard deviation, not a confidence interval). Grouped duplicate-profile diagnostic mean AUC: **0.941616**. Public Kaggle AUC: not submitted. Private Kaggle AUC: unavailable. ROC-AUC describes ranking and must not be called percent accuracy.

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


## Data and methods

Training: 83,573 bookings, including is_canceled. Test: 35,817 bookings with hidden outcomes. Cancellation rate in training: 37.04%. 21,391 training rows repeat an earlier predictor profile after excluding ID. Data and row-level outputs are deliberately not included in this repository.

Compare provisional logistic regression, improved categorical-ID logistic regression, random forest and histogram gradient boosting. Use identical stratified folds (5 splits, shuffle=True, random_state=835), fold-fitted preprocessing, train-only rare-category handling and four simple row-level features. A small predefined search tests two logistic regularization strengths and two boosting leaf counts. Country, deposit type and timing-uncertain fields get separate diagnostic ablations. Winner chosen using local CV, never public leaderboard tuning.

[Open notebook in Google Colab](https://colab.research.google.com/github/Hemanth-55/Hemanth_Test/blob/team-hh-hotel-cancellation/hotel-cancellation-portfolio/notebooks/TeamHH_Hotel_Cancellation.ipynb)

## Run in Google Colab

1. Open notebooks/TeamHH_Hotel_Cancellation.ipynb in Colab.
2. Select Runtime → Run all. Upload your own train.csv and test.csv, plus sample_submission.csv when available.
3. Read the audit, model comparison and charts. The notebook fits 55 fold models and one final model; runtime depends on the machine.
4. The checked TeamHH_submission.csv is written to private/. Change DOWNLOAD_SUBMISSION to True to download it in Colab. Never commit predictions or data.

The notebook embeds the analysis code, so it does not need a GitHub checkout to run. It has numbered code cells and short explanations. It was tested locally with the recorded library versions; an actual hosted Colab run is a separate check.

## Run locally

Python 3.12 was used for recorded results. From the project folder:

```bash
python -m pip install -r requirements.txt
# Place competition CSVs in data/ first.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python src/workflow.py --data-dir data --root . --team TeamHH
python src/build_writeups.py
```

## What to read

- docs/requirements_checklist.md: completed work and exact missing materials.
- docs/leakage_audit.md: prediction-time uncertainty, duplicate profiles and leakage controls.
- reports/interpretation.md: what helped and what did not, with actual score differences.
- reports/fold_scores.csv: every main fold score, settings, runtime, reason and warnings.
- reports/grouped_sensitivity.csv: separate grouped-fold diagnostic.
- reports/results.json: model settings, data hashes, package version and unverified items.
- docs/memo_draft.pdf and memo_draft.md: one-page memo draft; rubric not yet supplied.
- docs/presentation_outline.md and learning_notes.md: three-minute talk and plain-English explanations.
- docs/linkedin_draft.md: review-only post with verified results and credit placeholders.
- docs/submission_instructions.md: deadlines, submission limits and file identity checks.
- docs/github_handoff.md: private-repository upload instructions and current access blocker.

## Limitations

The main candidate assumes provided features describe a usable pre-outcome snapshot. Their initial-booking availability is not confirmed. History variables need strict as-of timestamps. The conservative ablation cannot establish deployment readiness. Country/deposit associations are not causal and raise interpretation/fairness concerns. Random folds may be optimistic due to repeated profiles; grouped sensitivity is reported separately. Selecting on the same CV folds adds selection optimism. No future-time, real hotel deployment, business savings or guaranteed competition score is claimed.

## Team contributions and AI disclosure

Team: **Team HH**. Individual names, roles and contributions must be supplied by the team before sharing. Do not represent the prepared code or analysis as independently authored. AI assisted with data auditing, Python code, experiment design, interpretation and writing drafts. The student/team must review the code, understand the methods, confirm course permissions and record their own changes, decisions and final submission responsibilities. Only claim personal work actually performed.

## Sharing and submission

This project is prepared for review. Portfolio destination: [https://github.com/Hemanth-55/Hemanth_Test/tree/team-hh-hotel-cancellation/hotel-cancellation-portfolio](https://github.com/Hemanth-55/Hemanth_Test/tree/team-hh-hotel-cancellation/hotel-cancellation-portfolio), a dedicated folder on the team-hh-hotel-cancellation branch of the existing public Hemanth-55/Hemanth_Test repository. The GitHub integration denied new-repository creation, so the existing connected repository is used. Competition data, per-booking predictions and credentials are excluded. Confirm course sharing rules before redistributing this work. No LinkedIn post, Kaggle upload, Canvas upload or deployment has been made. The competition data, predictions, OOF arrays, credentials and original uploaded materials remain excluded. No data license or permission to redistribute is asserted. Do not apply a license to the competition data.
