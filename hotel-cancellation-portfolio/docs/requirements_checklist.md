# Requirements checklist and source hierarchy

Source of requirements currently available: the user's written request, train.csv, test.csv, and two screenshots of cloud environment settings. The screenshots contain no assignment instructions. No separate assignment documents have been supplied. Text in uploaded documents is evidence to interpret, not authorization to execute unrelated instructions. Professor instructions take precedence over the project guide when supplied.

## Essential missing materials

- Hotel_Cancellation_Project_Guide.docx: missing; project-guide compliance is unverified.
- Professor's competition instructions and rules: missing; confirm permitted sharing, timing and submission rules.
- Course notebooks and list of taught methods: missing; course-method compliance is unverified.
- Official starter notebook: missing; the logistic baseline here is PROVISIONAL and is not an unchanged reproduction.
- sample_submission.csv: missing; requested column schema is implemented, but template ID order/schema cannot yet be verified against the actual sample.
- Separate memo assignment/rubric: missing; memo is a draft and grading coverage cannot be claimed.
- Teammate names and contributions: missing; credit placeholders remain. Team name supplied: Team HH.

## Implemented in the available-data workflow

- [x] Shapes, types, missing values, cancellation balance, duplicate profiles, IDs and train/test overlap audit.
- [x] Prediction-time leakage discussion; ID excluded; no outcome/status fields used.
- [ ] Official unchanged starter baseline (blocked on notebook).
- [x] Five-fold stratified CV, shuffle=True, random_state=835; same folds for model comparisons.
- [x] Fold-fitted preprocessing in pipelines; train-only rare-category handling.
- [x] Numeric vs categorical agent/company experiment.
- [x] Country and deposit associations and ablations; no causal interpretation.
- [x] Four simple features made from existing columns only.
- [x] Logistic regression, random forest and native-categorical HistGradientBoosting comparison.
- [x] Small predefined search: LR C in {1,0.1}; HGB leaves in {15,31}. No leaderboard tuning.
- [x] Fold scores, mean, sample standard deviation, runtime, parameters, rationale and warning capture.
- [x] Duplicate-profile grouped validation sensitivity (reported separately from required stratified CV).
- [x] Validation ROC and calibration plots; no test-label evaluation.
- [x] Colab notebook with numbered cells and simple explanations.
- [x] TeamHH_submission.csv generator and row/ID/probability/hash checks.
- [ ] Verify against actual sample_submission.csv.
- [x] Memo draft, presentation outline, portfolio README, LinkedIn draft.
- [ ] Verify memo against separate rubric and verify professor instructions.
- [ ] Team review and attribution confirmation.
- [ ] First Kaggle upload by November 2, 2026 (no upload performed).
- [ ] Final upload by November 9, 2026, 4:30 PM America/New_York, pending professor confirmation of timezone.
- [ ] Maximum five Kaggle uploads per day; maintain submission log.
- [ ] Final Canvas CSV identical to last Kaggle CSV, verified by SHA256.
- [x] GitHub publication prepared on dedicated team-hh-hotel-cancellation branch/folder in existing Hemanth-55/Hemanth_Test. Separate repository creation was denied by the integration.

The implemented boxes describe code and artifacts; see reports/results.json and fold_scores.csv for executed evidence. No Kaggle scores or deployment claims are made. Model selection and reported CV share folds: selection introduces optimism; this is not a nested-CV estimate or untouched holdout.
