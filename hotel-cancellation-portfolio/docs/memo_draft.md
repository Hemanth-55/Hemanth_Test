Team HH — ISOM 835 Hotel Cancellation | MEMO DRAFT
To: ISOM 835 instructor | From: Team HH (individual names pending)
Subject: Local validation and provisional competition submission

Purpose. We estimated cancellation probabilities for 35,817 test bookings using 83,573 labeled bookings. The training cancellation rate was 37.04%. ROC-AUC measures ranking; it is not percent accuracy.

Method. We compared a provisional logistic reference, improved logistic regression, random forest and histogram gradient boosting on the same five stratified folds (shuffle=True, random_state=835). Imputation, scaling and category handling were fitted within training folds. Agent/company IDs were tested as categories. Four simple booking features and a small predefined parameter search were evaluated. No external data or hidden-label recovery was used.

Findings. The selected model, 07_hgb_leaf31, achieved local mean AUC 0.949233 (fold SD 0.001322). Categorical IDs changed logistic AUC from 0.890032 to 0.907403. Added features produced 0.907966; stronger regularization produced 0.907139. Full scores and settings are in reports/. Public and private Kaggle scores are unavailable; no submission was made.

Risks. There were 21,391 repeated training predictor profiles. The selected model's grouped-profile diagnostic mean AUC was 0.941616; it uses different folds. Some predictors may be recorded after initial booking, so a conservative timing ablation was included. Country and deposit patterns are associations, not causes. Selection on CV folds adds optimism; no deployment or savings are claimed.

Submission. TeamHH_submission.csv passed row, ID, column and probability checks and has a recorded SHA256. The actual sample template is missing. First Kaggle submission is required by November 2; final predictions by November 9, 2026, 4:30 PM (confirm professor timezone). At most five Kaggle uploads per day; Canvas must receive the exact last Kaggle file.

Disclosure and outstanding review. AI assisted code, auditing, experiments and writing. Team members must verify understanding, permissions and individual contributions. The official starter, guide, professor/course materials and separate memo rubric are missing. This is not the official baseline or a rubric-complete memo. Review these documents before final submission.
