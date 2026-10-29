# LinkedIn draft — review before posting

I’m building a hotel booking cancellation project for ISOM 835 at Suffolk University with Team HH.

The task is to predict cancellation probabilities using competition-provided booking data. The project compares logistic regression, random forest and histogram gradient boosting using the same five cross-validation folds, with preprocessing learned only from each training fold.

In the executed local analysis, the selected model achieved mean ROC-AUC 0.9492 (fold standard deviation 0.0013). A separate diagnostic that kept identical booking profiles together achieved mean AUC 0.9416. These are local validation results; no Kaggle score or deployment result is claimed.

One useful finding: agent and company codes behave like category IDs, not numeric amounts. Another lesson is that repeated profiles and the timing of booking fields can change how we interpret validation scores. Country and deposit patterns are associations, not causal explanations.

AI assisted with code, analysis and writing drafts. I’m reviewing the methods and course requirements before final submission and public sharing.

Team credit: Team HH — [ADD CONFIRMED NAMES AND EACH PERSON'S ACTUAL CONTRIBUTION]. My contribution: [DESCRIBE ONLY WORK I PERSONALLY COMPLETED AND CAN EXPLAIN].

Repository: [ADD LINK AFTER PRIVATE UPLOAD, SHARING-RULE REVIEW AND PUBLICATION DECISION]. Competition data and row-level predictions will not be shared.

#Python #MachineLearning #DataAnalytics #SuffolkUniversity

Review note (remove before posting): Professor instructions, official starter, course notebooks, project guide, sample submission and memo rubric are still missing. Do not imply this draft completes all graded requirements. Do not post until teammates approve credit and sharing is permitted.
