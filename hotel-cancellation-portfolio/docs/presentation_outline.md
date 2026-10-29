# Three-minute presentation outline — Team HH

**0:00–0:25 — Problem.** “We predict the chance that a hotel booking will be canceled. We have 83,573 labeled bookings and 35,817 hidden-outcome bookings. We evaluate ranking with ROC-AUC, not percent accuracy.”

**0:25–0:55 — Audit and leakage.** Show reports/audit.json and mention the 37.04% cancellation rate. “IDs are excluded. Duplicate profiles can make random validation optimistic. Some booking fields may change after booking, so their timing needs confirmation.”

**0:55–1:25 — Fair comparison.** Show reports/model_comparison.png. Explain five folds in one sentence: “Each booking is evaluated by a model that did not train on that row.” Mention fold-fitted preprocessing, categorical agent/company IDs and the limited predefined search.

**1:25–2:00 — Actual findings.** “Categorical IDs changed logistic AUC from 0.8900 to 0.9074. Our selected 07_hgb_leaf31 scored 0.9492 mean local AUC with fold SD 0.0013. Grouped-profile validation gave 0.9416; this is a different test of generalization.” Read reports/interpretation.md for features and settings that did not help.

**2:00–2:30 — Meaning and limits.** Show validation_roc.png or validation_calibration.png. “Good ranking does not guarantee perfectly calibrated probabilities. Deposit/country patterns are associations, not causal claims. There are no Kaggle or deployment results yet.”

**2:30–3:00 — Handoff and credit.** “We checked the 35,817 probabilities, IDs, columns and file hash. Next we must review the official starter, course methods, sample and memo rubric, then submit the exact same final file to Kaggle and Canvas. AI assisted the draft; team members will explain their verified contributions.” Add confirmed teammate names and roles before speaking.
