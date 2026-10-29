# Submission handoff

Team name: Team HH. File-safe spelling: TeamHH_submission.csv. Actual predictions stay in private/ and are excluded from GitHub, even for a private repository. Do not share competition data publicly.

1. Obtain missing professor documents, official starter and sample template. Confirm allowed methods, academic integrity and sharing rules. The current candidate is provisional.
2. Run the Colab notebook. If sample_submission.csv exists, its columns and ID set are checked; output remains in the original test row order.
3. Inspect private/submission_validation.json. It records 35,817 rows, two columns (id,is_canceled), unique complete test IDs in test order, finite probabilities in [0,1] and file SHA256. No extra index is written.
4. Upload the chosen CSV manually to Kaggle. Do not exceed five submissions per day. Log the exact filename, hash, time and actual public score in docs/submission_log.csv. Keep public and private Kaggle AUC separate from local CV AUC.
5. First submission deadline: November 2, 2026. Final: November 9, 2026, 4:30 PM. Dates come from the user; confirm timezone and precise cutoff against professor instructions. This project uses America/New_York for user-facing schedules.
6. After the LAST Kaggle upload, preserve that exact CSV unchanged for Canvas. Compare SHA256 of both copies; matching filenames alone are insufficient. Do not rerun and overwrite the final file after submission.

No Kaggle or Canvas submissions have been made by this project. Probabilities are not binary labels. ROC-AUC measures ranking across thresholds; it is not percent accuracy. A model may rank well while its probabilities need calibration.
