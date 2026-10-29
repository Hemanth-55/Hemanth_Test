# Explain it in class

- A row is one booking. The target is 1 if canceled and 0 otherwise.
- Logistic regression produces a probability from a weighted combination of inputs. It is a useful, understandable reference.
- A random forest averages many decision trees. It can learn nonlinear patterns and interactions.
- Histogram gradient boosting builds small trees sequentially to correct earlier errors. The histogram makes training efficient.
- Cross-validation splits the training bookings into five parts. Train on four and predict the held-out part; repeat until each row has a prediction made without training on that row.
- ROC-AUC asks how often a randomly selected canceled booking is ranked above a randomly selected non-canceled booking. AUC 0.5 is random ranking; it is never called percent accuracy.
- A pipeline fits cleaning steps on training rows only. Fitting medians or category frequencies on validation rows would contaminate validation.
- Agent 240 is an identifier, not 24 times agent 10. Categorical encoding represents separate identities. Rare groups reduce unstable estimates for scarcely observed IDs.
- The standard deviation describes variation across these folds. The folds share training data, so it is not a confidence interval or proof of significance.
- A calibration chart compares predicted probabilities with actual cancellation fractions. AUC and calibration answer different questions.
- Trying many settings and choosing the best on the same folds makes the best reported score somewhat optimistic. This project limits the search and reports all experiments.
- Duplicate profiles make random validation easier. Grouped validation asks whether similar performance holds when an identical profile cannot cross folds.
- Association is not causation. Deposit type can be related to customer mix, hotel policy or unobserved timing. Country is not a causal explanation.

Before presenting: explain each feature, read the actual results table, review the code you submit, and confirm your course's AI and collaboration rules. Do not claim that AI-authored work is independently authored.

The added features are: total_nights = weekend nights + weekday nights; total_guests = adults + children + babies; has_children = whether children or babies are present; weekend_share = weekend nights / total nights. For zero-night stays, weekend_share is missing and is imputed using the training fold. Missing children is not silently assumed to mean no children. These features do not learn any cross-row statistics.
