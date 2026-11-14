# Task 2 comparison evidence

These two files are unchanged copies from `../../output/task2_report_support/`:

- `task2_predictions.csv` was `test_predictions.csv`: the Random Forest's 1,022 test labels and probabilities, with original dataset indices.
- `task2_shap.npz` was `stroke_shap.npz`: stroke-class Tree SHAP values, baselines, feature names and dataset indices.

The revised Task 6 notebook checks the test order and labels before comparison. It selects the Random Forest explanations for exactly the same 100 sampled test records used in the SVM global explanation. The earlier Random Forest is not retrained or changed.
