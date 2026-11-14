# XAI Assignment 1 — Benjy

Code for exploratory data analysis (Task 1), Random Forest classification with SHAP explanations (Task 2), a local SHAP force plot (Task 3), global SHAP summary plots (Task 4), and an SVM comparison with Kernel SHAP (Task 6).

## Files

- `Task2_Task3_Task4.ipynb`: complete notebook for Tasks 2–4, with all Task 4 code embedded and execution outputs. Includes model training, evaluation, local explanations and global analysis. Task 4 exports top-ten and all-21-feature beeswarm plots, an importance bar chart, the full ranking, feature-value direction checks, local/global comparison and verification metadata under `output/task4/`.
- `Task6_SVM_KernelSHAP.ipynb`: executed SVM evaluation and local/global Kernel SHAP explanations. Uses the shared DataLoader and the saved Random Forest evidence in `reference/`.
- `Task1_EDA.ipynb`: dataset exploration, missing values, class balance, outcome comparisons and correlations. Saves tables and figures under `output/task1/` when run.
- `00_data_exploration.py`: supplied EDA script, retained unchanged from the working project.
- `utils.py`: adapted supplied `DataLoader`. Tasks 2–4 use its split, training-fitted preprocessing and seeded oversampling methods. The original zero-filled `preprocess_data()` is retained for the Task 1 demonstration.
- `requirements.txt`: package versions from the existing Python 3.9.6 XAI environment, including notebook kernel support.
- `data/README.md`: dataset source, expected filename and checksum.

## Setup

Use Python 3.9 (the existing analyses used Python 3.9.6). From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name xai-assignment1 --display-name "XAI Assignment 1"
```

Download the CSV as described in `data/README.md`. Open the notebooks in VS Code or an existing Jupyter installation, select the **XAI Assignment 1** kernel and run each notebook from top to bottom. Jupyter's browser interface is not included in `requirements.txt`; use your existing notebook editor.

To run the supplied EDA script separately:

```bash
python 00_data_exploration.py
```

No arguments are required. The script displays histograms and prints dataset information; it does not save those figures itself. The Task 1 notebook also runs the script and captures its figures and text under `output/task1/source_outputs/`.

## Interpretation and reproducibility

The supplied EDA path `DataLoader.preprocess_data()` replaces missing BMI values with zero. Task 1 retains this behaviour. For Task 2, the adapted loader splits first, fits the BMI median and encoded columns with `preprocess_split()`, then oversamples training records with a fixed seed. These are distinct preprocessing paths.

Task 2 uses random seed 2021 and keeps oversampling within the training set. It reports accuracy, macro F1, per-class precision/recall/F1 and the confusion matrix, and verifies that stroke-class SHAP contributions reconstruct predicted stroke probabilities. The notebook does not currently calculate ROC-AUC; avoid claiming that it does in the report.

The classification target records stroke status. These experiments do not establish prospective stroke prediction performance or causal effects.

## Source and changes

The working source was copied from the local `xai-series-master` folder. Its original GitHub URL, download method and upstream commit were not recorded; the folder name alone does not verify them. Add the original repository URL and applicable licence information when available. No new licence is asserted for the supplied code.

For this submission, notebook dataset paths were adapted to this folder, machine-specific instructions were replaced, saved notebook outputs were cleared and setup documentation was added. The modelling and analysis calculations were retained. Reports, raw data, generated outputs and model binaries are excluded. The small saved comparison files in `reference/` are included because Task 6 needs them.

A commit in this repository identifies this submitted code; it does not identify the original upstream revision. After committing, record its hash with:

```bash
git rev-parse HEAD
```

Record the actual execution date and command in your report. Do not use the commit date as a substitute for the execution date.

## Submission validation

On 4 November 2026, both submission notebooks were executed from top to bottom in an isolated copy using the existing Python 3.9.6 XAI environment and the dataset checksum documented in `data/README.md`. Task 1 produced 21 figure files. Task 2 reproduced the existing confusion matrix `[[960, 12], [47, 3]]` (rows: actual no stroke/stroke; columns: predicted no stroke/stroke), and its SHAP probability reconstruction check passed. This confirms execution in that environment; a fresh installation on another operating system has not been tested. Task 1 retains cleared notebook outputs; see the additional Tasks 2 and 3 validation below.


### Tasks 2 and 3 completion

The combined notebook selects the first false negative in fixed test-set order (test position 32; dataset row index 173). Its stroke-class probability is 0.23. Task 3 reuses the Task 2 model and SHAP arrays and exports a static force plot, interactive HTML, full contribution table, compact report table, prediction summary and original recorded values under `output/task3/`. The interpretation matches the Task 3 reference document.

The combined notebook was executed from top to bottom on 9 November 2026 in the existing Python 3.9.6 XAI environment. All 14 code cells completed, the confusion matrix remained `[[960, 12], [47, 3]]`, the SHAP reconstruction checks passed, and all six Task 3 evidence files were generated. Execution outputs, including the force plot, are saved in the combined notebook.

### Task 4 completion

`Task2_Task3_Task4.ipynb` was executed from top to bottom on 9 November 2026 in the existing Python 3.9.6 XAI environment. All 17 code cells passed after embedding the Task 4 code directly in the notebook. Recomputed SHAP arrays matched the saved Task 2 evidence to absolute tolerance `1e-12`. The test set remains 1,022 records, including 50 stroke records; maximum probability reconstruction error is `3.22e-15`. The top five encoded predictors by mean absolute SHAP are age, average glucose, BMI, hypertension and heart disease.

Task 4 retains all test records and the fitted forest's implicit tree-path background. It explains class `1` probabilities and retains separate indicator columns. The main beeswarm displays ten features; an additional plot displays all 21. CSV exports retain every feature. Narrative interpretation and the comparison with dataset row 173 are included in the notebook. The dot-layout seed is 2021; this does not sample the test set.

To reproduce Tasks 2–4, execute `Task2_Task3_Task4.ipynb` from top to bottom. All Task 4 functions are defined inside the notebook, and no separate Task 4 Python file or parent evidence folder is required.

### Task 2 DataLoader revision

On 9 November 2026, Task 2 was refactored to call the supplied helper structure: `DataLoader.load_dataset()`, `get_data_split(stratify=True, random_state=2021)`, `preprocess_split()` and `oversample(random_state=2021)`. The adapted helper preserves the original Task 1 `preprocess_data()` output.

The complete Tasks 2–4 notebook was rerun. All 17 code cells passed. Training/test frames, labels, encoded order, oversampled rows and sample indices exactly matched the earlier inline code. Maximum differences in the saved model probabilities and SHAP values were both zero. The confusion matrix remains `[[960, 12], [47, 3]]`; all report metrics, the selected case and the local/global explanation values are unchanged. Restart the notebook kernel after changing the helper.

### Task 6 completion

`Task6_SVM_KernelSHAP.ipynb` uses the shared DataLoader with the same seed-2021 stratified split, training-median BMI, encoded columns and training-only oversampling as Task 2. StandardScaler precedes an RBF SVC with C=1.0, gamma="scale", probability=True and random_state=2021. There is no hyperparameter search or separate calibration wrapper. Reported labels use P(stroke = 1) > 0.5, with ties assigned to class 0; 34 native SVC labels differ from that probability rule.

All eight code cells were executed on 9 November 2026 in the existing Python 3.9.6 environment. The SVM confusion matrix was `[[774, 198], [22, 28]]`, versus `[[960, 12], [47, 3]]` for the forest. Stroke recall increased from 6% to 56%, while false positives increased from 12 to 198. SVM ROC AUC was 0.792510 and average precision 0.163409. These scores describe recorded stroke classification, not validated future risk.

Kernel SHAP explains stroke-class probability with an identity link, 50 sampled oversampled-training background records, 512 coalition samples and l1_reg=0. Global explanations cover a fixed stratified sample of 100 test records, including five stroke cases; all 1,022 test records are evaluated. Dataset row 173 is explained locally using the same coalition budget. The forest comparison uses saved Task 2 predictions and SHAP arrays on the identical test sample.

The required files `reference/task2_predictions.csv` and `reference/task2_shap.npz` are included in Git. Their provenance is recorded in `reference/README.md`. Run the notebook from this folder; it exports plots, contribution tables, backgrounds, sampled indices, predictions, SHAP arrays, comparison metrics and `summary.json` under `output/task6/`.

The maximum global Kernel SHAP reconstruction residual was 1.67e-16. Numerical matrix-multiplication warnings occurred, but saved arrays were finite and reconstruction checks passed. A finite residual does not establish attribution stability. A fresh installation on another platform has not been tested.
