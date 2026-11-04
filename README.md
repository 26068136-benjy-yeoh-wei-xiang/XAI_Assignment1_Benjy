# XAI Assignment 1 — Benjy

Code for exploratory data analysis (Task 1) and Random Forest classification with SHAP explanations (Task 2).

## Files

- `Task1_EDA.ipynb`: dataset exploration, missing values, class balance, outcome comparisons and correlations. Saves tables and figures under `output/task1/` when run.
- `Task2.ipynb`: stratified train/test split, training-only BMI median imputation, categorical encoding, training-only random oversampling, Random Forest evaluation and stroke-class SHAP explanations.
- `00_data_exploration.py`: supplied EDA script, retained unchanged from the working project.
- `utils.py`: supplied data-loading and preprocessing helper, retained unchanged from the working project.
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

The supplied `utils.py` replaces missing BMI values with zero. Task 1 retains this behaviour when demonstrating the supplied script. Task 2 uses a BMI median learned from the training set instead. These are distinct preprocessing paths.

Task 2 uses random seed 2021 and keeps oversampling within the training set. It reports accuracy, macro F1, per-class precision/recall/F1 and the confusion matrix, and verifies that stroke-class SHAP contributions reconstruct predicted stroke probabilities. The notebook does not currently calculate ROC-AUC; avoid claiming that it does in the report.

The classification target records stroke status. These experiments do not establish prospective stroke prediction performance or causal effects.

## Source and changes

The working source was copied from the local `xai-series-master` folder. Its original GitHub URL, download method and upstream commit were not recorded; the folder name alone does not verify them. Add the original repository URL and applicable licence information when available. No new licence is asserted for the supplied code.

For this submission, notebook dataset paths were adapted to this folder, machine-specific instructions were replaced, saved notebook outputs were cleared and setup documentation was added. The modelling and analysis calculations were retained. Reports, raw data, generated outputs and model binaries are excluded.

A commit in this repository identifies this submitted code; it does not identify the original upstream revision. After committing, record its hash with:

```bash
git rev-parse HEAD
```

Record the actual execution date and command in your report. Do not use the commit date as a substitute for the execution date.

## Submission validation

On 4 November 2026, both submission notebooks were executed from top to bottom in an isolated copy using the existing Python 3.9.6 XAI environment and the dataset checksum documented in `data/README.md`. Task 1 produced 21 figure files. Task 2 reproduced the existing confusion matrix `[[960, 12], [47, 3]]` (rows: actual no stroke/stroke; columns: predicted no stroke/stroke), and its SHAP probability reconstruction check passed. This confirms execution in that environment; a fresh installation on another operating system has not been tested. Saved notebook outputs remain cleared in the submission.
