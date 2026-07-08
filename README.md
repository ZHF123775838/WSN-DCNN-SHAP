# Network Traffic Anomaly Detection with SHAP and DCNN+SA

This repository provides the source code, data split, and reproduction workflow for the manuscript
**"An efficient method for network traffic anomaly detection based on SHAP and deep learning"**.
The target task is binary DoS anomaly detection on the UNSW-NB15 split included in this repository.

## Method Overview

- Fixed train/test split:
  - `trainset.csv`: 56,000 normal and 12,264 DoS samples
  - `testset.csv`: 37,000 normal and 4,089 DoS samples
- Preprocessing without test leakage:
  - missing-value imputation
  - numeric winsorization using training-set quantiles
  - Min-Max normalization fitted on training data only
  - one-hot encoding for categorical variables
- SHAP feature selection:
  - tree model trained on the training split
  - TreeExplainer SHAP values computed on training data only
  - global importance = mean absolute SHAP value
  - selected subset = minimum ranked features reaching 95% cumulative importance
- DCNN+SA classifier:
  - two 1D convolutional layers
  - self-attention block replacing max pooling
  - three fully connected layers including the binary output layer
  - Adam optimizer, learning rate 0.0005, batch size 128, 50 epochs
  - class-weighted binary cross entropy, early stopping, ReduceLROnPlateau
- Baselines:
  - SVM, Random Forest, and XGBoost using the same selected features

## Install

```bash
pip install -r requirements.txt
```

The original manuscript environment was Python 3.8.13, TensorFlow 2.9.1, and scikit-learn 1.2.0.

## Reproduction Workflow

### 1. Run SHAP Feature Selection

```bash
python shap_sort.py
```

Outputs are written to `artifacts/shap/`:

- `feature_ranking.csv`
- `selected_features_95.json`
- `shap_feature_importance.png`
- `cumulative_importance.png`

To also export the 90%, 95%, and 98% threshold subsets used for sensitivity analysis:

```bash
python sort_test.py
```

### 2. Train the DCNN+SA Model

```bash
python test.py
```

Outputs are written to `artifacts/dcnn_sa/`:

- `metrics.json`
- `predictions.csv`
- `history.csv`
- `dcnn_sa.keras`

For the DCNN ablation without self-attention:

```bash
python test.py --no-attention --output-dir artifacts/dcnn
```

### 3. Train Baselines

```bash
python baselines.py
```

Outputs are written to `artifacts/baselines/`.

For a quick validation run of the baseline code paths:

```bash
python baselines.py --selected-features artifacts/shap/selected_features_95.json --max-train-samples 2000
```

### 4. Recalculate Metrics

```bash
python result_calc.py --predictions artifacts/dcnn_sa/predictions.csv
```

You can also calculate metrics directly from a confusion matrix:

```bash
python result_calc.py --confusion TN FP FN TP
```

## Optional Preprocessed NPZ Export

The main workflow reads the committed CSV files directly. If NPZ files are needed for additional experiments:

```bash
python prepare_data.py
```

## Expected Results

The manuscript reports the following representative held-out test results for the SHAP-selected models:

| Method | Class | Recall | Precision | F1-score | Accuracy | Specificity | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM-SHAP | Normal | 98.32% | 99.26% | 98.78% | 97.82% | 93.32% | 88.38% |
| SVM-SHAP | DoS | 93.32% | 85.98% | 89.50% | 97.82% | 98.32% | 88.38% |
| XGBoost-SHAP | Normal | 98.91% | 99.71% | 99.31% | 98.76% | 97.38% | 93.36% |
| XGBoost-SHAP | DoS | 97.38% | 90.81% | 93.98% | 98.76% | 98.91% | 93.36% |
| DCNN+SA-SHAP | Normal | 99.04% | 99.63% | 99.34% | 98.81% | 96.70% | 93.55% |
| DCNN+SA-SHAP | DoS | 96.70% | 91.78% | 94.18% | 98.81% | 99.04% | 93.55% |

Small numeric differences can occur across TensorFlow, CUDA, LightGBM/XGBoost, and CPU/GPU versions.

## File Guide

| File | Purpose |
| --- | --- |
| `data_utils.py` | Shared CSV loading, preprocessing, feature selection JSON helpers |
| `shap_sort.py` | SHAP ranking and 95% cumulative-importance feature selection |
| `models.py` | DCNN and DCNN+SA architecture |
| `test.py` | Main DCNN+SA training and evaluation entry point |
| `baselines.py` | SVM, RF, and XGBoost baseline experiments |
| `sort_test.py` | SHAP threshold subset export for sensitivity analysis |
| `result_calc.py` | Metric calculation from predictions or confusion matrices |
| `prepare_data.py` | Optional legacy NPZ export from CSV |
| `trainset.csv`, `testset.csv` | Reproducible UNSW-NB15 DoS split |
