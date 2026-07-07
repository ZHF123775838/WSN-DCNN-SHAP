# Manuscript-to-Repository Alignment Notes

This file records how the repository now maps to the manuscript
`Manuscript-20260627.docx` and flags items that still need author-level
confirmation.

## Implemented Alignment

| Manuscript item | Repository implementation |
| --- | --- |
| UNSW-NB15 DoS binary detection | `trainset.csv` and `testset.csv` are the primary inputs. |
| No test-set leakage during preprocessing | `data_utils.py` fits imputers, winsorization limits, one-hot encoder, and Min-Max scaler on training data only. |
| SHAP-based feature ranking | `shap_sort.py` trains a tree model on training data and computes mean absolute SHAP values. |
| 95% cumulative SHAP threshold | `shap_sort.py` exports `artifacts/shap/selected_features_95.json`. |
| 90%, 95%, 98% threshold sensitivity | `sort_test.py` exports feature subsets and counts for those thresholds. |
| DCNN+SA classifier | `models.py` defines two Conv1D layers, one self-attention block, and three fully connected layers. |
| Attention ablation | `python test.py --no-attention --output-dir artifacts/dcnn` runs the DCNN variant with max pooling. |
| Final held-out evaluation | `test.py` trains on the committed training split and evaluates on the committed test split. |
| SVM/RF/XGBoost baselines | `baselines.py` trains manuscript-style baseline models on the SHAP-selected feature subset. |
| Metrics | `metrics_utils.py` and `result_calc.py` compute accuracy, specificity, recall, precision, F1, MCC, AUROC, and AUPRC. |

## Important Manuscript/Data Mismatches

1. Training normal-sample count:
   - Manuscript says: 55,998 normal and 12,264 DoS training samples.
   - Current `trainset.csv` contains: 56,000 normal and 12,264 DoS training samples.
   - Testing counts match the manuscript: 37,000 normal and 4,089 DoS.

2. Split protocol wording:
   - Section 2.1 mentions a 70%/30% stratified split.
   - Section 3.1 describes an approximately 60%/40% split.
   - Section 3.3 describes an outer stratified 5-fold cross-validation protocol.
   - The repository currently provides and uses a fixed committed train/test split, matching the Data Availability statement and README workflow.

3. Feature count wording:
   - The manuscript says the original data has 47 feature types and that SHAP reduces the final set to 22 features.
   - The committed CSV files contain 45 columns including `id`, `attack_cat`, and `label`; after dropping metadata/target columns and one-hot encoding categorical variables, the effective model feature space differs from the raw column count.

4. Environment availability:
   - A local `.venv` was created during smoke testing and dependencies from `requirements.txt` were installed successfully.
   - Smoke tests were run for SHAP feature selection, DCNN+SA training, metric calculation, and baseline models.
   - Full 50-epoch training was not run during this quick validation pass.

## Recommended Author Decision

To make the manuscript and repository fully consistent, choose one of these paths:

1. Keep the committed CSV files authoritative and revise the manuscript training normal count to 56,000.
2. Keep the manuscript count authoritative and regenerate `trainset.csv` with exactly 55,998 normal samples, then rerun SHAP selection and all model metrics.

Path 1 is less invasive because it preserves the public split already committed to GitHub.
