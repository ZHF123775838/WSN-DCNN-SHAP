# Network Traffic Anomaly Detection with SHAP and Attention-based DCNN

## Overview
This repository contains the implementation of an efficient anomaly detection framework that synergistically integrates SHAP (SHapley Additive exPlanations) for feature interpretation and a Deep Convolutional Neural Network (DCNN) with self-attention mechanism for Denial-of-Service (DoS) attack detection in wireless sensor networks (WSNs).

**Key Features:**
- SHAP-based feature selection eliminating the need for traditional dimensionality reduction
- Attention-enhanced DCNN architecture for improved spatiotemporal pattern learning
- Comprehensive evaluation on UNSW-NB15 dataset
- State-of-the-art performance: AUROC=0.999, AUPRC=0.992

## Repository Files

| File | Description |
|------|-------------|
| `trainset.csv` / `trainset.xlsx` | Training dataset (55,998 normal + 12,264 DoS samples) |
| `testset.csv` / `testset.xlsx` | Testing dataset (37,000 normal + 4,089 DoS samples) |
| `shap_sort.py` | SHAP-based feature selection and ranking |
| `result_calc.py` | Performance metrics calculation and analysis |
| `sort_test.py` | Feature sorting validation tests |
| `test.py` | Main model training and evaluation script |
| `import_scores.npy` | Pre-computed evaluation scores |
| `import_sort.npy` | Pre-computed feature rankings |
| `sorted.npz.npy` | Sorted feature data after SHAP selection |
| `result_rf.npy` | Random Forest baseline results |
| `fs_gridsearch.rar` | Feature selection grid search results |
| `nohup.out` | Execution logs and output |

## Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow scikit-learn shap pandas numpy matplotlib seaborn
```

### 2. Run Feature Selection
```bash
python shap_sort.py
```
This performs SHAP-based feature selection and generates feature rankings.

### 3. Train and Evaluate Model
```bash
python test.py
```
This executes the main training pipeline with the selected features.

### 4. Calculate Results
```bash
python result_calc.py
```
Generates performance metrics and comparative analysis.

## Expected Results
- **AUROC**: 0.999
- **AUPRC**: 0.992  
- **MCC**: ~93.55%
- **Precision (DoS)**: ~91.78%
- **Recall (DoS)**: ~96.70%

## File Descriptions

### Core Scripts
- **`shap_sort.py`**: Implements SHAP feature selection using LightGBM, calculates Shapley values, and selects optimal feature subset based on 95% cumulative importance threshold
- **`test.py`**: Main training script implementing DCNN with self-attention mechanism, includes hyperparameter optimization and model evaluation
- **`result_calc.py`**: Computes comprehensive performance metrics and statistical comparisons

### Data Files
- **`trainset.csv` / `testset.csv`**: Pre-processed UNSW-NB15 dataset splits with feature engineering applied
- **`*.npy` files**: Pre-computed results and intermediate data for reproducibility

### Additional Files
- **`fs_gridsearch.rar`**: Contains feature selection grid search results
- **`nohup.out`**: Execution logs from previous runs
- **`MACOSX.rar`**: System files (can be ignored)

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{your2023efficient,
  title={An efficient method for network traffic anomaly detection based on SHAP and deep learning},
  author={Your Name and Coauthors},
  journal={Journal Name},
  year={2023}
}
```
