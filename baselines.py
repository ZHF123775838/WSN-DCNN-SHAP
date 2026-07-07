"""Train manuscript baseline models on the SHAP-selected feature subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data_utils import RANDOM_SEED, load_preprocessed_data, load_selected_features, set_global_seed
from metrics_utils import binary_metrics, write_json


def make_models(seed: int):
    models = {
        "SVM": SVC(C=10, gamma=1e-3, kernel="rbf", class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            criterion="gini",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        pass
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="trainset.csv")
    parser.add_argument("--test-csv", default="testset.csv")
    parser.add_argument("--selected-features", default="artifacts/shap/selected_features_95.json")
    parser.add_argument("--output-dir", default="artifacts/baselines")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed_data(args.train_csv, args.test_csv)
    selected_path = Path(args.selected_features)
    cols = load_selected_features(selected_path, data.feature_names) if selected_path.exists() else list(range(len(data.feature_names)))

    X_train = data.X_train[:, cols]
    X_test = data.X_test[:, cols]
    y_train = data.y_train

    if args.max_train_samples and args.max_train_samples < len(X_train):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_train), size=args.max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    rows = []
    for name, model in make_models(args.seed).items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        metrics = binary_metrics(data.y_test, y_score)
        metrics["model"] = name
        rows.append(metrics)
        write_json(output_dir / f"{name.lower()}_metrics.json", metrics)

    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
