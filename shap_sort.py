"""Compute SHAP feature rankings and select the manuscript's feature subset.

The paper specifies SHAP TreeExplainer-based global feature importance,
computed only on the training split, with features retained until their
cumulative mean absolute SHAP importance reaches 95%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import RANDOM_SEED, load_preprocessed_data, save_feature_selection, set_global_seed


def build_tree_model(seed: int = RANDOM_SEED):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=seed,
            n_jobs=-1,
        )


def compute_mean_abs_shap(model, X: np.ndarray, sample_size: int | None, seed: int) -> np.ndarray:
    import shap

    rng = np.random.default_rng(seed)
    if sample_size and sample_size < len(X):
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_for_shap = X[idx]
    else:
        X_for_shap = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_for_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    return np.mean(np.abs(shap_values), axis=0)


def write_plots(ranking: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = ranking.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4, top_n * 0.28)))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#3267a8")
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ranking["rank"], ranking["cumulative_importance"], marker="o", markersize=3)
    plt.axhline(0.95, color="#c43c39", linestyle="--", label="95% threshold")
    plt.xlabel("Number of selected features")
    plt.ylabel("Cumulative SHAP importance")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_importance.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="trainset.csv")
    parser.add_argument("--test-csv", default="testset.csv")
    parser.add_argument("--output-dir", default="artifacts/shap")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--shap-sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--plot-top-n", type=int, default=22)
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed_data(args.train_csv, args.test_csv)
    model = build_tree_model(args.seed)
    model.fit(data.X_train, data.y_train)

    mean_abs = compute_mean_abs_shap(model, data.X_train, args.shap_sample_size, args.seed)
    order = np.argsort(mean_abs)[::-1]
    sorted_importance = mean_abs[order]
    total = sorted_importance.sum()
    cumulative = np.cumsum(sorted_importance) / total

    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "feature": [data.feature_names[i] for i in order],
            "feature_index": order,
            "mean_abs_shap": sorted_importance,
            "cumulative_importance": cumulative,
        }
    )
    ranking.to_csv(output_dir / "feature_ranking.csv", index=False)
    np.save(output_dir / "feature_order_desc.npy", order)

    selected_count = int(np.searchsorted(cumulative, args.threshold) + 1)
    selected_features = ranking.head(selected_count)["feature"].tolist()
    save_feature_selection(
        output_dir / f"selected_features_{int(args.threshold * 100)}.json",
        args.threshold,
        selected_features,
    )
    write_plots(ranking, output_dir, args.plot_top_n)

    print(f"Saved SHAP ranking to {output_dir / 'feature_ranking.csv'}")
    print(f"Selected {selected_count} features at {args.threshold:.0%} cumulative importance")


if __name__ == "__main__":
    main()
