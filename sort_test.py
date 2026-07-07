"""Inspect SHAP cumulative-importance thresholds used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_utils import save_feature_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", default="artifacts/shap/feature_ranking.csv")
    parser.add_argument("--output-dir", default="artifacts/shap")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.90, 0.95, 0.98])
    args = parser.parse_args()

    ranking = pd.read_csv(args.ranking)
    output_dir = Path(args.output_dir)
    rows = []
    for threshold in args.thresholds:
        selected = ranking[ranking["cumulative_importance"] <= threshold]
        if len(selected) < len(ranking):
            selected = ranking.head(len(selected) + 1)
        features = selected["feature"].tolist()
        save_feature_selection(output_dir / f"selected_features_{int(threshold * 100)}.json", threshold, features)
        rows.append({"threshold": threshold, "num_features": len(features)})

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "threshold_summary.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
