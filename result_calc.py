"""Recalculate metrics from saved predictions or a confusion matrix."""

from __future__ import annotations

import argparse

import pandas as pd

from metrics_utils import binary_metrics


def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> dict:
    y_true = [0] * (tn + fp) + [1] * (fn + tp)
    y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
    return binary_metrics(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/dcnn_sa/predictions.csv")
    parser.add_argument("--confusion", nargs=4, type=int, metavar=("TN", "FP", "FN", "TP"))
    args = parser.parse_args()

    if args.confusion:
        metrics = metrics_from_confusion(*args.confusion)
    else:
        df = pd.read_csv(args.predictions)
        metrics = binary_metrics(df["y_true"], df["y_score"])

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
