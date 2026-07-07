"""Optional helper to materialize preprocessed NPZ files from the CSV split."""

from __future__ import annotations

import argparse

from data_utils import load_preprocessed_data, save_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="trainset.csv")
    parser.add_argument("--test-csv", default="testset.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    bundle = load_preprocessed_data(args.train_csv, args.test_csv)
    save_npz(bundle, args.output_dir)
    print(f"Saved trainset.npz, testset.npz, and feature_names.json to {args.output_dir}")


if __name__ == "__main__":
    main()
