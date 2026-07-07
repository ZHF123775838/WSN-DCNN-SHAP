"""Shared data loading and preprocessing utilities for WSN-DCNN-SHAP.

The manuscript uses the UNSW-NB15 training/test split committed in this
repository.  This module keeps that split fixed and applies preprocessing
without leaking information from the test set into the training pipeline.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


TARGET_COLUMN = "label"
DROP_COLUMNS = ("id", "attack_cat")
RANDOM_SEED = 42


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer


def set_global_seed(seed: int = RANDOM_SEED, include_tensorflow: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if not include_tensorflow:
        return
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    drop_cols = [c for c in (*DROP_COLUMNS, TARGET_COLUMN) if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def winsorize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: Iterable[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in numeric_cols:
        lower = train_df[col].quantile(lower_q)
        upper = train_df[col].quantile(upper_q)
        train_df[col] = train_df[col].clip(lower, upper)
        test_df[col] = test_df[col].clip(lower, upper)
    return train_df, test_df


def load_preprocessed_data(
    train_csv: str | Path = "trainset.csv",
    test_csv: str | Path = "testset.csv",
) -> DatasetBundle:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train_raw, y_train = split_features_labels(train_df)
    X_test_raw, y_test = split_features_labels(test_df)

    categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_train_raw.columns if c not in categorical_cols]

    X_train_raw, X_test_raw = winsorize_train_test(X_train_raw, X_test_raw, numeric_cols)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    return DatasetBundle(
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=y_train.astype(np.int32),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_test=y_test.astype(np.int32),
        feature_names=feature_names,
        preprocessor=preprocessor,
    )


def save_feature_selection(path: str | Path, threshold: float, features: list[str]) -> None:
    payload = {
        "threshold": threshold,
        "num_features": len(features),
        "features": features,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_selected_features(path: str | Path, feature_names: list[str]) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    wanted = payload["features"]
    lookup = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise ValueError(f"Selected feature file contains unknown features: {missing[:5]}")
    return [lookup[name] for name in wanted]


def save_npz(bundle: DatasetBundle, output_dir: str | Path = ".") -> None:
    output_dir = Path(output_dir)
    np.savez_compressed(output_dir / "trainset.npz", data=bundle.X_train, label=bundle.y_train)
    np.savez_compressed(output_dir / "testset.npz", data=bundle.X_test, label=bundle.y_test)
    (output_dir / "feature_names.json").write_text(
        json.dumps(bundle.feature_names, indent=2),
        encoding="utf-8",
    )
