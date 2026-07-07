"""Train and evaluate the manuscript's DCNN+SA model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from data_utils import RANDOM_SEED, load_preprocessed_data, load_selected_features, set_global_seed
from metrics_utils import binary_metrics, write_json
from models import build_dcnn_sa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="trainset.csv")
    parser.add_argument("--test-csv", default="testset.csv")
    parser.add_argument("--selected-features", default="artifacts/shap/selected_features_95.json")
    parser.add_argument("--output-dir", default="artifacts/dcnn_sa")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-attention", action="store_true", help="Run the DCNN ablation without SA.")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for DCNN+SA training. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    set_global_seed(args.seed, include_tensorflow=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed_data(args.train_csv, args.test_csv)
    selected_path = Path(args.selected_features)
    if selected_path.exists():
        cols = load_selected_features(selected_path, data.feature_names)
    else:
        print(f"Warning: {selected_path} not found; using all preprocessed features.")
        cols = list(range(len(data.feature_names)))

    X_train = data.X_train[:, cols]
    X_test = data.X_test[:, cols]
    selected_names = [data.feature_names[i] for i in cols]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        data.y_train,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=data.y_train,
    )

    classes = np.unique(y_fit)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_fit)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))

    model = build_dcnn_sa(
        input_features=X_fit.shape[1],
        learning_rate=args.learning_rate,
        attention_heads=args.attention_heads,
        weight_decay=args.weight_decay,
        use_attention=not args.no_attention,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.1,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    y_score = model.predict(X_test, batch_size=args.batch_size).reshape(-1)
    metrics = binary_metrics(data.y_test, y_score)
    metrics.update(
        {
            "model": "DCNN+SA" if not args.no_attention else "DCNN",
            "num_selected_features": len(selected_names),
            "selected_features": selected_names,
        }
    )

    write_json(output_dir / "metrics.json", metrics)
    pd.DataFrame({"y_true": data.y_test, "y_score": y_score}).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    model.save(output_dir / ("dcnn_sa.keras" if not args.no_attention else "dcnn.keras"))

    print(metrics)


if __name__ == "__main__":
    main()
