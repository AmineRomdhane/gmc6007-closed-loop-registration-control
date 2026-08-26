from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def evaluate_binary_classifier(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    return {
        "threshold": threshold,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    config = load_config("configs/default.yaml")

    dataset_path = Path(config["paths"]["correspondence_dataset"])
    output_dir = Path(config["paths"]["output_dir"])
    models_dir = Path("models")
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = config["features"]["input_columns"]
    target_col = config["features"]["target_column"]
    group_col = config["features"]["group_column"]
    scenario_col = config["features"]["scenario_column"]
    random_seed = int(config["experiment"]["random_seed"])

    print("\n=== Train Reliability MLP ===\n")
    print(f"Dataset: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required_cols = feature_cols + [target_col, group_col, scenario_col]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()

    X = df[feature_cols].astype(float).to_numpy()
    y = df[target_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_seed,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1.0e-4,
                    batch_size=256,
                    learning_rate_init=1.0e-3,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=random_seed,
                    verbose=False,
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    y_score_test = model.predict_proba(X_test)[:, 1]

    metrics_03 = evaluate_binary_classifier(y_test, y_score_test, threshold=0.30)
    metrics_05 = evaluate_binary_classifier(y_test, y_score_test, threshold=0.50)

    metrics = {
        "dataset_path": str(dataset_path),
        "num_rows_total": int(len(df)),
        "num_rows_train": int(len(train_df)),
        "num_rows_test": int(len(test_df)),
        "num_train_samples": int(train_df[group_col].nunique()),
        "num_test_samples": int(test_df[group_col].nunique()),
        "feature_columns": feature_cols,
        "target_column": target_col,
        "group_column": group_col,
        "architecture": "StandardScaler + MLPClassifier(5 -> 32 -> 16 -> 1)",
        "threshold_0_30": metrics_03,
        "threshold_0_50": metrics_05,
    }

    model_path = models_dir / "reliability_mlp_sklearn.joblib"
    metadata_path = models_dir / "reliability_mlp_sklearn_metadata.json"
    predictions_path = tables_dir / "test_predictions_reliability_mlp.csv"

    joblib.dump(model, model_path)

    with open(metadata_path, "w") as f:
        json.dump(metrics, f, indent=2)

    test_predictions = test_df[
        [group_col, scenario_col, target_col] + feature_cols
    ].copy()
    test_predictions["predicted_weight"] = y_score_test
    test_predictions.to_csv(predictions_path, index=False)

    print("\nThreshold 0.30 metrics:")
    print(json.dumps(metrics_03, indent=2))

    print("\nThreshold 0.50 metrics:")
    print(json.dumps(metrics_05, indent=2))

    print("\nSaved:")
    print(f" - {model_path}")
    print(f" - {metadata_path}")
    print(f" - {predictions_path}")

    print("\nTraining finished successfully.\n")


if __name__ == "__main__":
    main()
