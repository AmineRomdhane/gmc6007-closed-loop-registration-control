from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def safe_mean(values: np.ndarray, default_value: float) -> float:
    if len(values) == 0:
        return float(default_value)
    return float(np.mean(values))


def compute_classification_metrics(y_true: np.ndarray, accepted: np.ndarray) -> dict:
    tp = int(np.sum((accepted == 1) & (y_true == 1)))
    fp = int(np.sum((accepted == 1) & (y_true == 0)))
    fn = int(np.sum((accepted == 0) & (y_true == 1)))
    tn = int(np.sum((accepted == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_threshold_control_for_sample(
    sample_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    residual_col: str,
    label_distance_col: str,
    lambda_filter: float,
    max_iterations: int,
    tolerance: float,
    tau_initial: float,
    tau_min: float,
    tau_max: float,
    residual_reference: float,
    min_acceptance_ratio: float,
    kp_residual: float,
    kp_acceptance: float,
) -> tuple[list[dict], dict]:
    tau = float(tau_initial)
    previous_error = None

    history = []

    scores = sample_df[score_col].to_numpy(dtype=float)
    y_true = sample_df[target_col].to_numpy(dtype=int)
    residuals = sample_df[residual_col].to_numpy(dtype=float)
    label_distances = sample_df[label_distance_col].to_numpy(dtype=float)

    for k in range(max_iterations):
        accepted = scores >= tau
        accepted_int = accepted.astype(int)

        acceptance_ratio = float(np.mean(accepted_int))
        mean_residual = safe_mean(
            residuals[accepted],
            default_value=float(np.mean(residuals)),
        )
        mean_label_distance = safe_mean(
            label_distances[accepted],
            default_value=float(np.mean(label_distances)),
        )

        residual_error = max(0.0, mean_residual - residual_reference)
        acceptance_error = max(0.0, min_acceptance_ratio - acceptance_ratio)

        control_error = residual_error + acceptance_error

        class_metrics = compute_classification_metrics(y_true, accepted_int)

        history_row = {
            "iteration": k,
            "tau": tau,
            "acceptance_ratio": acceptance_ratio,
            "mean_residual": mean_residual,
            "mean_label_distance": mean_label_distance,
            "residual_error": residual_error,
            "acceptance_error": acceptance_error,
            "control_error": control_error,
            **class_metrics,
        }
        history.append(history_row)

        if previous_error is not None and abs(previous_error - control_error) < tolerance:
            break

        previous_error = control_error

        residual_action = kp_residual * residual_error
        acceptance_action = -kp_acceptance * acceptance_error

        delta_tau_command = residual_action + acceptance_action
        delta_tau_filtered = lambda_filter * delta_tau_command

        tau = float(np.clip(tau + delta_tau_filtered, tau_min, tau_max))

    errors = np.array([row["control_error"] for row in history], dtype=float)
    taus = np.array([row["tau"] for row in history], dtype=float)

    final_row = history[-1]

    if len(taus) > 1:
        total_variation = float(np.sum(np.abs(np.diff(taus))))
    else:
        total_variation = 0.0

    summary = {
        "num_iterations": int(len(history)),
        "final_tau": float(final_row["tau"]),
        "final_acceptance_ratio": float(final_row["acceptance_ratio"]),
        "final_mean_residual": float(final_row["mean_residual"]),
        "final_mean_label_distance": float(final_row["mean_label_distance"]),
        "final_control_error": float(final_row["control_error"]),
        "final_precision": float(final_row["precision"]),
        "final_recall": float(final_row["recall"]),
        "final_f1": float(final_row["f1"]),
        "iae": float(np.sum(np.abs(errors))),
        "ise": float(np.sum(errors**2)),
        "total_variation_tau": total_variation,
    }

    return history, summary


def main() -> None:
    config = load_config("configs/default.yaml")

    dataset_path = Path(config["paths"]["correspondence_dataset"])
    output_dir = Path(config["paths"]["output_dir"])
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = config["features"]["input_columns"]
    target_col = config["features"]["target_column"]
    group_col = config["features"]["group_column"]
    scenario_col = config["features"]["scenario_column"]

    model_path = Path("models/reliability_mlp_sklearn.joblib")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run: python src/train_reliability_mlp.py"
        )

    df = pd.read_csv(dataset_path).dropna(
        subset=feature_cols
        + [target_col, group_col, scenario_col, "point_to_plane_residual", "label_distance"]
    ).copy()

    model = joblib.load(model_path)

    X = df[feature_cols].astype(float).to_numpy()
    df["predicted_weight"] = model.predict_proba(X)[:, 1]

    max_iterations = int(config["registration_control"]["max_iterations"])
    tolerance = float(config["registration_control"]["convergence_tolerance"])

    lambda_values = config["control"]["lambda_values"]

    residual_reference = 0.04
    min_acceptance_ratio = 0.20
    tau_initial = 0.50
    tau_min = 0.05
    tau_max = 0.95
    kp_residual = 2.0
    kp_acceptance = 0.5

    all_history = []
    all_summary = []

    print("\n=== Closed-Loop Threshold Control Simulation ===\n")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Number of samples: {df[group_col].nunique()}")
    print(f"Lambda values: {lambda_values}")

    grouped = df.groupby(group_col)

    for lambda_filter in lambda_values:
        print(f"\nRunning lambda = {lambda_filter}")

        for sample_id, sample_df in grouped:
            scenario = str(sample_df[scenario_col].iloc[0])

            history, summary = run_threshold_control_for_sample(
                sample_df=sample_df,
                score_col="predicted_weight",
                target_col=target_col,
                residual_col="point_to_plane_residual",
                label_distance_col="label_distance",
                lambda_filter=float(lambda_filter),
                max_iterations=max_iterations,
                tolerance=tolerance,
                tau_initial=tau_initial,
                tau_min=tau_min,
                tau_max=tau_max,
                residual_reference=residual_reference,
                min_acceptance_ratio=min_acceptance_ratio,
                kp_residual=kp_residual,
                kp_acceptance=kp_acceptance,
            )

            for row in history:
                row["sample_id"] = sample_id
                row["scenario"] = scenario
                row["lambda_filter"] = lambda_filter
                all_history.append(row)

            summary["sample_id"] = sample_id
            summary["scenario"] = scenario
            summary["lambda_filter"] = lambda_filter
            all_summary.append(summary)

    history_df = pd.DataFrame(all_history)
    summary_df = pd.DataFrame(all_summary)

    history_path = tables_dir / "closed_loop_threshold_history.csv"
    summary_path = tables_dir / "closed_loop_threshold_summary.csv"
    scenario_summary_path = tables_dir / "closed_loop_threshold_scenario_summary.csv"
    metadata_path = logs_dir / "closed_loop_threshold_control_metadata.json"

    scenario_summary = (
        summary_df.groupby(["scenario", "lambda_filter"])
        .agg(
            mean_iterations=("num_iterations", "mean"),
            mean_final_tau=("final_tau", "mean"),
            mean_acceptance_ratio=("final_acceptance_ratio", "mean"),
            mean_residual=("final_mean_residual", "mean"),
            mean_label_distance=("final_mean_label_distance", "mean"),
            mean_control_error=("final_control_error", "mean"),
            mean_precision=("final_precision", "mean"),
            mean_recall=("final_recall", "mean"),
            mean_f1=("final_f1", "mean"),
            mean_iae=("iae", "mean"),
            mean_ise=("ise", "mean"),
            mean_tv_tau=("total_variation_tau", "mean"),
        )
        .reset_index()
    )

    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    scenario_summary.to_csv(scenario_summary_path, index=False)

    metadata = {
        "controlled_variable": "correspondence residual quality with minimum acceptance constraint",
        "control_input": "acceptance threshold tau",
        "internal_model": "MLP predicted correspondence reliability",
        "imc_like_filter": "lambda_filter applied to threshold correction",
        "residual_reference": residual_reference,
        "min_acceptance_ratio": min_acceptance_ratio,
        "tau_initial": tau_initial,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "kp_residual": kp_residual,
        "kp_acceptance": kp_acceptance,
        "max_iterations": max_iterations,
        "convergence_tolerance": tolerance,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nScenario-level results:")
    print(scenario_summary)

    print("\nSaved:")
    print(f" - {history_path}")
    print(f" - {summary_path}")
    print(f" - {scenario_summary_path}")
    print(f" - {metadata_path}")

    print("\nClosed-loop simulation finished successfully.\n")


if __name__ == "__main__":
    main()
