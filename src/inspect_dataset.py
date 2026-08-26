from pathlib import Path
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


def main() -> None:
    config = load_config("configs/default.yaml")

    dataset_path = Path(config["paths"]["correspondence_dataset"])
    output_dir = Path(config["paths"]["output_dir"])
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = config["features"]["input_columns"]
    diagnostic_cols = config["features"]["diagnostic_columns"]
    target_col = config["features"]["target_column"]
    group_col = config["features"]["group_column"]
    scenario_col = config["features"]["scenario_column"]

    print("\n=== GMC6007 Dataset Inspection ===\n")
    print(f"Dataset: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print(f"Shape: {df.shape}")

    required_cols = feature_cols + diagnostic_cols + [target_col, group_col, scenario_col]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nTarget distribution:")
    print(df[target_col].value_counts(dropna=False))

    scenario_summary = (
        df.groupby(scenario_col)
        .agg(
            num_correspondences=(target_col, "size"),
            positive_ratio=(target_col, "mean"),
            mean_label_distance=("label_distance", "mean"),
            mean_distance_T0=("distance_T0", "mean"),
            mean_point_to_plane_residual=("point_to_plane_residual", "mean"),
        )
        .reset_index()
        .sort_values("num_correspondences", ascending=False)
    )

    sample_summary = (
        df.groupby([group_col, scenario_col])
        .agg(
            num_correspondences=(target_col, "size"),
            positive_ratio=(target_col, "mean"),
            mean_label_distance=("label_distance", "mean"),
            mean_distance_T0=("distance_T0", "mean"),
            mean_point_to_plane_residual=("point_to_plane_residual", "mean"),
        )
        .reset_index()
        .sort_values("num_correspondences", ascending=False)
    )

    feature_summary = df[feature_cols + diagnostic_cols + [target_col]].describe().T

    scenario_summary_path = tables_dir / "scenario_summary.csv"
    sample_summary_path = tables_dir / "sample_summary.csv"
    feature_summary_path = tables_dir / "feature_summary.csv"

    scenario_summary.to_csv(scenario_summary_path, index=False)
    sample_summary.to_csv(sample_summary_path, index=False)
    feature_summary.to_csv(feature_summary_path)

    print("\nScenario summary:")
    print(scenario_summary)

    print("\nSaved:")
    print(f" - {scenario_summary_path}")
    print(f" - {sample_summary_path}")
    print(f" - {feature_summary_path}")

    print("\nInspection finished successfully.\n")


if __name__ == "__main__":
    main()
