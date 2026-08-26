from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def clean_label(text: str) -> str:
    return str(text).replace("_", " ")


def plot_mlp_threshold_metrics(metadata: dict, figures_dir: Path) -> dict:
    metrics = ["accuracy", "precision", "recall", "f1"]
    threshold_keys = ["threshold_0_30", "threshold_0_50"]
    threshold_labels = ["Threshold 0.30", "Threshold 0.50"]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))

    for i, key in enumerate(threshold_keys):
        values = [metadata[key][m] for m in metrics]
        plt.bar(x + (i - 0.5) * width, values, width, label=threshold_labels[i])

    plt.xticks(x, [m.upper() if m == "f1" else m.capitalize() for m in metrics])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Metric value")
    plt.title("MLP reliability model performance")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    output_path = figures_dir / "fig_01_mlp_threshold_metrics.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "description": "Comparison of accuracy, precision, recall, and F1 for MLP reliability thresholds 0.30 and 0.50.",
    }


def plot_ise_vs_lambda(scenario_summary: pd.DataFrame, figures_dir: Path) -> dict:
    plt.figure(figsize=(9, 5))

    for scenario, group in scenario_summary.groupby("scenario"):
        group = group.sort_values("lambda_filter")
        plt.plot(
            group["lambda_filter"],
            group["mean_ise"],
            marker="o",
            label=clean_label(scenario),
        )

    plt.xlabel("IMC-like filter gain lambda")
    plt.ylabel("Mean ISE")
    plt.title("Closed-loop control error versus filter gain")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="best")

    output_path = figures_dir / "fig_02_ise_vs_lambda_by_scenario.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "description": "Mean integrated squared control error as a function of lambda for each scenario.",
    }


def plot_tv_tau_vs_lambda(scenario_summary: pd.DataFrame, figures_dir: Path) -> dict:
    plt.figure(figsize=(9, 5))

    for scenario, group in scenario_summary.groupby("scenario"):
        group = group.sort_values("lambda_filter")
        plt.plot(
            group["lambda_filter"],
            group["mean_tv_tau"],
            marker="o",
            label=clean_label(scenario),
        )

    plt.xlabel("IMC-like filter gain lambda")
    plt.ylabel("Mean total variation of threshold")
    plt.title("Control effort versus filter gain")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="best")

    output_path = figures_dir / "fig_03_tv_tau_vs_lambda_by_scenario.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "description": "Mean total variation of the threshold tau as a function of lambda for each scenario.",
    }


def plot_f1_vs_lambda(scenario_summary: pd.DataFrame, figures_dir: Path) -> dict:
    plt.figure(figsize=(9, 5))

    for scenario, group in scenario_summary.groupby("scenario"):
        group = group.sort_values("lambda_filter")
        plt.plot(
            group["lambda_filter"],
            group["mean_f1"],
            marker="o",
            label=clean_label(scenario),
        )

    plt.xlabel("IMC-like filter gain lambda")
    plt.ylabel("Mean final F1 score")
    plt.title("Final correspondence selection F1 versus filter gain")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="best")

    output_path = figures_dir / "fig_04_f1_vs_lambda_by_scenario.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "description": "Final F1 score of accepted correspondences as a function of lambda for each scenario.",
    }


def plot_global_lambda_summary(summary: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> dict:
    global_summary = (
        summary.groupby("lambda_filter")
        .agg(
            mean_iae=("iae", "mean"),
            mean_ise=("ise", "mean"),
            mean_tv_tau=("total_variation_tau", "mean"),
            mean_final_f1=("final_f1", "mean"),
            mean_final_precision=("final_precision", "mean"),
            mean_final_recall=("final_recall", "mean"),
            mean_iterations=("num_iterations", "mean"),
        )
        .reset_index()
        .sort_values("lambda_filter")
    )

    output_table = tables_dir / "control_lambda_global_summary.csv"
    global_summary.to_csv(output_table, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(global_summary["lambda_filter"], global_summary["mean_ise"], marker="o")
    plt.xlabel("IMC-like filter gain lambda")
    plt.ylabel("Mean ISE over all samples")
    plt.title("Global closed-loop ISE versus lambda")
    plt.grid(True, alpha=0.3)

    output_path = figures_dir / "fig_05_global_ise_vs_lambda.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "table": str(output_table),
        "description": "Global average ISE versus lambda, computed over all samples.",
    }


def plot_global_control_effort(summary: pd.DataFrame, figures_dir: Path) -> dict:
    global_summary = (
        summary.groupby("lambda_filter")
        .agg(mean_tv_tau=("total_variation_tau", "mean"))
        .reset_index()
        .sort_values("lambda_filter")
    )

    plt.figure(figsize=(7, 5))
    plt.plot(global_summary["lambda_filter"], global_summary["mean_tv_tau"], marker="o")
    plt.xlabel("IMC-like filter gain lambda")
    plt.ylabel("Mean total variation of tau")
    plt.title("Global threshold control effort versus lambda")
    plt.grid(True, alpha=0.3)

    output_path = figures_dir / "fig_06_global_tv_tau_vs_lambda.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "description": "Global average total variation of threshold tau versus lambda.",
    }


def plot_example_tau_history(history: pd.DataFrame, figures_dir: Path) -> dict:
    preferred_scenario = "static_noise"

    if preferred_scenario in set(history["scenario"]):
        scenario_history = history[history["scenario"] == preferred_scenario].copy()
    else:
        preferred_scenario = str(history["scenario"].iloc[0])
        scenario_history = history[history["scenario"] == preferred_scenario].copy()

    lambda_025 = scenario_history[scenario_history["lambda_filter"] == 0.25].copy()

    if len(lambda_025) == 0:
        candidate_counts = scenario_history.groupby("sample_id").size().sort_values(ascending=False)
    else:
        candidate_counts = lambda_025.groupby("sample_id").size().sort_values(ascending=False)

    sample_id = str(candidate_counts.index[0])
    sample_history = scenario_history[scenario_history["sample_id"] == sample_id].copy()

    plt.figure(figsize=(8, 5))

    for lambda_filter, group in sample_history.groupby("lambda_filter"):
        group = group.sort_values("iteration")
        plt.plot(
            group["iteration"],
            group["tau"],
            marker="o",
            label=f"lambda = {lambda_filter}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Threshold tau")
    plt.title(f"Threshold evolution on one {clean_label(preferred_scenario)} sample")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = figures_dir / "fig_07_example_tau_history.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "sample_id": sample_id,
        "scenario": preferred_scenario,
        "description": "Example evolution of the controlled threshold tau over iterations for different lambda values.",
    }


def plot_example_error_history(history: pd.DataFrame, figures_dir: Path) -> dict:
    preferred_scenario = "static_noise"

    if preferred_scenario in set(history["scenario"]):
        scenario_history = history[history["scenario"] == preferred_scenario].copy()
    else:
        preferred_scenario = str(history["scenario"].iloc[0])
        scenario_history = history[history["scenario"] == preferred_scenario].copy()

    lambda_025 = scenario_history[scenario_history["lambda_filter"] == 0.25].copy()

    if len(lambda_025) == 0:
        candidate_counts = scenario_history.groupby("sample_id").size().sort_values(ascending=False)
    else:
        candidate_counts = lambda_025.groupby("sample_id").size().sort_values(ascending=False)

    sample_id = str(candidate_counts.index[0])
    sample_history = scenario_history[scenario_history["sample_id"] == sample_id].copy()

    plt.figure(figsize=(8, 5))

    for lambda_filter, group in sample_history.groupby("lambda_filter"):
        group = group.sort_values("iteration")
        plt.plot(
            group["iteration"],
            group["control_error"],
            marker="o",
            label=f"lambda = {lambda_filter}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Control error")
    plt.title(f"Control error evolution on one {clean_label(preferred_scenario)} sample")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = figures_dir / "fig_08_example_control_error_history.png"
    save_figure(output_path)

    return {
        "file": str(output_path),
        "sample_id": sample_id,
        "scenario": preferred_scenario,
        "description": "Example evolution of the closed-loop control error over iterations for different lambda values.",
    }


def create_best_lambda_table(scenario_summary: pd.DataFrame, tables_dir: Path) -> dict:
    best_rows = []

    for scenario, group in scenario_summary.groupby("scenario"):
        best_idx = group["mean_ise"].idxmin()
        best_row = group.loc[best_idx].copy()
        best_rows.append(best_row)

    best_lambda = pd.DataFrame(best_rows).sort_values("scenario")

    output_path = tables_dir / "best_lambda_by_scenario.csv"
    best_lambda.to_csv(output_path, index=False)

    return {
        "file": str(output_path),
        "description": "Best lambda value per scenario according to minimum mean ISE.",
    }


def main() -> None:
    outputs_dir = Path("outputs")
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    logs_dir = outputs_dir / "logs"

    metadata_path = Path("models/reliability_mlp_sklearn_metadata.json")
    scenario_summary_path = tables_dir / "closed_loop_threshold_scenario_summary.csv"
    summary_path = tables_dir / "closed_loop_threshold_summary.csv"
    history_path = tables_dir / "closed_loop_threshold_history.csv"

    required_files = [
        metadata_path,
        scenario_summary_path,
        summary_path,
        history_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}. "
                "Run the training and closed-loop simulation scripts first."
            )

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    scenario_summary = pd.read_csv(scenario_summary_path)
    summary = pd.read_csv(summary_path)
    history = pd.read_csv(history_path)

    manifest = []

    manifest.append(plot_mlp_threshold_metrics(metadata, figures_dir))
    manifest.append(plot_ise_vs_lambda(scenario_summary, figures_dir))
    manifest.append(plot_tv_tau_vs_lambda(scenario_summary, figures_dir))
    manifest.append(plot_f1_vs_lambda(scenario_summary, figures_dir))
    manifest.append(plot_global_lambda_summary(summary, figures_dir, tables_dir))
    manifest.append(plot_global_control_effort(summary, figures_dir))
    manifest.append(plot_example_tau_history(history, figures_dir))
    manifest.append(plot_example_error_history(history, figures_dir))
    manifest.append(create_best_lambda_table(scenario_summary, tables_dir))

    manifest_path = logs_dir / "report_figures_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Report Figures Generated ===\n")

    for item in manifest:
        if "file" in item:
            print(f"- {item['file']}")
        if "table" in item:
            print(f"- {item['table']}")

    print(f"\nManifest saved to: {manifest_path}")
    print("\nFinished successfully.\n")


if __name__ == "__main__":
    main()
