from pathlib import Path
import json
import struct

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from simulate_transform_control import (
    apply_transform,
    compute_alignment_metrics,
    compute_correspondence_features,
    estimate_local_geometry,
    estimate_weighted_transform,
    interpolate_transforms,
)


def read_ply_xyz(path: Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with open(path, "rb") as f:
        header_lines = []

        while True:
            line = f.readline()

            if not line:
                raise ValueError(f"Invalid PLY file, missing end_header: {path}")

            decoded = line.decode("utf-8", errors="replace").strip()
            header_lines.append(decoded)

            if decoded == "end_header":
                break

        if len(header_lines) == 0 or header_lines[0] != "ply":
            raise ValueError(f"Not a valid PLY file: {path}")

        fmt = None
        vertex_count = None
        vertex_properties = []
        in_vertex_element = False

        for line in header_lines:
            parts = line.split()

            if len(parts) == 0:
                continue

            if parts[0] == "format":
                fmt = parts[1]

            elif parts[0] == "element":
                in_vertex_element = parts[1] == "vertex"

                if in_vertex_element:
                    vertex_count = int(parts[2])

            elif parts[0] == "property" and in_vertex_element:
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_properties.append((parts[1], parts[2]))

        if fmt is None:
            raise ValueError(f"PLY format not found: {path}")

        if vertex_count is None:
            raise ValueError(f"PLY vertex count not found: {path}")

        prop_names = [name for _, name in vertex_properties]

        for required in ["x", "y", "z"]:
            if required not in prop_names:
                raise ValueError(f"PLY file missing property '{required}': {path}")

        x_idx = prop_names.index("x")
        y_idx = prop_names.index("y")
        z_idx = prop_names.index("z")

        if fmt == "ascii":
            points = []

            for _ in range(vertex_count):
                line = f.readline().decode("utf-8", errors="replace").strip()
                values = line.split()

                if len(values) < len(vertex_properties):
                    continue

                points.append(
                    [
                        float(values[x_idx]),
                        float(values[y_idx]),
                        float(values[z_idx]),
                    ]
                )

            points = np.asarray(points, dtype=float)

        elif fmt in ["binary_little_endian", "binary_big_endian"]:
            endian = "<" if fmt == "binary_little_endian" else ">"

            type_map = {
                "char": "i1",
                "uchar": "u1",
                "int8": "i1",
                "uint8": "u1",
                "short": "i2",
                "ushort": "u2",
                "int16": "i2",
                "uint16": "u2",
                "int": "i4",
                "uint": "u4",
                "int32": "i4",
                "uint32": "u4",
                "float": "f4",
                "float32": "f4",
                "double": "f8",
                "float64": "f8",
            }

            dtype_fields = []

            for prop_type, prop_name in vertex_properties:
                if prop_type not in type_map:
                    raise ValueError(
                        f"Unsupported PLY property type '{prop_type}' in {path}"
                    )

                dtype_fields.append((prop_name, endian + type_map[prop_type]))

            dtype = np.dtype(dtype_fields)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)

            points = np.column_stack(
                [
                    data["x"].astype(float),
                    data["y"].astype(float),
                    data["z"].astype(float),
                ]
            )

        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    points = points[np.all(np.isfinite(points), axis=1)]

    if len(points) < 3:
        raise ValueError(f"PLY file has fewer than 3 valid points: {path}")

    return points


def load_transform(path: Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")

    T = np.loadtxt(path)

    if T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {T.shape}: {path}")

    return T.astype(float)


def maybe_subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)

    return points[indices]


def find_real_folds(folds_dir: Path, max_cases: int | None = None) -> list[dict]:
    folds_dir = Path(folds_dir)

    if not folds_dir.exists():
        raise FileNotFoundError(f"Folds directory not found: {folds_dir}")

    cases = []

    for case_dir in sorted(folds_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        reference_path = case_dir / "reference_downsampled.ply"
        observation_path = case_dir / "observation_downsampled.ply"
        initial_path = case_dir / "T_initial_T0.txt"

        pseudo_gt_path = case_dir / "T_pseudo_gt_label.txt"

        if not pseudo_gt_path.exists():
            pseudo_gt_path = case_dir / "T_mlp_weighted_svd.txt"

        if (
            reference_path.exists()
            and observation_path.exists()
            and initial_path.exists()
            and pseudo_gt_path.exists()
        ):
            cases.append(
                {
                    "case_name": case_dir.name,
                    "case_dir": str(case_dir),
                    "reference_path": str(reference_path),
                    "observation_path": str(observation_path),
                    "initial_transform_path": str(initial_path),
                    "pseudo_gt_transform_path": str(pseudo_gt_path),
                }
            )

    if max_cases is not None:
        cases = cases[:max_cases]

    if len(cases) == 0:
        raise RuntimeError(f"No usable real folds found in: {folds_dir}")

    return cases


def select_real_correspondences(
    distances: np.ndarray,
    weights: np.ndarray,
    distance_gate: float,
    min_correspondences: int,
    max_correspondences: int,
) -> np.ndarray:
    valid = np.where(distances <= distance_gate)[0]

    if len(valid) < min_correspondences:
        valid = np.argsort(distances)[: min(min_correspondences, len(distances))]

    if len(valid) > max_correspondences:
        valid_weights = weights[valid]
        order = np.argsort(valid_weights)[::-1]
        valid = valid[order[:max_correspondences]]

    return np.asarray(valid, dtype=int)


def run_real_case(
    model,
    case: dict,
    method: str,
    lambda_gain: float,
    num_iterations: int,
    num_neighbors: int,
    distance_gate: float,
    distance_sigma: float,
    min_correspondences: int,
    max_correspondences: int,
    fitness_threshold: float,
    max_points: int,
    seed: int,
) -> list[dict]:
    case_name = case["case_name"]

    reference = read_ply_xyz(Path(case["reference_path"]))
    observation = read_ply_xyz(Path(case["observation_path"]))

    reference = maybe_subsample(reference, max_points=max_points, seed=seed + 100)
    observation = maybe_subsample(observation, max_points=max_points, seed=seed + 200)

    T_initial = load_transform(Path(case["initial_transform_path"]))
    T_pseudo_gt = load_transform(Path(case["pseudo_gt_transform_path"]))

    source = observation
    target = reference

    source_normals, source_descriptors, source_density = estimate_local_geometry(
        source,
        num_neighbors=num_neighbors,
    )

    target_normals, target_descriptors, target_density = estimate_local_geometry(
        target,
        num_neighbors=num_neighbors,
    )

    target_tree = NearestNeighbors(n_neighbors=1)
    target_tree.fit(target)

    T_current = T_initial.copy()

    history = []

    previous_translation_update = 0.0
    previous_rotation_update_deg = 0.0

    for iteration in range(num_iterations + 1):
        metrics = compute_alignment_metrics(
            source=source,
            target=target,
            T_current=T_current,
            T_gt=T_pseudo_gt,
            target_tree=target_tree,
            fitness_threshold=fitness_threshold,
        )

        features, distances, nn_indices, point_to_plane_residual = compute_correspondence_features(
            source=source,
            target=target,
            source_normals=source_normals,
            target_normals=target_normals,
            source_descriptors=source_descriptors,
            target_descriptors=target_descriptors,
            source_density=source_density,
            target_density=target_density,
            T_current=T_current,
            target_tree=target_tree,
        )

        predicted_weights = model.predict_proba(features)[:, 1]
        distance_weights = np.exp(-((distances / distance_sigma) ** 2))

        if method == "unweighted_icp":
            solver_weights = np.ones_like(predicted_weights)

        elif method == "distance_weighted_icp":
            solver_weights = distance_weights

        elif method == "mlp_weighted_icp":
            solver_weights = np.clip(predicted_weights, 0.02, 1.0) * distance_weights

        else:
            raise ValueError(f"Unknown method: {method}")

        selected = select_real_correspondences(
            distances=distances,
            weights=solver_weights,
            distance_gate=distance_gate,
            min_correspondences=min_correspondences,
            max_correspondences=max_correspondences,
        )

        selected_source = source[selected]
        selected_target = target[nn_indices[selected]]
        selected_weights = solver_weights[selected]

        mean_predicted_weight = float(np.mean(predicted_weights[selected]))
        mean_selected_distance = float(np.mean(distances[selected]))
        mean_selected_point_to_plane = float(np.mean(point_to_plane_residual[selected]))

        history.append(
            {
                "case_name": case_name,
                "method": method,
                "lambda_gain": lambda_gain,
                "iteration": iteration,
                "num_source_points": len(source),
                "num_target_points": len(target),
                "num_selected_correspondences": len(selected),
                "mean_predicted_weight": mean_predicted_weight,
                "mean_selected_distance": mean_selected_distance,
                "mean_selected_point_to_plane": mean_selected_point_to_plane,
                "translation_update_norm": previous_translation_update,
                "rotation_update_deg": previous_rotation_update_deg,
                "reference_path": case["reference_path"],
                "observation_path": case["observation_path"],
                "initial_transform_path": case["initial_transform_path"],
                "pseudo_gt_transform_path": case["pseudo_gt_transform_path"],
                **metrics,
            }
        )

        if iteration == num_iterations:
            break

        T_est = estimate_weighted_transform(
            source_points=selected_source,
            target_points=selected_target,
            weights=selected_weights,
        )

        if T_est is None:
            previous_translation_update = 0.0
            previous_rotation_update_deg = 0.0
            continue

        T_next = interpolate_transforms(
            T_current=T_current,
            T_target=T_est,
            gain=lambda_gain,
        )

        previous_translation_update = float(
            np.linalg.norm(T_next[:3, 3] - T_current[:3, 3])
        )

        rotation_trace = np.trace(T_next[:3, :3] @ T_current[:3, :3].T)
        rotation_trace = float(np.clip((rotation_trace - 1.0) / 2.0, -1.0, 1.0))
        previous_rotation_update_deg = float(np.degrees(np.arccos(rotation_trace)))

        T_current = T_next

    return history


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_real_figures(
    history_df: pd.DataFrame,
    global_summary: pd.DataFrame,
    best_by_case: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures = []

    plt.figure(figsize=(8, 5))
    for method, group in global_summary.groupby("method"):
        group = group.sort_values("lambda_gain")
        plt.plot(
            group["lambda_gain"],
            group["mean_final_alignment_error"],
            marker="o",
            label=method.replace("_", " "),
        )

    plt.xscale("log")
    plt.xlabel("Transform filter gain lambda")
    plt.ylabel("Mean final pseudo-GT alignment error")
    plt.title("Real cases: final alignment error versus lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = figures_dir / "fig_14_real_final_alignment_error_vs_lambda.png"
    save_figure(path)
    figures.append(str(path))

    plt.figure(figsize=(8, 5))
    for method, group in global_summary.groupby("method"):
        group = group.sort_values("lambda_gain")
        plt.plot(
            group["lambda_gain"],
            group["mean_final_rmse_all"],
            marker="o",
            label=method.replace("_", " "),
        )

    plt.xscale("log")
    plt.xlabel("Transform filter gain lambda")
    plt.ylabel("Mean final nearest-neighbor RMSE")
    plt.title("Real cases: final RMSE versus lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = figures_dir / "fig_15_real_final_rmse_vs_lambda.png"
    save_figure(path)
    figures.append(str(path))

    plt.figure(figsize=(8, 5))
    for method, group in global_summary.groupby("method"):
        group = group.sort_values("lambda_gain")
        plt.plot(
            group["lambda_gain"],
            group["mean_final_translation_error"],
            marker="o",
            label=method.replace("_", " "),
        )

    plt.xscale("log")
    plt.xlabel("Transform filter gain lambda")
    plt.ylabel("Mean final translation error to pseudo-GT")
    plt.title("Real cases: translation error versus lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = figures_dir / "fig_16_real_translation_error_vs_lambda.png"
    save_figure(path)
    figures.append(str(path))

    plt.figure(figsize=(8, 5))
    for method, group in global_summary.groupby("method"):
        group = group.sort_values("lambda_gain")
        plt.plot(
            group["lambda_gain"],
            group["mean_final_rotation_error_deg"],
            marker="o",
            label=method.replace("_", " "),
        )

    plt.xscale("log")
    plt.xlabel("Transform filter gain lambda")
    plt.ylabel("Mean final rotation error to pseudo-GT (deg)")
    plt.title("Real cases: rotation error versus lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = figures_dir / "fig_17_real_rotation_error_vs_lambda.png"
    save_figure(path)
    figures.append(str(path))

    first_case = str(history_df["case_name"].iloc[0])
    example = history_df[
        (history_df["case_name"] == first_case)
        & (history_df["lambda_gain"].isin([0.2, 0.75, 1.0]))
    ].copy()

    plt.figure(figsize=(9, 5))
    for (method, lambda_gain), group in example.groupby(["method", "lambda_gain"]):
        group = group.sort_values("iteration")
        plt.plot(
            group["iteration"],
            group["alignment_error"],
            marker="o",
            label=f"{method.replace('_', ' ')}, lambda={lambda_gain}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Pseudo-GT alignment error")
    plt.title(f"Real example: alignment error history\n{first_case}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    path = figures_dir / "fig_18_real_example_alignment_history.png"
    save_figure(path)
    figures.append(str(path))

    plot_df = best_by_case.copy()
    plot_df["label"] = plot_df["case_name"].str.replace("real_", "", regex=False)
    plot_df["label"] = plot_df["label"].str.slice(0, 32)

    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["label"], plot_df["final_alignment_error"])
    plt.xlabel("Best final pseudo-GT alignment error")
    plt.ylabel("Real case")
    plt.title("Best real-case result per fold")
    plt.grid(axis="x", alpha=0.3)
    path = figures_dir / "fig_19_real_best_case_alignment_error.png"
    save_figure(path)
    figures.append(str(path))

    return figures


def main() -> None:
    output_dir = Path("outputs")
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path("models/reliability_mlp_sklearn.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run src/train_reliability_mlp.py first."
        )

    folds_dir = Path(
        "/home/amine/GMC6003_registration/results/by_test/"
        "v3_leave_one_case_registration_comparison/folds"
    )

    model = joblib.load(model_path)

    methods = [
        "unweighted_icp",
        "distance_weighted_icp",
        "mlp_weighted_icp",
    ]

    lambda_values = [
        0.1,
        0.2,
        0.4,
        0.75,
        1.0,
    ]

    parameters = {
        "folds_dir": str(folds_dir),
        "model_path": str(model_path),
        "methods": methods,
        "lambda_values": lambda_values,
        "num_iterations": 25,
        "num_neighbors": 20,
        "distance_gate": 0.20,
        "distance_sigma": 0.05,
        "min_correspondences": 120,
        "max_correspondences": 2500,
        "fitness_threshold": 0.05,
        "max_points": 3000,
        "seed": 42,
        "evaluation_reference": "T_pseudo_gt_label.txt when available, otherwise T_mlp_weighted_svd.txt",
        "important_note": "Real errors are measured against pseudo-ground-truth transforms, not physical ground truth.",
    }

    cases = find_real_folds(folds_dir=folds_dir)

    print("\n=== Real Transform-Level Closed-Loop Registration Control ===\n")
    print(f"Found {len(cases)} real fold cases.")
    print(f"Model: {model_path}")
    print(f"Methods: {methods}")
    print(f"Lambda values: {lambda_values}")
    print("\nReal-case errors are measured relative to pseudo-GT transforms.\n")

    all_history = []

    for case in cases:
        for method in methods:
            for lambda_gain in lambda_values:
                print(
                    f"Running case={case['case_name']}, "
                    f"method={method}, lambda={lambda_gain}"
                )

                history = run_real_case(
                    model=model,
                    case=case,
                    method=method,
                    lambda_gain=float(lambda_gain),
                    num_iterations=parameters["num_iterations"],
                    num_neighbors=parameters["num_neighbors"],
                    distance_gate=parameters["distance_gate"],
                    distance_sigma=parameters["distance_sigma"],
                    min_correspondences=parameters["min_correspondences"],
                    max_correspondences=parameters["max_correspondences"],
                    fitness_threshold=parameters["fitness_threshold"],
                    max_points=parameters["max_points"],
                    seed=parameters["seed"],
                )

                all_history.extend(history)

    history_df = pd.DataFrame(all_history)

    run_keys = ["case_name", "method", "lambda_gain"]

    summary_rows = []

    for keys, group in history_df.groupby(run_keys):
        group = group.sort_values("iteration")
        final = group.iloc[-1]

        alignment_error = group["alignment_error"].to_numpy(dtype=float)
        translation_updates = group["translation_update_norm"].to_numpy(dtype=float)
        rotation_updates = group["rotation_update_deg"].to_numpy(dtype=float)

        summary_rows.append(
            {
                "case_name": keys[0],
                "method": keys[1],
                "lambda_gain": keys[2],
                "final_rmse_all": float(final["rmse_all"]),
                "final_rmse_inlier": float(final["rmse_inlier"]),
                "final_fitness": float(final["fitness"]),
                "final_rotation_error_deg": float(final["rotation_error_deg"]),
                "final_translation_error": float(final["translation_error"]),
                "final_alignment_error": float(final["alignment_error"]),
                "mean_alignment_error": float(np.mean(alignment_error)),
                "iae_alignment": float(np.sum(np.abs(alignment_error))),
                "ise_alignment": float(np.sum(alignment_error**2)),
                "tv_translation_update": float(np.sum(np.abs(np.diff(translation_updates)))),
                "tv_rotation_update": float(np.sum(np.abs(np.diff(rotation_updates)))),
                "mean_selected_correspondences": float(
                    np.mean(group["num_selected_correspondences"])
                ),
                "mean_predicted_weight": float(np.mean(group["mean_predicted_weight"])),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    global_summary = (
        summary_df.groupby(["method", "lambda_gain"])
        .agg(
            mean_final_rmse_all=("final_rmse_all", "mean"),
            mean_final_fitness=("final_fitness", "mean"),
            mean_final_rotation_error_deg=("final_rotation_error_deg", "mean"),
            mean_final_translation_error=("final_translation_error", "mean"),
            mean_final_alignment_error=("final_alignment_error", "mean"),
            mean_alignment_error=("mean_alignment_error", "mean"),
            mean_iae_alignment=("iae_alignment", "mean"),
            mean_ise_alignment=("ise_alignment", "mean"),
            mean_tv_translation_update=("tv_translation_update", "mean"),
            mean_tv_rotation_update=("tv_rotation_update", "mean"),
            mean_selected_correspondences=("mean_selected_correspondences", "mean"),
            mean_predicted_weight=("mean_predicted_weight", "mean"),
        )
        .reset_index()
        .sort_values(["method", "lambda_gain"])
    )

    best_by_method = (
        global_summary.sort_values("mean_final_alignment_error")
        .groupby("method")
        .head(1)
        .reset_index(drop=True)
    )

    best_by_case = (
        summary_df.sort_values("final_alignment_error")
        .groupby("case_name")
        .head(1)
        .reset_index(drop=True)
        .sort_values("final_alignment_error")
    )

    history_path = tables_dir / "real_transform_control_history.csv"
    summary_path = tables_dir / "real_transform_control_summary.csv"
    global_summary_path = tables_dir / "real_transform_control_global_summary.csv"
    best_by_method_path = tables_dir / "real_transform_control_best_lambda_by_method.csv"
    best_by_case_path = tables_dir / "real_transform_control_best_by_case.csv"
    metadata_path = logs_dir / "real_transform_control_metadata.json"

    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    global_summary.to_csv(global_summary_path, index=False)
    best_by_method.to_csv(best_by_method_path, index=False)
    best_by_case.to_csv(best_by_case_path, index=False)

    metadata = {
        "parameters": parameters,
        "cases": cases,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    figures = generate_real_figures(
        history_df=history_df,
        global_summary=global_summary,
        best_by_case=best_by_case,
        figures_dir=figures_dir,
    )

    print("\nGlobal real transform-control summary:")
    print(global_summary.to_string(index=False))

    print("\nBest lambda by method:")
    print(best_by_method.to_string(index=False))

    print("\nBest method per real case:")
    print(best_by_case.to_string(index=False))

    print("\nSaved tables:")
    print(f" - {history_path}")
    print(f" - {summary_path}")
    print(f" - {global_summary_path}")
    print(f" - {best_by_method_path}")
    print(f" - {best_by_case_path}")
    print(f" - {metadata_path}")

    print("\nSaved figures:")
    for figure in figures:
        print(f" - {figure}")

    print("\nReal transform-level closed-loop simulation finished successfully.\n")


if __name__ == "__main__":
    main()
