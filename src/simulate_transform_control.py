from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


FEATURE_COLUMNS = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def make_transform(rotation_xyz_deg: list[float], translation: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rotation_xyz_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def transform_normals(normals: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return normals @ R.T


def rotation_error_deg(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    R_err = T_est[:3, :3] @ T_gt[:3, :3].T
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec())))


def translation_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def interpolate_transforms(T_current: np.ndarray, T_target: np.ndarray, gain: float) -> np.ndarray:
    gain = float(np.clip(gain, 0.0, 1.0))

    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]

    R_delta = R_target @ R_current.T
    delta_rotvec = Rotation.from_matrix(R_delta).as_rotvec()

    R_next = Rotation.from_rotvec(gain * delta_rotvec).as_matrix() @ R_current
    t_next = T_current[:3, 3] + gain * (T_target[:3, 3] - T_current[:3, 3])

    T_next = np.eye(4)
    T_next[:3, :3] = R_next
    T_next[:3, 3] = t_next

    return T_next


def sample_box_surface(
    center: np.ndarray,
    size: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)

    points = np.zeros((num_points, 3), dtype=float)
    faces = rng.integers(0, 6, size=num_points)

    for i, face in enumerate(faces):
        local = rng.uniform(-0.5, 0.5, size=3) * size

        axis = face // 2
        sign = -1.0 if face % 2 == 0 else 1.0
        local[axis] = sign * size[axis] / 2.0

        points[i] = center + local

    return points


def make_asymmetric_cad_cloud(num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    parts = [
        (np.array([0.00, 0.00, 0.00]), np.array([1.20, 0.70, 0.35]), 0.40),
        (np.array([0.35, 0.20, 0.55]), np.array([0.35, 0.30, 1.10]), 0.25),
        (np.array([-0.45, -0.10, 0.35]), np.array([0.25, 0.80, 0.35]), 0.20),
        (np.array([0.20, -0.35, 0.85]), np.array([0.45, 0.20, 0.25]), 0.15),
    ]

    clouds = []

    for center, size, ratio in parts:
        n = max(20, int(num_points * ratio))
        clouds.append(sample_box_surface(center, size, n, rng))

    cloud = np.vstack(clouds)

    if len(cloud) > num_points:
        ids = rng.choice(len(cloud), size=num_points, replace=False)
        cloud = cloud[ids]

    cloud = cloud + rng.normal(0.0, 0.001, size=cloud.shape)

    return cloud


def create_target_cloud(
    source: np.ndarray,
    T_gt: np.ndarray,
    scenario: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000)

    target_clean = apply_transform(source, T_gt)

    if scenario == "clean":
        noise_std = 0.003
        outlier_ratio = 0.00
        keep_ratio = 1.00
    elif scenario == "static_noise":
        noise_std = 0.010
        outlier_ratio = 0.00
        keep_ratio = 1.00
    elif scenario == "strong_noise":
        noise_std = 0.025
        outlier_ratio = 0.05
        keep_ratio = 1.00
    elif scenario == "partial_overlap":
        noise_std = 0.010
        outlier_ratio = 0.02
        keep_ratio = 0.70
    elif scenario == "added_outliers":
        noise_std = 0.010
        outlier_ratio = 0.25
        keep_ratio = 1.00
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if keep_ratio < 1.0:
        x_values = target_clean[:, 0]
        cutoff = np.quantile(x_values, keep_ratio)
        keep_mask = x_values <= cutoff
        target = target_clean[keep_mask].copy()
    else:
        target = target_clean.copy()

    target = target + rng.normal(0.0, noise_std, size=target.shape)

    if outlier_ratio > 0:
        num_outliers = int(len(target) * outlier_ratio)

        mins = target.min(axis=0) - 0.35
        maxs = target.max(axis=0) + 0.35

        outliers = rng.uniform(mins, maxs, size=(num_outliers, 3))
        target = np.vstack([target, outliers])

    ids = rng.permutation(len(target))
    target = target[ids]

    return target


def estimate_local_geometry(
    points: np.ndarray,
    num_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1.0e-12

    n_neighbors = min(num_neighbors + 1, len(points))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(points)

    distances, indices = nn.kneighbors(points)

    normals = np.zeros_like(points)
    descriptors = np.zeros((len(points), 4), dtype=float)
    mean_neighbor_distance = np.mean(distances[:, 1:], axis=1)

    centroid = np.mean(points, axis=0)

    for i in range(len(points)):
        neigh = points[indices[i, 1:]]
        centered = neigh - np.mean(neigh, axis=0)

        covariance = centered.T @ centered / max(1, len(neigh) - 1)

        eigvals, eigvecs = np.linalg.eigh(covariance)
        order = np.argsort(eigvals)

        eigvals = np.maximum(eigvals[order], eps)
        eigvecs = eigvecs[:, order]

        normal = eigvecs[:, 0]

        if np.dot(normal, points[i] - centroid) < 0:
            normal = -normal

        normals[i] = normal

        l0, l1, l2 = eigvals
        descriptors[i, 0] = (l2 - l1) / (l2 + eps)
        descriptors[i, 1] = (l1 - l0) / (l2 + eps)
        descriptors[i, 2] = l0 / (l0 + l1 + l2 + eps)
        descriptors[i, 3] = l2

    density = 1.0 / (mean_neighbor_distance**3 + eps)

    descriptor_mean = np.mean(descriptors, axis=0)
    descriptor_std = np.std(descriptors, axis=0) + eps
    descriptors = (descriptors - descriptor_mean) / descriptor_std

    return normals, descriptors, density


def compute_correspondence_features(
    source: np.ndarray,
    target: np.ndarray,
    source_normals: np.ndarray,
    target_normals: np.ndarray,
    source_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    source_density: np.ndarray,
    target_density: np.ndarray,
    T_current: np.ndarray,
    target_tree: NearestNeighbors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1.0e-12

    source_transformed = apply_transform(source, T_current)
    source_normals_transformed = transform_normals(source_normals, T_current)

    distances, nn_indices = target_tree.kneighbors(source_transformed)
    distances = distances[:, 0]
    nn_indices = nn_indices[:, 0]

    target_corr = target[nn_indices]
    target_normals_corr = target_normals[nn_indices]

    source_tree = NearestNeighbors(n_neighbors=1)
    source_tree.fit(source_transformed)
    back_indices = source_tree.kneighbors(target, return_distance=False)[:, 0]
    is_mutual = (back_indices[nn_indices] == np.arange(len(source))).astype(float)

    normal_dot_abs = np.abs(
        np.sum(source_normals_transformed * target_normals_corr, axis=1)
    )
    normal_dot_abs = np.clip(normal_dot_abs, 0.0, 1.0)

    descriptor_dist = np.linalg.norm(
        source_descriptors - target_descriptors[nn_indices],
        axis=1,
    )

    log_density_ratio = np.log(
        (source_density + eps) / (target_density[nn_indices] + eps)
    )

    features = np.column_stack(
        [
            distances,
            normal_dot_abs,
            descriptor_dist,
            log_density_ratio,
            is_mutual,
        ]
    )

    point_to_plane_residual = np.abs(
        np.sum((target_corr - source_transformed) * target_normals_corr, axis=1)
    )

    return features, distances, nn_indices, point_to_plane_residual


def estimate_weighted_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | None:
    eps = 1.0e-12

    if len(source_points) < 3:
        return None

    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, eps)

    weight_sum = float(np.sum(weights))

    if weight_sum <= eps:
        return None

    weights = weights / weight_sum

    source_centroid = np.sum(source_points * weights[:, None], axis=0)
    target_centroid = np.sum(target_points * weights[:, None], axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    H = source_centered.T @ (target_centered * weights[:, None])

    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None

    R_est = Vt.T @ U.T

    if np.linalg.det(R_est) < 0:
        Vt[-1, :] *= -1.0
        R_est = Vt.T @ U.T

    t_est = target_centroid - R_est @ source_centroid

    T_est = np.eye(4)
    T_est[:3, :3] = R_est
    T_est[:3, 3] = t_est

    return T_est


def compute_alignment_metrics(
    source: np.ndarray,
    target: np.ndarray,
    T_current: np.ndarray,
    T_gt: np.ndarray,
    target_tree: NearestNeighbors,
    fitness_threshold: float,
) -> dict:
    source_transformed = apply_transform(source, T_current)

    distances, _ = target_tree.kneighbors(source_transformed)
    distances = distances[:, 0]

    rmse_all = float(np.sqrt(np.mean(distances**2)))

    inlier_mask = distances <= fitness_threshold
    fitness = float(np.mean(inlier_mask))

    if np.any(inlier_mask):
        rmse_inlier = float(np.sqrt(np.mean(distances[inlier_mask] ** 2)))
    else:
        rmse_inlier = float("nan")

    rot_err = rotation_error_deg(T_current, T_gt)
    trans_err = translation_error(T_current, T_gt)

    alignment_error = rmse_all + trans_err + 0.01 * rot_err

    return {
        "rmse_all": rmse_all,
        "rmse_inlier": rmse_inlier,
        "fitness": fitness,
        "rotation_error_deg": rot_err,
        "translation_error": trans_err,
        "alignment_error": alignment_error,
    }


def select_correspondences(
    distances: np.ndarray,
    weights: np.ndarray,
    distance_gate: float,
    min_correspondences: int,
) -> np.ndarray:
    valid = np.where(distances <= distance_gate)[0]

    if len(valid) >= min_correspondences:
        return valid

    order = np.argsort(distances)
    count = min(min_correspondences, len(order))

    return order[:count]


def run_single_transform_control(
    model,
    scenario: str,
    seed: int,
    method: str,
    lambda_gain: float,
    num_points: int,
    num_iterations: int,
    num_neighbors: int,
    distance_gate: float,
    distance_sigma: float,
    min_correspondences: int,
    fitness_threshold: float,
) -> list[dict]:
    source = make_asymmetric_cad_cloud(num_points=num_points, seed=seed)

    T_gt = make_transform(
        rotation_xyz_deg=[12.0, -6.0, 18.0],
        translation=[0.45, -0.25, 0.18],
    )

    target = create_target_cloud(
        source=source,
        T_gt=T_gt,
        scenario=scenario,
        seed=seed,
    )

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

    T_perturb = make_transform(
        rotation_xyz_deg=[8.0, -4.0, 7.0],
        translation=[0.12, -0.08, 0.05],
    )

    T_current = T_perturb @ T_gt

    history = []

    previous_translation_update = 0.0
    previous_rotation_update_deg = 0.0

    for iteration in range(num_iterations + 1):
        metrics = compute_alignment_metrics(
            source=source,
            target=target,
            T_current=T_current,
            T_gt=T_gt,
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

        selected = select_correspondences(
            distances=distances,
            weights=solver_weights,
            distance_gate=distance_gate,
            min_correspondences=min_correspondences,
        )

        selected_weights = solver_weights[selected]
        selected_source = source[selected]
        selected_target = target[nn_indices[selected]]

        mean_predicted_weight = float(np.mean(predicted_weights[selected]))
        mean_selected_distance = float(np.mean(distances[selected]))
        mean_selected_point_to_plane = float(np.mean(point_to_plane_residual[selected]))

        history.append(
            {
                "scenario": scenario,
                "seed": seed,
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

        R_delta = T_next[:3, :3] @ T_current[:3, :3].T
        previous_rotation_update_deg = float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(R_delta).as_rotvec()))
        )

        T_current = T_next

    return history


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_transform_figures(
    history_df: pd.DataFrame,
    global_summary: pd.DataFrame,
    figures_dir: Path,
) -> list[dict]:
    figures = []

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
    plt.ylabel("Mean final RMSE")
    plt.title("Transform-control final RMSE versus lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = figures_dir / "fig_09_transform_final_rmse_vs_lambda.png"
    save_figure(path)
    figures.append(str(path))

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
    plt.ylabel("Mean final alignment error")
    plt.title("Transform-control final alignment error versus lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = figures_dir / "fig_10_transform_final_alignment_error_vs_lambda.png"
    save_figure(path)
    figures.append(str(path))

    example = history_df[
        (history_df["scenario"] == "static_noise")
        & (history_df["seed"] == 0)
        & (history_df["lambda_gain"].isin([0.05, 0.2, 1.0]))
    ].copy()

    if len(example) == 0:
        example = history_df[
            (history_df["seed"] == history_df["seed"].min())
            & (history_df["lambda_gain"].isin([0.05, 0.2, 1.0]))
        ].copy()

    plt.figure(figsize=(9, 5))
    for (method, lambda_gain), group in example.groupby(["method", "lambda_gain"]):
        group = group.sort_values("iteration")
        plt.plot(
            group["iteration"],
            group["rmse_all"],
            marker="o",
            label=f"{method.replace('_', ' ')}, lambda={lambda_gain}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("Example transform-control RMSE evolution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    path = figures_dir / "fig_11_transform_example_rmse_history.png"
    save_figure(path)
    figures.append(str(path))

    plt.figure(figsize=(9, 5))
    for (method, lambda_gain), group in example.groupby(["method", "lambda_gain"]):
        group = group.sort_values("iteration")
        plt.plot(
            group["iteration"],
            group["translation_error"],
            marker="o",
            label=f"{method.replace('_', ' ')}, lambda={lambda_gain}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Translation error")
    plt.title("Example transform-control translation error evolution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    path = figures_dir / "fig_12_transform_example_translation_error_history.png"
    save_figure(path)
    figures.append(str(path))

    plt.figure(figsize=(9, 5))
    for (method, lambda_gain), group in example.groupby(["method", "lambda_gain"]):
        group = group.sort_values("iteration")
        plt.plot(
            group["iteration"],
            group["rotation_error_deg"],
            marker="o",
            label=f"{method.replace('_', ' ')}, lambda={lambda_gain}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Rotation error (deg)")
    plt.title("Example transform-control rotation error evolution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    path = figures_dir / "fig_13_transform_example_rotation_error_history.png"
    save_figure(path)
    figures.append(str(path))

    return figures


def main() -> None:
    config = load_config("configs/default.yaml")

    output_dir = Path(config["paths"]["output_dir"])
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

    model = joblib.load(model_path)

    scenarios = [
        "clean",
        "static_noise",
        "strong_noise",
        "partial_overlap",
        "added_outliers",
    ]

    methods = [
        "unweighted_icp",
        "distance_weighted_icp",
        "mlp_weighted_icp",
    ]

    lambda_values = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
        0.40,
        0.75,
        1.00,
    ]

    seeds = [0, 1, 2]

    parameters = {
        "num_points": 1800,
        "num_iterations": 30,
        "num_neighbors": 20,
        "distance_gate": 0.35,
        "distance_sigma": 0.15,
        "min_correspondences": 250,
        "fitness_threshold": 0.05,
        "lambda_values": lambda_values,
        "scenarios": scenarios,
        "methods": methods,
        "seeds": seeds,
        "control_objective": "reduce transform-level alignment error",
        "controlled_variable": "alignment error measured from RMSE, translation error, and rotation error",
        "control_input": "filtered rigid transform correction",
    }

    all_history = []

    print("\n=== Transform-Level Closed-Loop Registration Control ===\n")
    print(f"Model: {model_path}")
    print(f"Scenarios: {scenarios}")
    print(f"Methods: {methods}")
    print(f"Lambda values: {lambda_values}")
    print(f"Seeds: {seeds}\n")

    for scenario in scenarios:
        for seed in seeds:
            for method in methods:
                for lambda_gain in lambda_values:
                    print(
                        f"Running scenario={scenario}, seed={seed}, "
                        f"method={method}, lambda={lambda_gain}"
                    )

                    history = run_single_transform_control(
                        model=model,
                        scenario=scenario,
                        seed=seed,
                        method=method,
                        lambda_gain=float(lambda_gain),
                        num_points=parameters["num_points"],
                        num_iterations=parameters["num_iterations"],
                        num_neighbors=parameters["num_neighbors"],
                        distance_gate=parameters["distance_gate"],
                        distance_sigma=parameters["distance_sigma"],
                        min_correspondences=parameters["min_correspondences"],
                        fitness_threshold=parameters["fitness_threshold"],
                    )

                    all_history.extend(history)

    history_df = pd.DataFrame(all_history)

    run_keys = ["scenario", "seed", "method", "lambda_gain"]

    summary_rows = []

    for keys, group in history_df.groupby(run_keys):
        group = group.sort_values("iteration")
        final = group.iloc[-1]

        alignment_error = group["alignment_error"].to_numpy(dtype=float)
        translation_updates = group["translation_update_norm"].to_numpy(dtype=float)
        rotation_updates = group["rotation_update_deg"].to_numpy(dtype=float)

        summary = {
            "scenario": keys[0],
            "seed": keys[1],
            "method": keys[2],
            "lambda_gain": keys[3],
            "final_rmse_all": float(final["rmse_all"]),
            "final_rmse_inlier": float(final["rmse_inlier"]),
            "final_fitness": float(final["fitness"]),
            "final_rotation_error_deg": float(final["rotation_error_deg"]),
            "final_translation_error": float(final["translation_error"]),
            "final_alignment_error": float(final["alignment_error"]),
            "iae_alignment": float(np.sum(np.abs(alignment_error))),
            "ise_alignment": float(np.sum(alignment_error**2)),
            "mean_alignment_error": float(np.mean(alignment_error)),
            "tv_translation_update": float(np.sum(np.abs(np.diff(translation_updates)))),
            "tv_rotation_update": float(np.sum(np.abs(np.diff(rotation_updates)))),
            "mean_selected_correspondences": float(
                np.mean(group["num_selected_correspondences"])
            ),
            "mean_predicted_weight": float(np.mean(group["mean_predicted_weight"])),
        }

        summary_rows.append(summary)

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

    scenario_summary = (
        summary_df.groupby(["scenario", "method", "lambda_gain"])
        .agg(
            mean_final_rmse_all=("final_rmse_all", "mean"),
            mean_final_fitness=("final_fitness", "mean"),
            mean_final_rotation_error_deg=("final_rotation_error_deg", "mean"),
            mean_final_translation_error=("final_translation_error", "mean"),
            mean_final_alignment_error=("final_alignment_error", "mean"),
            mean_alignment_error=("mean_alignment_error", "mean"),
            mean_iae_alignment=("iae_alignment", "mean"),
            mean_ise_alignment=("ise_alignment", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "method", "lambda_gain"])
    )

    best_by_method = (
        global_summary.sort_values("mean_final_alignment_error")
        .groupby("method")
        .head(1)
        .reset_index(drop=True)
    )

    history_path = tables_dir / "transform_control_history.csv"
    summary_path = tables_dir / "transform_control_summary.csv"
    global_summary_path = tables_dir / "transform_control_global_summary.csv"
    scenario_summary_path = tables_dir / "transform_control_scenario_summary.csv"
    best_path = tables_dir / "transform_control_best_lambda_by_method.csv"
    metadata_path = logs_dir / "transform_control_metadata.json"

    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    global_summary.to_csv(global_summary_path, index=False)
    scenario_summary.to_csv(scenario_summary_path, index=False)
    best_by_method.to_csv(best_path, index=False)

    with open(metadata_path, "w") as f:
        json.dump(parameters, f, indent=2)

    figures = generate_transform_figures(
        history_df=history_df,
        global_summary=global_summary,
        figures_dir=figures_dir,
    )

    print("\nGlobal transform-control summary:")
    print(global_summary.to_string(index=False))

    print("\nBest lambda by method:")
    print(best_by_method.to_string(index=False))

    print("\nSaved tables:")
    print(f" - {history_path}")
    print(f" - {summary_path}")
    print(f" - {global_summary_path}")
    print(f" - {scenario_summary_path}")
    print(f" - {best_path}")
    print(f" - {metadata_path}")

    print("\nSaved figures:")
    for figure in figures:
        print(f" - {figure}")

    print("\nTransform-level closed-loop simulation finished successfully.\n")


if __name__ == "__main__":
    main()
