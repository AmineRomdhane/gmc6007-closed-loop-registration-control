# GMC6007 - Closed-Loop Control of Weighted Point Cloud Registration

This project studies closed-loop control for weighted point cloud registration.

Point cloud registration is an important operation in SLAM because scan alignment affects pose estimation and map consistency. The project extends a previously trained correspondence-reliability multilayer perceptron (MLP) by integrating it into an iterative registration loop.

## Objective

The objective is to evaluate whether learned correspondence reliability can improve rigid point cloud registration when used inside a closed-loop transform update.

Three registration methods are compared:

- Unweighted ICP
- Distance-weighted ICP
- MLP-weighted ICP

The project includes:

- a preliminary MLP threshold-control experiment;
- a transform-level closed-loop controller;
- synthetic experiments with known ground truth;
- real CAD-to-scan experiments using pseudo-ground-truth transforms.

---

## Repository Structure

```text
gmc6007-closed-loop-registration-control/
├── configs/
│   └── default.yaml
├── data/
├── models/
│   ├── reliability_mlp_sklearn.joblib
│   └── reliability_mlp_sklearn_metadata.json
├── outputs/
│   ├── figures/
│   ├── logs/
│   ├── matrices/
│   └── tables/
├── report/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── src/
│   ├── generate_report_figures.py
│   ├── inspect_dataset.py
│   ├── simulate_closed_loop_threshold_control.py
│   ├── simulate_real_transform_control.py
│   ├── simulate_transform_control.py
│   └── train_reliability_mlp.py
├── requirements.txt
└── README.md
```

---

# Reproducible Example

The synthetic transform-control experiment can be reproduced directly from this repository.

The trained MLP model is already included in:

```text
models/reliability_mlp_sklearn.joblib
```

The synthetic experiment generates its own point clouds and evaluates the three registration methods using known ground truth.

## 1. Clone the repository

```bash
git clone https://github.com/AmineRomdhane/gmc6007-closed-loop-registration-control.git
cd gmc6007-closed-loop-registration-control
```

## 2. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the synthetic closed-loop experiment

From the repository root:

```bash
python src/simulate_transform_control.py
```

The experiment compares:

- unweighted ICP;
- distance-weighted ICP;
- MLP-weighted ICP.

The tested transform-update gains are:

```text
0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.75, 1.00
```

The synthetic scenarios include:

- clean;
- static noise;
- strong noise;
- partial overlap;
- added outliers.

The experiment uses known ground-truth transforms, allowing direct evaluation of rotation and translation errors.

## 5. Generate the report figures

```bash
python src/generate_report_figures.py
```

Generated figures are stored in:

```text
outputs/figures/
```

The numerical results are stored in:

```text
outputs/tables/
```

and experiment metadata are stored in:

```text
outputs/logs/
```

## 6. Inspect the main synthetic results

The main result files are:

```text
outputs/tables/transform_control_summary.csv
outputs/tables/transform_control_global_summary.csv
outputs/tables/transform_control_best_lambda_by_method.csv
outputs/tables/transform_control_scenario_summary.csv
outputs/tables/transform_control_history.csv
```

The main generated figures include:

```text
outputs/figures/fig_09_transform_final_rmse_vs_lambda.png
outputs/figures/fig_10_transform_final_alignment_error_vs_lambda.png
outputs/figures/fig_11_transform_example_rmse_history.png
outputs/figures/fig_12_transform_example_translation_error_history.png
outputs/figures/fig_13_transform_example_rotation_error_history.png
```

---

# Registration Loop

At each iteration:

1. The source cloud is transformed using the current transform.
2. Nearest-neighbor correspondences are computed.
3. Correspondence features are evaluated.
4. The MLP predicts correspondence reliability.
5. Solver weights are computed.
6. Weighted SVD estimates a rigid transform.
7. A filter gain controls how much of the estimated correction is applied.
8. The new transform is used at the next iteration.

The transform update is

\[
T_{k+1}
=
\mathrm{Interp}
\left(
T_k,
T_{\mathrm{est}},
\lambda
\right).
\]

The parameter \(\lambda\) controls how much of the estimated transform correction is applied at each iteration.

---

# Preliminary Threshold-Control Experiment

The preliminary experiment controls the MLP correspondence acceptance threshold.

It can be run with:

```bash
python src/simulate_closed_loop_threshold_control.py
```

This experiment requires the correspondence dataset produced in the previous assignment.

The dataset used for the original experiments is:

```text
all_correspondences.csv
```

from the previous GMC6003/GMC6007 correspondence-learning dataset.

The original development path was:

```text
/home/amine/GMC6003_registration/results/
learning_data_synthetic_plus_real_curated_clean_v3/
all_correspondences.csv
```

This external dataset is not included in this repository.

Therefore, reproducing the threshold-control experiment requires access to the original correspondence dataset.

The threshold experiment generates results including:

```text
outputs/tables/closed_loop_threshold_history.csv
outputs/tables/closed_loop_threshold_summary.csv
outputs/tables/closed_loop_threshold_scenario_summary.csv
outputs/tables/control_lambda_global_summary.csv
```

and figures including:

```text
outputs/figures/fig_02_ise_vs_lambda_by_scenario.png
outputs/figures/fig_03_tv_tau_vs_lambda_by_scenario.png
outputs/figures/fig_04_f1_vs_lambda_by_scenario.png
outputs/figures/fig_05_global_ise_vs_lambda.png
outputs/figures/fig_06_global_tv_tau_vs_lambda.png
outputs/figures/fig_07_example_tau_history.png
outputs/figures/fig_08_example_control_error_history.png
```

---

# MLP Training

The reliability MLP can be trained with:

```bash
python src/train_reliability_mlp.py
```

Training also requires the correspondence dataset from the previous assignment.

The trained model used by the experiments is already included in the repository:

```text
models/reliability_mlp_sklearn.joblib
```

The model architecture is:

```text
5 -> 32 -> 16 -> 1
```

The five correspondence features are:

```text
distance_T0
normal_dot_abs
fpfh_distance
log_normalized_density_ratio
is_mutual_nn
```

---

# Real CAD-to-Scan Experiment

The real transform-control experiment can be launched with:

```bash
python src/simulate_real_transform_control.py
```

The real experiment requires the original leave-one-real-case data from the previous registration project.

Each real case uses:

```text
reference_downsampled.ply
observation_downsampled.ply
T_initial_T0.txt
T_pseudo_gt_label.txt
```

These point-cloud datasets are not included in this repository.

The generated real-case results are stored in:

```text
outputs/tables/real_transform_control_summary.csv
outputs/tables/real_transform_control_global_summary.csv
outputs/tables/real_transform_control_best_lambda_by_method.csv
outputs/tables/real_transform_control_best_by_case.csv
outputs/tables/real_transform_control_case_winners.csv
outputs/tables/real_transform_control_win_count.csv
outputs/tables/real_transform_control_history.csv
```

---

# Main Results

## Synthetic Experiments

Best tested configurations:

| Method | Lambda | RMSE | Fitness | Rotation error | Translation error | Alignment error |
|---|---:|---:|---:|---:|---:|---:|
| Distance-weighted ICP | 1.00 | 0.0310 | 0.9396 | 0.376 deg | 0.00537 m | 0.0402 |
| MLP-weighted ICP | 0.75 | 0.0320 | 0.9394 | 0.196 deg | 0.00246 m | 0.0365 |
| Unweighted ICP | 1.00 | 0.0294 | 0.8961 | 0.620 deg | 0.01856 m | 0.0541 |

MLP-weighted ICP gives the lowest transform-level alignment error in the synthetic experiments.

## Real Experiments

Best average configurations:

| Method | Lambda | RMSE | Fitness | Rotation error | Translation error | Alignment error |
|---|---:|---:|---:|---:|---:|---:|
| Unweighted ICP | 0.75 | 0.1027 | 0.5229 | 3.246 deg | 0.0986 m | 0.2338 |
| Distance-weighted ICP | 1.00 | 0.1094 | 0.5319 | 2.463 deg | 0.1254 m | 0.2594 |
| MLP-weighted ICP | 1.00 | 0.1126 | 0.5273 | 3.262 deg | 0.1408 m | 0.2860 |

Winner count over the 12 real cases:

```text
Unweighted ICP        7/12
MLP-weighted ICP      3/12
Distance-weighted ICP 2/12
```

The learned weighting improves some real registrations, while unweighted ICP gives the best average result on the current real dataset.

---

# Reproducibility Summary

The repository contains everything required to reproduce the **synthetic transform-control experiment**, including:

- source code;
- configuration;
- trained MLP model;
- Python dependencies;
- fixed experimental setup;
- generated reference outputs.

The following experiments require external data from the previous project:

| Experiment | External data required |
|---|---|
| Synthetic transform control | No |
| Report figure generation | No |
| MLP training | Yes |
| Threshold control | Yes |
| Real transform control | Yes |

This separation allows the main closed-loop registration example to be reproduced without access to the original real-world dataset.

---

# Report

The final GMC6007 report is included in:

```text
report/main.tex
report/references.bib
```

The figures used by the report are stored in:

```text
report/figures/
```

---

# Author

Amine Romdhane  
Université du Québec à Trois-Rivières (UQTR)
