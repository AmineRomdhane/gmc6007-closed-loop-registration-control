# GMC6007 - Closed-Loop Control of Weighted Point Cloud Registration

This project studies closed-loop control for weighted point cloud registration.

Point cloud registration is an important operation in SLAM because scan alignment directly affects pose estimation and map consistency. The project extends a previously trained correspondence-reliability MLP by integrating it into an iterative registration loop.

## Objective

The objective is to evaluate whether learned correspondence reliability can improve rigid point cloud registration when used inside a closed-loop transform update.

Three registration methods are compared:

- Unweighted ICP
- Distance-weighted ICP
- MLP-weighted ICP

The experiments include:

- a preliminary MLP threshold-control loop;
- a transform-level closed-loop controller;
- synthetic point-cloud tests with known ground truth;
- real CAD-to-scan tests using pseudo-ground-truth transforms.

## Method

At each transform-control iteration:

1. The source cloud is transformed using the current transform.
2. Nearest-neighbor correspondences are computed.
3. Correspondence features are evaluated.
4. The MLP predicts correspondence reliability.
5. Solver weights are computed.
6. Weighted SVD estimates a new rigid transform.
7. A filter gain controls how much of the estimated correction is applied.
8. The new transform is reused at the next iteration.

The transform update is written as

\[
T_{k+1} = \mathrm{Interp}(T_k,T_{\mathrm{est}},\lambda)
\]

where \(\lambda\) controls the applied correction.

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
├── src/
│   ├── generate_report_figures.py
│   ├── inspect_dataset.py
│   ├── simulate_closed_loop_threshold_control.py
│   ├── simulate_real_transform_control.py
│   ├── simulate_transform_control.py
│   └── train_reliability_mlp.py
├── requirements.txt
└── README.md
