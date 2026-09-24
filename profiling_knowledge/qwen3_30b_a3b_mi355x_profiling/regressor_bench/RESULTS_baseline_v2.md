# Baseline v2: RF, XGBoost (exact), linear interpolation, and the measurement noise floor

Sealed random-geometry shape-level split (2,496 train / 499 test token values, identical for all 15 regressors); feature
`num_tokens`; label `time_stats.<op>.median` (ms); 10 shape-level CV folds on train; test scored once. **Holdout not read.**
Produced by `baseline_v2.py`; raw outputs in `results/baseline_v2/`.

* **XGBoost**: `tree_method="exact"`, fresh search over max_depth {4,6,8,10,12} × learning_rate {0.02,0.05,0.1,0.2} ×
  reg_lambda {0,0.01,0.1,1} × min_child_weight {1,2,5} × subsample {0.8,1.0} (480), 48 sampled without replacement,
  n_estimators cap 10,000 with early stopping (50 rounds) on a fixed 15 % shape-level carve-out, refit on all train rows at
  the best iteration. No winner hit the cap.
* **RF**: rows carried over from `results/baseline_rf_xgb` (identical protocol, deterministic); refit for predictions.
* **interp**: piecewise-linear interpolation over the training grid; no parameters.
* **Noise floor**: MAPE between the medians of two independent collections with the same measurement fixes on the
  test-tier tokens — job 21519 (work backlog, settle 3) vs job 21539 (settle 8, the training file). It is the error of one
  collection predicting the other and therefore carries both collections' noise; the single-collection floor is ≈ that / √2.

## Test-tier MAPE (%) next to the noise floor

| op | TP | noise floor (21519 vs 21539) | noise floor / √2 | interp test | RF test | XGB exact test |
|---|---|---|---|---|---|---|
| attn_post_proj | 1 | 1.29 | 0.91 | 1.38 | 1.35 | 1.47 |
| attn_post_proj | 2 | 1.23 | 0.87 | 1.51 | 1.42 | 1.46 |
| attn_post_proj | 4 | 1.02 | 0.72 | 1.25 | 1.24 | 1.38 |
| attn_post_proj | 8 | 1.22 | 0.86 | 1.52 | 1.32 | 1.38 |
| attn_pre_proj | 1 | 1.07 | 0.76 | 1.35 | 1.33 | 1.33 |
| attn_pre_proj | 2 | 0.96 | 0.68 | 1.19 | 1.25 | 1.22 |
| attn_pre_proj | 4 | 0.92 | 0.65 | 1.32 | 1.20 | 1.21 |
| attn_pre_proj | 8 | 0.89 | 0.63 | 1.40 | 1.32 | 1.25 |
| attn_rope | 1 | 1.52 | 1.08 | 1.54 | 1.25 | 1.28 |
| attn_rope | 2 | 1.45 | 1.02 | 1.39 | 1.33 | 1.30 |
| attn_rope | 4 | 1.25 | 0.88 | 1.21 | 1.02 | 1.06 |
| attn_rope | 8 | 1.16 | 0.82 | 1.23 | 0.99 | 1.02 |
| emb | 1 | 1.48 | 1.05 | 1.66 | 1.45 | 1.41 |
| input_layernorm | 1 | 1.18 | 0.83 | 1.41 | 1.24 | 1.24 |
| post_attention_layernorm | 1 | 1.19 | 0.84 | 1.21 | 1.02 | 1.07 |

Means over the 15 regressors:

| model | cv_mape_mean | train_mape | test_mape | test_cliff_mape | noise_floor_mape | noise_floor_single |
|---|---|---|---|---|---|---|
| interp | 1.41 | 0.00 | 1.37 | 1.43 | 1.19 | 0.84 |
| rf | 1.28 | 0.93 | 1.25 | 1.18 | 1.19 | 0.84 |
| xgb_exact | 1.29 | 1.02 | 1.27 | 1.18 | 1.19 | 0.84 |

## CV MAPE of the selected configurations

| op | TP | interp CV | RF CV | XGB exact CV |
|---|---|---|---|---|
| attn_post_proj | 1 | 1.51 | 1.43 | 1.41 |
| attn_post_proj | 2 | 1.51 | 1.44 | 1.42 |
| attn_post_proj | 4 | 1.49 | 1.52 | 1.54 |
| attn_post_proj | 8 | 1.58 | 1.47 | 1.54 |
| attn_pre_proj | 1 | 1.34 | 1.25 | 1.25 |
| attn_pre_proj | 2 | 1.12 | 1.10 | 1.10 |
| attn_pre_proj | 4 | 1.18 | 1.09 | 1.07 |
| attn_pre_proj | 8 | 1.28 | 1.17 | 1.13 |
| attn_rope | 1 | 1.62 | 1.36 | 1.37 |
| attn_rope | 2 | 1.39 | 1.36 | 1.35 |
| attn_rope | 4 | 1.23 | 1.04 | 1.09 |
| attn_rope | 8 | 1.21 | 0.98 | 1.02 |
| emb | 1 | 1.75 | 1.51 | 1.48 |
| input_layernorm | 1 | 1.55 | 1.33 | 1.31 |
| post_attention_layernorm | 1 | 1.43 | 1.23 | 1.22 |

## Cliff-window rows (the 6 test tokens inside 11800-12300; no test token falls in the four narrow windows)

| op | TP | interp cliff | RF cliff | XGB exact cliff | noise cliff |
|---|---|---|---|---|---|
| attn_post_proj | 1 | 0.94 | 0.87 | 0.79 | 0.87 |
| attn_post_proj | 2 | 0.71 | 0.67 | 0.49 | 0.66 |
| attn_post_proj | 4 | 0.53 | 0.66 | 0.53 | 0.27 |
| attn_post_proj | 8 | 1.57 | 1.43 | 1.17 | 1.29 |
| attn_pre_proj | 1 | 0.89 | 0.82 | 0.78 | 0.46 |
| attn_pre_proj | 2 | 1.71 | 1.50 | 1.49 | 0.94 |
| attn_pre_proj | 4 | 1.20 | 1.11 | 0.99 | 0.70 |
| attn_pre_proj | 8 | 2.16 | 1.79 | 1.77 | 0.80 |
| attn_rope | 1 | 1.56 | 1.36 | 1.43 | 0.68 |
| attn_rope | 2 | 0.65 | 0.63 | 0.80 | 0.81 |
| attn_rope | 4 | 1.62 | 0.88 | 1.04 | 0.95 |
| attn_rope | 8 | 0.70 | 0.66 | 0.55 | 1.29 |
| emb | 1 | 2.31 | 1.91 | 2.01 | 2.59 |
| input_layernorm | 1 | 3.64 | 2.53 | 2.82 | 0.90 |
| post_attention_layernorm | 1 | 1.22 | 0.86 | 0.99 | 0.42 |

## XGBoost (exact) winners

| op | tp | best_params | xgb_best_iteration | hit_cap | edge_params | cv_mape_mean | test_mape |
|---|---|---|---|---|---|---|---|
| attn_post_proj | 1 | {"max_depth": 8, "learning_rate": 0.05, "reg_lambda": 0.01, "min_child_weight": 5, "subsample": 0.8} | 221.00 | False | min_child_weight=5(max); subsample=0.8(min) | 1.41 | 1.47 |
| attn_post_proj | 2 | {"max_depth": 8, "learning_rate": 0.05, "reg_lambda": 0.01, "min_child_weight": 1, "subsample": 0.8} | 204.00 | False | min_child_weight=1(min); subsample=0.8(min) | 1.42 | 1.46 |
| attn_post_proj | 4 | {"max_depth": 12, "learning_rate": 0.1, "reg_lambda": 0, "min_child_weight": 1, "subsample": 1.0} | 67.00 | False | max_depth=12(max); reg_lambda=0(min); min_child_weight=1(min); subsample=1.0(max) | 1.54 | 1.38 |
| attn_post_proj | 8 | {"max_depth": 12, "learning_rate": 0.02, "reg_lambda": 0.01, "min_child_weight": 2, "subsample": 0.8} | 644.00 | False | max_depth=12(max); learning_rate=0.02(min); subsample=0.8(min) | 1.54 | 1.38 |
| attn_pre_proj | 1 | {"max_depth": 10, "learning_rate": 0.02, "reg_lambda": 1, "min_child_weight": 5, "subsample": 0.8} | 495.00 | False | learning_rate=0.02(min); reg_lambda=1(max); min_child_weight=5(max); subsample=0.8(min) | 1.25 | 1.33 |
| attn_pre_proj | 2 | {"max_depth": 6, "learning_rate": 0.05, "reg_lambda": 0.1, "min_child_weight": 2, "subsample": 0.8} | 167.00 | False | subsample=0.8(min) | 1.10 | 1.22 |
| attn_pre_proj | 4 | {"max_depth": 8, "learning_rate": 0.1, "reg_lambda": 0.1, "min_child_weight": 1, "subsample": 0.8} | 99.00 | False | min_child_weight=1(min); subsample=0.8(min) | 1.07 | 1.21 |
| attn_pre_proj | 8 | {"max_depth": 8, "learning_rate": 0.1, "reg_lambda": 0.1, "min_child_weight": 1, "subsample": 0.8} | 216.00 | False | min_child_weight=1(min); subsample=0.8(min) | 1.13 | 1.25 |
| attn_rope | 1 | {"max_depth": 8, "learning_rate": 0.05, "reg_lambda": 0.01, "min_child_weight": 5, "subsample": 0.8} | 217.00 | False | min_child_weight=5(max); subsample=0.8(min) | 1.37 | 1.28 |
| attn_rope | 2 | {"max_depth": 6, "learning_rate": 0.05, "reg_lambda": 0.1, "min_child_weight": 2, "subsample": 0.8} | 141.00 | False | subsample=0.8(min) | 1.35 | 1.30 |
| attn_rope | 4 | {"max_depth": 6, "learning_rate": 0.05, "reg_lambda": 0.1, "min_child_weight": 2, "subsample": 0.8} | 195.00 | False | subsample=0.8(min) | 1.09 | 1.06 |
| attn_rope | 8 | {"max_depth": 12, "learning_rate": 0.02, "reg_lambda": 0.01, "min_child_weight": 2, "subsample": 0.8} | 386.00 | False | max_depth=12(max); learning_rate=0.02(min); subsample=0.8(min) | 1.02 | 1.02 |
| emb | 1 | {"max_depth": 10, "learning_rate": 0.1, "reg_lambda": 0.1, "min_child_weight": 5, "subsample": 0.8} | 179.00 | False | min_child_weight=5(max); subsample=0.8(min) | 1.48 | 1.41 |
| input_layernorm | 1 | {"max_depth": 8, "learning_rate": 0.1, "reg_lambda": 0.01, "min_child_weight": 5, "subsample": 1.0} | 69.00 | False | min_child_weight=5(max); subsample=1.0(max) | 1.31 | 1.24 |
| post_attention_layernorm | 1 | {"max_depth": 8, "learning_rate": 0.1, "reg_lambda": 0, "min_child_weight": 5, "subsample": 0.8} | 96.00 | False | reg_lambda=0(min); min_child_weight=5(max); subsample=0.8(min) | 1.22 | 1.07 |

## Steps detected on the train tier, attn_post_proj

Rule: median label of the 4 grid points after a gap vs the 4 before differs by ≥ 6 %, tokens ≥ 256; adjacent detections merged.
`known_window` marks the five hipBLASLt windows (±8 tokens). Full list for all pairs in `results/baseline_v2/train_steps.csv`.

| op | tp | from_tokens | to_tokens | jump_pct | known_window |
|---|---|---|---|---|---|
| attn_post_proj | 1 | 256 | 257 | 19.57 |  |
| attn_post_proj | 1 | 515 | 516 | 12.04 |  |
| attn_post_proj | 1 | 768 | 769 | 9.98 |  |
| attn_post_proj | 1 | 1024 | 1026 | 17.21 |  |
| attn_post_proj | 1 | 1246 | 1247 | -6.13 |  |
| attn_post_proj | 1 | 1536 | 1537 | 10.88 |  |
| attn_post_proj | 1 | 2045 | 2046 | 13.50 |  |
| attn_post_proj | 1 | 2296 | 2304 | -7.65 |  |
| attn_post_proj | 1 | 2568 | 2576 | 14.49 |  |
| attn_post_proj | 1 | 3088 | 3096 | 10.38 |  |
| attn_post_proj | 1 | 4232 | 4240 | 16.51 |  |
| attn_post_proj | 1 | 4416 | 4424 | -9.06 |  |
| attn_post_proj | 1 | 4816 | 4824 | 10.94 |  |
| attn_post_proj | 1 | 5120 | 5128 | 19.70 |  |
| attn_post_proj | 1 | 5376 | 5384 | -9.88 | 5376-5384 |
| attn_post_proj | 1 | 6392 | 6416 | 6.04 |  |
| attn_post_proj | 1 | 8184 | 8192 | 30.56 |  |
| attn_post_proj | 1 | 8704 | 8736 | 12.76 |  |
| attn_post_proj | 1 | 9232 | 9248 | 11.43 |  |
| attn_post_proj | 1 | 9968 | 9984 | -10.16 |  |
| attn_post_proj | 1 | 10784 | 10800 | 12.67 |  |
| attn_post_proj | 1 | 10992 | 11040 | -7.73 |  |
| attn_post_proj | 2 | 385 | 386 | 8.91 |  |
| attn_post_proj | 2 | 768 | 769 | 9.45 |  |
| attn_post_proj | 2 | 1024 | 1026 | 19.44 |  |
| attn_post_proj | 2 | 1341 | 1344 | -9.94 |  |
| attn_post_proj | 2 | 2216 | 2224 | 13.69 |  |
| attn_post_proj | 2 | 2560 | 2568 | 21.49 |  |
| attn_post_proj | 2 | 4320 | 4328 | 23.41 |  |
| attn_post_proj | 2 | 4816 | 4824 | 18.74 |  |
| attn_post_proj | 2 | 5112 | 5120 | 28.28 |  |
| attn_post_proj | 2 | 5376 | 5384 | -19.84 | 5376-5384 |
| attn_post_proj | 2 | 6384 | 6392 | 11.06 |  |
| attn_post_proj | 2 | 8208 | 8224 | 18.02 |  |
| attn_post_proj | 2 | 8704 | 8736 | 19.44 |  |
| attn_post_proj | 2 | 8960 | 8976 | -12.93 |  |
| attn_post_proj | 2 | 9232 | 9248 | 18.72 |  |
| attn_post_proj | 2 | 9968 | 9984 | -13.90 |  |
| attn_post_proj | 2 | 13056 | 13072 | 9.67 |  |
| attn_post_proj | 2 | 14576 | 14592 | -8.80 |  |
| attn_post_proj | 4 | 511 | 512 | 7.90 |  |
| attn_post_proj | 4 | 1088 | 1089 | 9.53 |  |
| attn_post_proj | 4 | 2568 | 2576 | 28.70 |  |
| attn_post_proj | 4 | 2936 | 2944 | -16.50 |  |
| attn_post_proj | 4 | 4352 | 4360 | 33.63 | 4352-4360 |
| attn_post_proj | 4 | 4864 | 4880 | -25.04 | 4864-4872 |
| attn_post_proj | 4 | 5184 | 5192 | 34.07 |  |
| attn_post_proj | 4 | 5376 | 5384 | -26.58 | 5376-5384 |
| attn_post_proj | 4 | 8192 | 8208 | 65.53 |  |
| attn_post_proj | 4 | 8656 | 8704 | -6.31 |  |
| attn_post_proj | 4 | 8912 | 8928 | -18.82 |  |
| attn_post_proj | 4 | 9216 | 9232 | 28.95 |  |
| attn_post_proj | 4 | 9456 | 9472 | -21.77 |  |
| attn_post_proj | 4 | 10256 | 10288 | 26.51 |  |
| attn_post_proj | 4 | 10752 | 10784 | 20.19 |  |
| attn_post_proj | 4 | 11280 | 11312 | 10.00 |  |
| attn_post_proj | 4 | 11760 | 11776 | -21.79 |  |
| attn_post_proj | 4 | 13056 | 13072 | 14.81 |  |
| attn_post_proj | 8 | 512 | 515 | 15.54 |  |
| attn_post_proj | 8 | 704 | 705 | -6.13 |  |
| attn_post_proj | 8 | 1024 | 1026 | 27.18 |  |
| attn_post_proj | 8 | 2046 | 2056 | 32.02 |  |
| attn_post_proj | 8 | 3088 | 3096 | 41.94 |  |
| attn_post_proj | 8 | 4232 | 4240 | 46.56 |  |
| attn_post_proj | 8 | 5128 | 5136 | 10.80 |  |
| attn_post_proj | 8 | 8184 | 8192 | 23.14 |  |
| attn_post_proj | 8 | 8816 | 8848 | -9.85 |  |
| attn_post_proj | 8 | 9232 | 9248 | 49.28 |  |
| attn_post_proj | 8 | 10224 | 10256 | -10.75 |  |
| attn_post_proj | 8 | 10752 | 10784 | 31.61 |  |
| attn_post_proj | 8 | 11136 | 11184 | -25.95 |  |
| attn_post_proj | 8 | 11424 | 11440 | 37.72 |  |
| attn_post_proj | 8 | 12272 | 12288 | -41.58 | 11800-12300 |
| attn_post_proj | 8 | 14256 | 14320 | 9.34 |  |
| attn_post_proj | 8 | 14576 | 14592 | -7.22 |  |
