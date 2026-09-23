# Baseline RF / XGBoost per (op, TP), random-geometry shape-level split

Feature `num_tokens`; label `time_stats.<op>.median` (ms); 10 shape-level CV folds on train; test scored once; holdout not read. MAPE in %. `cliff` = rows with num_tokens in 4352-4360, 4480-4488, 4864-4872, 5376-5384, 11800-12300; `excl<10us` = rows with label >= 0.010 ms only; `cov_*` = train-tier num_tokens values inside or within 8 tokens of the window.

## Random Forest: metrics

| op | tp | cv_mape_mean | cv_mape_std | test_mape | test_cliff_mape | test_n_cliff_rows | test_mape_excl_sub10us | test_n_sub10us_rows | cv_cliff_mape_mean | cv_cliff_mape_std | cliff_gt_2x_overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_post_proj | 1 | 1.43 | 0.12 | 1.35 | 0.87 | 6 | 1.29 | 1 | 1.30 | 0.71 | False |
| attn_post_proj | 2 | 1.44 | 0.11 | 1.42 | 0.67 | 6 | 1.40 | 2 | 2.43 | 1.62 | False |
| attn_post_proj | 4 | 1.52 | 0.20 | 1.24 | 0.66 | 6 | 1.22 | 7 | 3.59 | 3.62 | False |
| attn_post_proj | 8 | 1.47 | 0.14 | 1.32 | 1.43 | 6 | 1.28 | 43 | 4.39 | 4.12 | False |
| attn_pre_proj | 1 | 1.25 | 0.06 | 1.33 | 0.82 | 6 | 1.33 | 0 | 1.12 | 0.58 | False |
| attn_pre_proj | 2 | 1.10 | 0.10 | 1.25 | 1.50 | 6 | 1.25 | 0 | 1.27 | 0.45 | False |
| attn_pre_proj | 4 | 1.09 | 0.06 | 1.20 | 1.11 | 6 | 1.20 | 0 | 0.93 | 0.43 | False |
| attn_pre_proj | 8 | 1.17 | 0.05 | 1.32 | 1.79 | 6 | 1.32 | 0 | 1.24 | 0.72 | False |
| attn_rope | 1 | 1.36 | 0.08 | 1.25 | 1.36 | 6 | 1.18 | 116 | 1.03 | 0.48 | False |
| attn_rope | 2 | 1.36 | 0.06 | 1.33 | 0.63 | 6 | 1.17 | 262 | 0.89 | 0.42 | False |
| attn_rope | 4 | 1.04 | 0.07 | 1.02 | 0.88 | 6 | 1.07 | 328 | 1.01 | 0.35 | False |
| attn_rope | 8 | 0.98 | 0.05 | 0.99 | 0.66 | 6 | 0.85 | 377 | 0.85 | 0.45 | False |
| emb | 1 | 1.51 | 0.11 | 1.45 | 1.91 | 6 | 1.96 | 345 | 1.73 | 0.86 | False |
| input_layernorm | 1 | 1.33 | 0.07 | 1.24 | 2.53 | 6 | 1.25 | 66 | 1.50 | 1.07 | True |
| post_attention_layernorm | 1 | 1.23 | 0.12 | 1.02 | 0.86 | 6 | 0.98 | 57 | 1.04 | 0.57 | False |

## Random Forest: selected hyper-parameters

| op | tp | best_params | cv_mape_mean |
|---|---|---|---|
| attn_post_proj | 1 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 1} | 1.43 |
| attn_post_proj | 2 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 1} | 1.44 |
| attn_post_proj | 4 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 1} | 1.52 |
| attn_post_proj | 8 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 1} | 1.47 |
| attn_pre_proj | 1 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 2} | 1.25 |
| attn_pre_proj | 2 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 2} | 1.10 |
| attn_pre_proj | 4 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 1} | 1.09 |
| attn_pre_proj | 8 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 5} | 1.17 |
| attn_rope | 1 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5} | 1.36 |
| attn_rope | 2 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5} | 1.36 |
| attn_rope | 4 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 10} | 1.04 |
| attn_rope | 8 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5} | 0.98 |
| emb | 1 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 2} | 1.51 |
| input_layernorm | 1 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 5} | 1.33 |
| post_attention_layernorm | 1 | {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 5} | 1.23 |

## XGBoost: metrics

| op | tp | cv_mape_mean | cv_mape_std | test_mape | test_cliff_mape | test_n_cliff_rows | test_mape_excl_sub10us | test_n_sub10us_rows | cv_cliff_mape_mean | cv_cliff_mape_std | cliff_gt_2x_overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_post_proj | 1 | 1.76 | 0.11 | 1.82 | 1.06 | 6 | 1.71 | 1 | 1.75 | 1.26 | False |
| attn_post_proj | 2 | 1.87 | 0.16 | 2.05 | 1.47 | 6 | 1.86 | 2 | 3.52 | 2.79 | False |
| attn_post_proj | 4 | 2.03 | 0.23 | 2.01 | 5.80 | 6 | 1.90 | 7 | 5.09 | 4.47 | True |
| attn_post_proj | 8 | 2.25 | 0.32 | 2.42 | 4.77 | 6 | 2.45 | 43 | 5.73 | 5.09 | False |
| attn_pre_proj | 1 | 1.48 | 0.14 | 1.61 | 1.80 | 6 | 1.61 | 0 | 1.40 | 0.89 | False |
| attn_pre_proj | 2 | 1.30 | 0.14 | 1.46 | 2.22 | 6 | 1.46 | 0 | 1.31 | 0.82 | False |
| attn_pre_proj | 4 | 1.21 | 0.10 | 1.49 | 2.18 | 6 | 1.49 | 0 | 1.24 | 0.49 | False |
| attn_pre_proj | 8 | 1.24 | 0.07 | 1.43 | 2.03 | 6 | 1.43 | 0 | 1.59 | 1.20 | False |
| attn_rope | 1 | 1.48 | 0.09 | 1.37 | 2.16 | 6 | 1.25 | 116 | 1.26 | 0.64 | False |
| attn_rope | 2 | 1.39 | 0.07 | 1.35 | 1.15 | 6 | 1.24 | 262 | 1.06 | 0.65 | False |
| attn_rope | 4 | 1.10 | 0.08 | 1.05 | 0.66 | 6 | 1.05 | 328 | 1.07 | 0.53 | False |
| attn_rope | 8 | 1.03 | 0.06 | 1.04 | 0.87 | 6 | 0.99 | 377 | 0.87 | 0.61 | False |
| emb | 1 | 1.57 | 0.14 | 1.56 | 1.83 | 6 | 1.95 | 345 | 1.85 | 0.94 | False |
| input_layernorm | 1 | 1.55 | 0.11 | 1.58 | 3.09 | 6 | 1.47 | 66 | 1.99 | 1.20 | False |
| post_attention_layernorm | 1 | 1.40 | 0.18 | 1.40 | 1.15 | 6 | 1.23 | 57 | 1.33 | 0.87 | False |

## XGBoost: selected hyper-parameters

| op | tp | best_params | cv_mape_mean | xgb_best_iteration |
|---|---|---|---|---|
| attn_post_proj | 1 | {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 1000, "subsample": 0.6, "min_child_weight": 5} | 1.76 | 110.00 |
| attn_post_proj | 2 | {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 200, "subsample": 1.0, "min_child_weight": 1} | 1.87 | 193.00 |
| attn_post_proj | 4 | {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8, "min_child_weight": 1} | 2.03 | 135.00 |
| attn_post_proj | 8 | {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8, "min_child_weight": 1} | 2.25 | 280.00 |
| attn_pre_proj | 1 | {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 1000, "subsample": 0.6, "min_child_weight": 5} | 1.48 | 717.00 |
| attn_pre_proj | 2 | {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 1000, "subsample": 1.0, "min_child_weight": 1} | 1.30 | 686.00 |
| attn_pre_proj | 4 | {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 1.0, "min_child_weight": 1} | 1.21 | 156.00 |
| attn_pre_proj | 8 | {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 500, "subsample": 1.0, "min_child_weight": 10} | 1.24 | 71.00 |
| attn_rope | 1 | {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 1000, "subsample": 0.6, "min_child_weight": 5} | 1.48 | 999.00 |
| attn_rope | 2 | {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.6, "min_child_weight": 5} | 1.39 | 150.00 |
| attn_rope | 4 | {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 1000, "subsample": 0.8, "min_child_weight": 10} | 1.10 | 999.00 |
| attn_rope | 8 | {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8, "min_child_weight": 1} | 1.03 | 376.00 |
| emb | 1 | {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.6, "min_child_weight": 5} | 1.57 | 178.00 |
| input_layernorm | 1 | {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.6, "min_child_weight": 5} | 1.55 | 218.00 |
| post_attention_layernorm | 1 | {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 1000, "subsample": 0.6, "min_child_weight": 5} | 1.40 | 969.00 |

## Train-tier coverage of the cliff windows (values inside or within 8 tokens; identical for every op at a TP)

| op | tp | 4352-4360 | 4480-4488 | 4864-4872 | 5376-5384 | 11800-12300 |
|---|---|---|---|---|---|---|
| attn_post_proj | 1 | 3 | 3 | 3 | 4 | 23 |
| attn_post_proj | 2 | 3 | 3 | 3 | 4 | 23 |
| attn_post_proj | 4 | 3 | 3 | 3 | 4 | 23 |
| attn_post_proj | 8 | 3 | 3 | 3 | 4 | 23 |
| attn_pre_proj | 1 | 3 | 3 | 3 | 4 | 23 |
| attn_pre_proj | 2 | 3 | 3 | 3 | 4 | 23 |
| attn_pre_proj | 4 | 3 | 3 | 3 | 4 | 23 |
| attn_pre_proj | 8 | 3 | 3 | 3 | 4 | 23 |
| attn_rope | 1 | 3 | 3 | 3 | 4 | 23 |
| attn_rope | 2 | 3 | 3 | 3 | 4 | 23 |
| attn_rope | 4 | 3 | 3 | 3 | 4 | 23 |
| attn_rope | 8 | 3 | 3 | 3 | 4 | 23 |
| emb | 1 | 3 | 3 | 3 | 4 | 23 |
| input_layernorm | 1 | 3 | 3 | 3 | 4 | 23 |
| post_attention_layernorm | 1 | 3 | 3 | 3 | 4 | 23 |

## Flags: cliff-window test MAPE > 2x overall test MAPE

| op | tp | model | test_mape | test_cliff_mape | test_n_cliff_rows |
|---|---|---|---|---|---|
| attn_post_proj | 4 | xgb | 2.01 | 5.80 | 6 |
| input_layernorm | 1 | rf | 1.24 | 2.53 | 6 |

## CV fold-level MAPE (winning config), per fold

| model | op | tp | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rf | attn_post_proj | 1 | 1.38 | 1.32 | 1.58 | 1.37 | 1.38 | 1.53 | 1.50 | 1.31 | 1.62 | 1.31 |
| rf | attn_post_proj | 2 | 1.57 | 1.52 | 1.39 | 1.53 | 1.24 | 1.44 | 1.33 | 1.50 | 1.48 | 1.34 |
| rf | attn_post_proj | 4 | 1.64 | 1.33 | 1.85 | 1.64 | 1.29 | 1.72 | 1.22 | 1.49 | 1.61 | 1.44 |
| rf | attn_post_proj | 8 | 1.59 | 1.25 | 1.46 | 1.41 | 1.34 | 1.45 | 1.46 | 1.53 | 1.46 | 1.78 |
| rf | attn_pre_proj | 1 | 1.28 | 1.22 | 1.37 | 1.23 | 1.24 | 1.23 | 1.12 | 1.29 | 1.29 | 1.22 |
| rf | attn_pre_proj | 2 | 1.17 | 1.19 | 1.01 | 1.20 | 1.06 | 0.93 | 1.17 | 1.12 | 1.16 | 0.99 |
| rf | attn_pre_proj | 4 | 1.18 | 1.09 | 1.09 | 1.12 | 1.09 | 1.05 | 1.05 | 1.09 | 1.16 | 0.98 |
| rf | attn_pre_proj | 8 | 1.20 | 1.14 | 1.11 | 1.07 | 1.17 | 1.21 | 1.14 | 1.19 | 1.25 | 1.16 |
| rf | attn_rope | 1 | 1.30 | 1.37 | 1.39 | 1.25 | 1.40 | 1.36 | 1.41 | 1.28 | 1.51 | 1.32 |
| rf | attn_rope | 2 | 1.39 | 1.38 | 1.28 | 1.47 | 1.33 | 1.36 | 1.42 | 1.38 | 1.32 | 1.28 |
| rf | attn_rope | 4 | 1.00 | 1.10 | 1.02 | 1.14 | 0.96 | 1.06 | 0.95 | 0.96 | 1.09 | 1.11 |
| rf | attn_rope | 8 | 1.00 | 1.03 | 1.00 | 1.02 | 0.87 | 0.97 | 0.99 | 0.91 | 0.99 | 1.04 |
| rf | emb | 1 | 1.37 | 1.43 | 1.44 | 1.69 | 1.59 | 1.50 | 1.36 | 1.58 | 1.54 | 1.56 |
| rf | input_layernorm | 1 | 1.38 | 1.29 | 1.30 | 1.47 | 1.34 | 1.35 | 1.24 | 1.26 | 1.29 | 1.33 |
| rf | post_attention_layernorm | 1 | 1.16 | 1.26 | 1.19 | 1.22 | 1.52 | 1.23 | 1.24 | 1.04 | 1.20 | 1.21 |
| xgb | attn_post_proj | 1 | 1.74 | 1.65 | 1.80 | 1.58 | 1.76 | 1.77 | 1.90 | 1.69 | 1.95 | 1.71 |
| xgb | attn_post_proj | 2 | 2.17 | 1.99 | 1.93 | 1.73 | 1.71 | 1.69 | 1.78 | 1.75 | 2.02 | 1.94 |
| xgb | attn_post_proj | 4 | 2.50 | 1.81 | 2.19 | 1.97 | 1.91 | 2.00 | 1.85 | 1.74 | 2.19 | 2.14 |
| xgb | attn_post_proj | 8 | 2.74 | 2.04 | 2.41 | 1.93 | 1.99 | 1.87 | 2.43 | 2.35 | 2.01 | 2.69 |
| xgb | attn_pre_proj | 1 | 1.59 | 1.48 | 1.74 | 1.52 | 1.39 | 1.45 | 1.33 | 1.34 | 1.63 | 1.38 |
| xgb | attn_pre_proj | 2 | 1.26 | 1.28 | 1.22 | 1.54 | 1.32 | 1.09 | 1.35 | 1.28 | 1.52 | 1.18 |
| xgb | attn_pre_proj | 4 | 1.24 | 1.21 | 1.30 | 1.28 | 1.11 | 1.17 | 1.14 | 1.11 | 1.43 | 1.15 |
| xgb | attn_pre_proj | 8 | 1.28 | 1.22 | 1.24 | 1.11 | 1.28 | 1.26 | 1.16 | 1.33 | 1.31 | 1.20 |
| xgb | attn_rope | 1 | 1.47 | 1.41 | 1.47 | 1.40 | 1.56 | 1.52 | 1.56 | 1.32 | 1.64 | 1.43 |
| xgb | attn_rope | 2 | 1.42 | 1.36 | 1.29 | 1.52 | 1.34 | 1.38 | 1.42 | 1.45 | 1.37 | 1.32 |
| xgb | attn_rope | 4 | 1.05 | 1.10 | 1.10 | 1.25 | 1.00 | 1.10 | 1.02 | 1.00 | 1.16 | 1.17 |
| xgb | attn_rope | 8 | 0.99 | 1.05 | 1.02 | 1.06 | 0.93 | 1.07 | 1.04 | 0.98 | 1.05 | 1.13 |
| xgb | emb | 1 | 1.39 | 1.35 | 1.51 | 1.74 | 1.61 | 1.68 | 1.49 | 1.65 | 1.74 | 1.56 |
| xgb | input_layernorm | 1 | 1.54 | 1.56 | 1.58 | 1.74 | 1.59 | 1.64 | 1.44 | 1.61 | 1.37 | 1.47 |
| xgb | post_attention_layernorm | 1 | 1.29 | 1.35 | 1.41 | 1.40 | 1.86 | 1.37 | 1.47 | 1.28 | 1.23 | 1.32 |

### CV fold-level cliff-window MAPE (winning config), per fold (– = no cliff rows in that fold)

| model | op | tp | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rf | attn_post_proj | 1 | 3.08 | 1.53 | 0.94 | 1.49 | 0.39 | 1.17 | 1.07 | 0.87 | 1.38 | 1.10 |
| rf | attn_post_proj | 2 | 5.49 | 1.92 | 2.39 | 1.50 | 0.64 | 3.79 | 1.10 | 4.52 | 1.76 | 1.21 |
| rf | attn_post_proj | 4 | 9.34 | 2.18 | 11.12 | 2.76 | 2.26 | 3.32 | 0.96 | 1.09 | 2.41 | 0.50 |
| rf | attn_post_proj | 8 | 3.44 | 6.11 | 12.66 | 1.44 | 1.08 | 1.66 | 5.54 | 9.85 | 1.46 | 0.70 |
| rf | attn_pre_proj | 1 | 0.73 | 0.59 | 1.79 | 0.71 | 0.15 | 1.24 | 1.82 | 1.85 | 1.22 | 1.14 |
| rf | attn_pre_proj | 2 | 1.10 | 1.13 | 0.83 | 2.23 | 0.89 | 1.09 | 1.76 | 0.88 | 1.20 | 1.60 |
| rf | attn_pre_proj | 4 | 0.64 | 0.30 | 1.06 | 0.92 | 0.30 | 1.41 | 0.91 | 1.05 | 1.11 | 1.63 |
| rf | attn_pre_proj | 8 | 1.52 | 1.72 | 1.21 | 0.49 | 1.21 | 2.94 | 0.41 | 1.13 | 0.83 | 0.96 |
| rf | attn_rope | 1 | 0.55 | 0.53 | 0.79 | 1.28 | 0.94 | 0.46 | 2.01 | 1.43 | 1.20 | 1.06 |
| rf | attn_rope | 2 | 1.26 | 0.43 | 0.93 | 1.82 | 0.45 | 0.76 | 1.10 | 0.67 | 0.69 | 0.75 |
| rf | attn_rope | 4 | 1.15 | 0.92 | 1.14 | 1.67 | 1.00 | 0.53 | 1.12 | 1.25 | 0.82 | 0.48 |
| rf | attn_rope | 8 | 1.10 | 0.24 | 1.18 | 1.34 | 0.43 | 0.48 | 1.00 | 0.71 | 0.46 | 1.56 |
| rf | emb | 1 | 1.42 | 2.89 | 1.32 | 1.23 | 0.43 | 2.97 | 2.59 | 1.74 | 1.85 | 0.84 |
| rf | emb | 2 | – | – | – | – | – | – | – | – | – | – |
| rf | emb | 4 | – | – | – | – | – | – | – | – | – | – |
| rf | emb | 8 | – | – | – | – | – | – | – | – | – | – |
| rf | input_layernorm | 1 | 1.79 | 1.21 | 2.53 | 1.67 | 0.56 | 0.48 | 0.77 | 3.87 | 1.57 | 0.52 |
| rf | input_layernorm | 2 | – | – | – | – | – | – | – | – | – | – |
| rf | input_layernorm | 4 | – | – | – | – | – | – | – | – | – | – |
| rf | input_layernorm | 8 | – | – | – | – | – | – | – | – | – | – |
| rf | post_attention_layernorm | 1 | 0.69 | 0.89 | 0.86 | 2.10 | 0.32 | 0.36 | 1.21 | 1.75 | 0.88 | 1.34 |
| rf | post_attention_layernorm | 2 | – | – | – | – | – | – | – | – | – | – |
| rf | post_attention_layernorm | 4 | – | – | – | – | – | – | – | – | – | – |
| rf | post_attention_layernorm | 8 | – | – | – | – | – | – | – | – | – | – |
| xgb | attn_post_proj | 1 | 2.87 | 1.09 | 1.05 | 3.20 | 0.34 | 4.31 | 1.04 | 1.20 | 1.06 | 1.29 |
| xgb | attn_post_proj | 2 | 6.71 | 1.09 | 3.26 | 5.08 | 0.49 | 9.39 | 1.71 | 3.46 | 2.04 | 1.95 |
| xgb | attn_post_proj | 4 | 10.49 | 0.68 | 10.71 | 8.04 | 1.69 | 11.41 | 2.57 | 1.75 | 2.35 | 1.24 |
| xgb | attn_post_proj | 8 | 3.49 | 15.30 | 13.08 | 2.88 | 2.47 | 1.66 | 9.44 | 5.64 | 1.87 | 1.44 |
| xgb | attn_pre_proj | 1 | 0.65 | 0.64 | 2.28 | 2.94 | 0.28 | 1.36 | 2.54 | 1.25 | 0.93 | 1.12 |
| xgb | attn_pre_proj | 2 | 0.66 | 0.39 | 0.83 | 2.81 | 0.81 | 0.98 | 2.41 | 0.73 | 1.65 | 1.79 |
| xgb | attn_pre_proj | 4 | 1.02 | 1.58 | 1.51 | 1.36 | 0.32 | 1.21 | 1.88 | 1.01 | 0.69 | 1.80 |
| xgb | attn_pre_proj | 8 | 1.44 | 3.57 | 1.26 | 0.32 | 1.42 | 3.97 | 1.15 | 0.95 | 0.81 | 1.02 |
| xgb | attn_rope | 1 | 0.75 | 1.00 | 1.05 | 1.96 | 1.24 | 0.43 | 2.70 | 1.21 | 1.16 | 1.05 |
| xgb | attn_rope | 2 | 0.98 | 0.83 | 1.02 | 2.59 | 0.03 | 0.89 | 1.43 | 0.82 | 0.77 | 1.23 |
| xgb | attn_rope | 4 | 0.71 | 0.65 | 1.31 | 2.37 | 1.03 | 1.08 | 1.02 | 1.27 | 0.52 | 0.76 |
| xgb | attn_rope | 8 | 1.05 | 0.09 | 1.40 | 1.44 | 0.32 | 0.37 | 1.02 | 0.61 | 0.40 | 2.00 |
| xgb | emb | 1 | 1.83 | 2.47 | 1.48 | 1.02 | 0.33 | 3.81 | 2.41 | 2.09 | 1.67 | 1.42 |
| xgb | emb | 2 | – | – | – | – | – | – | – | – | – | – |
| xgb | emb | 4 | – | – | – | – | – | – | – | – | – | – |
| xgb | emb | 8 | – | – | – | – | – | – | – | – | – | – |
| xgb | input_layernorm | 1 | 1.85 | 3.80 | 2.67 | 1.43 | 0.98 | 0.59 | 1.57 | 4.02 | 2.23 | 0.71 |
| xgb | input_layernorm | 2 | – | – | – | – | – | – | – | – | – | – |
| xgb | input_layernorm | 4 | – | – | – | – | – | – | – | – | – | – |
| xgb | input_layernorm | 8 | – | – | – | – | – | – | – | – | – | – |
| xgb | post_attention_layernorm | 1 | 0.84 | 0.84 | 0.87 | 3.64 | 0.66 | 1.48 | 1.34 | 1.57 | 0.78 | 1.28 |
| xgb | post_attention_layernorm | 2 | – | – | – | – | – | – | – | – | – | – |
| xgb | post_attention_layernorm | 4 | – | – | – | – | – | – | – | – | – | – |
| xgb | post_attention_layernorm | 8 | – | – | – | – | – | – | – | – | – | – |
