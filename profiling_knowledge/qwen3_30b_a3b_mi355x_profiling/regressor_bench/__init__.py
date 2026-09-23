"""Regressor benchmark for the Qwen3-30B-A3B / MI355X linear_op dataset.

Split (train / test / sealed holdout) on the num_tokens grid, then train and compare regressor families per
(op, TP) against interpolation baselines. See README.md in this directory.
"""
