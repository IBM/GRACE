# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pct  = float(np.mean(100.0 * np.abs(y_true - y_pred) / y_true))
    r,   _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)
    return {
        'RMSE':        rmse,
        'MeanPctDiff': pct,
        'PearsonR':    float(r),
        'SpearmanR':   float(rho),
        'KendallTau':  float(tau),
    }


def generalization_gap(train_rmse, test_rmse):
    return float(test_rmse - train_rmse)


_METRIC_KEYS = [
    'train_RMSE', 'train_MeanPctDiff',
    'test_RMSE',  'test_MeanPctDiff',
    'test_PearsonR', 'test_SpearmanR', 'test_KendallTau',
    'generalization_gap',
]


def aggregate_seeds(rows, split=None, n_train_frac=None):
    out = {
        'split':        split,
        'n_train_frac': n_train_frac,
        'n_seeds':      len(rows),
    }
    for key in _METRIC_KEYS:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        out[f'{key}_mean'] = float(np.nanmean(vals))
        out[f'{key}_std']  = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float('nan')
    return out


def summarize_runs(all_rows, split_col='split', n_train_frac_col='n_train_frac'):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in all_rows:
        groups[row[split_col]].append(row)

    summary_rows = []
    for split_label, rows in groups.items():
        n_train_frac = rows[0].get(n_train_frac_col)
        summary_rows.append(aggregate_seeds(rows, split=split_label, n_train_frac=n_train_frac))

    df = pd.DataFrame(summary_rows)
    if n_train_frac_col in df.columns:
        df = df.sort_values(n_train_frac_col).reset_index(drop=True)
    return df


def seed_row(y_train_true, y_train_pred, y_test_true, y_test_pred,
             split=None, n_train_frac=None, seed=None):
    train = compute_metrics(y_train_true, y_train_pred)
    test  = compute_metrics(y_test_true,  y_test_pred)
    gap   = generalization_gap(train['RMSE'], test['RMSE'])
    return {
        'split':         split,
        'n_train_frac':  n_train_frac,
        'seed':          seed,
        'train_RMSE':         train['RMSE'],
        'train_MeanPctDiff':  train['MeanPctDiff'],
        'test_RMSE':          test['RMSE'],
        'test_MeanPctDiff':   test['MeanPctDiff'],
        'test_PearsonR':      test['PearsonR'],
        'test_SpearmanR':     test['SpearmanR'],
        'test_KendallTau':    test['KendallTau'],
        'generalization_gap': gap,
    }
