from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from scipy.stats import t

import pandas as pd 
import numpy as np
import pingouin as pg
from dowhy import CausalModel

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
)

import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')


def data_prep(mode: str):
    uber_rts_df = pd.read_parquet(mode + '/Uber_Routes_Data.parquet')
    uber_rts_df = uber_rts_df.drop(columns = 'service')
    uber_rts_df.columns = ['date_of_trip', 'day_of_week', 'time_of_day', 'PULocationID', 'DOLocationID', 'uber_vol', 'uber_trip_dist','uber_mph', 'uber_trip_dur', 'uber_wait_dur'
                            , 'uber_med_wait_dur', 'uber_wait_ratio', 'uber_med_wait_ratio', 'uber_fare_per_mile', 'uber_pay_per_mile', 'uber_adj_pay_per_mile', 'uber_fare_per_min'
                            , 'uber_pay_ratio', 'uber_med_pay_ratio_pay_per_min', 'uber_rev_pos', 'uber_rev_per_mile', 'uber_rev_per_min', 'uber_med_pay_ratio', 'uber_pay_per_min']

    uber_rts_df = uber_rts_df.loc[(uber_rts_df['day_of_week'] < 4) & (uber_rts_df['time_of_day'].isin([2, 4]))].reset_index(drop = True)
    
    lyft_rts_df = pd.read_parquet(mode + '/Lyft_Routes_Data.parquet')
    lyft_rts_df = lyft_rts_df.drop(columns = 'service')

    lyft_rts_df.columns = ['date_of_trip', 'day_of_week', 'time_of_day', 'PULocationID', 'DOLocationID', 'lyft_vol', 'lyft_trip_dist','lyft_mph', 'lyft_trip_dur', 'lyft_wait_dur'
                            , 'lyft_med_wait_dur', 'lyft_wait_ratio', 'lyft_med_wait_ratio', 'lyft_fare_per_mile', 'lyft_pay_per_mile', 'lyft_adj_pay_per_mile', 'lyft_fare_per_min'
                            , 'lyft_pay_ratio', 'lyft_med_pay_ratio_pay_per_min', 'lyft_rev_pos', 'lyft_rev_per_mile', 'lyft_rev_per_min', 'lyft_med_pay_ratio', 'lyft_pay_per_min']

    lyft_rts_df = lyft_rts_df.loc[(lyft_rts_df['day_of_week'] < 4) & (lyft_rts_df['time_of_day'].isin([2, 4]))].reset_index(drop = True)
    
    vol_rts_df = pd.merge(uber_rts_df[['date_of_trip', 'day_of_week', 'time_of_day', 'PULocationID', 'DOLocationID', 'uber_vol', 'uber_wait_ratio', 'uber_pay_ratio'
            , 'uber_fare_per_mile', 'uber_pay_per_mile','uber_adj_pay_per_mile', 'uber_rev_pos', 'uber_rev_per_mile', 'uber_trip_dur', 'uber_wait_dur']]
            , lyft_rts_df[['date_of_trip', 'day_of_week', 'time_of_day', 'PULocationID', 'DOLocationID', 'lyft_vol', 'lyft_wait_ratio', 'lyft_pay_ratio'
            , 'lyft_fare_per_mile', 'lyft_pay_per_mile', 'lyft_adj_pay_per_mile', 'lyft_rev_pos', 'lyft_rev_per_mile', 'lyft_trip_dur', 'lyft_wait_dur']], how = 'inner')


    vol_rts_df['uber_vol'] = vol_rts_df['uber_vol'].fillna(0)

    vol_rts_df['lyft_vol'] = vol_rts_df['lyft_vol'].fillna(0)

    vol_rts_df['tot_vol'] = vol_rts_df['uber_vol'] + vol_rts_df['lyft_vol']

    vol_rts_df = vol_rts_df.loc[(vol_rts_df['uber_vol'] > 10) & (vol_rts_df['lyft_vol'] > 10)].reset_index(drop = True)

    vol_rts_df['uber_pay'] = vol_rts_df['uber_pay_per_mile']
    vol_rts_df['uber_fare'] = vol_rts_df['uber_fare_per_mile']
    vol_rts_df['lyft_pay'] = vol_rts_df['lyft_pay_per_mile']
    vol_rts_df['lyft_fare'] = vol_rts_df['lyft_fare_per_mile']
    vol_rts_df['uber_wait'] = vol_rts_df['uber_wait_ratio']
    vol_rts_df['lyft_wait'] = vol_rts_df['lyft_wait_ratio']

    vol_rts_df['tot_vol'] = vol_rts_df['uber_vol'] + vol_rts_df['lyft_vol']
    vol_rts_df['lyft_share'] = vol_rts_df['lyft_vol'] / vol_rts_df['tot_vol']

    return vol_rts_df

def _pearsonr_test(x: np.ndarray, y: np.ndarray):
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 samples for correlation test.")
    r = np.corrcoef(x, y)[0, 1]
    # Guard against numerical issues at |r| ~ 1
    r = float(np.clip(r, -0.999999, 0.999999))
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p = 2 * t.sf(np.abs(t_stat), df=n - 2)
    return r, p


def _partial_corr_test(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    n = len(x)
    if n < 5:
        raise ValueError("Need at least 5 samples for partial correlation test.")

    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]

    # Guard against numerical issues
    r_xy = float(np.clip(r_xy, -0.999999, 0.999999))
    r_xz = float(np.clip(r_xz, -0.999999, 0.999999))
    r_yz = float(np.clip(r_yz, -0.999999, 0.999999))

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom <= 1e-12:
        # Degenerate case: Z almost perfectly correlated with X or Y
        return np.nan, 1.0

    r_xy_z = (r_xy - r_xz * r_yz) / denom
    r_xy_z = float(np.clip(r_xy_z, -0.999999, 0.999999))

    # t-statistic for partial corr with k=1 control variable: df = n - 3
    df = n - 3
    t_stat = r_xy_z * np.sqrt(df / (1 - r_xy_z**2))
    p = 2 * t.sf(np.abs(t_stat), df=df)

    return r_xy_z, p


def pc_three_vars(
    df,
    cols,
    alpha=0.05,
    verbose=True,
    max_samples=1000,
    random_state=None,
):
    if len(cols) != 3:
        raise ValueError("cols must be a list of exactly 3 column names.")
    if len(set(cols)) != 3:
        raise ValueError("cols must contain 3 distinct column names.")

    # 1) Drop rows with any NaNs in the 3 columns
    data = df[cols].dropna()

    # 2) Randomly trim to max_samples to avoid ultra-high power / p-hacking
    if len(data) > max_samples:
        data = data.sample(n=max_samples, random_state=random_state)

    n = len(data)
    if n < 5:
        raise ValueError(
            "Need at least 5 complete rows after trimming for basic PC "
            "with partial correlations."
        )

    names = list(cols)
    X = data[names[0]].to_numpy(dtype=float)
    Y = data[names[1]].to_numpy(dtype=float)
    Z = data[names[2]].to_numpy(dtype=float)

    # ----------------------------------------
    # Step 0: Initial complete undirected graph
    # ----------------------------------------
    edges = {
        (names[0], names[1]),
        (names[0], names[2]),
        (names[1], names[2]),
    }

    sep_set = {}
    tests = {"marginal": {}, "conditional": {}}

    # ----------------------------------------
    # Step 1: Marginal independence tests (ALL 3 printed)
    # ----------------------------------------
    pair_vars = {
        (names[0], names[1]): (X, Y),
        (names[0], names[2]): (X, Z),
        (names[1], names[2]): (Y, Z),
    }

    for (a, b), (xa, xb) in pair_vars.items():
        r, p = _pearsonr_test(xa, xb)
        tests["marginal"][(a, b)] = {"r": r, "p": p}
        if p > alpha:
            # Independent => delete edge
            if (a, b) in edges:
                edges.remove((a, b))
            elif (b, a) in edges:
                edges.remove((b, a))
            sep_set[frozenset({a, b})] = set()  # empty separating set

    # ----------------------------------------
    # Step 2: Conditional independence tests (ALL 3 computed & stored)
    # ----------------------------------------
    # All 3 possible (a,b|c) triples
    triples = [
        (names[0], names[1], names[2]),  # (X, Y | Z)
        (names[0], names[2], names[1]),  # (X, Z | Y)
        (names[1], names[2], names[0]),  # (Y, Z | X)
    ]

    # First compute and store ALL tests, regardless of edges
    for a, b, c in triples:
        # Map (a,b,c) to actual numpy arrays
        if {a, b, c} != set(names):
            raise RuntimeError("Internal bug: unexpected triple contents.")

        if (a, b) == (names[0], names[1]) or (a, b) == (names[1], names[0]):
            xa, xb, xc = X, Y, Z  # (X,Y|Z)
        elif (a, b) == (names[0], names[2]) or (a, b) == (names[2], names[0]):
            xa, xb, xc = X, Z, Y  # (X,Z|Y)
        else:
            xa, xb, xc = Y, Z, X  # (Y,Z|X)

        r_partial, p_partial = _partial_corr_test(xa, xb, xc)
        tests["conditional"][(a, b, c)] = {"r": r_partial, "p": p_partial}

    # Then use those tests to update the graph,
    # but only for pairs that still have an edge
    for a, b, c in triples:
        if (a, b) not in edges and (b, a) not in edges:
            continue  # no edge -> nothing to delete

        vals = tests["conditional"][(a, b, c)]
        r_partial, p_partial = vals["r"], vals["p"]

        if p_partial > alpha:
            # Independent given c => delete edge and record separating set {c}
            if (a, b) in edges:
                edges.remove((a, b))
            elif (b, a) in edges:
                edges.remove((b, a))
            sep_set[frozenset({a, b})] = {c}

    # ----------------------------------------
    # Step 3: Orient colliders A -> M <- B
    # ----------------------------------------
    directed_edges = []

    for m in names:
        others = [v for v in names if v != m]
        a, b = others[0], others[1]

        # a-m and b-m must both be adjacent
        if not ((a, m) in edges or (m, a) in edges):
            continue
        if not ((b, m) in edges or (m, b) in edges):
            continue

        # a and b must NOT be adjacent (unshielded triple)
        if (a, b) in edges or (b, a) in edges:
            continue

        # m must NOT be in the separating set of {a,b}
        sep = sep_set.get(frozenset({a, b}), None)
        if sep is not None and m in sep:
            continue

        # Then orient a -> m <- b
        directed_edges.append((a, m))
        directed_edges.append((b, m))

    # ----------------------------------------
    # Remaining edges are undirected
    # ----------------------------------------
    oriented_pairs = {(u, v) for (u, v) in directed_edges} | {
        (v, u) for (u, v) in directed_edges
    }

    undirected_edges = []
    for (u, v) in edges:
        if (u, v) not in oriented_pairs and (v, u) not in oriented_pairs:
            if (v, u) not in undirected_edges:
                undirected_edges.append((u, v))

    result = {
        "nodes": names,
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
        "tests": tests,
        "n_used": n,
    }

    if verbose:
        print("=== PC on 3 variables ===")
        print("Nodes:", names)
        print(f"Rows used (after dropna + trim): {n}")

        print("\nMarginal correlation tests:")
        for (a, b), vals in tests["marginal"].items():
            print(f"{a} -- {b}: r={vals['r']:.3f}, p={vals['p']:.3g}")

        print("\nConditional (partial) correlation tests:")
        for (a, b, c), vals in tests["conditional"].items():
            print(f"{a} -- {b} | {c}: r={vals['r']:.3f}, p={vals['p']:.3g}")

        print("\nLearned structure:")
        if directed_edges:
            for u, v in directed_edges:
                print(f"{u} -> {v}")
        if undirected_edges:
            for u, v in undirected_edges:
                print(f"{u} - {v}  (direction not identifiable)")

    return result

def strat_folds(
    df: pd.DataFrame,
    n_folds: int,
    stratify_cols: Optional[List[str]] = None,
    fold_col: str = "fold",
    seed: int = 42,
    shuffle_within_strata: bool = True,
    dropna_strata: bool = False,
) -> pd.DataFrame:
    
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"n_folds must be an int >= 2. Got: {n_folds}")
    if len(df) < n_folds:
        raise ValueError(f"n_folds ({n_folds}) cannot exceed number of rows ({len(df)}).")

    stratify_cols = stratify_cols or []
    for c in stratify_cols:
        if c not in df.columns:
            raise KeyError(f"Stratify column not found in df: {c}")

    rng = np.random.default_rng(seed)
    out = df.copy()

    # Default: random K-fold assignment (no stratification)
    if len(stratify_cols) == 0:
        perm = rng.permutation(out.index.to_numpy())
        folds = np.empty(len(out), dtype=int)
        folds[np.arange(len(out))] = np.arange(len(out)) % n_folds
        out.loc[perm, fold_col] = folds
        out[fold_col] = out[fold_col].astype(int)
        return out

    # Decide which rows participate in stratification (optionally exclude NaNs)
    if dropna_strata:
        mask_strat = out[stratify_cols].notna().all(axis=1)
        strat_part = out.loc[mask_strat]
        rest_part = out.loc[~mask_strat]
    else:
        mask_strat = pd.Series(True, index=out.index)
        strat_part = out
        rest_part = out.iloc[0:0]  # empty

    # Prepare stratum keys (NaNs treated as a category by converting to sentinel string)
    key_df = strat_part[stratify_cols].copy()
    for c in stratify_cols:
        if pd.api.types.is_categorical_dtype(key_df[c]):
            key_df[c] = key_df[c].astype(object)
        key_df[c] = key_df[c].where(key_df[c].notna(), "__NA__")

    # Assign within each stratum via round-robin (after optional shuffle)
    out[fold_col] = -1
    for _, idx in key_df.groupby(stratify_cols, sort=False).groups.items():
        idx = np.fromiter(idx, dtype=out.index.dtype, count=len(idx))
        if shuffle_within_strata and len(idx) > 1:
            idx = rng.permutation(idx)
        fold_vals = (np.arange(len(idx)) % n_folds).astype(int)
        out.loc[idx, fold_col] = fold_vals

    # If we excluded NaN-strata rows, assign them randomly (balanced) across folds
    if len(rest_part) > 0:
        rest_idx = rest_part.index.to_numpy()
        rest_idx = rng.permutation(rest_idx)
        rest_folds = (np.arange(len(rest_idx)) % n_folds).astype(int)
        out.loc[rest_idx, fold_col] = rest_folds

    out[fold_col] = out[fold_col].astype(int)
    return out

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])))


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1 or np.isnan(r2):
        return np.nan
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def _normalize_objective(obj: str) -> str:
    
    if obj is None:
        return "reg:squarederror"
    o = str(obj).strip().lower().replace(" ", "_")
    if o in {"squared_error", "squarederror", "squared"}:
        return "reg:squarederror"
    return obj  # assume user provided a valid XGBoost objective string


def _tree_importance_as_coeffs(
    model: XGBRegressor, predictor_cols: List[str], importance_type: str = "gain"
) -> np.ndarray:

    booster = model.get_booster()
    score = booster.get_score(importance_type=importance_type)  # keys like "f0", "f1", ...
    out = np.zeros(len(predictor_cols), dtype=float)
    for i in range(len(predictor_cols)):
        out[i] = float(score.get(f"f{i}", 0.0))
    return out


def xgb_coeffs(
    df: pd.DataFrame,
    fold_col: str,
    predictor_cols: List[str],
    target_col: str,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    n_estimators: int = 200,
    booster: str = "gbtree",
    objective: str = "reg:squarederror",  # squared error
    random_state: int = 42,
    n_jobs: int = -1,
    tree_importance_type: str = "gain",  # used only for booster in {"gbtree","dart"}
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if fold_col not in df.columns:
        raise KeyError(f"fold_col '{fold_col}' not found in df")
    missing_preds = [c for c in predictor_cols if c not in df.columns]
    if missing_preds:
        raise KeyError(f"predictor cols not found in df: {missing_preds}")
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in df")

    objective = _normalize_objective(objective)

    df_out = df.copy()
    pred_col = f"{target_col}_pred"
    err_col = f"{target_col}_error"

    # init output cols
    df_out[pred_col] = np.nan
    df_out[err_col] = np.nan
    
    # folds (ignore NaN folds)
    folds = df_out[fold_col].dropna().unique()
    try:
        folds = np.sort(folds.astype(int))
    except Exception:
        folds = np.sort(folds)

    report_rows: List[Dict[str, Any]] = []
    p = len(predictor_cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for k in folds:
            test_mask = df_out[fold_col] == k
            train_mask = ~test_mask

            train_df = df_out.loc[train_mask, predictor_cols + [target_col]].dropna()
            n_train = int(train_df.shape[0])

            test_X_df = df_out.loc[test_mask, predictor_cols]
            test_y_s = df_out.loc[test_mask, target_col]
            pred_ok_mask = test_X_df.notna().all(axis=1)
            test_eval_mask = pred_ok_mask & test_y_s.notna()

            n_test_total = int(test_mask.sum())
            n_test_pred = int(pred_ok_mask.sum())
            n_test_eval = int(test_eval_mask.sum())

            if n_train < 2 or n_test_pred < 1:
                report_rows.append(
                    {
                        "fold": k,
                        "n_train": n_train,
                        "n_test_total": n_test_total,
                        "n_test_pred": n_test_pred,
                        "n_test_eval": n_test_eval,
                        "status": "skipped (insufficient train/test)",
                        "booster": booster,
                        "objective": objective,
                    }
                )
                continue

            X_train = train_df[predictor_cols].to_numpy(dtype=float)
            y_train = train_df[target_col].to_numpy(dtype=float)

            model = XGBRegressor(
                booster=booster,
                objective=objective,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,  # relevant for gbtree/dart; ignored by gblinear
                # keep pure coefficient values if gblinear:
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            # Predict where predictors exist
            test_idx = df_out.index[test_mask]
            pred_idx = test_idx[pred_ok_mask.values]
            X_test = test_X_df.loc[pred_idx, predictor_cols].to_numpy(dtype=float)
            y_pred = model.predict(X_test).astype(float)

            df_out.loc[pred_idx, pred_col] = y_pred

            # Error (true - predicted) where y_true exists
            y_true_for_err = df_out.loc[pred_idx, target_col].to_numpy(dtype=float)
            err = y_true_for_err - y_pred
            err[np.isnan(y_true_for_err)] = np.nan
            df_out.loc[pred_idx, err_col] = err

            # Metrics: train + test
            yhat_train = model.predict(X_train).astype(float)

            train_mae = float(mean_absolute_error(y_train, yhat_train))
            train_mse = float(mean_squared_error(y_train, yhat_train))
            train_rmse = float(np.sqrt(train_mse))
            train_medae = float(median_absolute_error(y_train, yhat_train))
            train_r2 = float(r2_score(y_train, yhat_train))
            train_adj_r2 = _adjusted_r2(train_r2, n=len(y_train), p=p)
            train_evs = float(explained_variance_score(y_train, yhat_train))
            train_mape = _safe_mape(y_train, yhat_train)
            train_smape = _safe_smape(y_train, yhat_train)
            train_corr = _safe_corr(y_train, yhat_train)

            eval_idx = test_idx[test_eval_mask.values]
            if len(eval_idx) >= 1:
                y_test = df_out.loc[eval_idx, target_col].to_numpy(dtype=float)
                yhat_test = df_out.loc[eval_idx, pred_col].to_numpy(dtype=float)

                test_mae = float(mean_absolute_error(y_test, yhat_test))
                test_mse = float(mean_squared_error(y_test, yhat_test))
                test_rmse = float(np.sqrt(test_mse))
                test_medae = float(median_absolute_error(y_test, yhat_test))
                test_r2 = float(r2_score(y_test, yhat_test)) if len(eval_idx) >= 2 else np.nan
                test_adj_r2 = _adjusted_r2(test_r2, n=len(y_test), p=p)
                test_evs = float(explained_variance_score(y_test, yhat_test)) if len(eval_idx) >= 2 else np.nan
                test_mape = _safe_mape(y_test, yhat_test)
                test_smape = _safe_smape(y_test, yhat_test)
                test_corr = _safe_corr(y_test, yhat_test)
                err_mean = float(np.nanmean(y_test - yhat_test))
                err_std = float(np.nanstd(y_test - yhat_test, ddof=1)) if len(eval_idx) >= 2 else np.nan
            else:
                test_mae = test_mse = test_rmse = test_medae = np.nan
                test_r2 = test_adj_r2 = test_evs = np.nan
                test_mape = test_smape = test_corr = np.nan
                err_mean = err_std = np.nan

            report_rows.append(
                {
                    "fold": k,
                    "n_train": n_train,
                    "n_test_total": n_test_total,
                    "n_test_pred": n_test_pred,
                    "n_test_eval": n_test_eval,
                    # Train
                    "mae_train": train_mae,
                    "mae_test": test_mae,
                    "mse_train": train_mse,
                    "mse_test": test_mse,
                    "rmse_train": train_rmse,
                    "rmse_test": test_rmse,
                    "medae_train": train_medae,
                    "medae_test": test_medae,
                    "mape_train": train_mape,
                    "mape_test": test_mape,
                    "smape_train": train_smape,
                    "smape_test": test_smape,
                    "r2_train": train_r2,
                    "r2_test": test_r2,
                    "adj_r2_train": train_adj_r2,
                    "adj_r2_test": test_adj_r2,
                    "explained_var_train": train_evs,
                    "explained_var_test": test_evs,
                    "corr_train": train_corr,
                    "corr_test": test_corr,
                    # Residual summary (true - pred)
                    "error_mean_test": err_mean,
                    "error_std_test": err_std,
                    # Model info
                    "booster": booster,
                    "objective": objective,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "tree_importance_type": tree_importance_type if str(booster).lower() != "gblinear" else None,
                    "status": "ok",
                }
            )

    report_df = pd.DataFrame(report_rows).sort_values("fold").reset_index(drop=True)
    return df_out, report_df, model

def fit_linear_regression(
    df: pd.DataFrame,
    predictor_cols: List[str],
    target_col: str,
    intercept: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = df[predictor_cols].copy()
    y = df[target_col].copy()
    
    if intercept:
        X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results

def xgb_predictor(df, objective, predictor_cols, target_col, new_col,
                  learning_rate, max_depth, n_estimators, model=None):

    from xgboost import XGBRegressor

    X = df[predictor_cols]
    y = df[target_col]

    if model is None:
        model = XGBRegressor(
            objective=objective,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X, y)

    df[new_col] = model.predict(X)

    return df, model

def lin_regression(
    df,
    predictor_cols,
    target_col,
    objective='reg:squarederror',
    model=None
):
    X = df[predictor_cols]
    y = df[target_col]
    
    if model is not None:
        # Use pretrained model for predictions
        predictions = model.predict(X)
        return model, predictions
    
    # Fit OLS without intercept (no sm.add_constant)
    ols_model = sm.OLS(y, X)
    results = ols_model.fit()
    
    return ols_model, results