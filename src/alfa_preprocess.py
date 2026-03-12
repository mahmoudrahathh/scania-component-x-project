import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype, is_timedelta64_dtype


def preprocess_alfa_merged(merged_data: pd.DataFrame):
    drop_cols = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]

    unlabeled_mask = merged_data["RUL"] == -1
    labeled_mask = merged_data["RUL"] >= 0

    X_all = merged_data.drop(columns=drop_cols, errors="ignore").copy()
    y = merged_data.loc[labeled_mask, "RUL"].to_numpy(dtype=np.float32)

    # timedelta -> seconds, datetime -> drop
    dt_drop = []
    for c in X_all.columns:
        if is_timedelta64_dtype(X_all[c]):
            X_all[c] = X_all[c].dt.total_seconds()
        elif is_datetime64_any_dtype(X_all[c]) or is_datetime64tz_dtype(X_all[c]):
            dt_drop.append(c)
    if dt_drop:
        X_all = X_all.drop(columns=dt_drop, errors="ignore")

    # fill numerics
    num_cols = X_all.select_dtypes(include=[np.number]).columns
    X_all[num_cols] = X_all[num_cols].replace([np.inf, -np.inf], np.nan)
    X_all[num_cols] = X_all[num_cols].fillna(X_all[num_cols].mean())

    # encode categoricals robustly
    cat_cols = X_all.select_dtypes(include=["object", "string", "category"]).columns
    for c in cat_cols:
        le = LabelEncoder()
        X_all[c] = le.fit_transform(X_all[c].astype("string").fillna("Unknown").astype(str))

    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_unlabeled = X_all.loc[unlabeled_mask]
    X_labeled = X_all.loc[labeled_mask]

    scaler = MinMaxScaler()
    if len(X_unlabeled) > 0:
        X_unlabeled = scaler.fit_transform(X_unlabeled)
        X_labeled = scaler.transform(X_labeled)
    else:
        X_labeled = scaler.fit_transform(X_labeled)
        X_unlabeled = np.empty((0, X_labeled.shape[1]), dtype=np.float32)

    X_unlabeled = np.asarray(X_unlabeled, dtype=np.float32)
    X_labeled = np.asarray(X_labeled, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)

    return X_unlabeled, X_labeled, y