import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

def _build_mlp(input_dim: int):
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return m

def train_plain_baseline_mlp(
    merged_data: pd.DataFrame,
    train_idx,
    val_idx,
    test_idx,
    epochs: int = 10,
):
    drop_cols = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]
    labeled = merged_data[merged_data["RUL"] >= 0].copy()

    X = labeled.drop(columns=drop_cols).copy()
    y = labeled["RUL"].values.astype(np.float32)

    num_cols = X.select_dtypes(include=[np.number]).columns
    obj_cols = X.select_dtypes(include=["object"]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

    for c in obj_cols:
        X[c] = X[c].fillna("Unknown")
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"[PLAIN] Baseline MLP train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    model = _build_mlp(X_train.shape[1])
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )
    _, mae = model.evaluate(X_test, y_test, verbose=0)
    return model, float(mae)