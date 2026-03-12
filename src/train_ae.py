import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype, is_timedelta64_dtype

def _build_mlp(input_dim: int):
    mlp = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    mlp.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return mlp

def build_ae(input_dim: int, latent_dim: int = 50):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    encoder = models.Model(inp, z, name="ae_encoder")

    z_in = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(z_in)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(input_dim)(x)
    decoder = models.Model(z_in, out, name="ae_decoder")

    ae_out = decoder(encoder(inp))
    autoencoder = models.Model(inp, ae_out, name="autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder, decoder

def build_encoder_mlp(encoder):
    encoder.trainable = False  # freeze pretrained encoder
    inp = layers.Input(shape=encoder.input_shape[1:])
    z = encoder(inp, training=False)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out, name="ae_encoder_mlp")
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m

def train_ae_and_predict_rul(
    merged_data: pd.DataFrame,
    train_idx,
    val_idx,
    test_idx,
    latent_dim: int = 50,
    epochs_representation: int = 10,
    epochs_finetune: int = 10,
):
    drop_cols = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]

    unlabeled_mask = merged_data["RUL"] == -1
    labeled_mask = merged_data["RUL"] >= 0

    X_all = merged_data.drop(columns=drop_cols).copy()
    y = merged_data.loc[labeled_mask, "RUL"].values.astype(np.float32)

    num_cols = X_all.select_dtypes(include=[np.number]).columns
    obj_cols = X_all.select_dtypes(include=["object"]).columns
    X_all[num_cols] = X_all[num_cols].fillna(X_all[num_cols].mean())
    for c in obj_cols:
        X_all[c] = X_all[c].fillna("Unknown")

    cat_cols = [c for c in X_all.columns if X_all[c].dtype == "object" or c.startswith("Spec_")]
    for c in cat_cols:
        le = LabelEncoder()
        X_all[c] = le.fit_transform(X_all[c].astype(str))

    X_unlabeled = X_all.loc[unlabeled_mask]
    X_labeled = X_all.loc[labeled_mask]

    # right before MinMaxScaler().fit_transform(...)
    # sanitize datetime-like columns for sklearn
    td_cols = [c for c in X_unlabeled.columns if is_timedelta64_dtype(X_unlabeled[c])]
    for c in td_cols:
        X_unlabeled[c] = X_unlabeled[c].dt.total_seconds()
        if c in X_labeled.columns:
            X_labeled[c] = X_labeled[c].dt.total_seconds()

    dt_cols = [
        c for c in X_unlabeled.columns
        if is_datetime64_any_dtype(X_unlabeled[c]) or is_datetime64tz_dtype(X_unlabeled[c])
    ]
    if dt_cols:
        X_unlabeled = X_unlabeled.drop(columns=dt_cols, errors="ignore")
        X_labeled = X_labeled.drop(columns=dt_cols, errors="ignore")

    scaler = MinMaxScaler()
    X_unlabeled = scaler.fit_transform(X_unlabeled)
    X_labeled = scaler.transform(X_labeled)

    X_unlabeled = np.nan_to_num(X_unlabeled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    X_labeled = np.nan_to_num(X_labeled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)

    X_train = X_labeled[train_idx]
    X_val = X_labeled[val_idx]
    X_test = X_labeled[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"[AE] Unlabeled samples for AE pretraining: {len(X_unlabeled)}")
    print(f"[AE] AE+MLP train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    autoencoder, encoder, decoder = build_ae(input_dim=X_unlabeled.shape[1], latent_dim=latent_dim)
    autoencoder.fit(
        X_unlabeled,
        X_unlabeled,
        epochs=epochs_representation,
        batch_size=256,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )

    ae_mlp = build_encoder_mlp(encoder)
    ae_mlp.fit(
        X_train,
        y_train,
        epochs=epochs_finetune,
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )
    _, ae_mlp_mae = ae_mlp.evaluate(X_test, y_test, verbose=0)

    ae_metrics = {
        "ae_mlp_mae": float(ae_mlp_mae),
    }

    return autoencoder, encoder, ae_mlp, ae_metrics