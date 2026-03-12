import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_timedelta64_dtype,
)


# ─────────────────────────────────────────────
# Augmentation: add small Gaussian noise
# ─────────────────────────────────────────────
def augment(x: tf.Tensor, noise_std: float = 0.05) -> tf.Tensor:
    return x + tf.random.normal(shape=tf.shape(x), stddev=noise_std)


# ─────────────────────────────────────────────
# Encoder backbone
# ─────────────────────────────────────────────
def build_encoder(input_dim: int, latent_dim: int = 50) -> models.Model:
    inp = layers.Input(shape=(input_dim,), name="enc_input")
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    return models.Model(inp, z, name="contrastive_encoder")


# ─────────────────────────────────────────────
# Projection head (used only during contrastive
# pretraining, discarded afterwards)
# ─────────────────────────────────────────────
def build_projection_head(latent_dim: int = 50) -> models.Model:
    inp = layers.Input(shape=(latent_dim,), name="proj_input")
    x = layers.Dense(128, activation="relu")(inp)
    out = layers.Dense(64)(x)
    return models.Model(inp, out, name="projection_head")


# ─────────────────────────────────────────────
# NT-Xent contrastive loss (SimCLR style)
# ─────────────────────────────────────────────
def nt_xent_loss(z_i: tf.Tensor, z_j: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    batch_size = tf.shape(z_i)[0]

    # Normalize
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Concatenate [z_i; z_j]  → shape (2N, dim)
    z = tf.concat([z_i, z_j], axis=0)

    # Similarity matrix (2N x 2N)
    sim = tf.matmul(z, z, transpose_b=True) / temperature

    # Remove self-similarity
    mask = 1.0 - tf.eye(2 * batch_size)
    sim = sim * mask - 1e9 * (1.0 - mask)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels_top = tf.range(batch_size, 2 * batch_size)
    labels_bot = tf.range(0, batch_size)
    labels = tf.concat([labels_top, labels_bot], axis=0)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim)
    )
    return loss


# ─────────────────────────────────────────────
# Contrastive pretraining step
# ─────────────────────────────────────────────
@tf.function
def train_step(batch, encoder, proj_head, optimizer, temperature=0.1):
    with tf.GradientTape() as tape:
        z_i = proj_head(encoder(augment(batch), training=True), training=True)
        z_j = proj_head(encoder(augment(batch), training=True), training=True)
        loss = nt_xent_loss(z_i, z_j, temperature=temperature)
    grads = tape.gradient(loss, encoder.trainable_variables + proj_head.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, encoder.trainable_variables + proj_head.trainable_variables)
    )
    return loss


# ─────────────────────────────────────────────
# MLP head on top of frozen → fine-tuned encoder
# ─────────────────────────────────────────────
def build_encoder_mlp(encoder: models.Model) -> models.Model:
    encoder.trainable = False   # fine-tune encoder weights jointly
    inp = layers.Input(shape=encoder.input_shape[1:])
    z = encoder(inp)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out, name="contrastive_encoder_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────
# Preprocessing helper (shared logic)
# ─────────────────────────────────────────────
def _preprocess(merged_data: pd.DataFrame):
    drop_cols = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]

    unlabeled_mask = merged_data["RUL"] == -1
    labeled_mask = merged_data["RUL"] >= 0

    X_all = merged_data.drop(columns=drop_cols).copy()
    y = merged_data.loc[labeled_mask, "RUL"].values.astype(np.float32)

    # Handle datetime/timedelta columns before scaling
    dt_drop = []
    for c in X_all.columns:
        if is_timedelta64_dtype(X_all[c]):
            X_all[c] = X_all[c].dt.total_seconds()
        elif is_datetime64_any_dtype(X_all[c]) or is_datetime64tz_dtype(X_all[c]):
            dt_drop.append(c)

    if dt_drop:
        X_all = X_all.drop(columns=dt_drop)

    num_cols = X_all.select_dtypes(include=[np.number]).columns
    obj_cols = X_all.select_dtypes(include=["object", "string", "category"]).columns

    X_all[num_cols] = X_all[num_cols].fillna(X_all[num_cols].mean())
    for c in obj_cols:
        X_all[c] = X_all[c].fillna("Unknown")

    cat_cols = [c for c in X_all.columns if X_all[c].dtype in ["object", "string", "category"] or c.startswith("Spec_")]
    for c in cat_cols:
        le = LabelEncoder()
        X_all[c] = le.fit_transform(X_all[c].astype(str))

    X_unlabeled = X_all.loc[unlabeled_mask]
    X_labeled = X_all.loc[labeled_mask]

    scaler = MinMaxScaler()
    X_unlabeled = scaler.fit_transform(X_unlabeled)
    X_labeled = scaler.transform(X_labeled)

    X_unlabeled = np.nan_to_num(X_unlabeled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    X_labeled = np.nan_to_num(X_labeled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)

    return X_unlabeled, X_labeled, y


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train_contrastive_and_predict_rul(
    merged_data: pd.DataFrame,
    train_idx,
    val_idx,
    test_idx,
    latent_dim: int = 50,
    epochs_representation: int = 10,
    epochs_finetune: int = 50,
    batch_size: int = 256,
    temperature: float = 0.1,
):
    X_unlabeled, X_labeled, y = _preprocess(merged_data)

    X_train = X_labeled[train_idx]
    X_val   = X_labeled[val_idx]
    X_test  = X_labeled[test_idx]
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    print(f"\n[Contrastive] Unlabeled samples for pretraining: {len(X_unlabeled)}")
    print(f"[Contrastive] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── Step 1: Contrastive pretraining on unlabeled data ──
    input_dim = X_unlabeled.shape[1]
    encoder   = build_encoder(input_dim, latent_dim)
    proj_head = build_projection_head(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    dataset = (
        tf.data.Dataset.from_tensor_slices(X_unlabeled)
        .shuffle(10000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"\n[Contrastive] Pretraining encoder for {epochs_representation} epochs...")
    best_loss = float("inf")
    patience_counter = 0
    patience = 3

    for epoch in range(1, epochs_representation + 1):
        epoch_losses = []
        for batch in dataset:
            loss = train_step(batch, encoder, proj_head, optimizer, temperature)
            epoch_losses.append(float(loss))
        mean_loss = float(np.mean(epoch_losses))
        print(f"  Epoch {epoch}/{epochs_representation} — contrastive loss: {mean_loss:.6f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_encoder_weights = encoder.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best encoder weights
    encoder.set_weights(best_encoder_weights)

    # ── Step 2: Discard projection head, attach MLP, fine-tune ──
    print("\n[Contrastive] Pretraining done. Fine-tuning encoder + MLP on labeled samples...")
    cl_mlp = build_encoder_mlp(encoder)
    cl_mlp.fit(
        X_train,
        y_train,
        epochs=epochs_finetune,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    _, cl_mae = cl_mlp.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Contrastive] Contrastive + MLP Test MAE: {cl_mae:.4f}")

    cl_metrics = {
        "contrastive_mlp_mae": float(cl_mae),
    }

    return encoder, cl_mlp, cl_metrics