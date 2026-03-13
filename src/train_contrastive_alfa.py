import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alfa_preprocess import preprocess_alfa_merged


def _augment(x): return x + tf.random.normal(tf.shape(x), stddev=0.05)

def _enc(input_dim, latent_dim=50):
    i = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(i)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim)(x)
    return models.Model(i, z)

def _proj(latent_dim=50):
    i = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(i)
    o = layers.Dense(64)(x)
    return models.Model(i, o)

def _loss(z1, z2, t=0.1):
    n = tf.shape(z1)[0]
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    z = tf.concat([z1, z2], axis=0)
    sim = tf.matmul(z, z, transpose_b=True) / t
    mask = 1.0 - tf.eye(2 * n)
    sim = sim * mask - 1e9 * (1.0 - mask)
    labels = tf.concat([tf.range(n, 2 * n), tf.range(0, n)], axis=0)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim))

@tf.function
def _step(b, e, p, opt):
    with tf.GradientTape() as tape:
        z1 = p(e(_augment(b), training=True), training=True)
        z2 = p(e(_augment(b), training=True), training=True)
        l = _loss(z1, z2)
    vars_ = e.trainable_variables + p.trainable_variables
    opt.apply_gradients(zip(tape.gradient(l, vars_), vars_))
    return l

def _head(encoder, lr=1e-4, dropout=0.25):
    encoder.trainable = False  # permanently frozen
    i = layers.Input(shape=encoder.input_shape[1:])
    z = encoder(i, training=False)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    o = layers.Dense(1)(x)
    m = models.Model(i, o)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.Huber(), metrics=["mae"])
    return m

def train_contrastive_alfa_and_predict_rul(merged_data, train_idx, val_idx, test_idx, latent_dim=32, epochs_representation=40, epochs_finetune=30):
    X_unlabeled, X_labeled, y = preprocess_alfa_merged(merged_data)
    X_train, X_val, X_test = X_labeled[train_idx], X_labeled[val_idx], X_labeled[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    pre_x = X_unlabeled if len(X_unlabeled) > 0 else X_train
    e, p = _enc(X_labeled.shape[1], latent_dim), _proj(latent_dim)
    opt = tf.keras.optimizers.Adam(1e-3)

    ds = tf.data.Dataset.from_tensor_slices(pre_x).shuffle(10000).batch(256, drop_remainder=True)
    for _ in range(epochs_representation):
        for b in ds:
            _step(b, e, p, opt)

    m = _head(e, lr=1e-4)

    # single stage: frozen encoder, head only
    m.fit(
        X_train, y_train,
        epochs=epochs_finetune,
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True),
            ReduceLROnPlateau(patience=3, factor=0.5),
        ],
        verbose=1,
    )

    _, mae = m.evaluate(X_test, y_test, verbose=0)
    return e, m, {"contrastive_mlp_mae": float(mae)}