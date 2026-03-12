import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alfa_preprocess import preprocess_alfa_merged


def _sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


def _build_vae(input_dim: int, latent_dim: int = 50):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(_sampling, name="z")([z_mean, z_log_var])

    dz = layers.Input(shape=(latent_dim,))
    d = layers.Dense(128, activation="relu")(dz)
    d = layers.Dense(256, activation="relu")(d)
    dout = layers.Dense(input_dim)(d)
    decoder = models.Model(dz, dout, name="decoder_alfa")

    recon = decoder(z)
    vae = models.Model(inp, recon, name="vae_alfa")
    encoder = models.Model(inp, z_mean, name="vae_encoder_alfa")

    # Keras-3 safe compile path (no add_loss on Functional)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return vae, encoder


def _build_mlp(encoder, lr=3e-4, dropout=0.25):
    inp = layers.Input(shape=encoder.input_shape[1:])
    z = encoder(inp)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out, name="vae_mlp_alfa")
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.Huber(), metrics=["mae"])
    return m


def train_vae_alfa_and_predict_rul(merged_data, train_idx, val_idx, test_idx, latent_dim=32, epochs_representation=30, epochs_finetune=60):
    X_unlabeled, X_labeled, y = preprocess_alfa_merged(merged_data)
    X_train, X_val, X_test = X_labeled[train_idx], X_labeled[val_idx], X_labeled[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    vae, encoder = _build_vae(X_labeled.shape[1], latent_dim)
    pre_x = X_unlabeled if len(X_unlabeled) > 0 else X_train

    # reconstruction training needs targets
    vae.fit(
        pre_x, pre_x,
        epochs=epochs_representation,
        batch_size=256,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )

    vae_mlp = _build_mlp(encoder, lr=3e-4)

    encoder.trainable = False
    vae_mlp.fit(
        X_train, y_train, epochs=12, batch_size=256, validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True), ReduceLROnPlateau(patience=2, factor=0.5)],
        verbose=1,
    )

    encoder.trainable = True
    vae_mlp.compile(optimizer=tf.keras.optimizers.Adam(8e-5), loss=tf.keras.losses.Huber(), metrics=["mae"])
    vae_mlp.fit(
        X_train, y_train, epochs=epochs_finetune, batch_size=256, validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True), ReduceLROnPlateau(patience=3, factor=0.5)],
        verbose=1,
    )

    _, mae = vae_mlp.evaluate(X_test, y_test, verbose=0)
    return vae, encoder, vae_mlp, {"vae_mlp_mae": float(mae)}