import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alfa_preprocess import preprocess_alfa_merged


def _build_autoencoder(input_dim: int, latent_dim: int = 50):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(input_dim)(x)
    autoencoder = models.Model(inp, out, name="ae_alfa")
    encoder = models.Model(inp, z, name="ae_encoder_alfa")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder


def _build_mlp(encoder, lr=1e-4, dropout=0.25):
    encoder.trainable = False  # permanently frozen
    inp = layers.Input(shape=encoder.input_shape[1:])
    z = encoder(inp, training=False)
    x = layers.Dense(128, activation="relu")(z)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out, name="ae_mlp_alfa")
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.Huber(), metrics=["mae"])
    return m


def train_ae_alfa_and_predict_rul(merged_data, train_idx, val_idx, test_idx, latent_dim=32, epochs_representation=30, epochs_finetune=30):
    X_unlabeled, X_labeled, y = preprocess_alfa_merged(merged_data)
    X_train, X_val, X_test = X_labeled[train_idx], X_labeled[val_idx], X_labeled[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    autoencoder, encoder = _build_autoencoder(X_labeled.shape[1], latent_dim)
    pre_x = X_unlabeled if len(X_unlabeled) > 0 else X_train
    autoencoder.fit(
        pre_x, pre_x,
        epochs=epochs_representation,
        batch_size=256,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )

    encoder = models.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("latent").output,
        name="ae_encoder",
    )

    ae_mlp = _build_mlp(encoder, lr=1e-4)

    # single stage: frozen encoder, head only
    ae_mlp.fit(
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

    _, mae = ae_mlp.evaluate(X_test, y_test, verbose=0)
    return autoencoder, encoder, ae_mlp, {"ae_mlp_mae": float(mae)}