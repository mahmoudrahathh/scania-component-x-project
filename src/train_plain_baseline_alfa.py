import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from alfa_preprocess import preprocess_alfa_merged


def train_plain_baseline_alfa_mlp(merged_data, train_idx, val_idx, test_idx, epochs=30):
    _, X_labeled, y = preprocess_alfa_merged(merged_data)

    X_train, X_val, X_test = X_labeled[train_idx], X_labeled[val_idx], X_labeled[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    i = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256, activation="relu")(i)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    o = layers.Dense(1)(x)
    m = models.Model(i, o, name="plain_baseline_alfa")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])

    m.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )
    _, mae = m.evaluate(X_test, y_test, verbose=0)
    return m, float(mae)