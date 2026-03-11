import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

class Sampling(layers.Layer):
    """Custom layer for VAE sampling."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, 0.0, 1.0)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
        return reconstructed

def build_vae(input_dim, latent_dim=100):
    """Build a deeper VAE."""
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(256, activation="relu")(inputs)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(64, activation="relu")(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    h_dec = layers.Dense(64, activation="relu")(latent_inputs)
    h_dec = layers.Dropout(0.2)(h_dec)
    h_dec = layers.Dense(128, activation="relu")(h_dec)
    h_dec = layers.Dropout(0.2)(h_dec)
    h_dec = layers.Dense(256, activation="relu")(h_dec)
    outputs = layers.Dense(input_dim)(h_dec)
    decoder = models.Model(latent_inputs, outputs, name="decoder")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
    return vae, encoder, decoder

def build_vae_mlp(encoder, latent_dim):
    """Build VAE + MLP by attaching MLP head to the frozen encoder."""
    # Unfreeze encoder for fine-tuning
    encoder.trainable = True

    inputs = layers.Input(shape=encoder.input_shape[1:])
    z_mean, z_log_var, z = encoder(inputs)
    # MLP head on top of z_mean
    x = layers.Dense(128, activation="relu")(z_mean)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1)(x)
    model = models.Model(inputs, output, name="vae_mlp")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return model

def build_baseline_mlp(input_dim):
    """Build baseline MLP with same head architecture as VAE+MLP."""
    mlp = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return mlp

def train_vae_and_predict_rul(merged_data: pd.DataFrame):
    # Separate unlabeled (RUL = -1) and labeled (RUL >= 0)
    drop_cols = ['vehicle_id', 'time_step', 'length_of_study_time_step', 'in_study_repair', 'RUL']
    
    unlabeled_full = merged_data[merged_data['RUL'] == -1].drop(columns=drop_cols).copy()
    labeled = merged_data[merged_data['RUL'] >= 0].copy()
    labeled_features = labeled.drop(columns=drop_cols).copy()
    y = labeled['RUL'].values

    print(f"\nUnlabeled samples (for VAE): {len(unlabeled_full)}")
    print(f"Labeled samples (for MLP fine-tuning): {len(labeled)}")

    # Fill NaN: mean for numeric, 'Unknown' for categorical
    numeric_cols = unlabeled_full.select_dtypes(include=[np.number]).columns
    cat_cols = unlabeled_full.select_dtypes(include=['object']).columns

    unlabeled_full[numeric_cols] = unlabeled_full[numeric_cols].fillna(unlabeled_full[numeric_cols].mean())
    for col in cat_cols:
        unlabeled_full[col] = unlabeled_full[col].fillna('Unknown')

    # Encode categorical columns on full unlabeled data
    categorical_cols = [col for col in unlabeled_full.columns if col.startswith('Spec_')]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        unlabeled_full[col] = le.fit_transform(unlabeled_full[col])
        encoders[col] = le

    # Scale on full unlabeled data
    scaler = MinMaxScaler()
    unlabeled_scaled = scaler.fit_transform(unlabeled_full)
    unlabeled_scaled = np.nan_to_num(unlabeled_scaled, nan=0.0, posinf=1.0, neginf=0.0)

    # Train VAE on all unlabeled samples
    input_dim = unlabeled_scaled.shape[1]
    print(f"\nTraining VAE on {len(unlabeled_scaled)} unlabeled samples...")
    vae, encoder, decoder = build_vae(input_dim)
    vae.fit(
        unlabeled_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10)]
    )
    print("\nVAE training completed. Now fine-tuning encoder + MLP head on labeled samples...")

    # Prepare labeled features
    labeled_features[numeric_cols] = labeled_features[numeric_cols].fillna(labeled_features[numeric_cols].mean())
    for col in cat_cols:
        if col in labeled_features.columns:
            labeled_features[col] = labeled_features[col].fillna('Unknown')
    for col in categorical_cols:
        if col in labeled_features.columns:
            labeled_features[col] = encoders[col].transform(labeled_features[col])
    labeled_scaled = scaler.transform(labeled_features)
    labeled_scaled = np.nan_to_num(labeled_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    print(f"Labeled samples used for VAE+MLP fine-tuning: {len(labeled_scaled)}")
    print(f"Labeled samples used for Baseline MLP: {len(labeled_scaled)}")

    # Split — same split for all models for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_scaled, y, test_size=0.2, random_state=42
    )

    # Sanity-check baseline: predict mean training RUL for every test sample
    mean_rul_train = float(np.mean(y_train))
    y_pred_mean = np.full_like(y_test, fill_value=mean_rul_train, dtype=np.float64)
    mae_mean_baseline = float(np.mean(np.abs(y_test - y_pred_mean)))

    print(f"\nSanity baseline (predict mean train RUL): {mean_rul_train:.4f}")
    print(f"Sanity baseline MAE: {mae_mean_baseline:.4f}")

    # VAE + MLP: fine-tune encoder + MLP head jointly
    print(f"\nTraining VAE + MLP on {len(X_train)} samples (validation: {len(X_train) - int(len(X_train)*0.8)}, test: {len(X_test)})...")
    vae_mlp = build_vae_mlp(encoder, latent_dim=100)
    vae_mlp.fit(
        X_train, y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10)]
    )
    loss_vae, mae_vae = vae_mlp.evaluate(X_test, y_test)
    print(f"\nVAE + MLP Test MAE: {mae_vae:.4f}")

    # Baseline MLP: train directly on raw features
    print(f"\nTraining Baseline MLP on {len(X_train)} samples (validation: {len(X_train) - int(len(X_train)*0.8)}, test: {len(X_test)})...")
    mlp_baseline = build_baseline_mlp(input_dim)
    mlp_baseline.fit(
        X_train, y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10)]
    )
    loss_base, mae_base = mlp_baseline.evaluate(X_test, y_test)
    print(f"Baseline MLP Test MAE: {mae_base:.4f}")

    # Comparison
    print(f"\n=== Performance Comparison ===")
    print(f"Sanity Mean-RUL MAE: {mae_mean_baseline:.4f}")
    print(f"VAE + MLP MAE:       {mae_vae:.4f}")
    print(f"Baseline MLP MAE:    {mae_base:.4f}")

    return vae, encoder, vae_mlp, mlp_baseline