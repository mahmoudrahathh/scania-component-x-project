import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype, is_timedelta64_dtype
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _prepare_features(merged_data: pd.DataFrame):
    drop_cols = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]

    labeled_mask = merged_data["RUL"] >= 0
    unlabeled_mask = merged_data["RUL"] == -1

    X_all = merged_data.drop(columns=drop_cols, errors="ignore").copy()

    # Convert/drop datetime-like columns
    dt_drop = []
    for c in X_all.columns:
        if is_timedelta64_dtype(X_all[c]):
            X_all[c] = X_all[c].dt.total_seconds()
        elif is_datetime64_any_dtype(X_all[c]) or is_datetime64tz_dtype(X_all[c]):
            dt_drop.append(c)
    if dt_drop:
        X_all = X_all.drop(columns=dt_drop, errors="ignore")

    # Encode categoricals
    cat_cols = X_all.select_dtypes(include=["object", "string", "category"]).columns
    for c in cat_cols:
        le = LabelEncoder()
        X_all[c] = le.fit_transform(X_all[c].astype("string").fillna("Unknown").astype(str))

    # Numeric cleanup
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(X_all)

    X_labeled = X_all[labeled_mask.to_numpy()]
    X_unlabeled = X_all[unlabeled_mask.to_numpy()]
    return X_labeled, X_unlabeled


def _encode(encoder_model, X: np.ndarray) -> np.ndarray:
    z = encoder_model.predict(X, batch_size=1024, verbose=0)

    # VAE encoder may return multiple tensors (e.g., z_mean, z_log_var, z)
    if isinstance(z, (list, tuple)):
        if len(z) >= 3:
            z = z[2]
        else:
            z = z[0]

    z = np.asarray(z)
    if z.ndim > 2:
        z = z.reshape(z.shape[0], -1)
    return z


def visualize_latent_representations(
    merged_data: pd.DataFrame,
    ae_encoder,
    vae_encoder,
    contrastive_encoder,
    latent_dim: int = 50,
    random_state: int = 42,
    out_file: str = "outputs/latent_spaces_pca2d.jpg",
):
    X_labeled, X_unlabeled = _prepare_features(merged_data)

    n_labeled = len(X_labeled)
    n_unlabeled = len(X_unlabeled)
    n_take = min(n_labeled, n_unlabeled)

    rng = np.random.default_rng(random_state)
    unlabeled_idx = rng.choice(n_unlabeled, size=n_take, replace=False)

    X_lab = X_labeled[:n_take]
    X_unlab = X_unlabeled[unlabeled_idx]

    X_vis = np.vstack([X_lab, X_unlab])
    y_vis = np.array([1] * n_take + [0] * n_take)  # 1=labeled, 0=unlabeled

    models = [
        ("AE", ae_encoder),
        ("VAE", vae_encoder),
        ("Contrastive", contrastive_encoder),
    ]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model) in zip(axes, models):
        Z = _encode(model, X_vis)

        # optional safety if latent dimension differs
        if Z.shape[1] != latent_dim:
            pass

        Z_2d = PCA(n_components=2, random_state=random_state).fit_transform(Z)

        # Plot unlabeled first (background) with stronger transparency
        ax.scatter(
            Z_2d[y_vis == 0, 0],
            Z_2d[y_vis == 0, 1],
            s=8,
            alpha=0.08,
            label="Unlabeled (RUL=-1)",
            zorder=1,
        )

        # Plot labeled second (foreground), also transparent but clearer
        ax.scatter(
            Z_2d[y_vis == 1, 0],
            Z_2d[y_vis == 1, 1],
            s=10,
            alpha=0.35,
            label="Labeled (RUL>=0)",
            zorder=2,
        )

        ax.set_title(f"{name} latent → PCA(2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, format="jpg")
    print(f"[Visualization] Saved: {out_file}")
    plt.show()