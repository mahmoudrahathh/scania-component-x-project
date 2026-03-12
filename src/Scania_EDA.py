import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_timedelta64_dtype,
)

from data_loader import DataLoader
from analysis import DataAnalysis


def _prepare_features(merged_data: pd.DataFrame):
    if "RUL" not in merged_data.columns:
        raise ValueError("Merged Scania data must contain a 'RUL' column.")

    y = np.where(merged_data["RUL"].to_numpy() == -1, 0, 1)  # 0=Healthy, 1=Failed

    drop_cols = [
        "RUL",
        "vehicle_id",
        "time_step",
        "length_of_study_time_step",
        "in_study_repair",
    ]
    X = merged_data.drop(columns=[c for c in drop_cols if c in merged_data.columns], errors="ignore").copy()

    dt_drop = []
    for c in X.columns:
        if is_timedelta64_dtype(X[c]):
            X[c] = X[c].dt.total_seconds()
        elif is_datetime64_any_dtype(X[c]) or is_datetime64tz_dtype(X[c]):
            dt_drop.append(c)

    if dt_drop:
        X = X.drop(columns=dt_drop, errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns
    for c in cat_cols:
        X[c] = pd.factorize(X[c].astype("string").fillna("Unknown"))[0]

    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def _subsample_data(X: np.ndarray, y: np.ndarray, max_samples: int = 5000, random_state: int = 42):
    n = len(y)
    if n <= max_samples:
        return X, y

    rng = np.random.default_rng(random_state)

    healthy_idx = np.where(y == 0)[0]
    failed_idx = np.where(y == 1)[0]

    n_healthy = int(round(max_samples * len(healthy_idx) / n))
    n_failed = max_samples - n_healthy

    n_healthy = min(n_healthy, len(healthy_idx))
    n_failed = min(n_failed, len(failed_idx))

    if n_healthy + n_failed < max_samples:
        remaining = max_samples - (n_healthy + n_failed)
        extra_healthy = min(remaining, len(healthy_idx) - n_healthy)
        n_healthy += extra_healthy
        remaining -= extra_healthy
        extra_failed = min(remaining, len(failed_idx) - n_failed)
        n_failed += extra_failed

    keep_idx = np.concatenate([
        rng.choice(healthy_idx, size=n_healthy, replace=False) if n_healthy > 0 else np.array([], dtype=int),
        rng.choice(failed_idx, size=n_failed, replace=False) if n_failed > 0 else np.array([], dtype=int),
    ])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    data_loader = DataLoader()
    operational_readouts = data_loader.load_operational_readouts()
    specifications = data_loader.load_specifications()
    tte_data = data_loader.load_tte()

    data_analysis = DataAnalysis(operational_readouts, specifications, tte_data)
    merged_data = data_analysis.perform_analysis()

    print(f"[Scania EDA] Merged data shape: {merged_data.shape}")

    healthy_count = int((merged_data["RUL"] == -1).sum())
    failed_count = int((merged_data["RUL"] >= 0).sum())

    print(f"[Scania EDA] Healthy samples (RUL = -1): {healthy_count}")
    print(f"[Scania EDA] Failed samples (RUL >= 0): {failed_count}")

    X, y = _prepare_features(merged_data)
    X, y = _subsample_data(X, y, max_samples=5000, random_state=42)

    print(f"[Scania EDA] Using {len(y)} samples for t-SNE")

    X_2d = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    ).fit_transform(X)

    healthy_mask = y == 0
    failed_mask = y == 1

    plt.figure(figsize=(10, 8))

    plt.scatter(
        X_2d[healthy_mask, 0],
        X_2d[healthy_mask, 1],
        s=8,
        alpha=0.20,
        c="#4C78A8",
        label=f"Healthy (RUL = -1): {int(healthy_mask.sum())}",
        edgecolors="none",
    )
    plt.scatter(
        X_2d[failed_mask, 0],
        X_2d[failed_mask, 1],
        s=18,
        alpha=0.90,
        c="#E45756",
        label=f"Failed (RUL >= 0): {int(failed_mask.sum())}",
        edgecolors="none",
    )

    plt.title("Scania Samples in 2D (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(out_dir, "scania_eda_tsne_2d.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Scania EDA] Plot saved to: {out_file}")


if __name__ == "__main__":
    main()