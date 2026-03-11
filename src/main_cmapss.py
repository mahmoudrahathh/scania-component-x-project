import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train_vae import train_vae_and_predict_rul
from train_ae import train_ae_and_predict_rul
from train_plain_baseline import train_plain_baseline_mlp
from train_contrastive import train_contrastive_and_predict_rul
from visualize_representations import visualize_latent_representations


def _load_cmapss_train_files(cmapss_dir: str) -> pd.DataFrame:
    """
    Loads all train_FD00*.txt files from CMAPSSData and concatenates them.
    """
    train_files = sorted(glob.glob(os.path.join(cmapss_dir, "train_FD*.txt")))
    if not train_files:
        raise FileNotFoundError(f"No train_FD*.txt files found in: {cmapss_dir}")

    all_df = []
    for fp in train_files:
        # CMAPSS train files are space-separated with possible trailing blank columns
        df = pd.read_csv(fp, sep=r"\s+", header=None, engine="python")
        # Keep first 26 columns: [unit, cycle, op1, op2, op3, s1..s21]
        df = df.iloc[:, :26].copy()
        df.columns = (
            ["vehicle_id", "time_step", "op_setting_1", "op_setting_2", "op_setting_3"]
            + [f"sensor_{i}" for i in range(1, 22)]
        )
        df["source_file"] = os.path.basename(fp)
        all_df.append(df)

    out = pd.concat(all_df, axis=0, ignore_index=True)
    return out


def _build_merged_like_dataframe(cmapss_raw: pd.DataFrame, unlabeled_ratio: float = 0.5) -> pd.DataFrame:
    """
    Build Scania-like merged_data schema expected by existing training functions:
      - vehicle_id
      - time_step
      - length_of_study_time_step
      - in_study_repair
      - RUL
      - feature columns...

    Strategy for unsupervised part:
      For each engine trajectory, early cycles are marked as unlabeled (RUL=-1),
      late cycles keep true RUL labels.
    """
    df = cmapss_raw.copy()

    # Use source_file + unit id to keep unique ids across FD subsets
    df["vehicle_id"] = df["source_file"].astype(str) + "_" + df["vehicle_id"].astype(str)

    # Drop source_file so it does not enter model features as raw string
    df.drop(columns=["source_file"], inplace=True)

    # Compute sequence length per unit
    max_cycle = df.groupby("vehicle_id")["time_step"].transform("max")
    df["length_of_study_time_step"] = max_cycle.astype(int)
    df["in_study_repair"] = 0  # placeholder for compatibility

    # True RUL
    df["RUL_true"] = (max_cycle - df["time_step"]).astype(float)

    # Mark early part as unlabeled (RUL=-1), late part as labeled with true RUL
    # cutoff cycle per unit = floor(max_cycle * unlabeled_ratio)
    cutoff = (max_cycle * unlabeled_ratio).astype(int)
    is_unlabeled = df["time_step"] <= cutoff

    df["RUL"] = df["RUL_true"]
    df.loc[is_unlabeled, "RUL"] = -1.0
    df.drop(columns=["RUL_true"], inplace=True)

    # Reorder key columns first
    key_cols = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]
    other_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + other_cols]

    return df


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmapss_dir = os.path.join(project_root, "data", "CMAPSSData")

    print(f"[CMAPSS] Loading data from: {cmapss_dir}")
    cmapss_raw = _load_cmapss_train_files(cmapss_dir)
    merged_data = _build_merged_like_dataframe(cmapss_raw, unlabeled_ratio=0.5)

    print("\n=== CMAPSS Merged Data Sample ===")
    print(merged_data.head())

    rul_negative_one = (merged_data["RUL"] == -1).sum()
    rul_non_negative = (merged_data["RUL"] >= 0).sum()

    print("\n=== RUL Distribution (CMAPSS) ===")
    print(f"Samples with RUL = -1: {rul_negative_one}")
    print(f"Samples with RUL >= 0: {rul_non_negative}")
    print(f"Total samples: {len(merged_data)}")
    print(f"Merged data size: {merged_data.shape}")

    # Shared split (same rule as main.py)
    labeled_mask = merged_data["RUL"] >= 0
    labeled_indices = np.arange(int(labeled_mask.sum()))

    train_idx, test_idx = train_test_split(
        labeled_indices, test_size=0.2, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42
    )

    print(f"\n[Split] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Contrastive
    cl_encoder, cl_mlp, cl_metrics = train_contrastive_and_predict_rul(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=50,
        epochs_representation=10,
        epochs_finetune=50,
    )

    # AE
    autoencoder, ae_encoder, ae_mlp, ae_metrics = train_ae_and_predict_rul(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=50,
        epochs_representation=10,
        epochs_finetune=50,
    )

    # VAE
    vae, vae_encoder, vae_mlp, vae_metrics = train_vae_and_predict_rul(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=50,
        epochs_representation=10,
        epochs_finetune=50,
    )

    # Standalone baseline
    plain_baseline_model, plain_baseline_mae = train_plain_baseline_mlp(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=50,
    )

    # MeanValuePredictor
    labeled_rul = merged_data.loc[labeled_mask, "RUL"].to_numpy(dtype=float)
    mean_value = float(np.mean(labeled_rul[train_idx]))
    meanValuePredictor = float(np.mean(np.abs(labeled_rul[test_idx] - mean_value)))

    print("\n=== Final Performance Comparison (CMAPSS) ===")
    print(f"Contrastive + MLP MAE:       {cl_metrics['contrastive_mlp_mae']:.4f}")
    print(f"AE + MLP MAE:                {ae_metrics['ae_mlp_mae']:.4f}")
    print(f"VAE + MLP MAE:               {vae_metrics['vae_mlp_mae']:.4f}")
    print(f"Standalone Baseline MLP MAE: {plain_baseline_mae:.4f}")
    print(f"MeanValuePredictor MAE:      {meanValuePredictor:.4f}")

    visualize_latent_representations(
        merged_data=merged_data,
        ae_encoder=ae_encoder,
        vae_encoder=vae_encoder,
        contrastive_encoder=cl_encoder,
        latent_dim=50,
        random_state=42,
        out_file="outputs/latent_spaces_pca2d_cmapss.jpg",
    )


if __name__ == "__main__":
    main()