import pandas as pd
import numpy as np
from data_loader import DataLoader
from analysis import DataAnalysis
from train_vae import train_vae_and_predict_rul
from train_ae import train_ae_and_predict_rul
from train_plain_baseline import train_plain_baseline_mlp
from train_contrastive import train_contrastive_and_predict_rul
from sklearn.model_selection import train_test_split
from visualize_representations import visualize_latent_representations

def main():
    # Load the data
    data_loader = DataLoader()
    operational_readouts = data_loader.load_operational_readouts()
    specifications = data_loader.load_specifications()
    tte_data = data_loader.load_tte()

    # Inspect the top rows
    print("=== operational_readouts head ===")
    print(operational_readouts.head(), "\n")

    print("=== specifications head ===")
    print(specifications.head(), "\n")

    print("=== tte head ===")
    print(tte_data.head(), "\n")

    # Perform analysis and merge
    data_analysis = DataAnalysis(operational_readouts, specifications, tte_data)
    merged_data = data_analysis.perform_analysis()

    # Visualize results
    data_analysis.visualize_results()

    print("\n=== Merged Data Sample ===")
    print(merged_data[["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]].head(10))

    # Count RUL distribution
    rul_negative_one = (merged_data["RUL"] == -1).sum()
    rul_non_negative = (merged_data["RUL"] >= 0).sum()

    print("\n=== RUL Distribution ===")
    print(f"Samples with RUL = -1: {rul_negative_one}")
    print(f"Samples with RUL >= 0: {rul_non_negative}")
    print(f"Total samples: {len(merged_data)}")
    print(f"\nMerged data size: {merged_data.shape}")

    # Split once here and reuse everywhere
    labeled_mask = merged_data["RUL"] >= 0
    labeled_indices = np.arange(int(labeled_mask.sum()))

    train_idx, test_idx = train_test_split(
        labeled_indices, test_size=0.2, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42
    )

    print(f"\n[Split] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Contrastive learning experiment
    cl_result = train_contrastive_and_predict_rul(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=50,
        epochs_representation=10,
        epochs_finetune=50,
    )
    cl_encoder, cl_mlp, cl_metrics = cl_result

    # AE experiment
    ae_result = train_ae_and_predict_rul(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=50,
        epochs_representation=10,
        epochs_finetune=50,
    )
    autoencoder, ae_encoder, ae_mlp, ae_metrics = ae_result

    # VAE experiment
    vae_result = train_vae_and_predict_rul(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        latent_dim=50,
        epochs_representation=10,
        epochs_finetune=50,
    )
    vae, encoder, vae_mlp, vae_metrics = vae_result

    # Plain baseline
    plain_baseline_model, plain_baseline_mae = train_plain_baseline_mlp(
        merged_data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=50,
    )

    labeled_rul = merged_data.loc[labeled_mask, "RUL"].to_numpy(dtype=float)
    mean_value = float(np.mean(labeled_rul[train_idx]))
    meanValuePredictor = float(np.mean(np.abs(labeled_rul[test_idx] - mean_value)))

    print("\n=== Final Performance Comparison ===")
    print(f"Contrastive + MLP MAE:       {cl_metrics['contrastive_mlp_mae']:.4f}")
    print(f"AE + MLP MAE:                {ae_metrics['ae_mlp_mae']:.4f}")
    print(f"VAE + MLP MAE:               {vae_metrics['vae_mlp_mae']:.4f}")
    print(f"Standalone Baseline MLP MAE: {plain_baseline_mae:.4f}")
    print(f"MeanValuePredictor MAE:      {meanValuePredictor:.4f}")

    # Visualize latent spaces (same number labeled vs unlabeled)
    # visualize_latent_representations(
    #     merged_data=merged_data,
    #     ae_encoder=ae_encoder,
    #     vae_encoder=encoder,
    #     contrastive_encoder=cl_encoder,
    #     latent_dim=50,
    #     random_state=42,
    #     out_file="outputs/latent_spaces_pca2d.jpg",
    # )

if __name__ == "__main__":
    main()