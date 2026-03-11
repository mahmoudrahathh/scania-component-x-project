import pandas as pd
import numpy as np
from data_loader import DataLoader
from analysis import DataAnalysis
from train_vae import train_vae_and_predict_rul

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

    # Train VAE and predict RUL
    vae, encoder, mlp_vae, mlp_baseline = train_vae_and_predict_rul(merged_data)

if __name__ == "__main__":
    main()