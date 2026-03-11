import pandas as pd

class DataAnalysis:
    def __init__(self, operational_readouts: pd.DataFrame, specifications: pd.DataFrame, tte: pd.DataFrame):
        self.operational_readouts = operational_readouts
        self.specifications = specifications
        self.tte = tte
        self.merged_data = None

    def merge_data(self) -> pd.DataFrame:
        """Merge operational readouts with specifications on vehicle_id,
        then add RUL labels from tte."""
        
        # Merge operational_readouts with specifications on vehicle_id
        merged = self.operational_readouts.merge(
            self.specifications,
            on="vehicle_id",
            how="left"
        )
        
        # Merge with tte on vehicle_id
        merged = merged.merge(
            self.tte,
            on="vehicle_id",
            how="left"
        )
        
        # Add RUL column: 
        # For vehicles with in_study_repair == 1, RUL = length_of_study_time_step - time_step
        # For vehicles with in_study_repair == 0, RUL = -1
        merged["RUL"] = merged.apply(
            lambda row: row["length_of_study_time_step"] - row["time_step"]
            if row["in_study_repair"] == 1
            else -1,
            axis=1
        )
        
        self.merged_data = merged
        return merged

    def perform_analysis(self):
        """Perform the merge and return the merged dataframe."""
        return self.merge_data()

    def visualize_results(self):
        """Display basic info about the merged dataset."""
        if self.merged_data is None:
            print("No merged data available. Run perform_analysis() first.")
            return
        
        print("\n=== Merged Data Info ===")
        print(f"Shape: {self.merged_data.shape}")
        print(f"\nColumns: {self.merged_data.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(self.merged_data.head())
        print(f"\nRUL distribution:")
        print(self.merged_data["RUL"].describe())