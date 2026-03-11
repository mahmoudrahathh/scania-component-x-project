import os
import pandas as pd
from typing import Optional

class DataLoader:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(base, "data")
        self.data_dir = data_dir

    def load_operational_readouts(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "train_operational_readouts.csv")
        return pd.read_csv(path)

    def load_specifications(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "train_specifications.csv")
        return pd.read_csv(path)

    def load_tte(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "train_tte.csv")
        return pd.read_csv(path)