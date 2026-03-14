import os
import glob
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from sklearn.model_selection import train_test_split

from train_vae_alfa import train_vae_alfa_and_predict_rul
from train_ae_alfa import train_ae_alfa_and_predict_rul
from train_plain_baseline_alfa import train_plain_baseline_alfa_mlp
from train_contrastive_alfa import train_contrastive_alfa_and_predict_rul
from visualize_representations import visualize_latent_representations


def _find_col(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _read_alfa(alfa_dir: str):
    parquet_files = sorted(glob.glob(os.path.join(alfa_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {alfa_dir}")

    xlsx_path = os.path.join(alfa_dir, "column_descriptions.xlsx")
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Missing file: {xlsx_path}")

    df = pd.read_parquet(parquet_files[0])
    desc = pd.read_excel(xlsx_path)

    return df, desc, parquet_files[0], xlsx_path


def _pick_degradation_features(df: pd.DataFrame, desc: pd.DataFrame):
    # requested ALWA feature group
    exact_cols = [
        "alwa_group.bypass.duration",
        "alwa_group.bypass.volume",
        "alwa_group.high_flow.duration",
        "alwa_group.high_flow.volume",
        "alwa_group.lamp_fault.duration",
        "alwa_group.lamp_fault.volume",
        "alwa_group.low_uvi.duration",
        "alwa_group.low_uvi.volume",
        "alwa_group.no_cip.duration",
        "alwa_group.no_cip.volume",
    ]

    exact_present = [c for c in exact_cols if c in df.columns]

    # generic pattern: alwa.[code].duration
    alwa_code_duration_cols = [
        c for c in df.columns
        if re.match(r"^alwa\.[^.]+\.duration$", c) is not None
    ]

    picked = list(dict.fromkeys(alwa_code_duration_cols + exact_present))

    return picked, {
        "alwa_code_duration_cols": alwa_code_duration_cols,
        "alwa_group_exact_cols": exact_present,
    }


def _build_pseudo_rul_dataframe(
    raw: pd.DataFrame,
    degradation_features,
    device_col=None,
    time_col=None,
    target_unlabeled_ratio: float = 0.35,
    random_state: int = 42,
):
    df = raw.copy()

    # device id
    if device_col is None:
        device_col = _find_col(df, ["machine_id", "device_id", "unit_id", "asset_id", "serial_number"])
    if device_col is None:
        df["vehicle_id"] = "device_0"
    else:
        df["vehicle_id"] = df[device_col].astype(str)

    # time column
    if time_col is None:
        time_col = _find_col(df, ["time", "timestamp", "datetime", "date", "event_time", "ts"])
    if time_col is None:
        df["time"] = pd.NaT
        df["time_step"] = df.groupby("vehicle_id").cumcount() + 1
    else:
        df["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.sort_values(["vehicle_id", "time"]).copy()
        df["time_step"] = df.groupby("vehicle_id").cumcount() + 1

    degradation_features = [c for c in degradation_features if c in df.columns]
    if not degradation_features:
        raise ValueError("No requested ALWA degradation features were found in the ALFA dataframe.")

    for c in degradation_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # stricter event definition: persistent activation over consecutive rows
    active_matrix = (df[degradation_features] > 0).astype(int)
    df["raw_event_score"] = active_matrix.sum(axis=1)
    df["raw_event"] = df["raw_event_score"] >= 1

    min_consecutive_steps = 3
    rolling_hits = (
        df.groupby("vehicle_id")["raw_event"]
        .transform(lambda s: s.astype(int).rolling(min_consecutive_steps, min_periods=min_consecutive_steps).sum())
        .fillna(0)
    )

    df["degradation_event"] = rolling_hits >= min_consecutive_steps

    prev_event = df.groupby("vehicle_id")["degradation_event"].shift(1, fill_value=False)
    df["degradation_event_start"] = df["degradation_event"] & (~prev_event)

    # RUL = time/steps to next degradation episode start
    if df["time"].notna().any():
        df["event_time"] = df["time"].where(df["degradation_event_start"], pd.NaT)
        df["next_event_time"] = df.groupby("vehicle_id")["event_time"].transform(lambda s: s.bfill())
        rul_td = df["next_event_time"] - df["time"]
        df["RUL"] = rul_td.dt.total_seconds() / 3600.0
    else:
        df["event_step"] = np.where(df["degradation_event_start"], df["time_step"], np.nan)
        df["next_event_step"] = df.groupby("vehicle_id")["event_step"].transform(lambda s: s.bfill())
        df["RUL"] = df["next_event_step"] - df["time_step"]

    # rows already in degradation episode get RUL = 0
    df.loc[df["degradation_event"], "RUL"] = 0.0

    # censored samples -> unlabeled
    df["RUL"] = pd.to_numeric(df["RUL"], errors="coerce")
    df.loc[~np.isfinite(df["RUL"]), "RUL"] = -1.0
    df.loc[df["RUL"] < 0, "RUL"] = -1.0

    # keep supervised / unsupervised ratio close to target
    cur_unlabeled_ratio = float((df["RUL"] == -1).mean())
    if cur_unlabeled_ratio < target_unlabeled_ratio:
        need = int(np.ceil((target_unlabeled_ratio - cur_unlabeled_ratio) * len(df)))
        labeled_idx = df.index[df["RUL"] >= 0]
        if need > 0 and len(labeled_idx) > 0:
            candidates = df.loc[labeled_idx, "RUL"].sort_values(ascending=False).index.to_numpy()
            rng = np.random.default_rng(random_state)
            to_mask = rng.permutation(candidates[:need])[:need]
            df.loc[to_mask, "RUL"] = -1.0

    df["length_of_study_time_step"] = df.groupby("vehicle_id")["time_step"].transform("max").astype(int)
    df["in_study_repair"] = 0

    mandatory = ["vehicle_id", "time_step", "length_of_study_time_step", "in_study_repair", "RUL"]
    ordered = mandatory + [c for c in df.columns if c not in mandatory]
    return df[ordered]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alfa_dir = os.path.join(project_root, "data", "Alfa")

    raw, desc, parquet_path, xlsx_path = _read_alfa(alfa_dir)

    # Save column names to txt file
    cols_out = os.path.join(project_root, "outputs", "alfa_columns.txt")
    os.makedirs(os.path.dirname(cols_out), exist_ok=True)
    with open(cols_out, "w") as f:
        for col in raw.columns:
            f.write(f"{col}\n")
    print(f"[ALFA] Column list saved to: {cols_out}")

    print(f"[ALFA] Data: {parquet_path}")
    print(f"[ALFA] Descriptions: {xlsx_path}")
    print("\n=== Raw ALFA head ===")
    print(raw.head())

    degr_features, _ = _pick_degradation_features(raw, desc)
    print(f"\n[ALFA] Selected degradation features ({len(degr_features)}):")
    print(degr_features[:20])

    merged_data = _build_pseudo_rul_dataframe(
        raw,
        degr_features,
        target_unlabeled_ratio=0.35,  # try 0.30–0.45
        random_state=42,
    )

    rul_negative_one = (merged_data["RUL"] == -1).sum()
    rul_non_negative = (merged_data["RUL"] >= 0).sum()
    print("\n=== Pseudo-RUL Distribution (ALFA) ===")
    print(f"Samples with RUL = -1: {rul_negative_one}")
    print(f"Samples with RUL >= 0: {rul_non_negative}")
    print(f"Total samples: {len(merged_data)}")
    print(f"Merged data size: {merged_data.shape}")
    print(merged_data["RUL"].describe())
    print("\n[ALFA] Fraction with RUL = 0:", float((merged_data["RUL"] == 0).mean()))
    print("[ALFA] Fraction unlabeled:", float((merged_data["RUL"] == -1).mean()))

    labeled_mask = merged_data["RUL"] >= 0
    labeled_indices = np.arange(int(labeled_mask.sum()))
    if len(labeled_indices) < 100:
        raise ValueError("Too few labeled samples after pseudo-labeling. Adjust threshold/window.")

    train_idx, test_idx = train_test_split(labeled_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    print(f"\n[Split] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    cl_encoder, cl_mlp, cl_metrics = train_contrastive_alfa_and_predict_rul(
        merged_data, train_idx, val_idx, test_idx, latent_dim=32, epochs_representation=40, epochs_finetune=30
    )
    autoencoder, ae_encoder, ae_mlp, ae_metrics = train_ae_alfa_and_predict_rul(
        merged_data, train_idx, val_idx, test_idx, latent_dim=32, epochs_representation=30, epochs_finetune=30
    )
    vae, vae_encoder, vae_mlp, vae_metrics = train_vae_alfa_and_predict_rul(
        merged_data, train_idx, val_idx, test_idx, latent_dim=32, epochs_representation=30, epochs_finetune=30
    )
    _, plain_baseline_mae = train_plain_baseline_alfa_mlp(
        merged_data, train_idx, val_idx, test_idx, epochs=30
    )

    labeled_rul = merged_data.loc[labeled_mask, "RUL"].to_numpy(dtype=float)
    mean_val = float(np.mean(labeled_rul[train_idx]))
    meanValuePredictor = float(np.mean(np.abs(labeled_rul[test_idx] - mean_val)))

    print("\n=== Final Performance Comparison (ALFA pseudo-RUL) ===")
    print(f"Contrastive + MLP MAE:       {cl_metrics['contrastive_mlp_mae']:.4f}")
    print(f"AE + MLP MAE:                {ae_metrics['ae_mlp_mae']:.4f}")
    print(f"VAE + MLP MAE:               {vae_metrics['vae_mlp_mae']:.4f}")
    print(f"Standalone Baseline MLP MAE: {plain_baseline_mae:.4f}")
    print(f"MeanValuePredictor MAE:      {meanValuePredictor:.4f}")

    # visualize_latent_representations(
    #     merged_data=merged_data,
    #     ae_encoder=ae_encoder,
    #     vae_encoder=vae_encoder,
    #     contrastive_encoder=cl_encoder,
    #     latent_dim=50,
    #     random_state=42,
    #     out_file="outputs/latent_spaces_pca2d_alfa.jpg",
    # )


if __name__ == "__main__":
    main()