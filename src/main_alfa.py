import os
import glob
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
    desc_cols = list(desc.columns)
    if len(desc_cols) >= 2:
        c_name, c_desc = desc_cols[0], desc_cols[1]
        desc_map = {
            str(r[c_name]).strip(): str(r[c_desc]).strip().lower()
            for _, r in desc.iterrows()
            if pd.notna(r[c_name])
        }
    else:
        desc_map = {}

    # Feature families from your prototype
    uv_i_cols = [c for c in df.columns if c.startswith("uvi.") and c.endswith(".mean")]
    power_cols = [c for c in df.columns if c.startswith("power_per_lamp_kw.") and c.endswith(".median_of_mean")]
    runtime_uvr = [c for c in df.columns if c.startswith("uvr_usage_duration.")]
    pressure_diff_col = "pressure.filter_differential.mean" if "pressure.filter_differential.mean" in df.columns else None

    # Extra keyword-based numeric features as fallback
    keywords = ["alarm", "duration", "uvi", "warning", "pressure", "differential", "fault", "error", "anomaly"]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    keyword_cols = []
    for c in numeric_cols:
        txt = f"{c.lower()} {desc_map.get(c, '')}"
        if any(k in txt for k in keywords):
            keyword_cols.append(c)

    picked = list(dict.fromkeys(uv_i_cols + power_cols + runtime_uvr + keyword_cols))
    return picked, {
        "uv_i_cols": uv_i_cols,
        "power_cols": power_cols,
        "runtime_uvr": runtime_uvr,
        "pressure_diff_col": pressure_diff_col,
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
        # Always coerce to tz-aware UTC to avoid naive/aware mismatch
        df["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.sort_values(["vehicle_id", "time"]).copy()
        df["time_step"] = df.groupby("vehicle_id").cumcount() + 1

    # ---- degradation event construction (from your prototype) ----
    uv_i_cols = [c for c in df.columns if c.startswith("uvi.") and c.endswith(".mean")]
    power_cols = [c for c in df.columns if c.startswith("power_per_lamp_kw.") and c.endswith(".median_of_mean")]
    runtime_uvr = [c for c in df.columns if c.startswith("uvr_usage_duration.")]
    pressure_diff_col = "pressure.filter_differential.mean" if "pressure.filter_differential.mean" in df.columns else None

    for c in set(uv_i_cols + power_cols + runtime_uvr + ([pressure_diff_col] if pressure_diff_col else [])):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    LOW_Q, HIGH_Q = 0.10, 0.90

    def qflag(g: pd.DataFrame, cols: list, lower_q=None, upper_q=None):
        out = pd.DataFrame(index=g.index)
        for c in cols:
            if c not in g.columns:
                continue
            if lower_q is not None:
                out[f"{c}__lowq"] = g[c] <= g[c].quantile(lower_q)
            if upper_q is not None:
                out[f"{c}__highq"] = g[c] >= g[c].quantile(upper_q)
        return out

    uv_low_flags = []
    power_hi_flags = []
    for _, g in df.groupby("vehicle_id", sort=False):
        uv_low_flags.append(qflag(g, uv_i_cols, lower_q=LOW_Q))
        power_hi_flags.append(qflag(g, power_cols, upper_q=HIGH_Q))

    uv_low_flags = pd.concat(uv_low_flags, axis=0) if len(uv_low_flags) else pd.DataFrame(index=df.index)
    power_hi_flags = pd.concat(power_hi_flags, axis=0) if len(power_hi_flags) else pd.DataFrame(index=df.index)

    df["lamp_uv_low"] = uv_low_flags.any(axis=1) if not uv_low_flags.empty else False
    df["lamp_power_hi"] = power_hi_flags.any(axis=1) if not power_hi_flags.empty else False

    if runtime_uvr:
        for c in runtime_uvr:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["aging_uvr_sec"] = df[runtime_uvr].max(axis=1)
    else:
        df["aging_uvr_sec"] = df.groupby("vehicle_id").cumcount().astype(float)

    df["aging_present"] = df.groupby("vehicle_id")["aging_uvr_sec"].transform(
        lambda s: s >= s.quantile(0.50)
    )

    if pressure_diff_col is not None:
        df["filter_clogging"] = df.groupby("vehicle_id")[pressure_diff_col].transform(
            lambda s: s > (s.mean(skipna=True) + 2.0 * s.std(skipna=True))
        ).fillna(False)
    else:
        df["filter_clogging"] = False

    lamp_fault = "alwa_group.lamp_fault.duration"
    low_uvi = "alwa_group.low_uvi.duration"
    interrupted = "interrupted_by_alarm"

    df["lamp_fault_event"] = pd.to_numeric(df[lamp_fault], errors="coerce").fillna(0.0) > 0 if lamp_fault in df.columns else False
    df["low_uvi_event"] = pd.to_numeric(df[low_uvi], errors="coerce").fillna(0.0) > 0 if low_uvi in df.columns else False
    df["alarm_interrupt"] = df[interrupted].fillna(0).astype(bool) if interrupted in df.columns else False

    df["lamp_degradation"] = (df["lamp_uv_low"] | df["lamp_power_hi"]) & df["aging_present"]
    df["degradation_event"] = df[
        ["lamp_degradation", "filter_clogging", "low_uvi_event", "lamp_fault_event", "alarm_interrupt"]
    ].any(axis=1)

    # ---- pseudo-RUL (hours) = time to next degradation event ----
    if df["time"].notna().any():
        # keep same dtype/tz as df["time"] and avoid assignment dtype conflict
        df["event_time"] = df["time"].where(df["degradation_event"], pd.NaT)
        df["next_event_time"] = df.groupby("vehicle_id")["event_time"].transform(lambda s: s.bfill())
        rul_td = df["next_event_time"] - df["time"]
        df["RUL"] = rul_td.dt.total_seconds() / 3600.0
    else:
        # fallback without timestamps: step-based pseudo-RUL
        df["event_step"] = np.where(df["degradation_event"], df["time_step"], np.nan)
        df["next_event_step"] = df.groupby("vehicle_id")["event_step"].transform(lambda s: s.bfill())
        df["RUL"] = df["next_event_step"] - df["time_step"]

    # censored samples -> unlabeled
    df["RUL"] = pd.to_numeric(df["RUL"], errors="coerce")
    df.loc[~np.isfinite(df["RUL"]), "RUL"] = -1.0
    df.loc[df["RUL"] < 0, "RUL"] = -1.0

    # --- balance labeled/unlabeled by masking far-from-event labels ---
    # Keep low-RUL (near-event) labels, mask high-RUL rows to -1 until target ratio is reached.
    cur_unlabeled_ratio = float((df["RUL"] == -1).mean())
    if cur_unlabeled_ratio < target_unlabeled_ratio:
        need = int(np.ceil((target_unlabeled_ratio - cur_unlabeled_ratio) * len(df)))
        labeled_idx = df.index[df["RUL"] >= 0]
        if need > 0 and len(labeled_idx) > 0:
            # Prefer masking large RUL (far from event)
            candidates = df.loc[labeled_idx, "RUL"].sort_values(ascending=False).index.to_numpy()

            # small shuffle inside equal-ish regions for robustness
            rng = np.random.default_rng(random_state)
            head = rng.permutation(candidates[:need])
            to_mask = head[:need]

            df.loc[to_mask, "RUL"] = -1.0

    # required compatibility columns
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

    visualize_latent_representations(
        merged_data=merged_data,
        ae_encoder=ae_encoder,
        vae_encoder=vae_encoder,
        contrastive_encoder=cl_encoder,
        latent_dim=50,
        random_state=42,
        out_file="outputs/latent_spaces_pca2d_alfa.jpg",
    )


if __name__ == "__main__":
    main()