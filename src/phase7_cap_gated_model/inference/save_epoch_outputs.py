"""
PHASE 7 — STRICT MULTI-CHANNEL INFERENCE
Uses exact 45 Phase-5 selected features per channel.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from ..config import Config
from ..dataset import TemporalEEGDataset, collate_fn
from ..model import CAPGatedTemporalModel


# ==============================================================
# EXACT PHASE-5 SELECTED FEATURES (45 PER CHANNEL)
# ==============================================================

PHASE5_FEATURES = {

"C4-P4": [
"cap_present","cap_a_ratio","cap_A_duration","cap_A2_sequences","cap_A_start",
"cap_A3_sequences","cap_A1_ratio","cap_CAPsequences","cap_A2_time",
"cap_A3_ratio","cap_NREMtime","cap_CAPtime","cap_CAPrate","cap_A1_time",
"cap_A1_sequences","cap_A2_ratio","cap_A3_time","C4-P4_slope_sign_changes",
"global_std","C4-P4_peak_to_peak","C4-P4_var","C4-P4_num_troughs",
"global_mean","C4-P4_hjorth_complexity","C4-P4_perm_entropy",
"C4-P4_energy","C4-P4_skew","C4-P4_hjorth_activity","global_rms",
"C4-P4_zero_cross","epoch_end_s","cap_CAPdensity","epoch_start_s",
"C4-P4_min","C4-P4_hjorth_mobility","C4-P4_max","C4-P4_mad",
"C4-P4_hurst_exp","global_energy","C4-P4_std","C4-P4_iqr",
"C4-P4_approx_entropy","C4-P4_fractal_dimension","C4-P4_kurt","C4-P4_rms"
],

"F4-C4": [
"cap_present","cap_a_ratio","cap_A_duration","cap_A2_sequences","cap_A_start",
"cap_A3_sequences","cap_A1_ratio","cap_CAPsequences","cap_A2_time",
"cap_A3_ratio","cap_NREMtime","cap_CAPtime","cap_CAPrate","cap_A1_time",
"cap_A1_sequences","F4-C4_zero_cross","cap_A2_ratio","cap_A3_time",
"global_std","F4-C4_kurt","F4-C4_energy","F4-C4_mad","F4-C4_hurst_exp",
"global_mean","F4-C4_iqr","F4-C4_num_troughs","F4-C4_approx_entropy",
"F4-C4_rms","F4-C4_hjorth_complexity","global_rms","cap_CAPdensity",
"F4-C4_mean","epoch_end_s","epoch_start_s","F4-C4_slope_sign_changes",
"F4-C4_peak_to_peak","F4-C4_sample_entropy","F4-C4_perm_entropy",
"F4-C4_mean_abs_val","F4-C4_num_peaks","F4-C4_hjorth_activity",
"F4-C4_curve_length","global_energy","F4-C4_var","F4-C4_line_length"
],

"Fp2-F4": [
"cap_present","cap_a_ratio","cap_A_duration","FP2-F4_var",
"FP2-F4_hjorth_activity","FP2-F4_hjorth_complexity","FP2-F4_kurt",
"FP2-F4_rms","FP2-F4_mad","FP2-F4_line_length","FP2-F4_mean_abs_val",
"FP2-F4_skew","FP2-F4_energy","FP2-F4_std","FP2-F4_min","FP2-F4_iqr",
"cap_A_start","FP2-F4_num_peaks","FP2-F4_willison_amplitude",
"cap_CAPsequences","FP2-F4_hjorth_mobility","FP2-F4_curve_length",
"FP2-F4_zero_cross","cap_A1_sequences","FP2-F4_peak_to_peak",
"FP2-F4_fractal_dimension","cap_A2_sequences","cap_A3_sequences",
"FP2-F4_num_troughs","Fp2-F4_rms","cap_A1_time","FP2-F4_perm_entropy",
"FP2-F4_max","FP2-F4_sample_entropy","FP2-F4_slope_sign_changes",
"Fp2-F4_fractal_dimension","Fp2-F4_std","cap_A1_ratio","cap_A3_time",
"Fp2-F4_iqr","Fp2-F4_slope_sign_changes","Fp2-F4_zero_cross",
"cap_NREMtime","FP2-F4_approx_entropy","FP2-F4_mean"
],

"P4-O2": [
"cap_present","cap_a_ratio","cap_A_duration","cap_A2_sequences",
"cap_A_start","cap_A3_sequences","cap_A1_ratio","cap_CAPsequences",
"cap_A2_time","cap_A3_ratio","cap_NREMtime","cap_CAPtime",
"P4-O2_zero_cross","cap_A1_time","cap_CAPrate","P4-O2_num_peaks",
"cap_A1_sequences","cap_A2_ratio","P4-O2_fractal_dimension",
"cap_A3_time","global_std","P4-O2_skew","P4-O2_hjorth_complexity",
"P4-O2_hjorth_mobility","P4-O2_curve_length","P4-O2_iqr","P4-O2_rms",
"P4-O2_line_length","global_mean","global_rms","P4-O2_peak_to_peak",
"P4-O2_perm_entropy","P4-O2_kurt","cap_CAPdensity","P4-O2_mean_abs_val",
"P4-O2_mean","P4-O2_willison_amplitude","P4-O2_min","epoch_end_s",
"epoch_start_s","P4-O2_num_troughs","P4-O2_slope_sign_changes",
"global_energy","P4-O2_std","P4-O2_energy"
]

}


# ==============================================================
# INFERENCE LOOP
# ==============================================================

def run_epoch_inference():

    print("\n========== PHASE 7 MULTI-CHANNEL INFERENCE ==========\n")

    device = Config.DEVICE
    csv_files = list(Config.PHASE7_DATA_DIR.glob("*_clean.csv"))

    for csv_path in csv_files:

        channel = csv_path.stem.replace("_clean", "")
        print(f"\nProcessing: {channel}")

        if channel not in PHASE5_FEATURES:
            print("No feature list found. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        feature_names = PHASE5_FEATURES[channel]

        # Strict 45 feature enforcement
        df = df[[c for c in feature_names if c in df.columns] +
                [Config.SUBJECT_COL, Config.CAP_COL,
                 Config.EPOCH_COL, Config.LABEL_COL]]

        temp_path = csv_path.parent / f"TEMP_{channel}_45.csv"
        df.to_csv(temp_path, index=False)

        ds = TemporalEEGDataset(
            csv_path=temp_path,
            feature_names=feature_names
        )

        loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

        model = CAPGatedTemporalModel().to(device)
        ckpt_path = list(Config.CHECKPOINT_DIR.glob(f"{channel}*.pt"))[0]
        ckpt = torch.load(ckpt_path, map_location=device)

        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        results = []
        df_meta=pd.read_csv(temp_path)
        subject_ids = df_meta[Config.SUBJECT_COL].drop_duplicates().tolist()
        subject_idx = 0

        with torch.no_grad():
            for batch in loader:

                x = batch["features"].to(device)
                cap = batch["cap_flags"].to(device)
                mask = batch["mask"].to(device)

                recon, _ = model(x, cap, mask)
                error = ((recon - x) ** 2).mean(dim=2).cpu().numpy()[0]

                results.append({
                    "subject_id": subject_ids[subject_idx],
                    "mean_error": float(error.mean()),
                    "p95_error": float(np.percentile(error, 95)),
                    "max_error": float(error.max()),
                    "n_epochs": int(len(error))
                })

                subject_idx += 1

        out_df = pd.DataFrame(results)
        save_path = Config.OUTPUT_DIR / f"{channel}_epoch_reconstruction_summary.csv"
        out_df.to_csv(save_path, index=False)

        print(f"Saved → {save_path}")

    print("\n✅ Phase 7 multi-channel inference complete.\n")


if __name__ == "__main__":
    run_epoch_inference()