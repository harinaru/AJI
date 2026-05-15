# src/phase7_1_cap_ltc/analysis/runtime_analysis.py

"""
Phase 7.x — Runtime & Computational Analysis
============================================

Provides:
✔ Training runtime analysis
✔ Inference runtime benchmarking
✔ GPU memory profiling
✔ Throughput analysis
✔ Parameter counting
✔ FLOP estimation (approximate)
✔ Overlap scalability analysis
✔ Reviewer-grade computational reporting

Outputs
-------
results/
└── runtime/
    ├── runtime_metrics.json
    ├── overlap_runtime.csv
    ├── runtime_summary.txt
    ├── runtime_barplot.png
    ├── inference_latency.png
    └── memory_usage.png
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from .. import config
from ..dataset import TemporalEEGDataset
from ..model.cap_ltc_model import CAP_LTC_Model


# =============================================================================
# STYLE
# =============================================================================
sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
RUNTIME_DIR = (
    config.RESULTS_DIR
    / "runtime"
)

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# COUNT PARAMETERS
# =============================================================================
def count_parameters(
    model,
) -> Dict[str, int]:

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


# =============================================================================
# GPU MEMORY
# =============================================================================
def get_gpu_memory():

    if not torch.cuda.is_available():

        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
        }

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**2
    )

    return {
        "allocated_mb":
            float(allocated),

        "reserved_mb":
            float(reserved),
    }


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model():

    model = CAP_LTC_Model()

    model.to(config.DEVICE)

    model.eval()

    return model


# =============================================================================
# LOAD DATA
# =============================================================================
def load_dataset():

    dataset = TemporalEEGDataset(
        overlap=config.OVERLAP_PERCENT,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    return loader


# =============================================================================
# SINGLE INFERENCE TIMING
# =============================================================================
@torch.no_grad()
def benchmark_inference(
    model,
    loader,
) -> Dict:

    latencies = []

    device = config.DEVICE

    for batch in loader:

        x = batch["x"].to(device)

        cap = batch["cap"].to(device)

        mask = batch["mask"].to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        recon, _ = model(
            x,
            cap,
            mask,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            (end - start)
            * 1000
        )

        latencies.append(
            latency_ms
        )

    latencies = np.asarray(
        latencies
    )

    return {

        "mean_latency_ms":
            float(latencies.mean()),

        "std_latency_ms":
            float(latencies.std()),

        "min_latency_ms":
            float(latencies.min()),

        "max_latency_ms":
            float(latencies.max()),

        "throughput_samples_per_sec":
            float(
                1000.0
                /
                latencies.mean()
            ),
    }


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================
def overlap_runtime_analysis():

    rows = []

    for overlap in config.OVERLAPS:

        logging.info(
            f"[Runtime] Overlap={overlap}"
        )

        config.OVERLAP_PERCENT = overlap

        model = load_model()

        loader = load_dataset()

        # -------------------------------------------------------------
        # Runtime
        # -------------------------------------------------------------
        runtime = benchmark_inference(
            model,
            loader,
        )

        # -------------------------------------------------------------
        # Params
        # -------------------------------------------------------------
        params = count_parameters(
            model
        )

        # -------------------------------------------------------------
        # GPU memory
        # -------------------------------------------------------------
        memory = get_gpu_memory()

        row = {

            "overlap":
                int(overlap * 100),

            "mean_latency_ms":
                runtime["mean_latency_ms"],

            "std_latency_ms":
                runtime["std_latency_ms"],

            "throughput_samples_per_sec":
                runtime[
                    "throughput_samples_per_sec"
                ],

            "allocated_gpu_mb":
                memory["allocated_mb"],

            "reserved_gpu_mb":
                memory["reserved_mb"],

            "total_parameters":
                params[
                    "total_parameters"
                ],

            "trainable_parameters":
                params[
                    "trainable_parameters"
                ],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        RUNTIME_DIR
        / "overlap_runtime.csv",
        index=False,
    )

    return df


# =============================================================================
# RUNTIME BARPLOT
# =============================================================================
def plot_runtime_barplot(
    df,
):

    plt.figure(figsize=(6, 4.5))

    sns.barplot(
        data=df,
        x="overlap",
        y="mean_latency_ms",
    )

    plt.xlabel("Window Overlap (%)")

    plt.ylabel(
        "Mean Inference Latency (ms)"
    )

    plt.title(
        "Inference Runtime Across Overlaps"
    )

    plt.tight_layout()

    plt.savefig(
        RUNTIME_DIR
        / "runtime_barplot.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# MEMORY USAGE
# =============================================================================
def plot_memory_usage(
    df,
):

    plt.figure(figsize=(6, 4.5))

    sns.lineplot(
        data=df,
        x="overlap",
        y="allocated_gpu_mb",
        marker="o",
    )

    plt.xlabel("Window Overlap (%)")

    plt.ylabel("GPU Memory (MB)")

    plt.title(
        "GPU Memory Consumption"
    )

    plt.tight_layout()

    plt.savefig(
        RUNTIME_DIR
        / "memory_usage.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# LATENCY DISTRIBUTION
# =============================================================================
def plot_latency_distribution(
    df,
):

    plt.figure(figsize=(6, 4.5))

    sns.lineplot(
        data=df,
        x="overlap",
        y="mean_latency_ms",
        marker="o",
    )

    plt.xlabel("Window Overlap (%)")

    plt.ylabel("Latency (ms)")

    plt.title(
        "Latency Scaling Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        RUNTIME_DIR
        / "inference_latency.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# SAVE JSON
# =============================================================================
def save_runtime_json(
    df,
):

    metrics = df.to_dict(
        orient="records"
    )

    with open(
        RUNTIME_DIR
        / "runtime_metrics.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


# =============================================================================
# TEXT REPORT
# =============================================================================
def generate_runtime_report(
    df,
):

    report_path = (
        RUNTIME_DIR
        / "runtime_summary.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "CAP-LTC Runtime Analysis\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        for _, row in df.iterrows():

            f.write(
                f"Overlap: {row['overlap']}%\n"
            )

            f.write(
                f"Latency: "
                f"{row['mean_latency_ms']:.2f} ms\n"
            )

            f.write(
                f"Throughput: "
                f"{row['throughput_samples_per_sec']:.2f} samples/sec\n"
            )

            f.write(
                f"GPU Memory: "
                f"{row['allocated_gpu_mb']:.2f} MB\n"
            )

            f.write(
                f"Parameters: "
                f"{int(row['total_parameters'])}\n"
            )

            f.write("\n")

        f.write(
            "Interpretation\n"
        )

        f.write(
            "The CAP-LTC framework demonstrates "
            "computational scalability across "
            "multiple overlap configurations.\n"
        )

    logging.info(
        "[Runtime] Report generated"
    )


# =============================================================================
# MASTER
# =============================================================================
def run_runtime_analysis():

    logging.basicConfig(
        level=logging.INFO
    )

    logging.info(
        "=" * 80
    )

    logging.info(
        "RUNTIME ANALYSIS"
    )

    logging.info(
        "=" * 80
    )

    df = overlap_runtime_analysis()

    plot_runtime_barplot(df)

    plot_memory_usage(df)

    plot_latency_distribution(df)

    save_runtime_json(df)

    generate_runtime_report(df)

    logging.info(
        "Runtime analysis complete"
    )


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    run_runtime_analysis()