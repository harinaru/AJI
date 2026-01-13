import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu

def plot_uncertainty_box(df, channel, outdir):
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(
        x="group", y="uncertainty",
        data=df, showfliers=False, ax=ax
    )
    sns.stripplot(
        x="group", y="uncertainty",
        data=df, color="black", alpha=0.6, size=4, ax=ax
    )

    screen = df[df.group == "Screenable"]["uncertainty"]
    rare = df[df.group == "Rare"]["uncertainty"]

    u, p = mannwhitneyu(screen, rare, alternative="less")

    ymax = max(df.uncertainty)
    ax.text(0.5, ymax * 1.08,
            f"Mann–Whitney U, p = {p:.2e}",
            ha="center", fontsize=11, fontweight="bold")

    for i, g in enumerate(df.group.unique()):
        n = (df.group == g).sum()
        ax.text(i, ymax * 0.95, f"n={n}", ha="center")

    ax.set_title(f"{channel} — Subject-Level Uncertainty")
    ax.set_ylabel("Mahalanobis Distance")
    ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(outdir / f"uncertainty_box_{channel}.png", dpi=300)
    plt.close()


def plot_risk_coverage(control_uncertainties, channel, tau, outdir):
    taus = np.linspace(
        np.percentile(control_uncertainties, 50),
        np.percentile(control_uncertainties, 99.5),
        100
    )

    coverage, risk = [], []

    for t in taus:
        committed = control_uncertainties < t
        coverage.append(committed.mean() * 100)
        risk.append(control_uncertainties[committed].std())

    plt.figure(figsize=(8, 6))
    plt.plot(coverage, risk, lw=2)
    plt.scatter(
        (control_uncertainties < tau).mean() * 100,
        control_uncertainties[control_uncertainties < tau].std(),
        c="red", s=80, label=f"τ = {tau:.2f}"
    )

    plt.xlabel("Coverage (% Controls)")
    plt.ylabel("Risk (Std of Uncertainty)")
    plt.title(f"{channel} — Risk–Coverage Trade-off")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / f"risk_coverage_{channel}.png", dpi=300)
    plt.close()
