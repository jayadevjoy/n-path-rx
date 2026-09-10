import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fftfreq, fftshift

from .utilities import nfft

# Plot Power Spectral Density (PSD) of a signal
def plot_psd(signal, f_s, ax=None):
    if ax is None:
        ax = plt.gca()

    # Compute FFT and PSD
    Nfft = len(signal)
    f    = fftfreq(Nfft, 1 / f_s)
    S    = np.abs(nfft(signal)) ** 2

    # Convert to dB scale
    S = np.maximum(S, 1e-5)
    S = 10 * np.log10(S)

    # Plot PSD
    ax.plot(fftshift(f) / 1e6, fftshift(S))
    ax.grid(True)

# Plot achievable rate vs. INR for N-path and conventional receivers
def plot_rate(csv_path, file_path=None, n_rows=2):
    df         = pd.read_csv(csv_path)
    snr_values = sorted(df["snr"].unique())
    n_total    = len(snr_values)
    n_cols     = (n_total + n_rows - 1) // n_rows

    plt.rcParams.update({"font.family":       "serif",
                        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
                        "font.size":         8,
                        "axes.labelsize":    8,
                        "axes.titlesize":    8,
                        "xtick.labelsize":   7,
                        "ytick.labelsize":   7,
                        "lines.linewidth":   0.8,
                        "axes.linewidth":    0.6,
                        "xtick.major.width": 0.5,
                        "ytick.major.width": 0.5,
                        "grid.linewidth":    0.4,
                        "grid.alpha":        0.5,
                        "figure.dpi":        300})

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9, 2.5 * n_rows),
                             sharey=True, sharex=False,
                             constrained_layout=True,
                             gridspec_kw={"hspace": 0.04})
    axes = axes.flatten() if n_total > 1 else [axes]

    ls_npath = dict(color="C0", linewidth=0.8, linestyle="-",
                    marker="o", markersize=2.5, markeredgewidth=0.4)
    ls_conv  = dict(color="C1", linewidth=0.8, linestyle="--",
                    marker="s", markersize=2.5, markeredgewidth=0.4)

    for idx, snr in enumerate(snr_values):
        ax       = axes[idx]
        row      = idx // n_cols
        col      = idx % n_cols
        is_left  = (col == 0)
        is_bottom_row = (row == n_rows - 1) or (idx + n_cols >= n_total)

        subset = df[df["snr"] == snr].sort_values("inr")

        ax.plot(subset["inr"], subset["npath_rate"],
                label="$N$-Path RX", **ls_npath)
        ax.plot(subset["inr"], subset["conv_rate"],
                label="Conventional RX",   **ls_conv)

        ax.set_title(f"SNR = {snr} dB", fontsize=8, fontweight="bold", pad=4)
        ax.set_xlim(subset["inr"].min(), subset["inr"].max())
        ax.grid(True)

        is_mid_bottom = is_bottom_row and (col == n_cols // 2)
        ax.set_xlabel("INR (dB)" if is_mid_bottom else "")
        fig.supylabel("Rate (bits/s/Hz)", fontsize=7)

    for j in range(n_total, len(axes)):
        axes[j].set_visible(False)

    last_ax = axes[n_cols - 1]
    handles, labels = last_ax.get_legend_handles_labels()
    last_ax.legend(
        handles, labels,
        loc="upper right",
        fontsize=6,
        framealpha=0.8,
        edgecolor="0.5",
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4)

    fig.align_ylabels(axes[:n_cols])

    if file_path is None:
        file_path = "results/plots/N_path_rate_vs_inr.pdf"

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {file_path}")
    plt.show()

# Rate comparison plot for 4-path & 8-path vs conventional receivers
def plot_rate_comparison(path_4_csv, path_8_csv, file_path=None):
    df4        = pd.read_csv(path_4_csv)
    df8        = pd.read_csv(path_8_csv)
    snr_values = sorted(df4["snr"].unique())
    n_paths    = [4, 8]
    dfs        = {4: df4, 8: df8}
    n_cols     = len(snr_values)

    plt.rcParams.update({"font.family":       "serif",
                        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
                        "font.size":         8,
                        "axes.labelsize":    8,
                        "axes.titlesize":    8,
                        "xtick.labelsize":   7,
                        "ytick.labelsize":   7,
                        "lines.linewidth":   0.8,
                        "axes.linewidth":    0.6,
                        "xtick.major.width": 0.5,
                        "ytick.major.width": 0.5,
                        "grid.linewidth":    0.4,
                        "grid.alpha":        0.5,
                        "figure.dpi":        300})

    fig, axes = plt.subplots(2, n_cols, figsize=(9, 4.65),
                             sharey="row", sharex="col",
                             constrained_layout=True)

    ls_npath = dict(color="C0", linewidth=0.8, linestyle="-",
                    marker="o", markersize=2.5, markeredgewidth=0.4)
    ls_conv  = dict(color="C1", linewidth=0.8, linestyle="--",
                    marker="s", markersize=2.5, markeredgewidth=0.4)

    for r, N in enumerate(n_paths):
        df        = dfs[N]
        is_bottom = (r == len(n_paths) - 1)
        is_mid    = (n_cols // 2)

        for c, snr in enumerate(snr_values):
            ax      = axes[r][c]
            is_left = (c == 0)
            subset  = df[df["snr"] == snr].sort_values("inr")

            ax.plot(subset["inr"], subset["npath_rate"],
                    label=f"$N$-Path RX", **ls_npath)
            ax.plot(subset["inr"], subset["conv_rate"],
                    label="Conventional RX",  **ls_conv)
            
            if r == 0:
                ax.set_title(f"SNR = {snr} dB",
                             fontsize=8, fontweight="bold", pad=4)
            ax.set_xlim(subset["inr"].min(), subset["inr"].max())
            ax.grid(True)

            ax.set_xlabel("INR (dB)" if (is_bottom and c == is_mid) else "")

            if is_left:
                ax.set_ylabel(
                    f"Rate (bits/s/Hz)\n[$N={N}$]",
                    fontsize=7, labelpad=2)

    handles, labels = axes[0][-1].get_legend_handles_labels()
    axes[0][-1].legend(
        handles, labels,
        loc="upper right",
        fontsize=6,
        framealpha=0.8,
        edgecolor="0.5",
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4)

    fig.align_ylabels(axes[:, 0])

    if file_path is None:
        file_path = "results/plots/rate_vs_inr.pdf"

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {file_path}")
    plt.show()