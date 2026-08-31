import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Normalized FFT: unitary DFT with 1/sqrt(N) scaling
def nfft(x):
    return fft(x, axis=-1, norm="ortho")

# Normalized IFFT: unitary inverse DFT with 1/sqrt(N) scaling
def nifft(X):
    return ifft(X, axis=-1, norm="ortho")

# Compute the achievable rate between input x and output y
def achievable_rate(x, y):
    # Zero-mean the signals
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)

    # Cross-correlation
    corr  = np.abs(np.mean(x * y.conj(), axis=0, keepdims=True))
    x_std = np.sqrt(np.mean(np.abs(x)**2, axis=0, keepdims=True))
    y_std = np.sqrt(np.mean(np.abs(y)**2, axis=0, keepdims=True))

    # Achievable rate from correlation: R = -log2(1 - rho^2)
    denom = x_std * y_std
    rho_k  = corr / denom
    rate_k = -np.log2(1 - rho_k**2)
    rate = np.mean(rate_k)
    return rate

# Uniform complex quantizer (separate I/Q quantization)
def quantizer(y, delta, b=10):
    """
    Parameters:
        y     : Input signal to be quantized
        delta : Quantization step size
        b     : Number of bits per real dimension
    """
    max_val = 2**(b - 1) - 1
    min_val = -2**(b - 1)

    re = y.real
    im = y.imag

    q_re = np.clip(np.round(re / delta), min_val, max_val) * delta
    q_im = np.clip(np.round(im / delta), min_val, max_val) * delta
    return q_re + 1j * q_im

# Compute quantization step size using power-based backoff
def delta_backoff(y, b=10, backoff_db=12):
    """
    Parameters:
        y          : Input signal used to estimate power
        b          : Number of bits per real dimension
        backoff_db : Backoff margin in dB (prevents clipping)
    """
    # Power per dimension (I/Q split)
    pow_per_dim = np.mean(np.abs(y)**2) / 2.0

    # Backoff margin
    A = np.sqrt(pow_per_dim * 10.0**(0.1 * backoff_db))

    # Step size based on symmetric b-bit quantizer
    delta = A / (2**(b - 1))
    return float(delta)









# Rate comparison plot for N-path vs conventional receivers
def plot_rate_comparison(path_4_csv, path_8_csv, file_path=None):
    df4        = pd.read_csv(path_4_csv)
    df8        = pd.read_csv(path_8_csv)
    snr_values = sorted(df4["snr"].unique())
    n_paths    = [4, 8]
    dfs        = {4: df4, 8: df8}
    n_cols     = len(snr_values)

    plt.rcParams.update({
        "font.family":       "serif",
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

            # Column heading on top row only
            if r == 0:
                ax.set_title(f"SNR = {snr} dB",
                             fontsize=8, fontweight="bold", pad=4)
            ax.set_xlim(subset["inr"].min(), subset["inr"].max())
            ax.grid(True)

            # x-label on bottom row, centre column only
            ax.set_xlabel("INR (dB)" if (is_bottom and c == is_mid) else "")

            # y-label on left column only, embedding row identifier
            if is_left:
                ax.set_ylabel(
                    f"Rate (bits/s/Hz)\n[$N={N}$]",
                    fontsize=7, labelpad=2)

    # ── Legend inside last subplot (bottom-right), top-right corner ──
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
        file_path = "../results/plots/rate_vs_inr.pdf"

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {file_path}")
    plt.show()

# Plot Power Spectral Density (PSD) of a signal
def plot_psd(signal, f_s, ax=None):
    if ax is None:
        ax = plt.gca()

    # Compute FFT and PSD
    Nfft = len(signal)
    f    = np.fft.fftfreq(Nfft, 1 / f_s)
    S    = np.abs(nfft(signal)) ** 2

    # Convert to dB scale
    S = np.maximum(S, 1e-5)
    S = 10 * np.log10(S)

    # Plot PSD
    ax.plot(np.fft.fftshift(f) / 1e6, np.fft.fftshift(S))
    ax.grid(True)