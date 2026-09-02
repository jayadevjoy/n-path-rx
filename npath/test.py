import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fftfreq, fftshift

import importlib
import utilities
import npathrx

importlib.reload(utilities)
importlib.reload(npathrx)

from utilities import nfft, plot_psd, achievable_rate
from npathrx import SignalGenerator, NPathRX


class NPathRXTest:
    """
    End-to-end testing for evaluating N-path receiver performance.
    Handles signal generation, receiver processing, achievable rate computation,
    and visualization of results and spectra.
    """

    def __init__(self,
                 f_p1=8e9,        # Carrier frequency of desired signal [Hz]
                 f_p2=13e9,       # Carrier frequency of interferer [Hz]
                 f_s=32e9,        # Simulation sampling rate [Hz]
                 bw_ofdm=100e6,   # OFDM signal bandwidth [Hz]
                 Nfft=512,        # Number of OFDM subcarriers
                 Nsamp=100,       # Number of signal realizations (batch size)
                 Nrep=3,          # Number of OFDM symbol repetitions per realization
                 f_lo=8e9,        # LO frequency for N-path switching [Hz]
                 f_adc=400e6,     # ADC sampling rate [Hz]
                 N=4,             # Number of paths in N-path receiver
                 f_c1=400e6,      # Cut-off frequency of N-path filter [Hz]
                 f_c2=400e6,      # Cut-off frequency of post-amplifier LPF [Hz]
                 noise_db=0,      # Noise power [dBW]
                 sat_db=40,       # Amplifier saturation power above noise floor [dB]
                 baseband=True,   # If True, N-path filter outputs baseband signal
                 symmetry=True):  # If True, use symmetric OFDM subcarrier allocation around DC

        # Parameter initialization
        self.f_p1      = f_p1
        self.f_p2      = f_p2
        self.f_s       = f_s
        self.bw_ofdm   = bw_ofdm
        self.Nfft      = Nfft
        self.Nsamp     = Nsamp
        self.Nrep      = Nrep
        self.f_lo      = f_lo
        self.f_adc     = f_adc
        self.N         = N
        self.f_c1      = f_c1
        self.f_c2      = f_c2
        self.noise_db = noise_db
        self.sat_db    = sat_db
        self.baseband  = baseband
        self.symmetry  = symmetry

        # Number of ADC samples per OFDM symbol
        self.ofdm_adc = int(f_adc * Nfft / bw_ofdm)

        # Instantiate signal generator and N-path receiver
        self.sig_gen  = SignalGenerator(f_p1=f_p1, f_p2=f_p2, f_s=f_s, 
                                        bw_ofdm=bw_ofdm, Nfft=Nfft, Nsamp=Nsamp, 
                                        Nrep=Nrep, symmetry=symmetry)
        self.npath_rx = NPathRX(f_lo=f_lo, f_adc=f_adc, N=N, f_s=f_s, 
                                f_c1=f_c1, f_c2=f_c2, 
                                noise_db=noise_db, sat_db=sat_db, 
                                baseband=baseband)
        
        # Store results
        self.results = []

    def run(self, SNR_db, INR_db, file_path=None):
        """
        Sweep over SNR and INR values and compute achievable rates for both receivers.

        Parameters
        ----------
        SNR_db : array-like - SNR values to sweep over [dB]
        INR_db : array-like - INR values to sweep over [dB]
        """

        # Initialize rate matrices: rows = SNR index, cols = INR index
        npath_rates = np.zeros((len(SNR_db), len(INR_db)))
        conv_rates  = np.zeros((len(SNR_db), len(INR_db)))

        for i, snr_db in enumerate(SNR_db):
            for j, inr_db in enumerate(INR_db):
                print(f"[sim] SNR={snr_db} dB  INR={inr_db} dB")

                # Generate passband signal and reference baseband signal
                t, x_b1, x_b2, x_p1, x_p2 = self.sig_gen.generate(snr_db=snr_db, 
                                                                  inr_db=inr_db)

                # Pass through N-path and conventional receivers
                y_adc_quant, y_conv_adc_quant = self.npath_rx.run(x_p1=x_p1, x_p2=x_p2, 
                                                      f_p1=self.f_p1, f_p2=self.f_p2)

                # Downsample reference baseband signal to ADC rate
                x_adc = x_b1[:, ::self.npath_rx.adc]

                # Trim to last OFDM symbol
                x_mat      = x_adc[:, -self.ofdm_adc:].copy()
                y_mat      = y_adc_quant[:, -self.ofdm_adc:].copy()
                y_conv_mat = y_conv_adc_quant[:, -self.ofdm_adc:].copy()
                
                # Determine the desired frequency band based on the OFDM symmetry
                f = fftshift(fftfreq(self.ofdm_adc, 1 / self.f_adc))
                if self.symmetry:
                    f_mask = np.abs(f) <= (self.bw_ofdm / 2)      
                else:
                    f_mask = (f >= 0) and (f < self.bw_ofdm)

                # Convert to frequency domain and extract the desired components
                X_mat = fftshift(nfft(x_mat))[:,f_mask]
                Y_mat = fftshift(nfft(y_mat))[:,f_mask]
                Y_conv_mat = fftshift(nfft(y_conv_mat))[:,f_mask]

                # Compute achievable rates (in frequency domain)
                npath_rates[i, j] = achievable_rate(X_mat, Y_mat)
                conv_rates[i, j]  = achievable_rate(X_mat, Y_conv_mat)

                print(f"  Rate (bits/s/Hz) - N-path: {npath_rates[i,j]:.2f} | Conv: {conv_rates[i,j]:.2f}")

                self.results.append({
                    'snr': snr_db,
                    'inr': inr_db,
                    'npath_rate': npath_rates[i, j],
                    'conv_rate': conv_rates[i, j]})
        
        if file_path is None:
            file_path = f"../results/data/{self.N}_path_data.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

    def plot_spectrum(self, snr_db=20, inr_db=20, file_path=None):
        """
        Plot time-domain waveforms and PSDs for the reference, N-path, and
        conventional receiver outputs at a single (SNR, INR) operating point.

        Parameters
        ----------
        snr_db : float - Desired signal SNR [dB]
        inr_db : float - Interferer INR [dB]
        file_path : str - Path to save the plot (optional)
        """

        # Instantiate signal generator and N-path receiver
        sig_gen_temp  = SignalGenerator(f_p1=self.f_p1, f_p2=self.f_p2, f_s=self.f_s, 
                                        bw_ofdm=self.bw_ofdm, Nfft=self.Nfft, Nsamp=1, 
                                        Nrep=self.Nrep, symmetry=self.symmetry)
        npath_rx_temp = NPathRX(f_lo=self.f_lo, f_adc=self.f_adc, N=self.N, f_s=self.f_s, 
                                f_c1=self.f_c1, f_c2=self.f_c2, 
                                noise_db=self.noise_db, sat_db=self.sat_db, 
                                baseband=self.baseband)

        # Generate signals and process through both receivers
        t, x_b1, x_b2, x_p1, x_p2 = sig_gen_temp.generate(snr_db=snr_db, 
                                                          inr_db=inr_db)
        y_adc_quant, y_conv_adc_quant = npath_rx_temp.run(x_p1=x_p1, x_p2=x_p2, 
                                                           f_p1=self.f_p1, f_p2=self.f_p2)

        # Downsample reference to ADC rate
        x_adc = x_b1[:, ::npath_rx_temp.adc]
        t_adc = t[::npath_rx_temp.adc]

        # Trim all signals to last OFDM symbol
        t_mat = t_adc[-self.ofdm_adc:].copy()
        x_mat = x_adc[:, -self.ofdm_adc:].copy()
        y_mat = y_adc_quant[:, -self.ofdm_adc:].copy()
        y_conv_mat = y_conv_adc_quant[:, -self.ofdm_adc:].copy()

        # ── IEEE style ──────────────────────────────────────────────────
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
            "figure.dpi":        300,
        })

        # Full page width for IEEE double-column paper
        fig, axes = plt.subplots(3, 2, figsize=(9, 5), constrained_layout=True)

        # ── Column headings (set as titles on top-row axes only) ─────────────
        axes[0][0].set_title("Time-Domain Waveform (Real Part)", fontsize=8, fontweight="bold", pad=4)
        axes[0][1].set_title("Power Spectral Density (dB)",      fontsize=8, fontweight="bold", pad=4)

        # ── Data and row labels ───────────────────────────────────────────────
        rows = [
            ("Baseband Input",        x_mat[0, :],      x_mat[0, :]),
            (f"{self.N}-Path RX Chain",     y_mat[0, :],      y_mat[0, :]),
            ("Conventional RX Chain", y_conv_mat[0, :], y_conv_mat[0, :]),
        ]

        for row_idx, (row_label, sig_time, sig_freq) in enumerate(rows):
            ax_t = axes[row_idx, 0]
            ax_f = axes[row_idx, 1]
            is_bottom = (row_idx == 2)

            # Left: real part — row label baked into ylabel
            ax_t.plot(t_mat / 1e-6, np.real(sig_time), color="C0")
            ax_t.set_ylabel(f"{row_label}", fontsize=7, labelpad=2)
            ax_t.grid(True)
            ax_t.tick_params(labelbottom=is_bottom)
            ax_t.set_xlabel("Time (μs)" if is_bottom else "")

            # Right: PSD
            plot_psd(sig_freq, self.f_adc, ax=ax_f)
            ax_f.tick_params(labelbottom=is_bottom)
            ax_f.set_xlabel("Frequency (MHz)" if is_bottom else "")

        fig.align_ylabels([axes[r][0] for r in range(3)])  # left column
        fig.align_ylabels([axes[r][1] for r in range(3)])  # right column

        if file_path is None:
            file_path = f"../results/plots/{self.N}_path_{snr_db}_snr_{inr_db}_inr_spectrum.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {file_path}")
        plt.show()

    def plot_rate(self, csv_path, file_path=None):
        df         = pd.read_csv(csv_path)
        snr_values = sorted(df["snr"].unique())
        n_cols     = len(snr_values)

        # ── IEEE style (mirrors plot_spectrum) ───────────────────────────
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
            "figure.dpi":        300,
        })

        fig, axes = plt.subplots(1, n_cols, figsize=(9, 2.5),
                                sharey=True, constrained_layout=True)
        if n_cols == 1:
            axes = [axes]

        # ── Line styles ───────────────────────────────────────────────────
        ls_npath = dict(color="C0", linewidth=0.8, linestyle="-",
                        marker="o", markersize=2.5, markeredgewidth=0.4)
        ls_conv  = dict(color="C1", linewidth=0.8, linestyle="--",
                        marker="s", markersize=2.5, markeredgewidth=0.4)

        for c, (ax, snr) in enumerate(zip(axes, snr_values)):
            is_left  = (c == 0)
            is_mid   = (c == n_cols // 2)
            subset   = df[df["snr"] == snr].sort_values("inr")

            ax.plot(subset["inr"], subset["npath_rate"],
                    label=f"$N={self.N}$-Path RX", **ls_npath)
            ax.plot(subset["inr"], subset["conv_rate"],
                    label="Conventional RX",        **ls_conv)

            # Column heading on every subplot (mirrors plot_spectrum top-row titles)
            ax.set_title(f"SNR = {snr} dB", fontsize=8, fontweight="bold", pad=4)

            ax.set_xlim(subset["inr"].min(), subset["inr"].max())
            ax.grid(True)

            ax.set_xlabel("INR (dB)" if is_mid else "")
            if is_left:
                ax.set_ylabel("Rate (bits/s/Hz)", fontsize=7, labelpad=2)

        # ── Legend inside last subplot, top-right ────────────────────────
        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(
            handles, labels,
            loc="upper right",
            fontsize=6,
            framealpha=0.8,
            edgecolor="0.5",
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=1.5,
            handletextpad=0.4)

        fig.align_ylabels(axes)

        if file_path is None:
            file_path = f"../results/plots/{self.N}_path_rate_vs_inr.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {file_path}")
        plt.show()