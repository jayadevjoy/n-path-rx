import numpy as np
from scipy.fft import ifftshift

import importlib
import utilities
importlib.reload(utilities)

from utilities import nifft, quantizer, delta_backoff


class SignalGenerator:
    """
    Generates OFDM baseband signals, tiles them for multiple repetitions,
    and up-converts to passband with a desired signal and an interferer.
    """

    def __init__(self,
                 f_p1=8e9,       # Carrier frequency of desired signal [Hz]
                 f_p2=13e9,      # Carrier frequency of interferer [Hz]
                 f_s=32e9,       # Simulation sampling rate [Hz]
                 bw_ofdm=100e6,  # OFDM bandwidth [Hz]
                 Nfft=512,       # Number of OFDM subcarriers
                 Nsamp=50,       # Number of signal realizations
                 Nrep=3,         # Number of OFDM symbol repetitions per realization
                 symmetry=True): # If True, use symmetric OFDM subcarrier allocation around DC        

        self.f_p1    = f_p1
        self.f_p2    = f_p2
        self.f_s     = f_s
        self.bw_ofdm = bw_ofdm
        self.Nfft    = Nfft
        self.Nsamp   = Nsamp
        self.Nrep    = Nrep
        self.symmetry = symmetry

        self.N = int(self.Nfft * self.f_s / self.bw_ofdm)      # Total samples per OFDM symbol at f_s
        assert f_s % (bw_ofdm / Nfft) == 0, "Sampling rate must be an integer multiple of subcarrier spacing"

    def generate(self, snr_db=20, inr_db=20, noise_db=0):
        """
        Generate a passband signal combining a desired OFDM signal and an interferer.

        Parameters
        ----------
        snr_db   : Desired signal power relative to noise floor [dB]
        inr_db   : Interferer power relative to noise floor [dB]
        noise_db : Noise floor reference power [dB]

        Returns
        -------
        t    : Time vector at simulation rate [s]
        x_b1 : Desired baseband signal (tiled, complex)
        x_b2 : Interferer baseband signal (tiled, complex)
        x_p1 : Desired passband signal
        x_p2 : Interferer passband signal
        """

        # Compute absolute power levels
        sig_pow_db = snr_db  + noise_db   # Desired signal absolute power [dB]
        int_pow_db = inr_db  + noise_db   # Interferer absolute power [dB]

        # Generate random OFDM baseband signals at their respective power levels
        # Output shape: (Nsamp, Nrep * Nfft)
        x_b1 = self.rand_ofdm(pow_db=sig_pow_db)
        x_b2 = self.rand_ofdm(pow_db=int_pow_db)

        # Time vector
        t = np.arange(x_b1.shape[1]) / self.f_s

        # Up-convert each baseband signal to its passband carrier and sum
        x_p1 = np.real(x_b1 * np.exp(1j * 2 * np.pi * self.f_p1 * t))
        x_p2 = np.real(x_b2 * np.exp(1j * 2 * np.pi * self.f_p2 * t))

        return t, x_b1, x_b2, x_p1, x_p2

    def rand_ofdm(self, pow_db):
        # Power per subcarrier
        amp = np.sqrt(10**(pow_db/10))

        # Random QPSK symbols on Nfft subcarriers
        X = amp * self.rand_qpsk(shape=(self.Nsamp, self.Nfft))

        # Zero-pad to length N (oversampling)
        if self.symmetry:
            left = (self.N - self.Nfft) // 2
            right = self.N - self.Nfft - left
            X_padded = np.concatenate([np.zeros((self.Nsamp, left)),
                                    X,
                                    np.zeros((self.Nsamp, right))], axis=1)
            X_padded = ifftshift(X_padded, axes=1)
        else:
            X_padded = np.concatenate([X, np.zeros((self.Nsamp, self.N - self.Nfft))], axis=1)

        # IFFT and scale to maintain power
        x_b = np.sqrt(self.N / self.Nfft) * nifft(X_padded)
        x_b = np.tile(x_b, self.Nrep)
        return x_b

    def rand_qpsk(self, shape):
        bits_1 = np.random.randint(0, 2, shape)
        bits_2 = np.random.randint(0, 2, shape)
        sym = (1 - 2*bits_1) + 1j*(1 - 2*bits_2)
        sym /= np.sqrt(2)
        return np.atleast_2d(sym)


class NPathRX:
    """
    Simulates an N-path receiver vs a conventional receiver, both followed
    by a soft-saturation (tanh) nonlinearity and a 2nd-order RC low-pass filter.
    Returns ADC-rate signals for downstream rate computation.
    """

    def __init__(self,
                 f_lo=8e9,       # LO frequency for N-path switching [Hz]
                 f_adc=400e6,    # ADC sampling rate [Hz]
                 N=4,            # Number of paths in N-path receiver
                 f_s=32e9,       # Simulation sampling rate [Hz]
                 f_c1=400e6,     # Cut-off frequency of N-path RC filter [Hz]
                 f_c2=400e6,     # Cut-off frequency of post-amplifier LPF [Hz]
                 noise_db=0,     # Noise power [dBW]
                 sat_db=40,      # Saturation power relative to noise floor [dB]
                 baseband=True): # If True, N-path filter outputs baseband signal

        self.f_lo     = f_lo
        self.f_adc    = f_adc
        self.N        = N
        self.f_s      = f_s
        self.f_c1     = f_c1
        self.f_c2     = f_c2
        self.baseband = baseband

        self.noise = 10**(noise_db / 10)
        self.Psat_sqrt = np.sqrt(10**(sat_db / 10) * self.noise)
        self.adc = int(f_s / f_adc)

        assert f_s % f_adc == 0,         "Ensure integer downsampling factor"
        assert f_s % (N * f_lo) == 0,    "Sampling rate must be integer multiple of N * f_lo"

        self.R1 = 1                                            # N-path RC resistance [Ohm]
        self.C1 = 1 / (2 * np.pi * self.N * self.R1 * f_c1)    # N-path RC capacitance [F]
        self.R2 = 1                                            # Post-amplifier RC resistance [Ohm]   
        self.C2 = 1 / (2 * np.pi * self.R2 * f_c2)             # Post-amplifier RC capacitance [F]

    def run(self, x_p1, x_p2, f_p1, f_p2):
        """
        Process a passband signal through N-path and conventional receivers.

        Parameters
        ----------
        x_p1 : Real passband input signal of desired signal at simulation sampling rate f_s.
        x_p2 : Real passband input signal of interferer at simulation sampling rate f_s.
        f_p1 : Carrier frequency of desired signal [Hz]
        f_p2 : Carrier frequency of interferer [Hz]

        Returns
        -------
        y_adc_quant      : N-path receiver output at ADC rate (Quantized)
        y_conv_adc_quant : Conventional receiver output at ADC rate (Quantized)
        """
        noise = np.random.normal(loc=0.0, scale=np.sqrt(self.noise), size=x_p1.shape)
        self.Nsamp, self.length = x_p1.shape

        # ==============================
        # N-PATH RECEIVER
        # ==============================

        # N-path filter
        y_out =  self.npath_filter(x_p=x_p1, f_p=f_p1)
        y_out +=  self.npath_filter(x_p=x_p2, f_p=f_p2)
        y_out +=  self.npath_noise(x_n=noise)
        
        # Soft-saturation nonlinearity
        phase_term = np.exp(1j * np.angle(y_out))
        tanh_term = np.tanh(np.abs(y_out) / self.Psat_sqrt)
        y_out = self.Psat_sqrt * tanh_term * phase_term

        # 2nd-order RC LPF
        y_out = self.rc_lpf(x=y_out, R=self.R2, C=self.C2)
        # y_out = self.rc_lpf(x=y_out, R=self.R2, C=self.C2)

        # ADC downsampling
        y_adc = y_out[:, ::self.adc].copy()
        delta = delta_backoff(y_adc)
        y_adc_quant = quantizer(y_adc, delta)

        # ==============================
        # CONVENTIONAL RECEIVER
        # ==============================

        # Noise
        y_conv = x_p1 + x_p2 + noise

        # Soft-saturation nonlinearity
        phase_term = np.exp(1j * np.angle(y_conv))
        tanh_term = np.tanh(np.abs(y_conv) / self.Psat_sqrt)
        y_conv = self.Psat_sqrt * tanh_term * phase_term

        # Down-conversion by mixing with LO
        t = np.arange(x_p1.shape[1]) / self.f_s
        y_conv = y_conv * np.exp(-1j * 2 * np.pi * self.f_lo * t)

        # 2nd-order RC LPF
        y_conv = self.rc_lpf(x=y_conv, R=self.R2, C=self.C2)
        # y_conv = self.rc_lpf(x=y_conv, R=self.R2, C=self.C2)

        # ADC downsampling
        y_conv_adc = y_conv[:, ::self.adc].copy()
        delta = delta_backoff(y_conv_adc)
        y_conv_adc_quant = quantizer(y_conv_adc, delta)

        return y_adc_quant, y_conv_adc_quant

    def npath_rc_lpf_path(self, x_p, f_p, y_init=0.0):
        """
        Exact RC low-pass filter response for one path of an N-path receiver
        """
        x_p = np.atleast_2d(x_p)
        _, length = x_p.shape
        y = np.zeros_like(x_p, dtype=np.float64)

        # Product terms for the exact solution
        y_ss = x_p / (1.0 + 1j * 2 * np.pi * f_p * self.R1 * self.C1)
        phase_advance = np.exp(1j * 2 * np.pi * f_p / self.f_s)
        rc_decay = np.exp(-1.0 / (self.f_s * self.R1 * self.C1))

        # Initial condition contribution
        y[:, 0] = np.real(y_ss[:, 0] * phase_advance + (y_init - y_ss[:, 0]) * rc_decay)

        # Recursive update from previous sample
        for i in range(1, length):
            y[:, i] = np.real(y_ss[:, i] * phase_advance + (y[:, i - 1] - y_ss[:, i]) * rc_decay)

        return y

    def npath_filter(self, x_p, f_p):
        """
        Apply an N-path RC filter / Receiver to a passband signal transmitted at f_p.
        """
        x_p = np.atleast_2d(x_p)

        T_lo = 1 / self.f_lo   # LO period in seconds
        period_samples = T_lo * self.f_s   # LO period in samples
        window = period_samples / self.N   # Time each path is active per LO cycle
        idx = np.arange(self.length)   # Sample indices for the input signal

        # Generate N boolean switching waveforms (one per path)
        switches = np.zeros((self.N, self.length), dtype=bool)
        for k in range(self.N):
            shift = k * window
            switches[k, :] = np.mod(idx - shift, period_samples) < window

        # Apply RC LPF to samples of each path
        y_paths = np.zeros((self.Nsamp, self.N, self.length), dtype=complex)
        for k in range(self.N):
            x_on  = x_p[:, switches[k, :]]  # Samples active on this path 
            y_on  = self.npath_rc_lpf_path(x_p=x_on, f_p=f_p)   # LPF output
            switches_delay = np.concatenate([[False], switches[k, :-1]])
            y_paths[:, k, switches_delay] = y_on[:, :np.sum(switches_delay)]    # Place filtered samples back

        if self.baseband:
            # Compute I/Q components by mixing each path with LO phases
            c = np.cos(2 * np.pi * np.arange(self.N) / self.N)  # cosine weights
            s = np.sin(2 * np.pi * np.arange(self.N) / self.N)  # sine weights
            I = np.sum(y_paths * c[np.newaxis, :, np.newaxis], axis=1)
            Q = -np.sum(y_paths * s[np.newaxis, :, np.newaxis], axis=1)
            result = I + 1j * Q  # Return complex baseband output
        else:
            # Sum all paths to produce bandpass output
            result = np.sum(y_paths, axis=1)

        return result

    def rc_lpf(self, x, R, C, y_init=0.0):
        """
        First-order RC low-pass filter (discrete-time). Discretized using the bilinear 
        transform (Tustin's method) with frequency pre-warping.
        """
        f_c = 1.0 / (2.0 * np.pi * R * C)
        assert self.f_s > 2 * f_c, "Sampling rate must be greater than twice the cutoff frequency"

        x = np.atleast_2d(x)
        _, length = x.shape
        y = np.zeros_like(x, dtype=np.result_type(x, y_init))

        # Pre-warping & coefficients
        k = np.tan(np.pi * f_c / self.f_s)
        b = k / (1.0 + k)
        a = (1.0 - k) / (1.0 + k)
        y[:, 0] = y_init

        for i in range(1, length):
            y[:, i] = b * (x[:, i] + x[:, i - 1]) + a * y[:, i - 1]

        return y

    def npath_noise(self, x_n):
        """
        Apply an N-path RC filter / Receiver to a noise signal
        """
        x_n = np.atleast_2d(x_n)

        T_lo = 1 / self.f_lo   # LO period in seconds
        period_samples = T_lo * self.f_s   # LO period in samples
        window = period_samples / self.N   # Time each path is active per LO cycle
        idx = np.arange(self.length)   # Sample indices for the input signal
        
        # Generate N boolean switching waveforms (one per path)
        switches = np.zeros((self.N, self.length), dtype=bool)
        for k in range(self.N):
            shift = k * window
            switches[k, :] = np.mod(idx - shift, period_samples) < window

        # Apply RC LPF to samples of each path
        y_paths = np.zeros((self.Nsamp, self.N, self.length), dtype=complex)
        for k in range(self.N):
            x_on = x_n[:, switches[k, :]]  # Samples active on this path
            y_on = self.rc_lpf(x=x_on, R=self.R1, C=self.C1)  # LPF output
            y_paths[:, k, switches[k, :]] = y_on   # Place filtered samples back

        if self.baseband:
            # Compute I/Q components by mixing each path with LO phases
            c = np.cos(2 * np.pi * np.arange(self.N) / self.N)  # cosine weights
            s = np.sin(2 * np.pi * np.arange(self.N) / self.N)  # sine weights
            I = np.sum(y_paths * c[np.newaxis, :, np.newaxis], axis=1)
            Q = -np.sum(y_paths * s[np.newaxis, :, np.newaxis], axis=1)
            result = I + 1j * Q  # Return complex baseband output
        else:
            # Sum all paths to produce bandpass output
            result = np.sum(y_paths, axis=1)

        return result