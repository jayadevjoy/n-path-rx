import numpy as np
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