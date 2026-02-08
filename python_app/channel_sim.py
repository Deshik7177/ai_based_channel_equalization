import numpy as np
import random

def add_awgn(signal, snr_db=20):
    """Adds Additive White Gaussian Noise."""
    if np.all(signal == 0): return signal
    sig_avg_watts = np.mean(signal**2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    noise_volts = np.random.normal(0, np.sqrt(noise_avg_watts), len(signal))
    return signal + noise_volts

def apply_multipath(signal, delay=2, attenuation=0.5):
    """
    Simulates acoustic multipath/reverberation by adding a delayed echo.
    """
    echo = np.zeros_like(signal)
    echo[delay:] = signal[:-delay] * attenuation
    return signal + echo

def apply_packet_loss(signal, loss_rate=0.1):
    """
    Simulates packet loss by replacing values with NaN.
    """
    noisy_signal = signal.copy().astype(float)
    mask = np.random.rand(len(noisy_signal)) < loss_rate
    noisy_signal[mask] = np.nan
    return noisy_signal

def handle_missing_data(signal):
    """
    Simple interpolation to handle missing data before DL model.
    """
    ok = ~np.isnan(signal)
    xp = ok.ravel().nonzero()[0]
    fp = signal[~np.isnan(signal)]
    x = np.isnan(signal).ravel().nonzero()[0]
    
    signal[np.isnan(signal)] = np.interp(x, xp, fp)
    return signal

if __name__ == "__main__":
    # Test simulation
    t = np.linspace(0, 1, 100)
    clean = np.sin(2 * np.pi * 5 * t)
    noisy = add_awgn(clean, snr_db=15)
    with_loss = apply_packet_loss(noisy, loss_rate=0.1)
    restored = handle_missing_data(with_loss)
    
    print("Clean signal (first 5):", clean[:5])
    print("Signal with loss (first 5):", with_loss[:5])
    print("Interpolated (first 5):", restored[:5])
