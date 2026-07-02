import math
import torch
import numpy as np
import torch.fft
from matplotlib import pyplot as plt
from scipy import signal


def highpass_filter_fft_controlled(wavelet: torch.Tensor,
                                   dt: float,
                                   cutoff: float = 5.0,
                                   transition_band: float = 2.0,
                                   ) -> torch.Tensor:
    wavelet_np = wavelet.numpy().copy()
    n = len(wavelet_np)
    fs = 1.0 / dt  # sampling frequency

    # Calculate the Fourier transform of the original signal
    fft_original = np.fft.fft(wavelet_np)
    freqs = np.fft.fftfreq(n, d=dt)

    # Verify frequency range
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        raise ValueError(f"Cutoff frequency {cutoff} Hz cannot be greater than or equal to Nyquist frequency {nyquist} Hz")

    # Define the boundary of the transition zone
    f_stop = cutoff - transition_band / 2
    f_pass = cutoff + transition_band / 2

    if f_stop < 0:
        f_stop = 0.0  # Ensure that f_stop is not negative
        # Adjust f_pass again to maintain the width of the transition zone
        f_pass = f_stop + transition_band

    if f_pass > nyquist:
        f_pass = nyquist
        f_stop = f_pass - transition_band if f_pass - transition_band > 0 else 0.0

    # Create a frequency response curve with a default gain of 1.0 (kept constant)
    gain = np.ones_like(freqs, dtype=np.float64)

    # --- 1. stopband: Below f_stop, the precision is 0 (satisfying the requirement of "spectrum is 0") ---
    # Hard cutoff region: frequency absolute value less than f_stop
    gain[(np.abs(freqs) < f_stop)] = 0.0

    # --- 2. Transition zone: Smooth rise from 0 to 1.0 (satisfying the requirement of "smooth transition from 0") ---

    transition_span = f_pass - f_stop

    # Transition band processing of positive and negative frequencies
    mask_transition = (np.abs(freqs) >= f_stop) & (np.abs(freqs) < f_pass)

    if np.any(mask_transition):
        # Normalize the position and calculate the relative position from f_stop to f_pass
        # We mainly focus on the positive frequency part, while the negative frequency is processed through absolute values

        # Calculate the distance between 0 and transition_stpan for each frequency point
        pos_in_transition = np.abs(freqs[mask_transition]) - f_stop
        norm_pos = pos_in_transition / transition_span

        # Use cosine function to increase the gain from 0 (norm_pos=0, i.e. f=f_stop) to 1 (norm_pos=1, i.e. f=f_pass)
        # Gain=0.5 * (1+cos (pi * (1- norm_pos))
        new_gain = 0.5 * (1.0 + np.cos(np.pi * (1.0 - norm_pos)))

        gain[mask_transition] = new_gain

    # --- 3. passband: The gain above f_pass is strictly 1.0 (consistent with the original signal) ---
    gain[np.abs(freqs) >= f_pass] = 1.0

    # Apply gain curve to frequency domain signal
    fft_filtered = fft_original * gain

    # Inverse Fourier transform back to the time domain, ensuring that the result is a real number
    filtered_signal = np.fft.ifft(fft_filtered).real


    original_torch_dtype = wavelet.dtype
    filtered_signal_torch = torch.from_numpy(filtered_signal.copy())

    # Explicitly convert to raw data type
    return filtered_signal_torch.to(original_torch_dtype)

def spectrum_analysis(signal, dt):
    """Perform spectrum analysis and plot results"""
    n = len(signal)
    fft = torch.fft.fft(signal)
    freq = torch.fft.fftfreq(n, d=dt)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(signal.numpy())
    plt.title('Ricker Wavelet (Time Domain)')

    plt.subplot(122)
    plt.plot(freq[:n // 2].numpy(), torch.abs(fft[:n // 2]).numpy())
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

def spectrum_analysis_comparison(signal_original: torch.Tensor,
                                 signal_filtered: torch.Tensor,
                                 dt: float):
    """
    Perform spectral analysis and plot the amplitude spectra of the original signal and the filtered signal in the same image for comparison.

    Args:
        signal_original: The original input signal (torch.Tensor)。
        signal_filtered: The filtered signal (torch.Tensor)。
        dt: Sampling interval (s)。
    """
    n = len(signal_original)  # Assuming two signals have the same length
    if n != len(signal_filtered):
        raise ValueError("The length of the original signal and the filtered signal must be consistent!")

    # Calculate FFT
    fft_original = torch.fft.fft(signal_original)
    fft_filtered = torch.fft.fft(signal_filtered)

    # Calculate frequency axis
    freq = torch.fft.fftfreq(n, d=dt)

    # Only plot the positive frequency portion (n/2), as the FFT result is symmetric about 0.5fs
    n_half = n // 2
    freq_positive = freq[:n_half]
    amplitude_original = torch.abs(fft_original[:n_half])
    amplitude_filtered = torch.abs(fft_filtered[:n_half])

    plt.figure(figsize=(12, 6))
    # Draw amplitude spectrum comparison
    plt.subplot(1, 1, 1) # If only the spectrum is drawn, set it to 1,1,1
    plt.plot(freq_positive.numpy(), amplitude_original.numpy(), label='Original Spectrum')
    plt.plot(freq_positive.numpy(), amplitude_filtered.numpy(), label='Filtered Spectrum')
    plt.title('Amplitude Spectrum Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()