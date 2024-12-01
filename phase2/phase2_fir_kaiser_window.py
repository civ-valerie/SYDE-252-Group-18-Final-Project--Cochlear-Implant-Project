import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import firwin, lfilter, freqz

def create_fir_filter_bank(band_ranges, samplerate, num_taps=877, beta=6):
    """
    Creates FIR filters for each band in the filter bank using Kaiser window.
    
    Parameters:
        band_ranges (list): List of (low, high) frequency pairs for the passbands.
        samplerate (int): Sampling rate of the audio signal.
        num_taps (int): Number of taps for the FIR filter.
        beta (float): Kaiser window beta parameter.
        
    Returns:
        filters (list): List of FIR filter coefficients for each band.
    """
    filters = []
    nyquist = samplerate / 2  # Nyquist frequency
    for low, high in band_ranges:
        high = min(high, nyquist - 1e-6)  # Adjust the upper bound if needed
        if low <= 0 or high <= low:
            raise ValueError(f"Cutoff frequencies out of range: ({low}, {high}), Nyquist={nyquist}")
        normalized_low = low / nyquist
        normalized_high = high / nyquist
        fir_coeff = firwin(
            num_taps, 
            [normalized_low, normalized_high], 
            pass_zero=False, 
            window=("kaiser", beta)
        )
        filters.append(fir_coeff)
    return filters


def apply_filter_bank(filters, data):
    """
    Applies each filter in the filter bank to the audio signal.
    """
    filtered_signals = []
    for fir_coeff in filters:
        filtered_signal = lfilter(fir_coeff, 1.0, data)
        filtered_signals.append(filtered_signal)
    return filtered_signals

# Task 2.5 - Process the file and apply the filter bank
def process_with_filter_bank(file_name, band_ranges, samplerate=16000):
    """
    Processes the audio file through a filter bank and visualizes the results.
    """
    data, original_samplerate = sf.read(file_name)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
    if original_samplerate != samplerate:
        data = resample(data, int(len(data) * samplerate / original_samplerate))
    filters = create_fir_filter_bank(band_ranges, samplerate)
    filtered_signals = apply_filter_bank(filters, data)
    plt.figure(figsize=(15, 10))
    for i, signal in enumerate(filtered_signals):
        plt.subplot(len(filtered_signals), 1, i + 1)
        plt.plot(signal)
        plt.title(f"Band {i + 1}: {band_ranges[i]}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    return filtered_signals

# Task 2.6 - Plot the output signals of the lowest and highest frequency channels
def task_2_6_plot_extreme_channels(filtered_signals, band_ranges):
    """
    Plots the output signals of the lowest and highest frequency channels.
    """
    lowest_signal = filtered_signals[0]
    highest_signal = filtered_signals[-1]
    lowest_band = band_ranges[0]
    highest_band = band_ranges[-1]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(lowest_signal)
    plt.title(f"Lowest Frequency Channel: {lowest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(highest_signal)
    plt.title(f"Highest Frequency Channel: {highest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Task 7 - Rectify the output signals of all bandpass filters
def task_7_rectify_signals(filtered_signals):
    """
    Rectifies the output signals of all bandpass filters using absolute value.
    """
    rectified_signals = [np.abs(signal) for signal in filtered_signals]
    return rectified_signals

# Task 8 - Detect the envelopes of all rectified signals using a lowpass filter
def task_8_extract_envelopes(rectified_signals, samplerate, num_taps=101, cutoff=400, beta = 6):
    """
    Detects the envelopes of all rectified signals using a lowpass filter.
    
    Parameters:
        rectified_signals (list): List of rectified signals for each band.
        samplerate (int): Sampling rate of the audio signal.
        num_taps (int): Number of taps for the lowpass FIR filter.
        cutoff (float): Cutoff frequency of the lowpass filter in Hz.
        
    Returns:
        envelopes (list): List of envelope signals for each band.
    """
    # Normalize the cutoff frequency
    nyquist = samplerate / 2
    normalized_cutoff = cutoff / nyquist
    
    # Design the lowpass FIR filter
    lowpass_filter = firwin(num_taps, normalized_cutoff, pass_zero=True, window=("kaiser", beta))
    
    # Apply the lowpass filter to each rectified signal
    envelopes = []
    for signal in rectified_signals:
        envelope = lfilter(lowpass_filter, 1.0, signal)
        envelopes.append(envelope)
    
    return envelopes

# Task 2.9 - Plot the extracted envelope of the lowest and highest frequency channels
def task_2_9_plot_envelopes(envelopes, band_ranges):
    """
    Plots the extracted envelope of the lowest and highest frequency channels.
    
    Parameters:
        envelopes (list): List of envelope signals for each band.
        band_ranges (list of tuples): Passband ranges for the filters.
    """
    # Identify the lowest and highest frequency channels
    lowest_envelope = envelopes[0]
    highest_envelope = envelopes[-1]
    lowest_band = band_ranges[0]
    highest_band = band_ranges[-1]
    
    # Plot the lowest frequency envelope
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(lowest_envelope)
    plt.title(f"Envelope - Lowest Frequency Channel: {lowest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot the highest frequency envelope
    plt.subplot(2, 1, 2)
    plt.plot(highest_envelope)
    plt.title(f"Envelope - Highest Frequency Channel: {highest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_filter_response(filters, samplerate):
    """
    Plots the frequency response of the filters.
    """
    plt.figure(figsize=(10, 6))
    for i, fir_coeff in enumerate(filters):
        w, h = freqz(fir_coeff, worN=8000)
        plt.plot((w / np.pi) * (samplerate / 2), 20 * np.log10(np.abs(h)), label=f"Band {i + 1}")
    plt.title("Magnitude Response of 16-channel Bandpass Using Kaiser Window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_stopband_attenuation(filters, band_ranges, samplerate):
    """
    Calculates the stopband attenuation for each filter in the filter bank.
    
    Parameters:
        filters (list): List of FIR filter coefficients.
        band_ranges (list): List of (low, high) frequency pairs for the passbands.
        samplerate (int): Sampling rate of the audio signal.
        
    Returns:
        stopband_attenuations (list): List of stopband attenuation values for each filter in dB.
    """
    nyquist = samplerate / 2
    stopband_attenuations = []

    for i, fir_coeff in enumerate(filters):
        low, high = band_ranges[i]
        w, h = freqz(fir_coeff, worN=8000)  # Compute frequency response
        freqs = (w / np.pi) * nyquist       # Convert to Hz
        magnitude_db = 20 * np.log10(np.abs(h))  # Convert magnitude to dB
        
        # Define stopband regions
        stopband_indices = (freqs < low) | (freqs > high)
        stopband_magnitude = magnitude_db[stopband_indices]
        
        # Find minimum magnitude (maximum attenuation) in stopband
        stopband_attenuation = -np.max(-stopband_magnitude)  # Avoid -inf if no attenuation
        stopband_attenuations.append(stopband_attenuation)
    
    return stopband_attenuations


# Define the adjusted band ranges
band_ranges = [
    (100, 277), (277, 476), (476, 700), (700, 951), (951, 1235),
    (1235, 1553), (1553, 1912), (1912, 2315), (2315, 2769),
    (2769, 3280), (3280, 3854), (3854, 4500), (4500, 5227),
    (5227, 6045), (6045, 6965), (6965, 7999)
]

# Task 2.5 - Process the file
filtered_signals = process_with_filter_bank("Whispering_audio.wav", band_ranges)

# Example usage
filters = create_fir_filter_bank(band_ranges, samplerate=16000)
plot_filter_response(filters, samplerate=16000)

# Task 2.5 - Process the file
filtered_signals = process_with_filter_bank("Whispering_audio.wav", band_ranges)

# Task 2.6 - Plot the outputs for the lowest and highest frequency channels
task_2_6_plot_extreme_channels(filtered_signals, band_ranges)

# Task 7 - Rectify the output signals
rectified_signals = task_7_rectify_signals(filtered_signals)

# Task 8 - Detect the envelopes of all rectified signals
envelopes = task_8_extract_envelopes(rectified_signals, samplerate=16000)

# Task 2.9 - Plot the extracted envelopes for the lowest and highest frequency channels
task_2_9_plot_envelopes(envelopes, band_ranges)