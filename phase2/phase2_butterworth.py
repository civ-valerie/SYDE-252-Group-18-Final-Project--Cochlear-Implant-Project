import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter, freqz

def create_butter_filter_bank(band_ranges, samplerate, order=8):
    """
    Creates second-order Butterworth filters for each band in the filter bank.
    """
    filters = []
    nyquist = samplerate / 2  # Nyquist frequency
    for low, high in band_ranges:
        high = min(high, nyquist - 1e-6)  # Adjust the upper bound if needed
        if low <= 0 or high <= low:
            raise ValueError(f"Cutoff frequencies out of range: ({low}, {high}), Nyquist={nyquist}")
        # Design the Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='bandpass', fs=samplerate)
        filters.append((b, a))
    return filters

def apply_filter_bank(filters, data):
    """
    Applies each Butterworth filter in the filter bank to the audio signal.
    """
    filtered_signals = []
    for b, a in filters:
        filtered_signal = lfilter(b, a, data)
        filtered_signals.append(filtered_signal)
    return filtered_signals

# Task 2.5 - Process the file and apply the filter bank
def process_with_filter_bank(file_name, band_ranges, samplerate=16000):
    """
    Processes the audio file through a Butterworth filter bank and visualizes the results.
    """
    data, original_samplerate = sf.read(file_name)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
    if original_samplerate != samplerate:
        data = resample(data, int(len(data) * samplerate / original_samplerate))
    filters = create_butter_filter_bank(band_ranges, samplerate)
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
def task_8_extract_envelopes(rectified_signals, samplerate, order=2, cutoff=400):
    """
    Detects the envelopes of all rectified signals using a lowpass Butterworth filter.
    
    Parameters:
        rectified_signals (list): List of rectified signals for each band.
        samplerate (int): Sampling rate of the audio signal.
        order (int): Order of the lowpass Butterworth filter.
        cutoff (float): Cutoff frequency of the lowpass filter in Hz.
            
    Returns:
        envelopes (list): List of envelope signals for each band.
    """
    # Design the lowpass Butterworth filter
    b, a = butter(order, cutoff, btype='low', fs=samplerate)
    
    # Apply the lowpass filter to each rectified signal
    envelopes = []
    for signal in rectified_signals:
        envelope = lfilter(b, a, signal)
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

def plot_butter_filter_response(filters, samplerate):
    """
    Plots the frequency response of Butterworth filters.
    
    Parameters:
        filters (list): List of (b, a) tuples for each band.
        samplerate (int): Sampling rate of the audio signal.
    """
    plt.figure(figsize=(10, 6))
    for i, (b, a) in enumerate(filters):
        # Get the frequency response
        w, h = freqz(b, a, worN=8000)
        # Convert to frequency (Hz) and plot
        plt.plot((w / np.pi) * (samplerate / 2), 20 * np.log10(np.abs(h)), label=f"Band {i + 1}")
    plt.title("Magnitude Response of Butterworth Bandpass Filters")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Define the adjusted band ranges
band_ranges = [
    (100, 132),     # Band 1
    (132, 173),     # Band 2
    (173, 228),     # Band 3
    (228, 299),     # Band 4
    (299, 394),     # Band 5
    (394, 518),     # Band 6
    (518, 682),     # Band 7
    (682, 897),     # Band 8
    (897, 1181),    # Band 9
    (1181, 1553),   # Band 10
    (1553, 2043),   # Band 11
    (2043, 2686),   # Band 12
    (2686, 3534),   # Band 13
    (3534, 4645),   # Band 14
    (4645, 6115),   # Band 15
    (6115, 8039)    # Band 16
]

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

# Example usage
filters = create_butter_filter_bank(band_ranges, samplerate=16000)
plot_butter_filter_response(filters, samplerate=16000)

