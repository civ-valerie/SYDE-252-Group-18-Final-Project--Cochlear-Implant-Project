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

def generate_carrier_signals(band_ranges, signal_length, samplerate):
    """
    Generates carrier cosine signals for each bandpass filter channel.
    """
    carrier_signals = []
    t = np.arange(signal_length) / samplerate  # Time vector
    for low, high in band_ranges:
        center_freq = (low + high) / 2
        carrier = np.cos(2 * np.pi * center_freq * t)
        carrier_signals.append(carrier)
    return carrier_signals

def amplitude_modulate(envelopes, carrier_signals):
    """
    Amplitude modulates the carrier signals with the envelopes.
    """
    modulated_signals = []
    for envelope, carrier in zip(envelopes, carrier_signals):
        modulated_signal = envelope * carrier
        modulated_signals.append(modulated_signal)
    return modulated_signals

def sum_and_normalize(modulated_signals):
    """
    Sums all modulated signals and normalizes the result.
    """
    output_signal = np.sum(modulated_signals, axis=0)
    max_abs = np.max(np.abs(output_signal))
    if max_abs > 0:
        output_signal = output_signal / max_abs  # Normalize
    return output_signal

def play_and_save_output(output_signal, samplerate, output_filename="output.wav"):
    """
    Plays the output sound and writes it to a file.
    """
    # Play the sound
    sd.play(output_signal, samplerate)
    sd.wait()
    # Save to file
    sf.write(output_filename, output_signal, samplerate)


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

samplerate = 16000
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
file_name = "Whispering_audio.wav"
filtered_signals = process_with_filter_bank(file_name, band_ranges)

# Task 2.6 - Plot the outputs for the lowest and highest frequency channels
task_2_6_plot_extreme_channels(filtered_signals, band_ranges)

# Task 7 - Rectify the output signals
rectified_signals = task_7_rectify_signals(filtered_signals)

# Task 8 - Detect the envelopes of all rectified signals
envelopes = task_8_extract_envelopes(rectified_signals, samplerate=16000)

# Task 2.9 - Plot the extracted envelopes for the lowest and highest frequency channels
task_2_9_plot_envelopes(envelopes, band_ranges)

# Task 10 - Generate carrier signals
signal_length = len(envelopes[0])  # Assuming all envelopes are the same length
carrier_signals = generate_carrier_signals(band_ranges, signal_length, samplerate)

# Task 11 - Amplitude modulate the carrier signals with the envelopes
modulated_signals = amplitude_modulate(envelopes, carrier_signals)

# Task 12 - Sum the modulated signals and normalize
output_signal = sum_and_normalize(modulated_signals)

# Task 13 - Play and save the output sound
play_and_save_output(output_signal, samplerate, output_filename=f"output_bandpass_middle_bigger_{file_name}.wav")



# Example usage
filters = create_butter_filter_bank(band_ranges, samplerate=16000)
plot_butter_filter_response(filters, samplerate=16000)

