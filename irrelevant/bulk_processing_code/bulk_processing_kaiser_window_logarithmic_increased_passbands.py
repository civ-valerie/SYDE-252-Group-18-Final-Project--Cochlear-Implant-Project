import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import firwin, lfilter, freqz

def create_fir_filter_bank(band_ranges, samplerate, num_taps=877, beta=6):
    """
    Creates FIR filters for each band in the filter bank using Kaiser window.
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

def process_with_filter_bank(data, band_ranges, samplerate=16000):
    """
    Processes the audio data through a filter bank and returns filtered signals.
    """
    filters = create_fir_filter_bank(band_ranges, samplerate)
    filtered_signals = apply_filter_bank(filters, data)
    return filters, filtered_signals

def plot_filter_response(filters, samplerate):
    """
    Plots the frequency response of the filters.
    """
    plt.figure(figsize=(10, 6))
    for i, fir_coeff in enumerate(filters):
        w, h = freqz(fir_coeff, worN=8000)
        plt.plot((w / np.pi) * (samplerate / 2), 20 * np.log10(np.abs(h)))
    plt.title("Magnitude Response of 16-channel Bandpass Filters Using Kaiser Window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    plt.title(f"Lowest Frequency Channel Output: {lowest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(highest_signal)
    plt.title(f"Highest Frequency Channel Output: {highest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def task_7_rectify_signals(filtered_signals):
    """
    Rectifies the output signals of all bandpass filters using absolute value.
    """
    rectified_signals = [np.abs(signal) for signal in filtered_signals]
    return rectified_signals

def task_8_extract_envelopes(rectified_signals, samplerate, num_taps=101, cutoff=400, beta=6):
    """
    Detects the envelopes of all rectified signals using a lowpass filter.
    """
    nyquist = samplerate / 2
    normalized_cutoff = cutoff / nyquist
    lowpass_filter = firwin(num_taps, normalized_cutoff, pass_zero=True, window=("kaiser", beta))
    envelopes = []
    for signal in rectified_signals:
        envelope = lfilter(lowpass_filter, 1.0, signal)
        envelopes.append(envelope)
    return envelopes

def task_2_9_plot_envelopes(envelopes, band_ranges):
    """
    Plots the extracted envelopes of the lowest and highest frequency channels.
    """
    lowest_envelope = envelopes[0]
    highest_envelope = envelopes[-1]
    lowest_band = band_ranges[0]
    highest_band = band_ranges[-1]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(lowest_envelope)
    plt.title(f"Envelope - Lowest Frequency Channel: {lowest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(highest_envelope)
    plt.title(f"Envelope - Highest Frequency Channel: {highest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
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

def process_files(file_list, band_ranges, samplerate=16000, output_dir="processed_outputs"):
    """
    Processes a list of audio files using the FIR filter bank and saves the outputs.

    Parameters:
        file_list (list): List of file paths to process.
        band_ranges (list): List of tuples defining the band ranges for the filter bank.
        samplerate (int): Sampling rate for the processing pipeline.
        output_dir (str): Directory where the processed files will be saved.

    Returns:
        None
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in file_list:
        # Read and preprocess the audio file
        data, original_samplerate = sf.read(file_name)
        if len(data.shape) > 1 and data.shape[1] == 2:
            data = np.mean(data, axis=1)  # Convert stereo to mono
        if original_samplerate != samplerate:
            data = resample(data, int(len(data) * samplerate / original_samplerate))

        # Process the data and apply the filter bank
        filters, filtered_signals = process_with_filter_bank(data, band_ranges, samplerate)

        # Rectify the output signals
        rectified_signals = task_7_rectify_signals(filtered_signals)

        # Detect the envelopes of all rectified signals
        envelopes = task_8_extract_envelopes(rectified_signals, samplerate)

        # Generate carrier signals
        signal_length = len(envelopes[0])  # Assuming all envelopes are the same length
        carrier_signals = generate_carrier_signals(band_ranges, signal_length, samplerate)

        # Amplitude modulate the carrier signals with the envelopes
        modulated_signals = amplitude_modulate(envelopes, carrier_signals)

        # Sum the modulated signals and normalize
        output_signal = sum_and_normalize(modulated_signals)

        # Save the output sound to the specified directory
        output_file_name = os.path.join(output_dir, f"processed_{os.path.basename(file_name)}")
        sf.write(output_file_name, output_signal, samplerate)

        print(f"Processed and saved: {output_file_name}")


if __name__ == "__main__":
    samplerate = 16000
    band_ranges = [
        (100, 277), (277, 476), (476, 700), (700, 951), (951, 1235),
        (1235, 1553), (1553, 1912), (1912, 2315), (2315, 2769),
        (2769, 3280), (3280, 3854), (3854, 4500), (4500, 5227),
        (5227, 6045), (6045, 6965), (6965, 7999)
    ]

    # List of files to process
    file_list = [
        "Alarm_Clock.wav",
        "Classroom.wav",
        "Female_Speech.wav",
        "Instrumental_Piano_Sound.wav",
        "Male_Speech.wav",
        "One_Speaker_To_Multiple.wav",
        "Squash_Ball.wav",
        "Street.wav",
        "Talking_Normal_To_Whispering.wav",
    ]

    # Process the files and save the outputs
    process_files(file_list, band_ranges, samplerate, output_dir="processed_outputs")