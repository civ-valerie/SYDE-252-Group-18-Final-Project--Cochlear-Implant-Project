import numpy as np
from scipy.signal import resample, firwin, lfilter, freqz, spectrogram
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import os

def create_fir_filter_bank(band_ranges, samplerate, num_taps=877, beta=6):
    """
    Creates FIR filters for each band in the filter bank using a Kaiser window.
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

def sum_and_normalize(filtered_signals, gain=2.0):
    """
    Sums all filtered signals, normalizes the result, then applies a gain factor.
    """
    output_signal = np.sum(filtered_signals, axis=0)
    max_abs = np.max(np.abs(output_signal))
    if max_abs > 0:
        output_signal = output_signal / max_abs  # Normalize to [-1, 1] range
    output_signal *= gain  # Apply gain factor
    # Ensure the output is within [-1, 1] to prevent clipping
    output_signal = np.clip(output_signal, -1.0, 1.0)
    return output_signal

def play_and_save_output(output_signal, samplerate, output_filename="output.wav"):
    """
    Plays the output sound and writes it to a file.
    """
    # Play the sound
    # sd.play(output_signal, samplerate)
    # sd.wait()
    # Save to file
    sf.write(output_filename, output_signal, samplerate)

def process_files(file_list, band_ranges, samplerate=16000, output_dir="processed_outputs", gain=2.0):
    """
    Processes a list of audio files using the FIR filter bank and saves the outputs.

    Parameters:
        file_list (list): List of file paths to process.
        band_ranges (list): List of tuples defining the band ranges for the filter bank.
        samplerate (int): Sampling rate for the processing pipeline.
        output_dir (str): Directory where the processed files will be saved.
        gain (float): Gain factor to apply after normalization.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in file_list:
        print(f"Processing: {file_name}")
        # Read and preprocess the audio file
        data, original_samplerate = sf.read(file_name)
        if len(data.shape) > 1 and data.shape[1] == 2:
            data = np.mean(data, axis=1)  # Convert stereo to mono
        if original_samplerate != samplerate:
            data = resample(data, int(len(data) * samplerate / original_samplerate))

        # Process the data and apply the filter bank
        filters, filtered_signals = process_with_filter_bank(data, band_ranges, samplerate)

        # Sum the filtered signals, normalize, and apply gain
        output_signal = sum_and_normalize(filtered_signals, gain=gain)

        # Save the output sound to the specified directory
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        output_file_name = os.path.join(output_dir, f"processed_{base_name}.wav")
        play_and_save_output(output_signal, samplerate, output_filename=output_file_name)

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
    process_files(file_list, band_ranges, samplerate, output_dir="processed_outputs", gain=2.0)
