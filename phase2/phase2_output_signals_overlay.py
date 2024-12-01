import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter, firwin

def create_butter_filter_bank(band_ranges, samplerate, order=2):
    """
    Creates second-order Butterworth filters for each band in the filter bank.
    """
    filters = []
    nyquist = samplerate / 2  # Nyquist frequency
    for low, high in band_ranges:
        # Adjust the frequencies if needed
        low = max(low, 1e-6)
        high = min(high, nyquist - 1e-6)  # Ensure high < Nyquist frequency
        if low >= nyquist or high <= low:
            raise ValueError(f"Cutoff frequencies out of range: ({low}, {high}), Nyquist={nyquist}")
        # Design the Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='bandpass', fs=samplerate)
        filters.append((b, a))
    return filters

def create_fir_filter_bank(band_ranges, samplerate, num_taps=101, window='hamming', beta=None):
    """
    Creates FIR filters for each band in the filter bank using the specified window.
    """
    filters = []
    nyquist = samplerate / 2  # Nyquist frequency
    for low, high in band_ranges:
        # Adjust frequencies
        low = max(low, 1e-6)
        high = min(high, nyquist - 1e-6)
        normalized_low = low / nyquist
        normalized_high = high / nyquist
        if window == 'kaiser' and beta is not None:
            fir_coeff = firwin(num_taps, [normalized_low, normalized_high], pass_zero=False, window=(window, beta))
        else:
            fir_coeff = firwin(num_taps, [normalized_low, normalized_high], pass_zero=False, window=window)
        filters.append(fir_coeff)
    return filters

def apply_filter_bank(filters, data, filter_type='butter'):
    """
    Applies each filter in the filter bank to the audio signal.
    """
    filtered_signals = []
    if filter_type == 'butter':
        for b, a in filters:
            filtered_signal = lfilter(b, a, data)
            filtered_signals.append(filtered_signal)
    else:
        for fir_coeff in filters:
            filtered_signal = lfilter(fir_coeff, 1.0, data)
            filtered_signals.append(filtered_signal)
    return filtered_signals

def overlay_task_2_6_outputs(filtered_signals_dict, band_ranges):
    """
    Overlays the output signals of the lowest and highest frequency channels from different filter types.
    
    Parameters:
        filtered_signals_dict (dict): Dictionary containing filtered signals from different filter types.
        band_ranges (list of tuples): Passband ranges for the filters.
    """
    lowest_band = band_ranges[0]
    highest_band = band_ranges[-1]
    
    # Define line styles and colors for distinction
    line_styles = {
        'Butterworth': ('-', 'blue'),
        'Hamming FIR': ('--', 'green'),
        'Kaiser FIR': (':', 'red')
    }
    
    # Plot for the lowest frequency channel
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for label, signals in filtered_signals_dict.items():
        style, color = line_styles.get(label, ('-', 'black'))
        plt.plot(signals[0], linestyle=style, color=color, alpha=0.8, label=label)
    plt.title(f"Output Signal - Lowest Frequency Channel: {lowest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot for the highest frequency channel
    plt.subplot(2, 1, 2)
    for label, signals in filtered_signals_dict.items():
        style, color = line_styles.get(label, ('-', 'black'))
        plt.plot(signals[-1], linestyle=style, color=color, alpha=0.8, label=label)
    plt.title(f"Output Signal - Highest Frequency Channel: {highest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Display the plots
    plt.tight_layout()
    plt.show()

def main():
    samplerate = 16000  # Define the sampling rate
    
    # Read and preprocess the audio file
    file_name = "Whispering_audio.wav"
    data, original_samplerate = sf.read(file_name)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
    if original_samplerate != samplerate:
        data = resample(data, int(len(data) * samplerate / original_samplerate))
    
    # Dictionary to hold filtered signals from different filter types
    filtered_signals_dict = {}
    
    # Butterworth filter processing
    butter_filters = create_butter_filter_bank(band_ranges, samplerate, order=2)
    butter_filtered = apply_filter_bank(butter_filters, data, filter_type='butter')
    filtered_signals_dict['Butterworth'] = butter_filtered
    
    # Hamming window FIR filter processing
    hamming_filters = create_fir_filter_bank(band_ranges, samplerate, num_taps=101, window='hamming')
    hamming_filtered = apply_filter_bank(hamming_filters, data, filter_type='fir')
    filtered_signals_dict['Hamming FIR'] = hamming_filtered
    
    # Kaiser window FIR filter processing
    kaiser_beta = 6
    kaiser_filters = create_fir_filter_bank(band_ranges, samplerate, num_taps=877, window='kaiser', beta=kaiser_beta)
    kaiser_filtered = apply_filter_bank(kaiser_filters, data, filter_type='fir')
    filtered_signals_dict['Kaiser FIR'] = kaiser_filtered
    
    # Overlay the task 2.6 outputs
    overlay_task_2_6_outputs(filtered_signals_dict, band_ranges)
    
if __name__ == "__main__":
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
    main()
