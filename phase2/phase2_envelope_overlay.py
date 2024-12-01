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

def task_7_rectify_signals(filtered_signals):
    """
    Rectifies the output signals of all bandpass filters using absolute value.
    """
    rectified_signals = [np.abs(signal) for signal in filtered_signals]
    return rectified_signals

def task_8_extract_envelopes(rectified_signals, samplerate, filter_type='butter', order=2, num_taps=101, cutoff=400, beta=6):
    """
    Detects the envelopes of all rectified signals using a lowpass filter.
    """
    envelopes = []
    nyquist = samplerate / 2  # Nyquist frequency
    cutoff = min(cutoff, nyquist - 1e-6)
    if filter_type == 'butter':
        b, a = butter(order, cutoff, btype='low', fs=samplerate)
        for signal in rectified_signals:
            envelope = lfilter(b, a, signal)
            envelopes.append(envelope)
    else:
        normalized_cutoff = cutoff / nyquist
        if filter_type == 'kaiser':
            lowpass_filter = firwin(num_taps, normalized_cutoff, pass_zero=True, window=('kaiser', beta))
        else:
            lowpass_filter = firwin(num_taps, normalized_cutoff, pass_zero=True, window='hamming')
        for signal in rectified_signals:
            envelope = lfilter(lowpass_filter, 1.0, signal)
            envelopes.append(envelope)
    return envelopes

def overlay_envelopes(envelopes_dict, band_ranges):
    """
    Overlays the extracted envelopes of the lowest and highest frequency channels from different filter types.
    """
    lowest_band = band_ranges[0]
    highest_band = band_ranges[-1]
    
    # Define line styles and markers for distinction
    line_styles = {
        'Butterworth': ('-', 'o'),
        'Hamming FIR': ('--', 's'),
        'Kaiser FIR': (':', '^')
    }
    
    colors = {
        'Butterworth': 'blue',
        'Hamming FIR': 'green',
        'Kaiser FIR': 'red'
    }
    
    # Plot for the lowest frequency channel
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for label, envelopes in envelopes_dict.items():
        style, marker = line_styles.get(label, ('-', None))
        plt.plot(envelopes[0], linestyle=style, color=colors.get(label, None), alpha=0.8, label=label)
    plt.title(f"Envelope - Lowest Frequency Channel: {lowest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot for the highest frequency channel
    plt.subplot(2, 1, 2)
    for label, envelopes in envelopes_dict.items():
        style, marker = line_styles.get(label, ('-', None))
        plt.plot(envelopes[-1], linestyle=style, color=colors.get(label, None), alpha=0.8, label=label)
    plt.title(f"Envelope - Highest Frequency Channel: {highest_band}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Display the plots
    plt.tight_layout()
    plt.show()

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
def main():
    samplerate = 16000  # Define the sampling rate
    
    # Read and preprocess the audio file
    file_name = "Whispering_audio.wav"
    data, original_samplerate = sf.read(file_name)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
    if original_samplerate != samplerate:
        data = resample(data, int(len(data) * samplerate / original_samplerate))
    
    # Dictionary to hold envelopes from different filter types
    envelopes_dict = {}
    
    # Butterworth filter processing
    butter_filters = create_butter_filter_bank(band_ranges, samplerate, order=2)
    butter_filtered = apply_filter_bank(butter_filters, data, filter_type='butter')
    butter_rectified = task_7_rectify_signals(butter_filtered)
    butter_envelopes = task_8_extract_envelopes(butter_rectified, samplerate, filter_type='butter', order=2, cutoff=400)
    envelopes_dict['Butterworth'] = butter_envelopes
    
    # Hamming window FIR filter processing
    hamming_filters = create_fir_filter_bank(band_ranges, samplerate, num_taps=101, window='hamming')
    hamming_filtered = apply_filter_bank(hamming_filters, data, filter_type='fir')
    hamming_rectified = task_7_rectify_signals(hamming_filtered)
    hamming_envelopes = task_8_extract_envelopes(hamming_rectified, samplerate, filter_type='hamming', num_taps=101, cutoff=400)
    envelopes_dict['Hamming FIR'] = hamming_envelopes
    
    # Kaiser window FIR filter processing
    kaiser_beta = 6
    kaiser_filters = create_fir_filter_bank(band_ranges, samplerate, num_taps=877, window='kaiser', beta=kaiser_beta)
    kaiser_filtered = apply_filter_bank(kaiser_filters, data, filter_type='fir')
    kaiser_rectified = task_7_rectify_signals(kaiser_filtered)
    kaiser_envelopes = task_8_extract_envelopes(kaiser_rectified, samplerate, filter_type='kaiser', num_taps=101, cutoff=400, beta=kaiser_beta)
    envelopes_dict['Kaiser FIR'] = kaiser_envelopes
    
    # Overlay the envelopes
    overlay_envelopes(envelopes_dict, band_ranges)

if __name__ == "__main__":
    main()
