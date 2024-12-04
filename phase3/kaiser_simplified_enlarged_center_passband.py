import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import firwin, lfilter, freqz
from scipy.signal import spectrogram

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
    plt.title("Magnitude Response of 16-channel Bandpass Filters Using Kaiser Window - Center Passbands Enlarged")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def moving_rms(signal, window_size):
    return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid'))


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
    sd.play(output_signal, samplerate)
    sd.wait()
    # Save to file
    sf.write(output_filename, output_signal, samplerate)

def compress_signal(signal, compression_ratio):
    """
    Compresses the signal's dynamic range using logarithmic compression.
    """
    return np.sign(signal) * np.log1p(compression_ratio * np.abs(signal)) / np.log1p(compression_ratio)


def main():
    samplerate = 16000  # Define the sampling rate

    # Read and preprocess the audio file
    file_name = "OperaManSingingClapping.wav"
    data, original_samplerate = sf.read(file_name)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)  # Convert to mono
    if original_samplerate != samplerate:
        data = resample(data, int(len(data) * samplerate / original_samplerate))

    band_ranges = [
        (100, 219), (219, 385), (385, 615), (615, 935),
        (935, 1380), (1380, 1998), (1998, 2857), (2857, 4050),
        (4050, 5242), (5242, 6101), (6101, 6719), (6719, 7164),
        (7164, 7484), (7484, 7714), (7714, 7880), (7880, 7999)
    ]

    # Process the data and apply the filter bank
    filters, filtered_signals = process_with_filter_bank(data, band_ranges, samplerate)

    # Sum the filtered signals, normalize, and apply gain
    gain = 2.0  # Adjust as needed
    output_signal = sum_and_normalize(filtered_signals, gain=gain)

    # Apply compression to the output signal
    compression_ratio = 10  # Adjust as needed
    # output_signal = compress_signal(output_signal, compression_ratio)

    # Play and save the output sound
    play_and_save_output(output_signal, samplerate, output_filename=f"output_with_gain_{file_name}")

    # Plot the input and output signals
    # Normalize input signal for comparison
    max_abs_input = np.max(np.abs(data))
    if max_abs_input > 0:
        data_normalized = data / max_abs_input

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data_normalized)
    plt.title("Normalized Input Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(output_signal)
    plt.title(f"Output Signal with Gain {gain}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot RMS amplitude over time
    window_size = 1024
    input_rms = moving_rms(data_normalized, window_size)
    output_rms = moving_rms(output_signal, window_size)

    plt.figure(figsize=(12, 4))
    plt.plot(input_rms, label='Input RMS')
    plt.plot(output_rms, label='Output RMS', alpha=0.7)
    plt.title("RMS Amplitude Over Time")
    plt.xlabel("Sample")
    plt.ylabel("RMS Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Spectrogram of input signal
    f_in, t_in, Sxx_in = spectrogram(data_normalized, fs=samplerate)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t_in, f_in, 10 * np.log10(Sxx_in), shading='gouraud')
    plt.title('Spectrogram of Input Signal')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')

    # Spectrogram of output signal
    f_out, t_out, Sxx_out = spectrogram(output_signal, fs=samplerate)
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_out, f_out, 10 * np.log10(Sxx_out), shading='gouraud')
    plt.title('Spectrogram of Output Signal')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
