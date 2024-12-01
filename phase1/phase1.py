import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

def process_audio(file_name):
    # Step 3.1: Read file
    data, samplerate = sf.read(file_name)

    # Step 3.2: Convert to mono if stereo
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)

    # Step 3.3: Play sound
    # sd.play(data, samplerate)
    # sd.wait()

    # Step 3.4: Write to a new file
    sf.write("processed_" + file_name, data, samplerate)

    # Step 3.5: Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title("Waveform of " + file_name)
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Step 3.6: Downsample if necessary
    if samplerate != 16000:
        data = resample(data, int(len(data) * 16000 / samplerate))
        samplerate = 16000

    # Step 3.7: Generate and play 1 kHz cosine
    # t = np.linspace(0, len(data) / samplerate, len(data), endpoint=False)
    # cos_wave = np.cos(2 * np.pi * 1000 * t)
    # sd.play(cos_wave, samplerate)
    # sd.wait()

    # Plot two cycles of 1 kHz cosine wave
    # plt.figure(figsize=(10, 4))
    # plt.plot(t[:int(samplerate/500)], cos_wave[:int(samplerate/500)])
    # plt.title("1 kHz Cosine Wave (Two Cycles)")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.show()

# Function call
process_audio("Whispering_audio.wav")
