

#%% Make M4A-file into wav-file
from pydub import AudioSegment

audio = AudioSegment.from_file("Krage_kald.m4a", format="m4a")
audio.export("Krage_kald.wav", format="wav")




#%% Split wav-file into small 0.5 sek segments

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

# Load audio
y, sr = sf.read("Krage_kald.wav")

# If stereo → convert to mono
if len(y.shape) > 1:
    y = y.mean(axis=1)

# Envelope (absolute amplitude)
envelope = np.abs(y)

# Smooth envelope
window = int(sr * 0.01)  # 10 ms smoothing
envelope = np.convolve(envelope, np.ones(window)/window, mode='same')

# Detect peaks %% Chose peak height to select how sensitive it is
peaks, _ = find_peaks(envelope, height=0.2, distance=int(sr*0.2))

clip_samples = int(sr * 0.5)  # 0.5 second clips

for i, peak in enumerate(peaks):

    start = peak - clip_samples//2
    end = peak + clip_samples//2

    start = max(start, 0)
    end = min(end, len(y))

    clip = y[start:end]

    sf.write(f"Krage_kald_test_clip_{i}.wav", clip, sr)

print("Clips created:", len(peaks))


