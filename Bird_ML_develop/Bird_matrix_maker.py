
# %% test 2
import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# PARAMETERS
folder_path = r"C:\Users\Bruger\OneDrive - Aarhus universitet\Dokumenter\UNI\10. Semester\TinyML\TinyML_sampler\Microphone_PDM-main\server\out"
clip_duration = 0.5
sens = 3
peak_distance_sec = 0.4

all_data = []
features = []   

# Loop through all wav files
for file in os.listdir(folder_path):
    if file.endswith(".wav"):

        filepath = os.path.join(folder_path, file)
        label = file.split("_")[0]

        y, sr = sf.read(filepath)

        if len(y.shape) > 1:
            y = y.mean(axis=1)

        envelope = np.abs(y)
        window = int(sr * 0.01)
        envelope = np.convolve(envelope, np.ones(window)/window, mode='same')

        noise_level = np.mean(envelope)
        noise_std = np.std(envelope)

        adaptive_height = noise_level + sens * noise_std 

        peaks, _ = find_peaks(
            envelope,
          height=adaptive_height,
          distance=int(sr * peak_distance_sec)
        )

        clip_samples = int(sr * clip_duration)

        print(file, "peaks:", len(peaks))  # ✅ DEBUG

        for peak in peaks:
            start = max(0, peak - clip_samples // 2)
            end = min(len(y), peak + clip_samples // 2)

            clip = y[start:end]

            if len(clip) < clip_samples:
                clip = np.pad(clip, (0, clip_samples - len(clip)))

            # -------- RAW DATA --------
            row = np.concatenate(([label], clip))
            all_data.append(row)

            # -------- FEATURES --------
            peak_to_peak = np.max(clip) - np.min(clip)
            variance = np.var(clip)

            features.append({
                "label": label,
                "peak_to_peak": peak_to_peak,
                "variance": variance
            })

print("Total clips:", len(all_data))
print("Total features:", len(features))
# Print class distribution in features
feature_labels = [f["label"] for f in features]
unique, counts = np.unique(feature_labels, return_counts=True)
print("Feature class distribution:")
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")    



# RAW clips
df_raw = pd.DataFrame(all_data)
df_raw.to_csv("bird_dataset_5.csv", index=False)
np.save("bird_dataset_5.npy", df_raw.values)

# FEATURES
df_feat = pd.DataFrame(features)
df_feat.to_csv("bird_dataset_with_features_5.csv", index=False)
np.save("bird_dataset_with_features_5.npy", df_feat.values)

print("Datasets saved!")

df_feat["peak_to_peak"] = df_feat["peak_to_peak"].astype(float)
df_feat["variance"] = df_feat["variance"].astype(float)

plt.figure(figsize=(8,6))
plt.scatter(df_feat["peak_to_peak"], df_feat["variance"], alpha=0.5)
plt.xlabel("Peak-to-Peak")
plt.ylabel("Variance")
plt.title("Clip Feature Distribution")
plt.show()
print("Features plotted!")



# %%
