#%% Load dataset

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from scipy.signal import iirnotch, filtfilt, butter

data = np.load("C:\\Users\\Bruger\\OneDrive - Aarhus universitet\\Dokumenter\\UNI\\10. Semester\\TinyML\\Fuglelyde\\bird_dataset_5.npy", allow_pickle=True)
print(data.shape)
y = data[:, 0]                  # labels
X_audio_load = data[:, 1:].astype(float)   # audio clips
sr = 16000  # sample rate

#%%   preprocessing

X_audio=X_audio_load.copy() 
n_fft = 8000  # resolution
x_lim= 8000

for i in range(len(X_audio)):

    clip = X_audio[i]

    # #Notch filter
    for f in [50, 60]:
        b, a = iirnotch(f, 30, sr)
        clip = filtfilt(b, a, clip)

    # # HP-filter
    # b, a = butter(4, 100/(sr/2), btype='high')
    # clip = filtfilt(b, a, clip)

    X_audio[i] = clip

X_audio = X_audio_load.copy()

# # Normalize each clip individually
for i in range(len(X_audio)):

    max_val = np.max(np.abs(X_audio[i]))

    # Avoid divide-by-zero
    if max_val > 0:
        X_audio[i] = X_audio[i] / max_val   

classes = np.unique(y)

plt.figure(figsize=(10,6))

for label in classes:
    
    # Take first sample of this class
    idx = np.where(y == label)[0][0]
    clip = X_audio[idx]

    # FFT
    spectrum = np.abs(np.fft.rfft(clip, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)

    # Normalize (important for comparison)
    spectrum = spectrum / np.max(spectrum + 1e-10)

    plt.plot(freqs, spectrum, label=label)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.title("FFT Comparison per Class")
plt.xlim(0, x_lim)  # Nyquist frequency
plt.legend()
plt.show()



#%% print dataset info
print("Clips:", X_audio.shape)

# Feature extraction definition
bands = [
    (0, 500),
    (500, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 6000),
    (6000, 8000)
]

def extract_features(clip, sr=16000):  # adjust sr if needed
    
    features = {}

    # Time domain features
    features["ABS_mean"] = np.mean(abs(clip))
    #features["std"] = np.std(clip)
    # features["min"] = np.min(clip)
    # features["max"] = np.max(clip)
    # features["ptp"] = np.max(clip)- np.min(clip)
    # features["median"] = np.median(clip)
    # features["mad"] = np.median(np.abs(clip - np.median(clip)))
    features["IQR"] = np.percentile(clip, 75) - np.percentile(clip, 25)
    # features["max_mean"] = np.max(clip)/features["ABS_mean"]
    # RMS
    # features["rms"] = np.sqrt(np.sum(clip**2) / len(clip)) 

    # Zero crossing rate
    # features["zcr"] = np.mean(np.abs(np.diff(np.sign(clip))))

    # Peak count
    features["peak_count"] = len(find_peaks(clip)[0])

    # Shape
    # features["skew"] = stats.skew(clip)
    # features["kurtosis"] = stats.kurtosis(clip)


    # n_fft = 1048
    # spectrum = np.abs(np.fft.rfft(np.clip(clip, -1, 1), n=n_fft))
    # freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    # Frequency domain features
    spectrum = np.abs(np.fft.rfft(clip))
    freqs = np.fft.rfftfreq(len(clip), d=1/sr)

    if np.sum(spectrum) > 0:
        features["spectral_centroid"] = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        features["spectral_centroid"] = 0

    # features["dominant_freq_1"] = freqs[np.argmax(spectrum)]
    # features["dominant_freq_2"] = freqs[np.argsort(spectrum)[-2]]  # Second largest
    # features["dominant_freq_3"] = freqs[np.argsort(spectrum)[-3]]  # Third largest
    # features["dominant_freq_4"] = freqs[np.argsort(spectrum)[-4]]  # Fourth largest
    # features["dominant_freq_5"] = freqs[np.argsort(spectrum)[-5]]  # Fifth largest
    # features["dominant_freq_6"] = freqs[np.argsort(spectrum)[-6]]  # Sixth largest
    # features["dominant_freq_7"] = freqs[np.argsort(spectrum)[-7]]  # Seventh largest
    # features["dominant_freq_8"] = freqs[np.argsort(spectrum)[-8]]  # Eighth largest
    # features["dominant_freq_9"] = freqs[np.argsort(spectrum)[-9]]  # Ninth largest
    # features["dominant_freq_10"] = freqs[np.argsort(spectrum)[-10]]  # Tenth largest

    # Spectral rolloff (frequency below which 85% of the energy is contained)
    # cumulative_energy = np.cumsum(spectrum) / np.sum(spectrum)
    # features["spectral_rolloff"] = freqs[np.where(cumulative_energy >= 0.85)[0][0]]

    # # spectral variance and mean
    # if np.sum(spectrum) > 0:
    #     features["spectral_variance"] = np.sum(((freqs - features["spectral_centroid"])**2) * spectrum) / np.sum(spectrum)
    #     features["spectral_mean"] = np.sum(freqs * spectrum) / np.sum(spectrum)
    # else:
    #     features["spectral_variance"] = 0
    #     features["spectral_mean"] = 0
    
    # spectrum_safe = spectrum + 1e-10  # avoid log(0)
    # features["spectral_flatness"] = np.exp(np.mean(np.log(spectrum_safe))) / np.mean(spectrum_safe)
    
    # features["spectral_bandwidth"] = np.sqrt(features["spectral_variance"])

    for i, (f_low, f_high) in enumerate(bands):

        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]

        band_energy = np.sum(spectrum[idx]**2)

        features[f"band_energy_{f_low}_{f_high}"] = band_energy

    psd = spectrum / np.sum(spectrum)
    features["spectral_entropy"] = -np.sum(psd * np.log2(psd + 1e-10))
   
    return features


# Apply to all clips

feature_list = []

for i, clip in enumerate(X_audio):
    feats = extract_features(clip)
    feats["label"] = y[i]
    feature_list.append(feats)

df_features = pd.DataFrame(feature_list)

# number of each class
# print(df_features["label"].value_counts())
# print(df_features.head())
print("\n")

# ML random forrest

X= df_features.drop("label", axis=1)
y = df_features["label"]

# print("Feature columns:", X.columns)
# print("Number of features:", X.shape[1])
# print(y)

# Encode labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

rf = RandomForestClassifier( 
    n_estimators=50, 
    max_depth=8, # how large do we need it?
    random_state=42 
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Test scoring 
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
# make 80-20 train test split where the train and test set contain some of all classes

cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

#%% feature importance
importances = rf.feature_importances_
feat_names = X.columns

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

print(feat_imp)

feat_imp.plot(kind="bar", figsize=(10,5), title="Feature Importance")
plt.show()
print(len(feat_names))

# %% delete n number of least imortant features and retrain
# n_remove = 5

# rf_2 = RandomForestClassifier( 
#     n_estimators=100, 
#     max_depth=None, 
#     random_state=42 
# )

# least_important = feat_imp.tail(n_remove).index
# print(f"Removing least important features: {list(least_important)}")
# X_reduced = X.drop(columns=least_important)
# X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
#     X_reduced, y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded
# )


# rf_2.fit(X_train_red, y_train_red)
# y_pred_red = rf_2.predict(X_test_red)
# print("Test accuracy after feature removal:", accuracy_score(y_test_red, y_pred_red))

# cm = confusion_matrix(y_test_red, y_pred_red)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix after feature removal")
# plt.show()


# top_n = 15

# plt.figure(figsize=(12,6))

# feat_imp_sorted = feat_imp.sort_values(ascending=False).head(top_n)

# feat_imp_sorted.plot(kind="bar")

# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.title(f"Top {top_n} Feature Importance")

# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()



#%% EM learn export

# 1. Convert
import emlearn


wrapper = emlearn.convert(rf, method='inline')

# 2. Get the C code string
# The 'name' will be the prefix for your C functions
c_code = wrapper.save(name='Bird_recog_model')

# 3. Write to file
with open('Bird_recog_model.h', 'w') as f:
    f.write(c_code)

print("SUCCESS: Bird_recog_model.h is finally ready!")

# print number of lines in the generated C code
with open('Bird_recog_model.h', 'r') as f:
    lines = f.readlines()
    print(f"Number of lines in Bird_recog_model.h: {len(lines)}")



