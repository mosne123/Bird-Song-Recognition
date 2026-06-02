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

# data = np.load("C:\\Users\\Bruger\\OneDrive - Aarhus universitet\\Dokumenter\\UNI\\10. Semester\\TinyML\\Fuglelyde\\bird_dataset_5.npy", allow_pickle=True)
data = np.load("C:\\Users\\Bruger\\OneDrive - Aarhus universitet\\Dokumenter\\UNI\\10. Semester\\TinyML\\Bird_recog\\Bird-Song-Recognition\\Bird_ML_develop\\\\bird_dataset_5.npy", allow_pickle=True)
print(data.shape)
y = data[:, 0]                  # labels
X_audio_load = data[:, 1:].astype(float)   # audio clips
sr = 16000  # sample rate


# number of each class
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Class distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} samples")

#%% load wav file and plot signal and fft to check 50Hz noise

audio_field, sr_field = sf.read("C:\\Users\\Bruger\\OneDrive - Aarhus universitet\\Dokumenter\\UNI\\10. Semester\\TinyML\\TinyML_sampler\\Microphone_PDM-main\\server\\Unknown_vind.wav")
t_axis_field = np.arange(len(audio_field)) / sr_field
audio_field = audio_field / np.max(np.abs(audio_field))


plt.plot(t_axis_field, audio_field)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Signal taken on field")
plt.xlim(0, 5)  # Show only first second for clarity
plt.grid()
plt.show()

# reduce number of fft bins
n_fft = 8000
fft_field = np.fft.rfft(audio_field, n=n_fft)
freqs_field = np.fft.rfftfreq(n_fft, d=1/sr_field)


plt.plot(freqs_field, np.abs(fft_field))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Field Recording")
plt.xlim(0, 80)  
plt.grid()
plt.show()

audio_aarhus, sr_aarhus = sf.read("C:\\Users\\Bruger\\OneDrive - Aarhus universitet\\Dokumenter\\UNI\\10. Semester\\TinyML\\TinyML_sampler\\Microphone_PDM-main\\server\\out\\Unknown_ugle.wav")
t_axis_aarhus = np.arange(len(audio_aarhus)) / sr_aarhus
audio_aarhus = audio_aarhus / np.max(np.abs(audio_aarhus))

plt.plot(t_axis_aarhus, audio_aarhus)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Signal taken in apartment in Aarhus")
plt.xlim(0, 5)  # Show only first second for clarity
plt.grid()
plt.show()

fft_aarhus = np.fft.rfft(audio_aarhus)
freqs_aarhus = np.fft.rfftfreq(len(audio_aarhus), d=1/sr_field)

plt.plot(freqs_aarhus, np.abs(fft_aarhus))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Aarhus Recording")
plt.xlim(0, 70)  
plt.grid()
plt.show()


#%%   preprocessing

X_audio=X_audio_load.copy() 
n_fft = 8000  # resolution
x_lim= 8000

for i in range(len(X_audio)):

    clip = X_audio[i]

    # HP-filter
    b, a = butter(2, 100/(sr/2), btype='high')
    clip = filtfilt(b, a, clip)

    X_audio[i] = clip

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
    features["IQR"] = np.percentile(clip, 75) - np.percentile(clip, 25)
    features["peak_count"] = len(find_peaks(clip)[0])

    # Frequency domain features
    spectrum = np.abs(np.fft.rfft(clip))
    freqs = np.fft.rfftfreq(len(clip), d=1/sr)

    if np.sum(spectrum) > 0:
        features["spectral_centroid"] = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        features["spectral_centroid"] = 0

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
print("\n")

# Print one feature example per class
print("\nOne feature example per class:\n")

for label in df_features["label"].unique():
    row = df_features[df_features["label"] == label].iloc[0]

    print("=" * 50)
    print("Class:", label)
    print("=" * 50)

    for feature_name, value in row.drop("label").items():
        print(f"{feature_name:30s}: {value:.6f}")

    print("\n")


# ML random forrest
X= df_features.drop("label", axis=1)
y = df_features["label"]

# Encode labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42, #42
    stratify=y_encoded
)

rf = RandomForestClassifier( 
    n_estimators=50, #50
    max_depth=5, # 8
    random_state=42 
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Test scoring 
print("Test accuracy:", accuracy_score(y_test, y_pred))

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

#feature importance
importances = rf.feature_importances_
feat_names = X.columns

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

print(feat_imp)

feat_imp.plot(kind="bar", figsize=(10,5), title="Feature Importance")
plt.show()
print(len(feat_names))

#%% test of number of trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

n_trees = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

train_acc = []
test_acc = []

for n in n_trees:

    rf_md = RandomForestClassifier(
        n_estimators=n,
        max_depth=5,      # fixed depth
        random_state=42
    )

    rf_md.fit(X_train, y_train)

    y_train_pred = rf_md.predict(X_train)
    y_test_pred = rf_md.predict(X_test)

    train_acc.append(
        accuracy_score(y_train, y_train_pred)
    )

    test_acc.append(
        accuracy_score(y_test, y_test_pred)
    )

    print(
        f"Trees={n}, "
        f"Train={train_acc[-1]:.4f}, "
        f"Test={test_acc[-1]:.4f}"
    )

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    n_trees,
    train_acc,
    marker='o',
    label="Training Accuracy"
)

plt.plot(
    n_trees,
    test_acc,
    marker='s',
    label="Test Accuracy"
)

plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Trees (max_depth=5)")
plt.grid(True)
plt.legend()

plt.show()

#%% test of max depts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]

train_acc = []
test_acc = []

for d in depths:

    rf_nt = RandomForestClassifier(
        n_estimators=200,
        max_depth=d,
        random_state=42
    )

    rf_nt.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf_nt.predict(X_train)
    y_test_pred = rf_nt.predict(X_test)

    # Accuracies
    train_acc.append(
        accuracy_score(y_train, y_train_pred)
    )

    test_acc.append(
        accuracy_score(y_test, y_test_pred)
    )

    print(
        f"Depth={d}, "
        f"Train={train_acc[-1]:.4f}, "
        f"Test={test_acc[-1]:.4f}"
    )

#%% Plot
depth_labels = [str(d) for d in depths]

plt.figure(figsize=(8,5))

plt.plot(depth_labels, train_acc,
         marker='o',
         label="Training Accuracy")

plt.plot(depth_labels, test_acc,
         marker='s',
         label="Test Accuracy")

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Max Depth (200 Trees)")
plt.grid(True)
plt.legend()

plt.show()

#%% svm model and test 





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

#%% svm model test 



#%% EM learn export

# 1. Convert
import emlearn


wrapper = emlearn.convert(rf, method='inline', dtype='float32')

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





