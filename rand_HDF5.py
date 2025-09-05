

# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import parallel_backend
import config
from Library.model import ReduceMeanTime, ReduceMeanDepthTranspose

# ---------------- PATHS -----------------
EMBEDDING_MODEL_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\output 312031\train_dual\Embedding_Network_dual.keras"
NOISE_FILE = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\df_noise_test.csv"
EVENT_FILE = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\df_test.csv"
SAVE_DIR = os.path.join(config.OUTPUT_PATH, "test_classification")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- PARAMETERS -----------------
TARGET_LEN = 700
TIME_BINS, FREQ_BINS = 32, 65
BATCH_SIZE = 32

# ---------------- FUNCTIONS -----------------
def fix_length(arr, target_len=TARGET_LEN):
    """Pad or truncate waveform to fixed length."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), mode="constant")
    else:
        return arr

from scipy.signal import spectrogram

def compute_spectrogram(waveform, fs=100, nperseg=128, noverlap=64, time_bins=32, freq_bins=65):
    """Compute log-spectrogram and resize to fixed (time_bins, freq_bins)."""
    f, t, Sxx = spectrogram(waveform, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.log1p(Sxx)  # log scale

    # Convert to tensor and resize to (time_bins, freq_bins)
    Sxx = tf.convert_to_tensor(Sxx[np.newaxis, ..., np.newaxis], dtype=tf.float32)  # (1, f, t, 1)
    Sxx_resized = tf.image.resize(Sxx, (time_bins, freq_bins))  
    return Sxx_resized.numpy().squeeze()

def load_stead_pickle(file_path, label, use_channel="Z_channel"):
    """
    Load STEAD-style pickle file and return waveforms, labels, metadata.
    """
    df = pd.read_pickle(file_path)
    if use_channel not in df.columns:
        raise ValueError(f"Channel {use_channel} not found. Columns: {df.columns.tolist()}")

    traces, specs = [], []
    for _, row in df.iterrows():
        arr = fix_length(row[use_channel])
        traces.append(arr)

        spec = compute_spectrogram(arr, time_bins=TIME_BINS, freq_bins=FREQ_BINS)
        specs.append(spec)

    traces = np.array(traces, dtype=np.float32)[..., np.newaxis]   # (n_samples, TARGET_LEN, 1)
    specs = np.array(specs, dtype=np.float32)[..., np.newaxis]     # (n_samples, TIME_BINS, FREQ_BINS, 1)
    labels = np.full(len(traces), label, dtype=int)
    return traces, specs, labels, df["trace_name"].values

# ---------------- LOAD DATA -----------------
z_noise, s_noise, y_noise, meta_noise = load_stead_pickle(NOISE_FILE, label=0, use_channel="Z_channel")
z_event, s_event, y_event, meta_event = load_stead_pickle(EVENT_FILE, label=1, use_channel="Z_channel")

X_wave = np.vstack([z_noise, z_event])
X_spec = np.vstack([s_noise, s_event])
y = np.concatenate([y_noise, y_event])
metadata = np.concatenate([meta_noise, meta_event])

print("Combined dataset shapes:", X_wave.shape, X_spec.shape, y.shape)

# ---------------- CREATE TF DATASET -----------------
test_ds = tf.data.Dataset.from_tensor_slices((X_wave, X_spec)).batch(BATCH_SIZE)

# ---------------- LOAD EMBEDDING MODEL -----------------
embedding_model = keras.models.load_model(
    EMBEDDING_MODEL_PATH,
    compile=False,
    custom_objects={"ReduceMeanTime": ReduceMeanTime, "ReduceMeanDepthTranspose": ReduceMeanDepthTranspose}
)

embedding_layer_model = keras.Model(inputs=embedding_model.inputs, outputs=embedding_model.layers[-2].output)
final_output_model = keras.Model(inputs=embedding_model.inputs, outputs=embedding_model.layers[-1].output)

# ---------------- EXTRACT EMBEDDINGS -----------------
def extract_embeddings(dataset):
    all_embeddings, all_final_values = [], []
    for batch_wave, batch_spec in dataset:
        batch_embeddings = embedding_layer_model([batch_wave, batch_spec]).numpy()
        batch_final_values = final_output_model([batch_wave, batch_spec]).numpy()
        all_embeddings.append(batch_embeddings)
        all_final_values.append(batch_final_values)
    return np.vstack(all_embeddings), np.vstack(all_final_values)

embeddings_array, final_values_array = extract_embeddings(test_ds)

np.save(os.path.join(SAVE_DIR, "combined_embeddings.npy"), embeddings_array)
np.save(os.path.join(SAVE_DIR, "combined_final_values.npy"), final_values_array)
print("Embeddings and final values saved.")

# ---------------- CLASSIFICATION DATA -----------------
X_final = final_values_array.reshape(-1, 1)
y_true = y

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_true, test_size=0.3, random_state=42, stratify=y_true
)

# ---------------- SINGLE RANDOM FOREST -----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Best hyperparameters you found earlier
best_params_rf = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "log2",
    "bootstrap": True,
    "random_state": 42
}

# Train a single Random Forest with best params
best_rf = RandomForestClassifier(**best_params_rf)
best_rf.fit(X_train, y_train)

# Predictions
y_pred_rf = best_rf.predict(X_test)

# Evaluation
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["noise", "earthquake"]))
print("Confusion Matrix (RF):\n", confusion_matrix(y_test, y_pred_rf))


# ---------------- SVM WITH STANDARDIZATION -----------------
svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=["noise", "earthquake"]))
print("Confusion Matrix (SVM):\n", confusion_matrix(y_test, y_pred_svm))