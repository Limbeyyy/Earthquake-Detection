# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

import config
from Library.model import ReduceMeanTime, ReduceMeanDepthTranspose

# ---------------- PATHS -----------------
EMBEDDING_MODEL_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\output 242317\train_dual\Embedding_Network_dual.keras"
TEST_JSON_FILE = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\UUSS 3C data, test n2222 r100.json"
METADATA_CSV_FILE = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\UUSS 3C meta, test snr_db, n2222.csv"
SAVE_DIR = os.path.join(config.OUTPUT_PATH, "test_classification")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD TEST JSON -----------------
print("Loading test JSON data...")
with open(TEST_JSON_FILE, "r") as f:
    test_json = json.load(f)

print("Original JSON distribution:", Counter([r["type"] for r in test_json.values()]))

# Map JSON index to record
json_index_map = {int(k): v for k, v in test_json.items()}

# ---------------- LOAD METADATA -----------------
metadata = pd.read_csv(METADATA_CSV_FILE)
if "index_o" not in metadata.columns:
    raise ValueError("Metadata CSV must contain 'index_o' column to match JSON keys.")
metadata = metadata.set_index("index_o")


# ---------------- ALIGN JSON WITH METADATA -----------------
waveforms = []
labels = []

SAMPLING_RATE = 100
WINDOW_TIME = 7
TARGET_LEN = SAMPLING_RATE * WINDOW_TIME

def fix_length(arr, target_len=TARGET_LEN):
    if len(arr) > target_len:
        return arr[:target_len]  # truncate
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), mode="constant")
    else:
        return arr
    

import json
first_key = next(iter(json_index_map))
print("First JSON key:", first_key)
print("First JSON entry:\n", json.dumps(json_index_map[first_key], indent=2)[:1000])  # show first 1000 chars

# Align JSON records to CSV order
for idx in metadata.index:
    if idx in json_index_map:
        entry = json_index_map[idx]
        waveform_z = np.array(entry["Z"])  # Z-component only
        waveform_z = fix_length(waveform_z)         
        waveforms.append(waveform_z)
        labels.append(metadata.loc[idx, "label"])

waveforms = np.array(waveforms)                 # (n_samples, signal_length)
labels = np.array(labels)
waveforms = waveforms[..., np.newaxis]         # (n_samples, signal_length, 1)
       
print(f"Aligned records between CSV and JSON")
# Placeholder spectrograms
time_bins, freq_bins = 32, 65
spectrograms = np.zeros((len(waveforms), time_bins, freq_bins, 1), dtype=np.float32)

# Create TensorFlow dataset
test_ds = tf.data.Dataset.from_tensor_slices((waveforms, spectrograms)).batch(1)

# ---------------- LOAD EMBEDDING MODEL -----------------
print("Loading trained embedding network...")
embedding_model = keras.models.load_model(
    EMBEDDING_MODEL_PATH,
    compile=False,
    custom_objects={"ReduceMeanTime": ReduceMeanTime, "ReduceMeanDepthTranspose": ReduceMeanDepthTranspose}
)
embedding_model.summary()

# ---------------- CREATE EMBEDDING & FINAL OUTPUT MODELS -----------------
embedding_layer_model = keras.Model(
    inputs=embedding_model.inputs,
    outputs=embedding_model.layers[-2].output
)
final_output_model = keras.Model(
    inputs=embedding_model.inputs,
    outputs=embedding_model.layers[-1].output
)

# ---------------- EXTRACT EMBEDDINGS & FINAL VALUES -----------------
embeddings_list = []
final_values_list = []

for wave, spec in test_ds:
    embeddings = embedding_layer_model([wave, spec])
    final_val = final_output_model([wave, spec])
    embeddings_list.append(embeddings.numpy())
    final_values_list.append(final_val.numpy())

embeddings_array = np.vstack(embeddings_list)
final_values_array = np.vstack(final_values_list)

# Save embeddings and final scalar values
np.save(os.path.join(SAVE_DIR, "test_embeddings.npy"), embeddings_array)
np.save(os.path.join(SAVE_DIR, "test_final_values.npy"), final_values_array)
print("Embeddings and final values saved.")

# ---------------- CLASSIFICATION -----------------
# ---------------- TRAIN/TEST SPLIT -----------------
# Use embeddings instead of final scalar values
X = embeddings_array   # (n_samples, embedding_dim)
y = labels

from scipy.stats import norm
# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Compute class-wise mean and std
class_stats = {}
for cls in np.unique(y_train):
    X_cls = X_train[y_train == cls]
    mean_vec = np.mean(X_cls, axis=0)
    std_vec = np.std(X_cls, axis=0) + 1e-6   # avoid divide by zero
    class_stats[cls] = (mean_vec, std_vec)

# Predict using Gaussian likelihood
y_pred = []
for x in X_test:
    likelihoods = {}
    for cls, (mean_vec, std_vec) in class_stats.items():
        # Independent Gaussian assumption per feature
        pdfs = norm.pdf(x, mean_vec, std_vec)
        log_likelihood = np.sum(np.log(pdfs + 1e-12))  # sum log-probabilities
        likelihoods[cls] = log_likelihood
    pred_cls = max(likelihoods, key=likelihoods.get)
    y_pred.append(pred_cls)

# Detect labels present in test set 
present_labels = np.unique(y_test) 
print("Labels present in test set:", present_labels)

y_pred = np.array(y_pred)
label_name_map = {0: 'earthquake', 1: 'explosion', 2: 'noise'} 
present_label_names = [label_name_map[l] for l in present_labels]


# Save predictions 
predictions_df = pd.DataFrame({ "index": range(len(y_test)), "true_label": y_test, "predicted_label": y_pred }) 
predictions_df.to_csv(os.path.join(SAVE_DIR, "test_predictions_svm_embeddings.csv"), index=False)

 # ---------------- METRICS ----------------- 
print("\nClassification Report:") 
print(classification_report( y_test, y_pred, labels=present_labels, target_names=present_label_names )) 


print("\nConfusion Matrix:") 
print(confusion_matrix(y_test, y_pred, labels=present_labels))