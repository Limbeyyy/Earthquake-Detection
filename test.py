# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
from tensorflow import keras
from joblib import load

import config
from Library.model import ReduceMeanTime, ReduceMeanDepthTranspose


# ---------------- PATHS -----------------
EMBEDDING_MODEL_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\output 242317\train_dual\Embedding_Network_dual.keras"
SVM_MODEL_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\output 251154\test_classification\best_svm_model.joblib"

CSV_FILE = r"C:\Users\Hp\Desktop\Earthquake-Detection\earthquake_data_2025-08-25_12-21-23.csv"

signal_length = config.SIGNAL_LENGTH
SAVE_DIR = os.path.join(config.OUTPUT_PATH, "real_time_predictions")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD EMBEDDING MODEL -----------------
print("Loading trained embedding network...")
embedding_model = keras.models.load_model(
    EMBEDDING_MODEL_PATH,
    compile=False,
    custom_objects={"ReduceMeanTime": ReduceMeanTime,
                    "ReduceMeanDepthTranspose": ReduceMeanDepthTranspose}
)

# Embedding extractor (last hidden layer)
embedding_layer_model = keras.Model(
    inputs=embedding_model.inputs,
    outputs=embedding_model.layers[-2].output
)

# Final scalar output extractor
final_output_model = keras.Model(
    inputs=embedding_model.inputs,
    outputs=embedding_model.layers[-1].output
)

print("✅ Embedding and final output models ready.")

# ---------------- LOAD SVM MODEL -----------------
print("Loading trained SVM model...")
best_svm = load(SVM_MODEL_PATH)
print("✅ SVM model loaded successfully.")

# ---------------- PREPROCESS FUNCTIONS -----------------
def fix_length(arr, target_len=signal_length):
    """Pad or truncate waveform to fixed length."""
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), mode="constant")
    else:
        return arr

def preprocess_waveform(batch_data):
    """Detrend + normalize waveform."""
    batch_data = scipy.signal.detrend(batch_data)
    batch_data = (batch_data - np.mean(batch_data)) / (np.std(batch_data) + 1e-8)
    return batch_data

def waveform_to_spectrogram(waveform, time_bins=32, freq_bins=65):
    spec = tf.signal.stft(waveform, frame_length=256, frame_step=128, fft_length=256)
    spec = tf.abs(spec)
    spec = tf.image.resize(spec[..., tf.newaxis], [time_bins, freq_bins])
    return spec.numpy()

# ---------------- REAL-TIME PREDICTION LOOP -----------------
last_index = 0
buffer = []
confidence_window = []
WINDOW_SIZE = 5

label_name_map = {0: "explosion", 1: "earthquake", 2: "noise"}

print(f"🔄 Starting real-time prediction loop on {CSV_FILE}...")

try:
    while True:
        df = pd.read_csv(CSV_FILE)
        new_rows = df.iloc[last_index:]
        num_new = len(new_rows)

        if num_new > 0:
            buffer.extend(new_rows['vibration(10 decimal places)'].astype(np.float32).tolist())
            last_index += num_new

        while len(buffer) >= signal_length:
            batch_data = np.array(buffer[:signal_length], dtype=np.float32)
            buffer = buffer[signal_length:]

            # --- Preprocessing ---
            batch_data = preprocess_waveform(batch_data)

            # --- Dual-input tensors ---
            refer_wave = np.expand_dims(batch_data, axis=(0, -1))
            refer_spec = waveform_to_spectrogram(batch_data)[np.newaxis, ...]

            # --- Extract features ---
            final_val = final_output_model([refer_wave, refer_spec]).numpy().reshape(1, -1)

            # --- Predict with SVM ---
            y_pred = best_svm.predict(final_val)[0]
            y_prob = getattr(best_svm, "predict_proba", None)

            if y_prob is not None:
                confidence = np.max(best_svm.predict_proba(final_val))
            else:
                confidence = 1.0  # fallback if SVM has no probas

            # --- Smoothed confidence ---
            confidence_window.append(confidence)
            if len(confidence_window) > WINDOW_SIZE:
                confidence_window.pop(0)
            smoothed_conf = sum(confidence_window) / len(confidence_window)
            print(f"\nBatch Prediction: {label_name_map[y_pred]}")


               # --- Save prediction ---
            import csv
            import json
            import os

            PREDICTION_CSV = "prediction_batch.csv"
            PREDICTION_JSON = "prediction_batch.json"

            # Read CSV and convert to JSON
            with open(PREDICTION_CSV, mode="r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Save as JSON
            with open(PREDICTION_JSON, mode="w") as f:
                json.dump(rows, f, indent=4)

            



        print(f"\rWaiting for more data... buffer size: {len(buffer)}", end='', flush=True)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n Prediction loop stopped manually.")
