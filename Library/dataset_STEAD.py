# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import os


class Data_Generator:
    """
    Generates 3-waveform contrastive samples (refer, pos, neg) along with spectrograms.
    Only uses Z-component for spectrograms.
    """

    def __init__(self, data_frame, seed=None, spec_frame_length=256, spec_frame_step=128,
                 time_bins=32, freq_bins=65):
        """
        data_frame: dict with keys "EQ" and "Noise" containing waveform arrays
        """
        self.eq_data = data_frame.get("EQ", [])
        self.noise_data = data_frame.get("Noise", [])
        self.seed = seed
        self.spec_frame_length = spec_frame_length
        self.spec_frame_step = spec_frame_step
        self.time_bins = time_bins
        self.freq_bins = freq_bins
        if self.seed is not None:
            np.random.seed(self.seed)

    # ---------- NORMALIZATION ----------
    @staticmethod
    def _normalize_signal(signal):
        mean = tf.reduce_mean(signal)
        std = tf.math.reduce_std(signal)
        return tf.cond(
            tf.equal(std, 0.0),
            lambda: signal,
            lambda: (signal - mean) / std
        )

    # ---------- WAVEFORM TO SPECTROGRAM ----------
    @staticmethod
    def waveform_to_spectrogram(waveform, target_shape=(32, 65)):
        """
        Converts 1D waveform to spectrogram and resizes to target shape
        """
        waveform = tf.convert_to_tensor(waveform, tf.float32)
        spec = tf.signal.stft(
            waveform,
            frame_length=256,
            frame_step=128,
            fft_length=256
        )
        spec = tf.abs(spec)
        spec = tf.image.resize(spec[..., tf.newaxis], target_shape)
        return spec

    # ---------- SAMPLE GENERATOR ----------
    def get_next_record(self):
        """
        Yields a tuple: ((refer, refer_spec), (pos, pos_spec), (neg, neg_spec))
        """
        while True:
            # randomly sample anchor class
            if np.random.rand() < 0.5:
                # EQ as anchor
                refer = self.eq_data[np.random.randint(len(self.eq_data))]
                pos = self.eq_data[np.random.randint(len(self.eq_data))]
                neg = self.noise_data[np.random.randint(len(self.noise_data))]
            else:
                # Noise as anchor
                refer = self.noise_data[np.random.randint(len(self.noise_data))]
                pos = self.noise_data[np.random.randint(len(self.noise_data))]
                neg = self.eq_data[np.random.randint(len(self.eq_data))]

            # --- Use Z-component if multi-channel ---
            if refer.ndim > 1 and refer.shape[1] >= 3:
                refer = refer[:, 2]
            if pos.ndim > 1 and pos.shape[1] >= 3:
                pos = pos[:, 2]
            if neg.ndim > 1 and neg.shape[1] >= 3:
                neg = neg[:, 2]

            # normalize and expand dims
            refer_wave = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(refer, tf.float32)), -1)
            pos_wave = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(pos, tf.float32)), -1)
            neg_wave = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(neg, tf.float32)), -1)

            # spectrograms
            target_spec_shape = (32, 65)
            refer_spec = self.waveform_to_spectrogram(refer, target_shape=target_spec_shape)
            pos_spec = self.waveform_to_spectrogram(pos, target_shape=target_spec_shape)
            neg_spec = self.waveform_to_spectrogram(neg, target_shape=target_spec_shape)

            yield ((refer_wave, refer_spec),
                   (pos_wave, pos_spec),
                   (neg_wave, neg_spec))

    # ---------- FLATTEN SAMPLE ----------
    @staticmethod
    def flat_sample(nested_sample):
        """
        Flatten a nested sample to: refer, refer_spec, pos, pos_spec, neg, neg_spec
        """
        (refer, refer_spec), (pos, pos_spec), (neg, neg_spec) = nested_sample
        return (refer, refer_spec, pos, pos_spec, neg, neg_spec)


# ----------------- JSON UTILS -----------------
def load_json_data(file_path, id=None):
    with open(file_path, "r") as read_file:
        dataset = json.load(read_file)
    if id is not None:
        return dataset[str(id)]
    return dataset


def save_json_data(file_path, dataset, indent="\t"):
    with open(file_path, "w") as outfile:
        json.dump(dataset, outfile, indent=indent)


def load_embedding_data(data_dir, name):
    return load_json_data(os.path.join(data_dir, name))
