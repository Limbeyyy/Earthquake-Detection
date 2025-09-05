# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import json


class Data_Generator:
    """
    Yields pairs: (waveform, spectrogram). Use .flat_sample() wrapper when you
    want a flat tuple for TF dataset: (refer_wave, refer_spec, pos_wave, pos_spec, ...)
    """

    def __init__(self, data_frame, seed=None,
                 spec_frame_length=256, spec_frame_step=128,
                 time_bins=32, freq_bins=65):
        # NOTE: defaults chosen smaller than before to reduce memory
        self.data_frame = data_frame
        self.seed = seed
        self.spec_frame_length = spec_frame_length
        self.spec_frame_step = spec_frame_step
        self.time_bins = time_bins
        self.freq_bins = freq_bins
        if self.seed is not None:
            np.random.seed(self.seed)

    @staticmethod
    def _normalize_signal(signal):
        mean = tf.math.reduce_mean(signal)
        std = tf.math.reduce_std(signal)
        return tf.cond(
            tf.math.equal(std, 0.0),
            lambda: signal,
            lambda: (signal - mean) / std
        )

    def _waveform_to_spectrogram(self, waveform):
        """
        waveform: 1D tensor [signal_length]
        returns: resized spectrogram [time_bins, freq_bins, 1]
        """
        spec = tf.signal.stft(
            waveform,
            frame_length=self.spec_frame_length,
            frame_step=self.spec_frame_step,
            fft_length=self.spec_frame_length
        )
        spec = tf.abs(spec)  # magnitude spectrogram
        # ensure channels-last and resize to fixed dims
        spec = tf.image.resize(spec[..., tf.newaxis], [self.time_bins, self.freq_bins])
        return spec

    def get_next_record(self):
        """
        Yields nested tuple of pairs:
        ((refer_wave, refer_spec), (pos_wave, pos_spec), (neg_wave, neg_spec), (sil_wave, sil_spec))
        Each waveform has shape (signal_length, 1), each spec (time_bins, freq_bins, 1)
        """
        while True:
            idx = np.random.choice(len(self.data_frame))
            refer, pos, neg, sil = self.data_frame["data"].iloc[idx]

            # normalize and keep as column vector for Conv1D (channel-last)
            refer = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(refer, tf.float32)), -1)
            pos = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(pos, tf.float32)), -1)
            neg = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(neg, tf.float32)), -1)
            sil = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(sil, tf.float32)), -1)

            # spectrograms from the squeezed waveform
            refer_spec = self._waveform_to_spectrogram(tf.squeeze(refer, -1))
            pos_spec = self._waveform_to_spectrogram(tf.squeeze(pos, -1))
            neg_spec = self._waveform_to_spectrogram(tf.squeeze(neg, -1))
            sil_spec = self._waveform_to_spectrogram(tf.squeeze(sil, -1))

            yield ((refer, refer_spec),
                   (pos, pos_spec),
                   (neg, neg_spec),
                   (sil, sil_spec))

    def get_next_test(self):
        """
        Same as get_next_record but yields additional metadata at the end.
        """
        while True:
            idx = np.random.choice(len(self.data_frame))
            refer, pos, neg, sil = self.data_frame["data"].iloc[idx]
            typpe_str = self.data_frame["type"].iloc[idx]
            refer_names, pos_names, neg_names = self.data_frame["choices"].iloc[idx]

            refer = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(refer, tf.float32)), -1)
            pos = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(pos, tf.float32)), -1)
            neg = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(neg, tf.float32)), -1)
            sil = tf.expand_dims(self._normalize_signal(tf.convert_to_tensor(sil, tf.float32)), -1)

            refer_spec = self._waveform_to_spectrogram(tf.squeeze(refer, -1))
            pos_spec = self._waveform_to_spectrogram(tf.squeeze(pos, -1))
            neg_spec = self._waveform_to_spectrogram(tf.squeeze(neg, -1))
            sil_spec = self._waveform_to_spectrogram(tf.squeeze(sil, -1))

            yield ((refer, refer_spec),
                   (pos, pos_spec),
                   (neg, neg_spec),
                   (sil, sil_spec),
                   typpe_str,
                   refer_names, pos_names, neg_names)

    # Convenience helper if you want to convert one nested sample to a flat tuple:
    @staticmethod
    def flat_sample(nested_sample):
        """
        nested_sample = ((refer, refer_spec), (pos, pos_spec), (neg, neg_spec), (sil, sil_spec))
        returns:
          (refer, refer_spec, pos, pos_spec, neg, neg_spec, sil, sil_spec)
        """
        (refer, refer_spec), (pos, pos_spec), (neg, neg_spec), (sil, sil_spec) = nested_sample
        return (refer, refer_spec, pos, pos_spec, neg, neg_spec, sil, sil_spec)


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
