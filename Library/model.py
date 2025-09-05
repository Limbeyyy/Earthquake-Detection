# -*- coding: utf-8 -*-
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import (Conv1D, Conv2D, MaxPool1D, MaxPool2D,
                                     GlobalMaxPooling1D, GlobalMaxPooling2D,
                                     BatchNormalization, Activation, Dropout,
                                     Concatenate, Layer, GaussianNoise)
from tensorflow.keras.regularizers import l2
from Library.metrics import wasserstein_1D, euclidean
import config
from keras.saving import register_keras_serializable

L2_REG = 3  # Regularization coefficient
GAUSSIAN_STDDEV = 0.03  # Standard deviation of Gaussian noise

# ---------------- Custom Layers ----------------
@register_keras_serializable()
class ReduceMeanTime(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-2, keepdims=True)

@register_keras_serializable()
class ReduceMeanDepthTranspose(Layer):
    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        return tf.transpose(x, perm=[0, 2, 1])

@register_keras_serializable()
class TransposeLayer(Layer):
    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 1])

# ---------------- Unified Gene Parser ----------------
def parse_genes_unified(chromosome):
    conv_params = [round(chromosome[0])]
    conv_params.extend([1 if chromosome[i] > 0 else 0 for i in range(1, 6)])
    pooling_params = [round(chromosome[6]), round(chromosome[7])]

    mlp_params = []
    if round(chromosome[9]) <= 0:
        mlp_params.append(round(chromosome[8]))
    elif round(chromosome[10]) <= 0:
        mlp_params.extend([round(chromosome[8]), round(chromosome[9])])
    else:
        mlp_params.extend([round(chromosome[8]), round(chromosome[9]), round(chromosome[10])])
    
    return conv_params, pooling_params, mlp_params

# ---------------- Gaussian Noise Layer for Embeddings ----------------
@register_keras_serializable()
class EmbeddingNoise(Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs
    
# ---------------- Embedding Utilities ----------------
def transform_embedding(features, params):
    x = features
    x = keras.layers.Dense(256, activation=config.ACTIVATION,
                           kernel_regularizer=l2(L2_REG))(x)
    x = Dropout(0.4)(x)
    for i, units in enumerate(params):
        is_last = (i == len(params) - 1)
        x = keras.layers.Dense(units=units,
                               activation=config.ACTIVATION if not is_last else None,
                               kernel_regularizer=l2(L2_REG))(x)
        if not is_last:
            x = Dropout(0.4)(x)
    return x

# ---------------- Conv1D Feature Builder ----------------
def conv_stride(inputs, kernel_num, kernel_size, stride_conv, pool_size, stride_pool,
                activation="relu", conv_depth=4, base_dropout=0.2):
    x = inputs
    filters_1 = max(8, int(kernel_num))
    x = Conv1D(filters_1, kernel_size=kernel_size, strides=stride_conv,
               padding='same', kernel_regularizer=l2(L2_REG))(x)
    x = Activation(activation)(x)
    x = Dropout(base_dropout)(x)
    for i in range(1, conv_depth):
        filters = int(filters_1 * (2 ** i))
        k = max(3, kernel_size - 2 * i)
        x = Conv1D(filters, kernel_size=k, strides=1, padding='same',
                   kernel_regularizer=l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(min(0.5, base_dropout + 0.1 * i))(x)
    _merge_time = ReduceMeanTime()(x)
    _merge_depth = ReduceMeanDepthTranspose()(x)
    _merged = Concatenate(axis=-1)([_merge_time, _merge_depth])
    _merged = TransposeLayer()(_merged)
    output = MaxPool1D(pool_size=pool_size, strides=stride_pool,
                       padding='same', data_format='channels_first')(_merged)
    return output

def build_features(inputs, input_size, conv_params, pooling_params):
    spacial_resolution = round(input_size / conv_params[0])
    feature_list = []
    CONV_DEPTH = 3
    BASE_DROPOUT = 0.2
    kernel_sizes = [1, 3, 5, 7, 9]
    for i, flag in enumerate(conv_params[1:]):
        if flag == 1:
            _feature = conv_stride(inputs,
                                   kernel_num=spacial_resolution,
                                   kernel_size=kernel_sizes[i],
                                   stride_conv=conv_params[0],
                                   pool_size=pooling_params[0],
                                   stride_pool=pooling_params[1],
                                   activation=config.ACTIVATION,
                                   conv_depth=CONV_DEPTH,
                                   base_dropout=BASE_DROPOUT)
            feature_list.append(_feature)
    if len(feature_list) == 0:
        x = Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=l2(L2_REG))(inputs)
        x = BatchNormalization()(x)
        x = Activation(config.ACTIVATION)(x)
        return GlobalMaxPooling1D()(x)
    concat_features = Concatenate(axis=1)(feature_list)
    x = Conv1D(128, kernel_size=3, padding='same', activation=config.ACTIVATION,
               kernel_regularizer=l2(L2_REG))(concat_features)
    return GlobalMaxPooling1D()(x)

# ---------------- Conv2D Feature Builder ----------------
def build_features_2d(inputs, input_shape, conv_params, pooling_params):
    feature_list = []
    CONV_DEPTH = 3
    BASE_DROPOUT = 0.2
    kernel_sizes = [(1,1), (3,3), (5,5), (7,7), (9,9)]
    for i, flag in enumerate(conv_params[1:]):
        if flag == 1:
            x = inputs
            for d in range(CONV_DEPTH):
                filters = 32 * (2 ** d)
                k = kernel_sizes[i]
                x = Conv2D(filters, kernel_size=k, strides=(1,1),
                           padding='same', kernel_regularizer=l2(L2_REG))(x)
                x = BatchNormalization()(x)
                x = Activation(config.ACTIVATION)(x)
                x = Dropout(min(0.5, BASE_DROPOUT + 0.1 * d))(x)
            x = MaxPool2D(pool_size=(pooling_params[0], pooling_params[1]), padding='same')(x)
            feature_list.append(x)
    if len(feature_list) == 0:
        x = Conv2D(64, kernel_size=(3,3), padding='same', kernel_regularizer=l2(L2_REG))(inputs)
        x = BatchNormalization()(x)
        x = Activation(config.ACTIVATION)(x)
        return GlobalMaxPooling2D()(x)
    concat_features = Concatenate(axis=-1)(feature_list)
    x = Conv2D(128, kernel_size=(3,3), padding='same', activation=config.ACTIVATION,
               kernel_regularizer=l2(L2_REG))(concat_features)
    return GlobalMaxPooling2D()(x)

# ---------------- Dual-Input Embedding Network with Gaussian Noise ----------------
def Embedding_Network_dual(input_shape_wave, input_shape_spec, chromosome, noise_std=GAUSSIAN_STDDEV):
    conv_params, pooling_params, mlp_params = parse_genes_unified(chromosome)
    
    # Waveform branch
    input_wave = keras.Input(shape=input_shape_wave + (1,), name="waveform")
    features_wave = build_features(input_wave, input_shape_wave[0], conv_params, pooling_params)
    
    # Spectrogram branch
    input_spec = keras.Input(shape=input_shape_spec + (1,), name="spectrogram")
    features_spec = build_features_2d(input_spec, input_shape_spec, conv_params, pooling_params)
    
    # Merge embeddings
    merged = Concatenate()([features_wave, features_spec])
    codes = transform_embedding(merged, mlp_params)

    # Add Gaussian noise
    codes = EmbeddingNoise(stddev=noise_std)(codes)
    
    return keras.Model(inputs=[input_wave, input_spec], outputs=codes, name="embedding_dual")

# ---------------- Dual-Input Contrastive Network ----------------
def Contrastive_Network_dual(input_wave_size, input_spec_shape, embedding_model):
    inputs = {}
    for name in ["refer", "positive", "negative", "silence"]:
        inputs[name+"_wave"] = keras.Input(shape=(input_wave_size,1), name=name+"_wave")
        inputs[name+"_spec"] = keras.Input(shape=input_spec_shape + (1,), name=name+"_spec")
    
    emb = {}
    for name in ["refer", "positive", "negative", "silence"]:
        emb[name] = embedding_model([inputs[name+"_wave"], inputs[name+"_spec"]])
    
    return keras.Model(
        inputs=list(inputs.values()),
        outputs=[emb["refer"], emb["positive"], emb["negative"], emb["silence"]]
    )

# ---------------- Contrastive Model ----------------
class Contrastive_Model(keras.Model):
    def __init__(self, Contrastive_network, batch_size, loss_tracker, feature_distance,
                 metric_acc=None, margin=1, alpha=1):
        super().__init__()
        self.Contrastive_network = Contrastive_network
        self.margin = margin
        self.alpha = alpha
        self.loss_tracker = loss_tracker
        self.batch_size = batch_size
        self.metric_acc = metric_acc
        self.feature_distance = feature_distance
        if feature_distance == "Wasserstein":
            self.distance_func = wasserstein_1D
        elif feature_distance == "Euclidean":
            self.distance_func = euclidean
        else:
            raise ValueError("Invalid feature distance")

    def _compute_distance(self, inputs):
        emb_ref, emb_pos, emb_neg, emb_sil = self.Contrastive_network(inputs)
        return (self.distance_func(emb_ref, emb_pos),
                self.distance_func(emb_ref, emb_neg),
                self.distance_func(emb_ref, emb_sil))

    def _compute_loss(self, pos, neg, sil):
        loss1 = tf.maximum(pos - neg + self.margin, 0.0)
        loss2 = tf.maximum(pos - self.alpha * sil + self.margin, 0.0)
        return loss1 + loss2

    def _compute_acc(self, pos, neg, sil):
        y_pred = pos < neg
        y_true = tf.ones_like(neg, dtype=tf.bool)
        if self.metric_acc:
            self.metric_acc.update_state(y_true, y_pred)
            return self.metric_acc.result()
        else:
            return tf.reduce_mean(tf.cast(y_pred, tf.float32))

    def call(self, inputs):
        return self._compute_distance(inputs)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            pos, neg, sil = self._compute_distance(inputs)
            loss = self._compute_loss(pos, neg, sil)
        grads = tape.gradient(loss, self.Contrastive_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.Contrastive_network.trainable_variables))
        self.loss_tracker.update_state(loss)
        acc = self._compute_acc(pos, neg, sil)
        return {"loss": self.loss_tracker.result(), "acc": acc}

    def test_step(self, inputs):
        pos, neg, sil = self._compute_distance(inputs)
        loss = self._compute_loss(pos, neg, sil)
        self.loss_tracker.update_state(loss)
        acc = self._compute_acc(pos, neg, sil)
        return {"loss": self.loss_tracker.result(), "acc": acc}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metric_acc] if self.metric_acc else [self.loss_tracker]