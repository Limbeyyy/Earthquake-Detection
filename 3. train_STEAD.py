# -*- coding: utf-8 -*-
import os
import math
import logging
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

import configs
from Library.model_STEAD import Embedding_Network_dual, Contrastive_Network_dual, Contrastive_Model
from Library.dataset_STEAD import Data_Generator
from Library import utils

# ----------------- PATHS & LOGGING -----------------
SAVE_DIR = os.path.join(configs.OUTPUT_PATH, "train_dual")
os.makedirs(SAVE_DIR, exist_ok=True)

CONTRASTIVE_SAVE_PATH = os.path.join(SAVE_DIR, f"Contrastive_Network_{configs.FEATURE_DISTANCE}")
EMBEDDING_SAVE_PATH = os.path.join(SAVE_DIR, f"Embedding_Network_dual.keras")
os.makedirs(CONTRASTIVE_SAVE_PATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(SAVE_DIR, "Train_log.txt"),
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s]: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode='w'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# ----------------- LOAD HDF5 FILES -----------------
logger.info("Loading HDF5 files (waveforms streamed on-the-fly)...")
eq_h5 = h5py.File(configs.EQ_FILE_PATH, "r")
noise_h5 = h5py.File(configs.NOISE_FILE_PATH, "r")
eq_len = eq_h5["traces"].shape[0]
noise_len = noise_h5["traces"].shape[0]
logger.info(f"EQ samples: {eq_len}, Noise samples: {noise_len}")

# ----------------- HDF5 GENERATOR -----------------
class HDF5WaveGenerator:
    def __init__(self, eq_h5, noise_h5, seed=None, signal_length=700):
        self.eq_h5 = eq_h5
        self.noise_h5 = noise_h5
        self.eq_len = eq_h5["traces"].shape[0]
        self.noise_len = noise_h5["traces"].shape[0]
        self.seed = seed
        self.signal_length = signal_length
        if seed is not None:
            np.random.seed(seed)

    def get_next_record(self):
        while True:
            refer_idx, pos_idx = np.random.randint(0, self.eq_len, 2)
            neg_idx = np.random.randint(0, self.noise_len)

            refer = self.eq_h5["traces"][refer_idx][:self.signal_length]
            pos = self.eq_h5["traces"][pos_idx][:self.signal_length]
            neg = self.noise_h5["traces"][neg_idx][:self.signal_length]

            # --- Z component ---
            if refer.ndim > 1: refer = refer[:, 2]
            if pos.ndim > 1: pos = pos[:, 2]
            if neg.ndim > 1: neg = neg[:, 2]

            # normalize and expand dims for Conv1D
            refer_wave = tf.expand_dims(Data_Generator._normalize_signal(tf.convert_to_tensor(refer, tf.float32)), -1)
            pos_wave = tf.expand_dims(Data_Generator._normalize_signal(tf.convert_to_tensor(pos, tf.float32)), -1)
            neg_wave = tf.expand_dims(Data_Generator._normalize_signal(tf.convert_to_tensor(neg, tf.float32)), -1)

            # spectrograms
            target_spec_shape = (32, 65)
            refer_spec = Data_Generator.waveform_to_spectrogram(refer, target_spec_shape)
            pos_spec = Data_Generator.waveform_to_spectrogram(pos, target_spec_shape)
            neg_spec = Data_Generator.waveform_to_spectrogram(neg, target_spec_shape)

            yield ((refer_wave, refer_spec),
                   (pos_wave, pos_spec),
                   (neg_wave, neg_spec))

    @staticmethod
    def flat_sample(nested_sample):
        (refer, refer_spec), (pos, pos_spec), (neg, neg_spec) = nested_sample
        return (refer, refer_spec, pos, pos_spec, neg, neg_spec)

# ----------------- CREATE DATASETS -----------------
train_generator = HDF5WaveGenerator(eq_h5, noise_h5, seed=configs.SEED, signal_length=700)
val_generator = HDF5WaveGenerator(eq_h5, noise_h5, seed=configs.SEED, signal_length=700)

BATCH_SIZE_SAFE = min(configs.BATCH_SIZE, 8)

def flat_dataset_generator(gen):
    for nested in gen.get_next_record():
        yield HDF5WaveGenerator.flat_sample(nested)

train_ds = tf.data.Dataset.from_generator(
    lambda: flat_dataset_generator(train_generator),
    output_signature=(
        tf.TensorSpec(shape=(700,1), dtype=tf.float32),
        tf.TensorSpec(shape=(32,65,1), dtype=tf.float32),
        tf.TensorSpec(shape=(700,1), dtype=tf.float32),
        tf.TensorSpec(shape=(32,65,1), dtype=tf.float32),
        tf.TensorSpec(shape=(700,1), dtype=tf.float32),
        tf.TensorSpec(shape=(32,65,1), dtype=tf.float32),
    )
).shuffle(configs.BUFFER_SIZE).batch(BATCH_SIZE_SAFE).prefetch(configs.AUTO)

val_ds = tf.data.Dataset.from_generator(
    lambda: flat_dataset_generator(val_generator),
    output_signature=(
        tf.TensorSpec(shape=(700,1), dtype=tf.float32),
        tf.TensorSpec(shape=(32,65,1), dtype=tf.float32),
        tf.TensorSpec(shape=(700,1), dtype=tf.float32),
        tf.TensorSpec(shape=(32,65,1), dtype=tf.float32),
        tf.TensorSpec(shape=(700,1), dtype=tf.float32),
        tf.TensorSpec(shape=(32,65,1), dtype=tf.float32),
    )
).batch(BATCH_SIZE_SAFE).prefetch(configs.AUTO)

# ----------------- BUILD EMBEDDING & CONTRASTIVE NETWORKS -----------------
spec_shape = (32, 65)

embedding_network = Embedding_Network_dual(
    input_shape_wave=(700,),
    input_shape_spec=spec_shape,
    chromosome=configs.CHROMOSOME
)
embedding_network.summary(print_fn=logger.info)

contrastive_network = Contrastive_Network_dual(
    input_wave_size=700,
    input_spec_shape=spec_shape,
    embedding_model=embedding_network
)

contrastive_model = Contrastive_Model(
    Contrastive_network=contrastive_network,
    batch_size=BATCH_SIZE_SAFE,
    loss_tracker=keras.metrics.Mean(name="loss"),
    feature_distance=configs.FEATURE_DISTANCE,
    metric_acc=keras.metrics.Accuracy(name="acc"),
    margin=configs.MARGIN,
    
)

build_input_shapes = [
    (None, 700, 1), (None, spec_shape[0], spec_shape[1], 1),
    (None, 700, 1), (None, spec_shape[0], spec_shape[1], 1),
    (None, 700, 1), (None, spec_shape[0], spec_shape[1], 1),
]
contrastive_model.build(input_shape=build_input_shapes)

# ----------------- OPTIMIZER, LR SCHEDULE, CALLBACKS -----------------
class WarmupHoldSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, steps_per_epoch, warmup_epochs, hold_epochs, total_epochs, min_lr):
        super().__init__()
        self.base_lr = base_lr
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_epochs * self.steps_per_epoch, tf.float32)
        hold_steps = tf.cast(self.hold_epochs * self.steps_per_epoch, tf.float32)
        total_steps = tf.cast(self.total_epochs * self.steps_per_epoch, tf.float32)
        warmup_lr = self.base_lr * (step + 1.0) / warmup_steps
        progress = (step - hold_steps) / tf.maximum(1.0, (total_steps - hold_steps))
        cosine_lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < warmup_steps, warmup_lr,
                        tf.where(step < hold_steps, self.base_lr, cosine_lr))

lr_schedule = WarmupHoldSchedule(
    base_lr=configs.BASE_LR,
    steps_per_epoch=configs.STEP_PER_EPOCH,
    warmup_epochs=configs.WARMUP_EPOCHS,
    hold_epochs=configs.HOLD_EPOCHS,
    total_epochs=configs.EPOCHS,
    min_lr=configs.MIN_LR
)

optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
contrastive_model.compile(optimizer=optimizer)

checkpoint_filepath = os.path.join(CONTRASTIVE_SAVE_PATH, "checkpoint/best.weights.h5")
os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True
)

early_stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

class LRSchedulerLogger(tf.keras.callbacks.Callback):
    def __init__(self, lr_schedule, steps_per_epoch):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        step = epoch * self.steps_per_epoch
        lr = tf.keras.backend.get_value(self.lr_schedule(step))
        print(f"Epoch {epoch}: Learning Rate = {lr:.6f}")

lr_logger = LRSchedulerLogger(lr_schedule=lr_schedule, steps_per_epoch=configs.STEPS_PER_EPOCH)

# ----------------- TRAIN -----------------
logger.info("Starting training...")
history = contrastive_model.fit(
    train_ds,
    steps_per_epoch=configs.STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=configs.VALIDATION_STEPS,
    epochs=configs.EPOCHS,
    callbacks=[checkpoint_callback, early_stop_callback, lr_logger]
)
logger.info("Training completed.")

# ----------------- SAVE MODELS -----------------
logger.info("Loading best weights from checkpoint...")
contrastive_model.load_weights(checkpoint_filepath)

logger.info(f"Saving best contrastive network to {CONTRASTIVE_SAVE_PATH} ...")
keras.models.save_model(
    contrastive_model.Contrastive_network,
    filepath=os.path.join(CONTRASTIVE_SAVE_PATH, "Contrastive_Network_dual.keras"),
    include_optimizer=False
)

logger.info(f"Saving embedding network to {EMBEDDING_SAVE_PATH} ...")
keras.models.save_model(
    embedding_network,
    filepath=EMBEDDING_SAVE_PATH,
    include_optimizer=False
)

logger.info("Plotting training history...")
utils.plot_training(history, SAVE_DIR)

logger.info("Dual-input training completed successfully.")

