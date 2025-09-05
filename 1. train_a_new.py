# -*- coding: utf-8 -*-
import os
import math
import logging
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import config
from Library.model import Embedding_Network_dual, Contrastive_Network_dual, Contrastive_Model
from Library.dataset import Data_Generator
from Library import utils

# ----------------- PATHS & LOGGING -----------------
SAVE_DIR = os.path.join(config.OUTPUT_PATH, "train_dual")
os.makedirs(SAVE_DIR, exist_ok=True)

CONTRASTIVE_SAVE_PATH = os.path.join(SAVE_DIR, f"Contrastive_Network_{config.FEATURE_DISTANCE}")
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

# ----------------- LOAD DATA -----------------
logger.info("Loading train/validation data...")
train_file = pd.read_pickle(config.TRAIN_FILE_PATH)
val_file = pd.read_pickle(config.VAL_FILE_PATH)

signal_length = config.SIGNAL_LENGTH

# at top: imports unchanged

# --- after loading data and constructing generators ---
# instantiate Data_Generator with smaller spectrogram dims
train_generator = Data_Generator(train_file, seed=config.SEED,
                                 time_bins=32, freq_bins=65)
val_generator = Data_Generator(val_file, seed=config.SEED,
                               time_bins=32, freq_bins=65)

# choose a safe batch size to avoid OOM; if your config.BATCH_SIZE is large, override here
BATCH_SIZE_SAFE = min(config.BATCH_SIZE, 8)  # change 8 -> 4 if still OOM

# --- TF dataset generator that yields a flat tuple (8 tensors) ---
def flat_dataset_generator(gen):
    for nested in gen.get_next_record():
        # nested is ((refer, refer_spec), (pos, pos_spec), (neg, neg_spec), (sil, sil_spec))
        refer, refer_spec, pos, pos_spec, neg, neg_spec, sil, sil_spec = Data_Generator.flat_sample(nested)
        yield (refer, refer_spec, pos, pos_spec, neg, neg_spec, sil, sil_spec)

# create train_ds and val_ds with explicit output_signature (flat)
train_ds = tf.data.Dataset.from_generator(
    lambda: flat_dataset_generator(train_generator),
    output_signature=(
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),        # refer_wave
        tf.TensorSpec(shape=(train_generator.time_bins, train_generator.freq_bins, 1), dtype=tf.float32),  # refer_spec
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),        # pos_wave
        tf.TensorSpec(shape=(train_generator.time_bins, train_generator.freq_bins, 1), dtype=tf.float32),  # pos_spec
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),        # neg_wave
        tf.TensorSpec(shape=(train_generator.time_bins, train_generator.freq_bins, 1), dtype=tf.float32),  # neg_spec
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),        # sil_wave
        tf.TensorSpec(shape=(train_generator.time_bins, train_generator.freq_bins, 1), dtype=tf.float32),  # sil_spec
    )
).shuffle(config.BUFFER_SIZE).batch(BATCH_SIZE_SAFE).prefetch(config.AUTO)

val_ds = tf.data.Dataset.from_generator(
    lambda: flat_dataset_generator(val_generator),
    output_signature=(
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(val_generator.time_bins, val_generator.freq_bins, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(val_generator.time_bins, val_generator.freq_bins, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(val_generator.time_bins, val_generator.freq_bins, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(val_generator.time_bins, val_generator.freq_bins, 1), dtype=tf.float32),
    )
).batch(BATCH_SIZE_SAFE).prefetch(config.AUTO)

# --- Build embedding and contrastive networks ---
# compute spec shape and pass to your model constructors
spec_shape = (train_generator.time_bins, train_generator.freq_bins)

embedding_network = Embedding_Network_dual(
    input_shape_wave=(signal_length,),
    input_shape_spec=spec_shape,
    chromosome=config.CHROMOSOME
)
embedding_network.summary(print_fn=logger.info)

contrastive_network = Contrastive_Network_dual(
    input_wave_size=signal_length,
    input_spec_shape=spec_shape,
    embedding_model=embedding_network
)

contrastive_model = Contrastive_Model(
    Contrastive_network=contrastive_network,
    batch_size=BATCH_SIZE_SAFE,
    loss_tracker=keras.metrics.Mean(name="loss"),
    feature_distance=config.FEATURE_DISTANCE,
    metric_acc=keras.metrics.Accuracy(name="acc"),
    margin=config.MARGIN,
    alpha=config.ALPHA,
)

# When building the contrastive model, pass list of input shapes that match the flat dataset:
# format: [ (None, signal_length, 1), (None, time_bins, freq_bins, 1), ... ]  (for 8 inputs)
build_input_shapes = [
    (None, signal_length, 1), (None, spec_shape[0], spec_shape[1], 1),  # refer
    (None, signal_length, 1), (None, spec_shape[0], spec_shape[1], 1),  # pos
    (None, signal_length, 1), (None, spec_shape[0], spec_shape[1], 1),  # neg
    (None, signal_length, 1), (None, spec_shape[0], spec_shape[1], 1),  # sil
]
contrastive_model.build(input_shape=build_input_shapes)

# compile/train as before, but use BATCH_SIZE_SAFE in steps_per_epoch if needed.
logger.info("Data loaded and model built successfully.")
# ----------------- OPTIMIZER & LR -----------------
class WarmupHoldSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, steps_per_epoch, warmup_epochs, hold_epochs, total_epochs, min_lr=1e-8):
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
    base_lr=config.BASE_LR,
    steps_per_epoch=config.STEP_PER_EPOCH,
    warmup_epochs=config.WARMUP_EPOCHS,
    hold_epochs=config.HOLD_EPOCHS,
    total_epochs=config.EPOCHS,
    min_lr=config.MIN_LR
)

optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
contrastive_model.compile(optimizer=optimizer)

# ----------------- CALLBACKS -----------------
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

# ----------------- LR Logger Callback -----------------
class LRSchedulerLogger(tf.keras.callbacks.Callback):
    def __init__(self, lr_schedule, steps_per_epoch):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        step = epoch * self.steps_per_epoch
        lr = tf.keras.backend.get_value(self.lr_schedule(step))
        print(f"Epoch {epoch}: Learning Rate = {lr:.6f}")

lr_logger = LRSchedulerLogger(lr_schedule=lr_schedule, steps_per_epoch=config.STEPS_PER_EPOCH)


# ----------------- EMBEDDING STATS LOGGER -----------------
class EmbeddingStatsLogger(keras.callbacks.Callback):
    def __init__(self, embedding_model, sample_batch):
        """
        sample_batch should be a dict with keys:
            "refer_wave", "refer_spec",
            "positive_wave", "positive_spec",
            "negative_wave", "negative_spec",
            "silence_wave", "silence_spec"
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.sample_batch = sample_batch

    def on_epoch_end(self, epoch, logs=None):
        # Compute embeddings (without training noise)
        emb_ref = self.embedding_model([self.sample_batch["refer_wave"], self.sample_batch["refer_spec"]], training=False)
        emb_pos = self.embedding_model([self.sample_batch["positive_wave"], self.sample_batch["positive_spec"]], training=False)
        emb_neg = self.embedding_model([self.sample_batch["negative_wave"], self.sample_batch["negative_spec"]], training=False)
        emb_sil = self.embedding_model([self.sample_batch["silence_wave"], self.sample_batch["silence_spec"]], training=False)

        # Group by semantic class
        emb_eq = tf.concat([emb_ref, emb_pos], axis=0)   # Seismic event (EQ)
        emb_qb = emb_neg                                 # Quarry blast
        emb_ns = emb_sil                                 # Noise

        # Stats per class
        stats = {
            "EQ": (tf.reduce_mean(emb_eq).numpy(), tf.math.reduce_std(emb_eq).numpy()),
            "QB": (tf.reduce_mean(emb_qb).numpy(), tf.math.reduce_std(emb_qb).numpy()),
            "Noise": (tf.reduce_mean(emb_ns).numpy(), tf.math.reduce_std(emb_ns).numpy()),
        }

        print(f"\n[Epoch {epoch+1}] Embedding Stats:")
        for cls, (mean_val, std_val) in stats.items():
            print(f"  {cls:<5} → Mean: {mean_val:.4f}, Std: {std_val:.4f}")

# get one batch of (wave, spec) pairs for logging
# Take one batch from train_ds
for batch in train_ds.take(1):
    sample_batch = {
        "refer_wave": batch[0],
        "refer_spec": batch[1],
        "positive_wave": batch[2],
        "positive_spec": batch[3],
        "negative_wave": batch[4],
        "negative_spec": batch[5],
        "silence_wave": batch[6],
        "silence_spec": batch[7],
    }
    break

# Create logger
stats_logger = EmbeddingStatsLogger(embedding_network, sample_batch)

# ----------------- TRAIN -----------------
logger.info("Starting training...")
contrastive_model.build(input_shape=(None, signal_length, 1))

history = contrastive_model.fit(
    train_ds,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=config.VALIDATION_STEPS,
    epochs=config.EPOCHS,
    callbacks=[checkpoint_callback, early_stop_callback, lr_logger, stats_logger]
)
logger.info("Training completed.")

# ----------------- SAVE BEST MODELS -----------------
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

