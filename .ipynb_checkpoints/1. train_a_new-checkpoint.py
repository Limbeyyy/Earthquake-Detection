# -*- coding: utf-8 -*-
"""
Final version with Batch Normalization applied after each Conv1D layer and before activation.
"""

import pandas as pd
import config
import os
import time
import logging
import tensorflow as tf
from tensorflow import keras

from Library.model import Embedding_Network, Contrastive_Network, Contrastive_Model
from Library.dataset import Data_Generator
from Library import utils

#--------------
# Configuration
#--------------
save_dir = f"{config.OUTPUT_PATH} train a new/contrastive_model.keras"
Contrastive_SAVE_PATH = os.path.join(save_dir, "Contrastive_Network, {}".format(config.FEATURE_DISTANCE))
EMBEDDING_SAVE_PATH = os.path.join("output 041116 train a new", "contrastive_model.keras", "Embedding_Network_Wasserstein.keras")

os.makedirs(save_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(save_dir, "Train log.txt"),
                    level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    datefmt = "%Y-%m-%d %H:%M:%S",
                    filemode='w')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

#-------------
# Load dataset
#-------------
logger.info("Load train data ...")
val_file_path = r'C:\Users\Hp\Desktop\Earthquake Datasets\Coding and Figure Demo\Data train-demo\UUSS Validation demo dataset.pkl'
train_file_path= r'C:\Users\Hp\Desktop\Earthquake Datasets\Coding and Figure Demo\Data train-demo\UUSS Train demo dataset.pkl'


val_file = pd.read_pickle(val_file_path)

train_file = pd.read_pickle(train_file_path)


# Continue with rest of your code
logger.info("Build tensor dataset ...")
signal_length = len(train_file["data"].iloc[0][0])
train_generator = Data_Generator(train_file, seed=config.SEED)
val_generator = Data_Generator(val_file, seed=config.SEED)


train_ds = tf.data.Dataset.from_generator(
    generator=train_generator.get_next_record,
    output_signature=(
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
    )).batch(config.BATCH_SIZE).prefetch(config.AUTO)

val_ds = tf.data.Dataset.from_generator(
    generator=val_generator.get_next_record,
    output_signature=(
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
    )).batch(config.BATCH_SIZE).prefetch(config.AUTO)

#------------
# Build model
#------------
logger.info("Build model ...")
chromosome = [8.097911880061828, -0.30367944386676804, -0.35225592325209076, -0.5840144913416598,
              -0.13185410244369244, 0.7712233111496816, 7.997406552070309, 3.961237228964,
              22.835719873542722, 32.38916312891132, 1.146460404651366]

embedding_network = Embedding_Network(input_size=signal_length, chromosome=chromosome)
embedding_network.summary(print_fn=logger.info)
Contrastive_network = Contrastive_Network(signal_length, embedding_network)

Contrastive_model = Contrastive_Model(
    Contrastive_network=Contrastive_network,
    batch_size=config.BATCH_SIZE,
    loss_tracker=keras.metrics.Mean(name="loss"),
    feature_distance=config.FEATURE_DISTANCE,
    metric_acc=keras.metrics.Accuracy(name="acc"),
    margin=config.MARGIN,
    alpha=config.ALPHA)

logger.info(f"Architecture chromosome: {chromosome}")
logger.info("Training setting:\n"
            f"data_window_length={signal_length/config.SAMPLING_RATE}s, "
            f"margin={config.MARGIN}, alpha={config.ALPHA}, activation={config.ACTIVITION}, "
            f"learning_rate={config.LEARNING_RATE}, epochs={config.EPOCHS}, "
            f"steps_per_epoch={config.STEPS_PER_EPOCH}, "
            f"validate_steps={config.VALIDATION_STEPS}, "
            f"test_batch_size={config.TEST_BATCH_SIZE}, "
            f"feature distance: {config.FEATURE_DISTANCE}.")

Contrastive_model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=config.LEARNING_RATE,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7 
    )
)

checkpoint_filepath = os.path.join(Contrastive_SAVE_PATH, "checkpoint/best.weights.h5")
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

logger.info("Training the contrastive model...")
Contrastive_model.build(input_shape=(None, signal_length, 1))

history = Contrastive_model.fit(
    train_ds,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=config.VALIDATION_STEPS,
    epochs=config.EPOCHS,
    callbacks=[checkpoint_callback],
)

infer_start = time.time()
for i in range(config.VALIDATION_STEPS):
    data = next(iter(val_ds))
    _ = Contrastive_network(data)
infer_elapsed = time.time() - infer_start
infer_speed = round(infer_elapsed * 1000 / (config.TEST_BATCH_SIZE * config.VALIDATION_STEPS), 4)
logger.info(f"Inference speed: {infer_speed} ms/sample.")

logger.info(f"Reload the best weights from checkpoint ...")
Contrastive_model.load_weights(checkpoint_filepath)

logger.info(f"Save the best Contrastive network to {save_dir}...")
save_dir = "output 041105 train a new"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "contrastive_model_v2.keras")

keras.models.save_model(
    model=Contrastive_model.Contrastive_network,
    filepath=save_path,
    include_optimizer=False)

logger.info(f"Saving the best embedding network to {EMBEDDING_SAVE_PATH}...")
keras.models.save_model(
    model=embedding_network,
    filepath=EMBEDDING_SAVE_PATH,
    include_optimizer=False)

logger.info(f"Plotting training history...")
utils.plot_training(history, save_dir)
