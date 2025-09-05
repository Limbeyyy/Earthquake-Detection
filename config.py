import tensorflow as tf
import os
from datetime import datetime

SEED = 2023
SAMPLING_RATE = 1

# network
SPACIAL_RESOLUTION = 128
MARGIN = 0.1
ALPHA = 1

ACTIVATION = "relu"

# train
BATCH_SIZE = 64
BUFFER_SIZE = BATCH_SIZE * 2
AUTO = tf.data.AUTOTUNE   # define autotune

LEARNING_RATE = 0.0005

EPOCHS = 100 # 100
STEPS_PER_EPOCH = 100 #  100
VALIDATION_STEPS = 50 #  20

# metrics
FEATURE_DISTANCE = "Wasserstein"  # Wasserstein or Euclidean

# test
TEST_BATCH_SIZE = 16
CROP_CONFIDENCE = 0.5

# save
# create save folder
now = datetime.now()
time_str = now.strftime("%d%H%M")
OUTPUT_PATH = "output {}".format(time_str)


DATASET_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\earthquake_data (3).csv"
TRAIN_FILE_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\data files, train n65065 win7 r100.pkl"
VAL_FILE_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\data files, val n7165 win7 r100.pkl"


SIGNAL_LENGTH = 700  # length of the signal
TRAIN_SIGNAL_LENGTH = 700  # length of the signal for training      
CHROMOSOME= [
    8.097911880061828, -0.30367944386676804, -0.35225592325209076, 
    -0.5840144913416598, -0.13185410244369244, 0.7712233111496816, 
    7.997406552070309, 3.961237228964, 22.835719873542722, 
    32.38916312891132, 1.146460404651366
]



STEP_PER_EPOCH = max(1, STEPS_PER_EPOCH)
WARMUP_EPOCHS = 3
HOLD_EPOCHS = 10
BASE_LR = 3e-4
MIN_LR = 1e-4


CSV_DATA_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\UUSS 3C meta, test snr_db, n2222.csv"
TEST_DATA_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\UUSS 3C data, test n2222 r100.json"
SAVE_DIR = r"output 192156\train_dual"

Z_N = r"C:\Users\Hp\Desktop\Earthquake-Detection\Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909\Embedding data, Z.json.npy"
N_N = r"C:\Users\Hp\Desktop\Earthquake-Detection\Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909\Embedding data, N.json.npy"
E_N = r"C:\Users\Hp\Desktop\Earthquake-Detection\Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909\Embedding data, E.json.npy"


Z = r"C:\Users\Hp\Desktop\Earthquake-Detection\Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909\Embedding data, Z.json"
N = r"C:\Users\Hp\Desktop\Earthquake-Detection\Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909\Embedding data, N.json"
E = r"C:\Users\Hp\Desktop\Earthquake-Detection\Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909\Embedding data, E.json"


DEMO_VAL_FILE_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\UUSS Validation demo dataset.pkl"

EQ_FILE_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\data_STEAD\train.hdf5"
NOISE_FILE_PATH = r"C:\Users\Hp\Desktop\Earthquake-Detection\Data train-demo\data_STEAD\train_noise.hdf5"