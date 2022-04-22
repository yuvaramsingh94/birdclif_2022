import re
import os
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
from tqdm.auto import tqdm
import json
from datetime import datetime
from config.config import config
from utils import *
from model import *
import shutil
from tensorflow.keras import backend as K

seed_everything(config.SEED)

IS_COLAB = not os.path.exists("/kaggle/input")
print(IS_COLAB)

if IS_COLAB:
    if not os.path.exists(config.SAVE_DIR + config.WEIGHT_SAVE):
        os.mkdir(config.SAVE_DIR + config.WEIGHT_SAVE)

    if not os.path.exists(config.SAVE_DIR + config.WEIGHT_SAVE + "/weights/"):
        os.mkdir(config.SAVE_DIR + config.WEIGHT_SAVE + "/weights/")

'''
print("copy the code and supporting materials for reference")
if os.path.exists(config.SAVE_DIR + config.WEIGHT_SAVE + "/code"):
    shutil.rmtree(config.SAVE_DIR + config.WEIGHT_SAVE + "/code")
shutil.copytree("/content/code", config.SAVE_DIR + config.WEIGHT_SAVE + "/code")
'''

import tensorflow as tf

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
print("REPLICAS: ", strategy.num_replicas_in_sync)
config.BATCH_SIZE = config.BATCH_SIZE * strategy.num_replicas_in_sync


MODEL_NAME = None
if config.model_type == "effnetv1":
    MODEL_NAME = f"effnetv1_b{config.EFF_NET}"
elif config.model_type == "effnetv2":
    MODEL_NAME = f"effnetv2_{config.EFF_NETV2}"

config.MODEL_NAME = MODEL_NAME
print(MODEL_NAME)

print("Train image original ")
row = 10
col = 8
row = min(row, config.BATCH_SIZE // col)

"""
GCS_PATH = (
    config.DATA_LINK  # public one dietic crop
)
# GCS_PATH = 'gs://kds-8ab64f4f49e944c723b1ef14c537b9928a5a299d3bbc21e4d3cf32f5'
train_files = np.sort(
    np.array(tf.io.gfile.glob(GCS_PATH + "/happywhale-2022-train*.tfrec"))
)
test_files = np.sort(
    np.array(tf.io.gfile.glob(GCS_PATH + "/happywhale-2022-test*.tfrec"))
)

"""
train_files = [
    "data/tfrec/v1/happywhale-ff-2022-train-0-962.tfrec",#happywhale-ff-2022-train-0-962
    "data/tfrec/v1/happywhale-ff-2022-train-1-962.tfrec",
    "data/tfrec/v1/happywhale-ff-2022-train-2-961.tfrec",
    "data/tfrec/v1/happywhale-ff-2022-train-3-961.tfrec",
    "data/tfrec/v1/happywhale-ff-2022-train-4-961.tfrec",
    "data/tfrec/v1/happywhale-ff-2022-train-5-961.tfrec",
    "data/tfrec/v1/happywhale-ff-2022-train-6-961.tfrec",
    "data/tfrec/v1/happywhale-ff-2022-train-7-961.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-0-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-1-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-2-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-3-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-4-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-5-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-6-1000.tfrec",
    "data/tfrec/v1/happywhale-war-2022-train-7-1000.tfrec",
]
#"""

#ds = get_training_dataset(train_files)

'''
for (sample, label) in ds:
    print("this is sampe keys ", sample.keys())
    img = sample["inp1"]  # dict_keys(['inp_crop', 'inp_ori', 'inp2'])
    print(sample["inp2"])
    plt.figure(figsize=(25, int(25 * row / col)))
    for j in range(row * col):
        plt.subplot(row, col, j + 1)
        plt.title(label[j].numpy())
        plt.axis("off")
        plt.imshow(
            img[
                j,
            ]
        )
    plt.show()
    break
print(img.shape)
print('min max',tf.reduce_min(img),tf.reduce_max(img))
'''
"""

for fold in range(config.FOLDS):
    if not os.path.exists(config.save_dir + f"fold_{fold}"):
        os.mkdir(config.save_dir + f"fold_{fold}")
"""
##

#fold = config.FOLDS

TRAINING_FILENAMES = [x for i, x in enumerate(train_files) ]#if i != fold]
#VALIDATION_FILENAMES = [
#    x for i, x in enumerate(train_files) if i == fold
#]  # [x for i, x in enumerate(train_files) if i == fold]
"""
#print("Fold ", fold)
print(
    len(TRAINING_FILENAMES),
    len(VALIDATION_FILENAMES),
    count_data_items(TRAINING_FILENAMES),
    count_data_items(VALIDATION_FILENAMES),
)
"""
print("Training file", TRAINING_FILENAMES)
#print("validation file", VALIDATION_FILENAMES)

seed_everything(config.SEED)
VERBOSE = 1
train_dataset = get_training_dataset(TRAINING_FILENAMES)
#val_dataset = get_val_dataset(VALIDATION_FILENAMES)
STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES) // config.BATCH_SIZE
#VAL_STEPS_PER_EPOCH = count_data_items(VALIDATION_FILENAMES) // config.BATCH_SIZE
train_logger = tf.keras.callbacks.CSVLogger(
    config.SAVE_DIR + config.WEIGHT_SAVE + "/weights/" + f"/training-log.h5.csv"
)
# SAVE BEST MODEL EACH FOLD
sv_loss = tf.keras.callbacks.ModelCheckpoint(
    #config.SAVE_DIR + config.WEIGHT_SAVE + "/weights/" + "/best.hdf5",  # {epoch:02d}
    'weights/v1/best.h5',
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    save_freq="epoch",
)
# BUILD MODEL
K.clear_session()
model = get_model(strategy)
# model.summary()

#print(
#    "#### Image Size %i with EfficientNet B%i and batch_size %i"
#    % (config.IMAGE_SIZE, config.EFF_NET, config.BATCH_SIZE)
#)

history = model.fit(
    train_dataset,
    #validation_data=val_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    #validation_steps=VAL_STEPS_PER_EPOCH,
    epochs=config.EPOCHS,
    #callbacks=[get_lr_callback(), train_logger, sv_loss, Snapshot([1, 20, 30, 40])],#maybe remove  train_logger, sv_loss
    callbacks=[get_lr_callback(), sv_loss],#maybe remove  train_logger, sv_loss # Snapshot([1, 20, 30, 40])
    verbose=VERBOSE,
)