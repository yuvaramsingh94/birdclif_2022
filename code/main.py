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
import matplotlib.pyplot as plt

seed_everything(config.SEED)

IS_COLAB = not os.path.exists("/kaggle/input")
print(IS_COLAB)

"""
print("copy the code and supporting materials for reference")
if os.path.exists(config.SAVE_DIR + config.WEIGHT_SAVE + "/code"):
    shutil.rmtree(config.SAVE_DIR + config.WEIGHT_SAVE + "/code")
shutil.copytree("/content/code", config.SAVE_DIR + config.WEIGHT_SAVE + "/code")
"""

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
if not config.IS_COLAB:
    train_files = [
        config.DATA_PATH + i for i in os.listdir(config.DATA_PATH) if "tfrec" in i
    ]
else:
    train_file = np.sort(
        np.array(tf.io.gfile.glob(config.DATA_LINK + '/v1' + "/happywhale*.tfrec"))
    )


for fold in range (config.FOLD):


    if IS_COLAB:
        if not os.path.exists(
            config.SAVE_DIR + config.WEIGHT_SAVE + f"/fold_{str(fold)}/"
        ):
            os.makedirs(config.SAVE_DIR + config.WEIGHT_SAVE + f"/fold_{str(fold)}/")

        if not os.path.exists(
            config.SAVE_DIR
            + config.WEIGHT_SAVE
            + f"/fold_{str(fold)}/"
            + "/weights/"
        ):
            os.makedirs(
                config.SAVE_DIR
                + config.WEIGHT_SAVE
                + f"/fold_{str(fold)}/"
                + "/weights/"
            )


    print("copy the code and supporting materials for reference")
    if os.path.exists(config.SAVE_DIR + config.WEIGHT_SAVE + f"/fold_{str(fold)}/" + "/code"):
        shutil.rmtree(
            config.SAVE_DIR + config.WEIGHT_SAVE + f"/fold_{str(fold)}/" + "/code"
        )
        shutil.copytree(
        "/content/code",
        config.SAVE_DIR + config.WEIGHT_SAVE + f"/fold_{str(fold)}/" + "/code",
        )



    TRAINING_FILENAMES = [x for i, x in enumerate(train_file) if i != fold] 

    VALIDATION_FILENAMES = [x for i, x in enumerate(train_file) if i == fold]

    print("Training file", TRAINING_FILENAMES)
    print("validation file", VALIDATION_FILENAMES)

    seed_everything(config.SEED)
    VERBOSE = 1
    train_dataset = get_training_dataset(TRAINING_FILENAMES)
    val_dataset = get_valid_dataset(VALIDATION_FILENAMES)
    STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES) // config.BATCH_SIZE
    # VAL_STEPS_PER_EPOCH = count_data_items(VALIDATION_FILENAMES) // config.BATCH_SIZE
    train_logger = tf.keras.callbacks.CSVLogger(
        config.SAVE_DIR
        + config.WEIGHT_SAVE
        + f"/fold_{str(fold)}/"
        + "weights/"
        + f"training-log.h5.csv"
    )
    # SAVE BEST MODEL EACH FOLD
    sv_loss = tf.keras.callbacks.ModelCheckpoint(
        config.SAVE_DIR
        + config.WEIGHT_SAVE
        + f"/fold_{str(fold)}/"
        + "weights/"
        + "best.hdf5",  # {epoch:02d}
        # "weights/v1/best.h5",
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

    # print(
    #    "#### Image Size %i with EfficientNet B%i and batch_size %i"
    #    % (config.IMAGE_SIZE, config.EFF_NET, config.BATCH_SIZE)
    # )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        # validation_steps=VAL_STEPS_PER_EPOCH,
        epochs=config.EPOCHS,
        # callbacks=[get_lr_callback(), train_logger, sv_loss, Snapshot([1, 20, 30, 40])],#maybe remove  train_logger, sv_loss
        callbacks=[
            get_lr_callback(),
            sv_loss,
        ],  # maybe remove  train_logger, sv_loss # Snapshot([1, 20, 30, 40])
        verbose=VERBOSE,
    )

    print(history.history.keys())

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
