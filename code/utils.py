import os
import re
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
from config.config import config


AUTO = tf.data.experimental.AUTOTUNE


def get_lr_callback(plot=False):
    lr_start = config.LR_START  # 0.000001
    lr_max = config.LR_MAX * config.BATCH_SIZE
    lr_min = config.LR_MIN  # 0.000001
    lr_ramp_ep = config.LR_RAMP
    lr_sus_ep = 0
    lr_decay = 0.9

    def lrfn(epoch):
        if config.RESUME:
            epoch = epoch + config.RESUME_EPOCH
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min

        return lr

    if plot:
        epochs = list(range(config.EPOCHS))
        learning_rates = [lrfn(x) for x in epochs]
        plt.scatter(epochs, learning_rates)
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback


def get_cosine_lr_callback(plot=False):
    lr_start = 0.000001
    lr_max = 0.000005 * config.BATCH_SIZE
    lr_min = 0.000001
    lr_ramp_ep = 4
    lr_sus_ep = 0
    lr_decay = 0.9
    alpha = 0.0

    def lrfn(epoch):
        if config.RESUME:
            epoch = epoch + config.RESUME_EPOCH
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            # lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * epoch / config.EPOCH_MAX))
            decayed = (1 - alpha) * cosine_decay + alpha

            return lr_max * decayed
        return lr

    if plot:
        epochs = list(range(config.EPOCHS))
        learning_rates = [lrfn(x) for x in epochs]
        plt.scatter(epochs, learning_rates)
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    #image = tf.image.resize(image, [config.IMAGE_SIZE//2, config.IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    # normalization
    image = image - config.mean  # [0.485, 0.456, 0.406]
    image = image / config.std  # [0.229, 0.224, 0.225]
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        'label': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = example["label"]
    #label = tf.one_hot(tf.cast(label, tf.int32), depth = config.N_CLASSES)
    # image name
    image_name = example["image_name"]
    return image, label, image_name  # returns a dataset of (image, label) pairs


def load_dataset(filenames, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed

    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTO
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_training_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.map(
        lambda image, label, image_name: (image, label), num_parallel_calls=AUTO
    )
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(config.BATCH_SIZE, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_valid_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.map(
        lambda image, label, image_name: (image, label), num_parallel_calls=AUTO
    )
    #dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(config.BATCH_SIZE, num_parallel_calls=AUTO, drop_remainder=True)
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_eval_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.map(
        lambda image, label, image_name: (image, label,image_name), num_parallel_calls=AUTO
    )
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(config.BATCH_SIZE, num_parallel_calls=AUTO, drop_remainder=False)
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


class Snapshot(tf.keras.callbacks.Callback):
    def __init__(self, snapshot_epochs=[]):
        super(Snapshot, self).__init__()
        self.snapshot_epochs = snapshot_epochs

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        #         print(f"epoch: {epoch}, train_acc: {logs['acc']}, valid_acc: {logs['val_acc']}")
        if epoch in self.snapshot_epochs:  # your custom condition
            self.model.save_weights(
                config.SAVE_DIR + config.WEIGHT_SAVE + "/weights/" + f"/{epoch}.h5"
            )
        self.model.save_weights(
            config.SAVE_DIR
            + config.WEIGHT_SAVE
            + "/weights/"
            + f"/{config.MODEL_NAME}_last.h5"
        )
