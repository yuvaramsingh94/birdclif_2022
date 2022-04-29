## predict on the train soundscape data using the binary model that was trained

import sys
#sys.path.append('/content/code')
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
import gc

## we are evaluating on fold 0,1,2,3,4
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


if not config.IS_COLAB:
    train_files = [config.DATA_PATH+i for i in os.listdir(config.DATA_PATH) if "tfrec" in i]
else:
    train_files = np.sort(
        np.array(tf.io.gfile.glob(config.DATA_LINK + "/train-soundscape*.tfrec"))
    )
test_prediction = []
test_targets = []
test_file = []

## get the model for 8 fold

def model_get(strategy):
    model_list = []
    for fold in fold(0,8):
        model_list.append(get_model(strategy))
        model_list[fold].load_weights(
            config.SAVE_DIR + config.WEIGHT_SAVE + f'/fold_{fold}/'+ "weights/" + "best.hdf5",
        )
    return modle_list

for fold in range(0,10):
    ## model load
    K.clear_session()
    model_list = model_get(strategy)
    test_files = train_files#[train_ff_files[fold]] + [train_war_files[fold]]
    test_dataset = get_eval_dataset(test_files)
    actual_count = 0
    print('test_files',test_files)
    for i in test_files:
        print('i',i)
        actual_count += int(i.split(".")[-2].split("-")[-1])
    print('this is the actual count',actual_count)
    count = 0
    
    test_prediction_fold = []
    test_targets_fold = []
    test_file_fold = []
    for m in model_list:
        test_prediction_sub = []
        test_targets_sub = []
        test_file_sub = []
        for element in test_dataset:

            pred = model.predict(element[0])  # , verbose=0)
            test_prediction_sub.append(tf.math.sigmoid(pred))
            test_targets_sub.append(element[1].numpy())
            test_file_sub.append(element[2].numpy())

            if count * config.BATCH_SIZE > actual_count + 20:  # this 20 is for safty
                break
            count += 1

        test_prediction_fold.append(np.concatenate(test_prediction_sub)[:actual_count])
        test_targets_fold.append(np.concatenate(test_targets_sub)[:actual_count])
        test_file_fold.append(np.concatenate(test_file_sub)[:actual_count])
'''
test_prediction = np.concatenate(test_prediction)
test_targets = np.concatenate(test_targets)
test_file = np.concatenate(test_file)


oof_dict = {

    'itemid':test_file,
    'hasbird':test_targets,
    'prediction':np.squeeze(test_prediction,axis=1),
}

oof_df = pd.DataFrame.from_dict(oof_dict)
oof_df['itemid'] =  oof_df['itemid'].map(lambda x: x.decode("utf-8")) 
oof_df.to_csv(config.SAVE_DIR + config.WEIGHT_SAVE  + f"/oof.csv", index = False)
'''

