import sys
sys.path.append('/content/code')
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


## for 10 sec
#'gs://kds-2079ef885c3d4a49fef52a42707e12ddcc2612d1d145871df520e01a'
#config.IMG_FREQ = 128
#config.IMG_TIME = 862

## for 5 sec
#gs://kds-e9110ae66abbac40204455497f8dab2d00ea839009e45f7eb2ed0941
config.IMG_FREQ = 128
config.IMG_TIME = 431

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
        np.array(tf.io.gfile.glob('gs://kds-e9110ae66abbac40204455497f8dab2d00ea839009e45f7eb2ed0941' + "/train-soundscape*.tfrec"))
    )
test_prediction = []
test_targets = []
test_file = []

## get the model for 8 fold

def model_get(strategy):
    model_list = []
    for fold in range(0,8):#8
        model_list.append(get_model(strategy))
        _=model_list[fold].load_weights(
            config.SAVE_DIR + config.WEIGHT_SAVE + f'/fold_{fold}/'+ "weights/" + "best.hdf5",
        )
    return model_list

#for fold in range(0,10):
## model load
K.clear_session()
model_list = model_get(strategy)
test_files = train_files#[train_ff_files[fold]] + [train_war_files[fold]]
test_dataset = get_eval_dataset(test_files)
actual_count = 0
print('test_files',test_files)
for i in test_files:
    #print('i',i)
    actual_count += int(i.split(".")[-2].split("-")[-1])
print('this is the actual count',actual_count)


test_prediction_fold = []
test_targets_fold = []
test_file_fold = []
for rr, m in enumerate(model_list):
    count = 0
    print('model ',rr)
    test_prediction_sub = []
    test_targets_sub = []
    test_file_sub = []
    for element in test_dataset:

        pred = m.predict(element[0])  # , verbose=0)
        test_prediction_sub.append(tf.nn.softmax(pred)[:,1])
        test_targets_sub.append(np.argmax(element[1].numpy(),axis=1))
        test_file_sub.append(element[2].numpy())

        if count * config.BATCH_SIZE > actual_count + 20:  # this 20 is for safty
            print('the break ',count * config.BATCH_SIZE)
            break
        print('the out ',count * config.BATCH_SIZE)
        count += 1

    test_prediction_fold.append(np.concatenate(test_prediction_sub)[:actual_count])
    test_targets_fold.append(np.concatenate(test_targets_sub)[:actual_count])
    test_file_fold.append(np.concatenate(test_file_sub)[:actual_count])

test_prediction = np.array(test_prediction_fold).squeeze().mean(axis=0)
test_targets = test_targets_fold[0]
test_file = test_file_fold[0]


oof_dict = {

    'itemid':test_file,
    'hasbird':test_targets,
    'prediction':test_prediction,
}

oof_df = pd.DataFrame.from_dict(oof_dict)
oof_df['itemid'] =  oof_df['itemid'].map(lambda x: x.decode("utf-8")) 
oof_df.to_csv(config.SAVE_DIR + config.WEIGHT_SAVE  + f"/train_soundscape_5_sec.csv", index = False)

## 5 sec
## happywhale-tfrev-train-sound-v2
## gs://kds-22c2720af882134066efd729205635efc3a6ce12dfa19de78bd41475

## 10 sec
## happywhale-tfrev-train-sound-v1
## gs://kds-168dba29a1eb377402725ff4ec9b4a8afa032dfacaa1f1b84639cc59