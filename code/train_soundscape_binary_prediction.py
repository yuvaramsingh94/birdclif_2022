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
import shutil
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import gc


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)



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

test_prediction = []
test_targets = []
test_file = []

## get the model for 8 fold

def model_get():
    model_list = []
    for fold in range(0,8):#8
        model_list.append(tf.keras.models.load_model("../input/birdclif-saved-model/binary_v3_SED"))
        _=model_list[fold].load_weights(
        config.SAVE_DIR + f'/fold_{fold}/'+ "weights/" + "best.hdf5",
    )  ## work on the path
    return model_list

#for fold in range(0,10):
## model load
K.clear_session()
model_list = model_get()
test_dataset = get_eval_dataset()
actual_count = len(os.listdir('./test_5sec_mel_spec'))


test_prediction_fold = []
test_targets_fold = []
test_file_fold = []
#for rr, m in enumerate(model_list):
count = 0
#print('model ',rr)
test_prediction_sub_model_0 = []
test_prediction_sub_model_1 = []
test_prediction_sub_model_2 = []
test_prediction_sub_model_3 = []
test_prediction_sub_model_4 = []
test_prediction_sub_model_5 = []
test_prediction_sub_model_6 = []
test_prediction_sub_model_7 = []
test_targets_sub = []
test_file_sub = []
for element in test_dataset:

    test_prediction_sub_model_0.append(tf.nn.softmax(model_list[0].__call__(element[0]))[:,1])
    test_prediction_sub_model_1.append(tf.nn.softmax(model_list[1].__call__(element[0]))[:,1])
    test_prediction_sub_model_2.append(tf.nn.softmax(model_list[2].__call__(element[0]))[:,1])
    test_prediction_sub_model_3.append(tf.nn.softmax(model_list[3].__call__(element[0]))[:,1])
    test_prediction_sub_model_4.append(tf.nn.softmax(model_list[4].__call__(element[0]))[:,1])
    test_prediction_sub_model_5.append(tf.nn.softmax(model_list[5].__call__(element[0]))[:,1])
    test_prediction_sub_model_6.append(tf.nn.softmax(model_list[6].__call__(element[0]))[:,1])
    test_prediction_sub_model_7.append(tf.nn.softmax(model_list[7].__call__(element[0]))[:,1])
    #test_targets_sub.append(np.argmax(element[1].numpy(),axis=1))
    test_file_sub.append(element[1].numpy())

    if count * config.BATCH_SIZE > actual_count + 20:  # this 20 is for safty
        print('the break ',count * config.BATCH_SIZE)
        break
    print('the out ',count * config.BATCH_SIZE)
    count += 1

test_prediction_fold.append(np.concatenate(test_prediction_sub_model_0)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_4)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_5)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_6)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_7)[:actual_count])
#test_targets_fold.append(np.concatenate(test_targets_sub)[:actual_count])
test_file_fold.append(np.concatenate(test_file_sub)[:actual_count])

test_prediction = np.array(test_prediction_fold).squeeze().mean(axis=0)
#test_targets = test_targets_fold[0]
test_file = test_file_fold[0]


oof_dict = {

    'itemid':test_file,
    #'hasbird':test_targets,
    'prediction':test_prediction,
}

oof_df = pd.DataFrame.from_dict(oof_dict)
oof_df['itemid'] =  oof_df['itemid'].map(lambda x: x.decode("utf-8")) 
oof_df.to_csv(f"binary_prediction.csv", index = False)

## 5 sec
## happywhale-tfrev-train-sound-v2
## gs://kds-22c2720af882134066efd729205635efc3a6ce12dfa19de78bd41475

## 10 sec
## happywhale-tfrev-train-sound-v1
## gs://kds-168dba29a1eb377402725ff4ec9b4a8afa032dfacaa1f1b84639cc59