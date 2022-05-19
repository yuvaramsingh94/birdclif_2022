import sys
#sys.path.append('./code')
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
import tensorflow as tf
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

idx_to_bird = {118:'akiapo',
125:'aniani',
71:'apapan',
115:'barpet',
150:'crehon',
119:'elepai',
137:'ercfra',
99:'hawama',
101:'hawcre',
130:'hawgoo',
146:'hawhaw',
145:'hawpet1',
12:'houfin',
84:'iiwi',
53:'jabwar',
151:'maupar',
98:'omao',
147:'puaioh',
1:'skylar',
61:'warwhe1',
62:'yefcan'}



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

def model_get(model, save_dir):
    model_list = []
    for fold in range(0,8):#8
        model_list.append(tf.keras.models.load_model(model))
        _=model_list[fold].load_weights(
        save_dir + f'/fold_{fold}/'+ "weights/" + "best.hdf5",
    )  ## work on the path
    return model_list

#for fold in range(0,10):
## model load
K.clear_session()
model_list1 = model_get(model = "../input/birdclif-saved-model/comp_v4_SED",
                      save_dir = "../input/bird-clief-comp-v4-sed")
model_list2 = model_get(model = "../input/birdclif-saved-model/comp_v1_SED",
                      save_dir = "../input/bird-clief-comp-v1-sed")

model_list3 = model_get(model = "../input/birdclif-saved-model/comp_v6_SED_vis",
                      save_dir = "../input/bird-clief-comp-v6-sed-vis")

model_list = model_list1 + model_list2 + model_list3
test_dataset = get_eval_dataset()
actual_count = len(os.listdir('./test_5sec_mel_spec'))

test_prediction_fold = []

## model 1
test_prediction_sub_model_1_0 = []
test_prediction_sub_model_1_1 = []
test_prediction_sub_model_1_2 = []
test_prediction_sub_model_1_3 = []
test_prediction_sub_model_1_4 = []
test_prediction_sub_model_1_5 = []
test_prediction_sub_model_1_6 = []
test_prediction_sub_model_1_7 = []

## model 2
test_prediction_sub_model_2_0 = []
test_prediction_sub_model_2_1 = []
test_prediction_sub_model_2_2 = []
test_prediction_sub_model_2_3 = []
test_prediction_sub_model_2_4 = []
test_prediction_sub_model_2_5 = []
test_prediction_sub_model_2_6 = []
test_prediction_sub_model_2_7 = []

## model 3
test_prediction_sub_model_3_0 = []
test_prediction_sub_model_3_1 = []
test_prediction_sub_model_3_2 = []
test_prediction_sub_model_3_3 = []
test_prediction_sub_model_3_4 = []
test_prediction_sub_model_3_5 = []
test_prediction_sub_model_3_6 = []
test_prediction_sub_model_3_7 = []
test_targets_fold = []
test_file_fold = []
#for rr, m in enumerate(model_list):
count = 0
#print('model ',rr)
test_prediction_sub = []
test_targets_sub = []
test_file_sub = []
for element in test_dataset:

    #pred = m.__call__(element[0])  # , verbose=0)
    #test_prediction_sub.append(pred)
    test_prediction_sub_model_1_0.append(model_list[0].__call__(element[0]))
    test_prediction_sub_model_1_1.append(model_list[1].__call__(element[0]))
    test_prediction_sub_model_1_2.append(model_list[2].__call__(element[0]))
    test_prediction_sub_model_1_3.append(model_list[3].__call__(element[0]))
    test_prediction_sub_model_1_4.append(model_list[4].__call__(element[0]))
    test_prediction_sub_model_1_5.append(model_list[5].__call__(element[0]))
    test_prediction_sub_model_1_6.append(model_list[6].__call__(element[0]))
    test_prediction_sub_model_1_7.append(model_list[7].__call__(element[0]))

    test_prediction_sub_model_2_0.append(model_list[8].__call__(element[0]))
    test_prediction_sub_model_2_1.append(model_list[9].__call__(element[0]))
    test_prediction_sub_model_2_2.append(model_list[10].__call__(element[0]))
    test_prediction_sub_model_2_3.append(model_list[11].__call__(element[0]))
    test_prediction_sub_model_2_4.append(model_list[12].__call__(element[0]))
    test_prediction_sub_model_2_5.append(model_list[13].__call__(element[0]))
    test_prediction_sub_model_2_6.append(model_list[14].__call__(element[0]))
    test_prediction_sub_model_2_7.append(model_list[15].__call__(element[0]))

    test_prediction_sub_model_3_0.append(model_list[16].__call__(element[0]))
    test_prediction_sub_model_3_1.append(model_list[17].__call__(element[0]))
    test_prediction_sub_model_3_2.append(model_list[18].__call__(element[0]))
    test_prediction_sub_model_3_3.append(model_list[19].__call__(element[0]))
    test_prediction_sub_model_3_4.append(model_list[20].__call__(element[0]))
    test_prediction_sub_model_3_5.append(model_list[21].__call__(element[0]))
    test_prediction_sub_model_3_6.append(model_list[22].__call__(element[0]))
    test_prediction_sub_model_3_7.append(model_list[23].__call__(element[0]))


    #test_targets_sub.append(np.argmax(element[1].numpy(),axis=1))
    test_file_sub.append(element[1].numpy())

    if count * config.BATCH_SIZE > actual_count + 20:  # this 20 is for safty
        print('the break ',count * config.BATCH_SIZE)
        break
    print('the out ',count * config.BATCH_SIZE)
    count += 1

test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_0)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_1)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_2)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_3)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_4)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_5)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_6)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_1_7)[:actual_count])

test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_0)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_1)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_2)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_3)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_4)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_5)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_6)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_2_7)[:actual_count])

test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_0)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_1)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_2)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_3)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_4)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_5)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_6)[:actual_count])
test_prediction_fold.append(np.concatenate(test_prediction_sub_model_3_7)[:actual_count])
#test_targets_fold.append(np.concatenate(test_targets_sub)[:actual_count])
test_file_fold.append(np.concatenate(test_file_sub)[:actual_count])

test_prediction = np.array(test_prediction_fold).squeeze().mean(axis=0)
#test_targets = test_targets_fold[0]
test_file = test_file_fold[0]


oof_dict = {

    'itemid':test_file,
    #'hasbird':test_targets,

}

for i in range(0, config.N_CLASSES):
    oof_dict[i] = test_prediction[:,i]

oof_df = pd.DataFrame.from_dict(oof_dict)
oof_df['itemid'] =  oof_df['itemid'].map(lambda x: x.decode("utf-8")) 

## retain the unwanted birds
oof_df = oof_df.filter(items=list(idx_to_bird.keys())+['itemid'])

oof_df.to_csv(f"comp_prediction.csv", index = False)

## 5 sec
## happywhale-tfrev-train-sound-v2
## gs://kds-22c2720af882134066efd729205635efc3a6ce12dfa19de78bd41475

## 10 sec
## happywhale-tfrev-train-sound-v1
## gs://kds-168dba29a1eb377402725ff4ec9b4a8afa032dfacaa1f1b84639cc59