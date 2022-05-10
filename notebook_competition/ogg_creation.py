import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import cv2
from joblib import delayed,Parallel
import pandas as pd

import os, json, random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf, re, math
from tqdm import tqdm


train_df = pd.read_csv('../data/birdclef-2022/train_metadata.csv')


Target_sr = 44100
folder = '../data/birdclef-2022/train_audio/'
folder_wav = '../data/birdclef-2022/train_audio_ogg/'
def resample_to_wav(x):
    y, original_sr = librosa.load(folder+x, sr = None)
    resample_y = librosa.resample(y, orig_sr=original_sr, target_sr=Target_sr)
    f_name = x.split('.')[0]
    #librosa.output.write_wav(f'../data/train_soundscapes/{f_name}.wav', resample_y, Target_sr,) 
    sf.write(folder_wav + f'{f_name}.wav', resample_y, Target_sr)


#'''
_ = Parallel(n_jobs=8, verbose=1)(
    delayed(resample_to_wav)(
    file,
    ) for file in train_df['filename'].values#[:30]#[:5]
)
#'''
'''
for file in train_df['filename'].values[:30]:#[:5]
    resample_to_wav(file)
'''