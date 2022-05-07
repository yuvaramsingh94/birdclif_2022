class config:
    EPOCHS = 20  # 20
    BATCH_SIZE = 8#32
    WAVE_LENGTH = 441000
    IMAGE_SIZE = -1
    IMG_FREQ = 128
    IMG_TIME = 862
    FOLD = 8  # validation fold
    WEIGHT_SAVE = "competition_torch_v1"
    N_CLASSES = 152
    IS_COLAB = True
    DATA_LINK = "gs://kds-bf56271889f9bf5c5e02f2e650ed90452aa6a2f40d413357733a7830"
    DATA_PATH = "data/tfrec/v3/"
    DATAFRAME_PATH = "prediction/binary_v3_SED/train_competition_v1_binary_merged.csv"
    AUG_PER = 0.8
    SECONDAEY_EFF = 0.5
    SEED = 1
    MODEL_TYPE = "resnet50"
    WORKERS = 6
    sample_rate = 44100
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128
    LR_START = 0.000001  # 0.000001
    LR_MAX = 0.000005  # 0.00001
    LR_MIN = 0.0000001  # 0.000001
    LR_RAMP = 4
    SAVE_DIR = "/content/drive/MyDrive/Kaggle/birdclif-2022/"

    RESUME = False
    RESUME_EPOCH = 0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    sample_rate = 44100
    n_mels = 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
