class config:
    EPOCHS = 20  # 20
    BATCH_SIZE = 32
    WAVE_LENGTH = 441000
    IMAGE_SIZE = -1
    IMG_FREQ = 128
    IMG_TIME = 862
    FOLD = 7#validation fold
    WEIGHT_SAVE = "binary_v3_SED"
    N_CLASSES = 2
    IS_COLAB  = True
    DATA_LINK = "gs://kds-bf56271889f9bf5c5e02f2e650ed90452aa6a2f40d413357733a7830"
    DATA_PATH = "data/tfrec/v3/"
    SEED = 1
    model_type = "resnet50"
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

