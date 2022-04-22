class config:
    EPOCHS = 20  # 20
    BATCH_SIZE = 2
    MODEL = 'resnet18'
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
    SAVE_DIR = '.'
    WEIGHT_SAVE
