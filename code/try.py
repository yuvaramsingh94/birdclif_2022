from joblib import delayed, Parallel
import pandas as pd
import librosa
import librosa.display
import soundfile as sf

train_df = pd.read_csv("../data/birdclef-2022/train_metadata.csv")

folder_wav = "../data/wav/competition/v1/10_sec_1/"
for i in train_df["primary_label"].value_counts().index.values:
    os.mkdir(folder_wav + i)

SR = 44100
SEC = 10
atleast_sec = 3
folder_wav = "../data/wav/competition/v1/10_sec_1/"


def wav_10sec(
    file,
):
    file = file.replace("ogg", "wav")
    #print(file)
    y, sr = librosa.load(f"../data/birdclef-2022/train_audio_wav/{file}", sr=None)

    for i in range(0, int(len(y) / 44100), SEC):

        ## lets check if the sound track is atleast greater than 3 sec
        wave = y[i * SR : (i + SEC) * SR]

        if len(wave) > atleast_sec * SR:

            if len(wave) < SEC * SR:
                wave = np.concatenate(
                    [wave, np.array([0.0] * abs(wave.shape[0] - SEC * SR))]
                )

            ## wav 10 sec save
            sf.write(folder_wav + f"{file}_{i+SEC}.wav", wave, SR)

for file in tqdm(train_df["filename"].values):
    wav_10sec(file)