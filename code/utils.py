import torch
import torch.utils.data as data
import numpy as np
import os
import random
import torchaudio


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore


# https://github.com/pratogab/batch-transforms/blob/master/batch_transforms.py


class binarybird_dataset(data.Dataset):
    def __init__(
        self,
        main_df,
        augmentation=None,
        path=None,
        effective_sec=5,
        aug_per=0.0,
        is_validation=False,
    ):
        self.main_df = main_df
        self.augmentation = augmentation
        self.path = path
        self.effective_sec = effective_sec
        self.aug_per = aug_per
        self.is_validation = is_validation
        self.sample_rate = 44100

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["itemid"]
        dset = info["datasetid"]
        waveform, sample_rate = torchaudio.load(f"{self.path}/{dset}/audio/wav/{ids}.wav")
        if self.sample_rate != sample_rate:
            raise Exception("Sorry, sample_rate is not same as ", self.sample_rate)
        # print(sample_rate)
        if not self.is_validation:
            duration = len(waveform) // self.sample_rate  # in sec
            wave_x = waveform
            wave_x = wave_x.squeeze()  # since all of them are mono.

            if self.augmentation != None:
                if random.uniform(0.0, 1.0) < self.aug_per:
                    # print('augmentation')
                    wave_x = self.augmentation(wave_x.numpy())
                    wave_x = torch.from_numpy(wave_x)
            
            label = np.array(info["hasbird"])
            return {"waveform": wave_x, "targets": label}
        else:
            wave_x = waveform.squeeze()
            #wave_x = torch.from_numpy(wave_x)

        label = np.array(info["hasbird"])
        return {"waveform": wave_x, "targets": label, 'ids':ids}
