import torch
import torch.nn.functional as F
import torch.nn as nn
from config.config import config
import torchaudio
import torchaudio.transforms as T


class BirdBinaryModel(nn.Module):
    def __init__(
        self,
    ):
        super(BirdBinaryModel, self).__init__()

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=config.n_mels,
            mel_scale="htk",
        )  # .cuda()

        self.amp_to_db = T.AmplitudeToDB()  # .cuda()

        self.input_norm = nn.BatchNorm2d(3)

        ## spec augmentation
        ## normalization

    def input_viz(self, x):
        x = self.mel_spectrogram(x)
        x = self.amp_to_db(x)
        x = torch.stack([x, x, x], dim=-1).permute(0, 3, 1, 2) ## checked and it works perfectly
        x = self.input_norm(x)
        return x

    def forward(self, x):

        x = self.input_viz(x)
            
