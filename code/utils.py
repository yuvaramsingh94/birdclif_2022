import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import torchaudio
from sklearn.metrics import f1_score, roc_auc_score
from config.config import config

# import h5py


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore


#     torch.backends.cudnn.deterministic = True  # type: ignore
#     torch.backends.cudnn.benchmark = True  # type: ignore
# https://github.com/pratogab/batch-transforms/blob/master/batch_transforms.py


class birdclef_dataset(data.Dataset):
    def __init__(
        self,
        main_df,
        augmentation=None,
        path=None,
        aug_per=0.0,
        is_validation=False,
        secondary_effectiveness=0.5,
    ):
        self.main_df = main_df
        self.path = path
        self.augmentation = augmentation
        self.aug_per = aug_per
        self.is_validation = is_validation
        self.secondary_effectiveness = secondary_effectiveness
        self.n_classes = config.N_CLASSES

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["itemid"]

        # To try
        # waveform, sample_rate = torchaudio.load(f"{self.path}/{ids}.wav")
        waveform = np.zeros([1, 441000])
        # print(sample_rate)
        if not self.is_validation:

            wave_x = waveform.squeeze()

            if self.augmentation != None:
                if random.uniform(0.0, 1.0) < self.aug_per:
                    # print('augmentation')
                    #wave_x = self.augmentation(wave_x.numpy())
                    wave_x = self.augmentation(wave_x)
                wave_x = torch.from_numpy(wave_x)

            """
            if wave_x.shape[0] != self.effective_sec * sample_rate:
                #print
                result = torch.zeros(self.effective_sec * sample_rate)
                result[:wave_x.shape[0]] = wave_x
                wave_x = result
            """
        else:
            wave_x = waveform.squeeze()

        ## label manupulation
        info
        prediction = info["prediction"]
        primary_ = int(info["primary_label_index"])
        #print('primary_label', primary_label, type(primary_label))
        secondary_label = info["secondary_label_index"]
        
        primary_label = np.array([0.]*self.n_classes)
        primary_label[primary_] = 1 * prediction

        if len(secondary_label) > 0:
            for i in secondary_label:
                temp = np.array([0.]*self.n_classes)
                temp[i] = 1 * self.secondary_effectiveness
                primary_label += temp
        #return {"waveform": wave_x, "targets": primary_label}
        #print('THis is type ',type(wave_x))
        return  wave_x, torch.from_numpy(primary_label)


class focal_loss(nn.Module):
    def __init__(self, alpha, gamma=2, if_sigmoid=False):
        super(focal_loss, self).__init__()
        self.alpha = alpha#torch.Tensor([alpha, 1 - alpha])  # .cuda()
        # self.alpha = self.alpha.to(device)
        self.gamma = gamma
        self.if_sigmoid = if_sigmoid
        # self.bce_loss = nn.BCELoss(reduction="none")

    def forward(
        self,
        inputs,
        targets,
    ):
        if self.if_sigmoid:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        ## i guess this is going to give use some thing like [0.25,0.25,1-0.25]
        ## for target of [0,0,1]. i guess gather will do this for us
        at = self.alpha.gather(0, targets.data.view(-1))
        ## here we apply the exp for the log values
        ## this is very tricky. if you see this is for choosing p pr 1-p based on 0 or 1
        ## its an inteligent way to do the choosing and araiving at the value fast
        ## without this you have to do some hard engineering to get this value
        # print('bce ',BCE_loss.shape)

        pt = torch.exp(-BCE_loss)
        # print('rest ',(at * (1.0 - pt) ** self.gamma))

        F_loss = (at * (1.0 - pt) ** self.gamma) * (BCE_loss)
        return F_loss.mean()


class PANNsLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()

        self.bce = focal_loss(
            alpha=alpha, gamma=3, if_sigmoid = True
        )  # nn.BCELoss(weight = torch.tensor(class_weights, requires_grad = False))

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_), torch.zeros_like(input_), input_)
        input_ = torch.where(torch.isinf(input_), torch.zeros_like(input_), input_)

        target = target.float()
        # print(input_.shape)
        return self.bce(input_.view(-1), target.view(-1))


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
