from pytorch_lightning import LightningModule, Trainer
import pandas as pd
import numpy as np
from model import TimmSED
from utils import PANNsLoss, birdclef_dataset, set_seed
from config.config import config
import torch
import torch.utils.data as data
import torch.nn.functional as F
import augmentations as ag
set_seed(config.SEED)


AVAIL_GPUS = max(0, torch.cuda.device_count())


class LitBIRD_V1(LightningModule):
    def __init__(
        self,
        train_df,
        valid_df,
        learning_rate=2e-4,
    ):

        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.train_df = train_df
        self.valid_df = valid_df

        # Define PyTorch model
        self.model = TimmSED(
            base_model_name=config.MODEL_TYPE,
            pretrained=True,
            num_classes=config.N_CLASSES,
            in_channels=1,
        )
        
        print('this is device',self.device)
        alpha = 0.3
        #self.alpha = torch.Tensor([alpha, 1 - alpha])
        #self.loss_fn = PANNsLoss(alpha)#.cuda()
        

    def forward(self, x):
        x = x.float()
        #print(x)
        #print('X',x.shape)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        #print('batch',batch)
        #x, y = batch['waveform'], batch['targets']
        x, y = batch
        logits = self(x)
        #loss = self.loss_fn(logits, y)
        loss = F.binary_cross_entropy_with_logits(logits['clipwise_output'], y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        #loss = self.loss_fn(logits, y)
        loss = F.binary_cross_entropy_with_logits(logits['clipwise_output'], y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    # def test_step(self, batch, batch_idx):
    #    # Here we just reuse the validation_step for testing
    #    return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):

        transform = ag.Compose(
            [
                ag.OneOf(
                    [
                        ag.GaussianNoiseSNR(
                            min_snr=10, max_snr=20.0, p=1.0
                        ),  # this is the change
                        ag.PinkNoiseSNR(min_snr=10, max_snr=20.0, p=1.0),  #
                    ]
                )
            ]
        )

        train_dataset = birdclef_dataset(
            self.train_df[:50],
            augmentation=transform,
            path=config.DATA_PATH,
            aug_per=config.AUG_PER,
            is_validation=False,
            secondary_effectiveness=config.SECONDAEY_EFF,
        )

        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.WORKERS,
            drop_last=True,
            # pin_memory=True,
        )

        return train_dataloader

    def val_dataloader(self):
        valid_dataset = birdclef_dataset(
            self.valid_df[:50],
            augmentation=None,
            path=config.DATA_PATH,
            aug_per=0.0,
            is_validation=True,
            secondary_effectiveness=config.SECONDAEY_EFF,
        )

        valid_dataloader = data.DataLoader(
            valid_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            drop_last=True,
            # pin_memory=True,
        )

        return valid_dataloader

    # def test_dataloader(self):
    #    return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


main_df = pd.read_csv(config.DATAFRAME_PATH)
main_df["secondary_label_index"] = main_df["secondary_label_index"].apply(eval)
for fold in range(config.FOLD):
    train_df = main_df[main_df["fold"] != fold].reset_index(
        drop=True
    )  # actually its v2 with 5 fold
    valid_df = main_df[main_df["fold"] == fold].reset_index(
        drop=True
    )  # better_df[better_df['fold'] == fold]

    lightning_model = LitBIRD_V1(
        train_df,
        valid_df,
        learning_rate=config.LR_START,
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=config.EPOCHS,
        progress_bar_refresh_rate=20,
    )
    trainer.fit(lightning_model)
