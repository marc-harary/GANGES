import pytorch_lightning as pl
import torch
from torch import nn
from train_seg.layers import *
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex


class UNetLearner(pl.LightningModule):
    def __init__(self, n_channels=1, n_classes=3, bilinear=True):
        super().__init__()
        self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        self.metrics = dict(jac=MulticlassJaccardIndex(num_classes=3))
        for key, value in self.metrics.items():
            setattr(self, key, value)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, stage, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, targets)
        self.log(f'{stage}/loss', loss, prog_bar=True)
        for key, value in self.metrics.items():
            metric = getattr(self, key)
            self.log(f'{stage}/{key}', metric(outputs.flatten(-2, -1), targets.flatten(-2, -1)), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_step("test", batch, batch_idx)

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code("train_seg")
