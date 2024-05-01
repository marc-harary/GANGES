import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from train_seg.data import SegmentationDataModule
from train_seg.learner import UNetLearner

def main():
    cli = LightningCLI(UNetLearner, SegmentationDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    main()
