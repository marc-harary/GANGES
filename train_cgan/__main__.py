import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from train_cgan.data import SegmentationDataModule
from train_cgan.learner import cGANLearner

def main():
    cli = LightningCLI(cGANLearner, SegmentationDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    main()
