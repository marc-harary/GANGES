import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from train.data import GANDataModule
from train.learner import GAN

def main():
    cli = LightningCLI(GAN, GANDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    main()
