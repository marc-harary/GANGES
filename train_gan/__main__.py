import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from train_gan.data import GANDataModule
from train_gan.learner import CycleGAN

def main():
    cli = LightningCLI(CycleGAN, GANDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    main()
