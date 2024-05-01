import pytorch_lightning as pl
from train_gan.data import GANDataModule
from train_gan.layers import *
from torchmetrics.image import *
from piq import *
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn


class CycleGAN(pl.LightningModule):
    def __init__(
        self, input_channels=3, output_channels=3, lr=2e-4,
    ):
        super(CycleGAN, self).__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Generators
        self.gen_hne2ihc = UNetGenerator(
        )
        self.gen_ihc2hne = UNetGenerator(
        )

        # Discriminators
        self.disc_hne = Discriminator()
        self.disc_ihc = Discriminator()

        # Losses
        self.mae_loss = nn.L1Loss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.ssim_loss = SSIM(data_range=1.0)


    def training_step(self, batch, batch_idx):
        real_hne, real_ihc = batch

        # Get the optimizers
        opt_gen, opt_disc_hne, opt_disc_ihc = self.optimizers()

        # Generate fake images using generators
        fake_ihc = self.gen_hne2ihc(real_hne)
        fake_hne = self.gen_ihc2hne(real_ihc)

        # ---------------------
        #  Train Generators
        # ---------------------
        # Calculate cycle consistency loss
        cycle_hne = self.gen_ihc2hne(fake_ihc)
        cycle_ihc = self.gen_hne2ihc(fake_hne)
        cycle_loss = (self.mae_loss(cycle_hne, real_hne) + self.mae_loss(cycle_ihc, real_ihc)) / 2

        cycle_loss += (self.ssim_loss(cycle_hne, real_hne) + self.ssim_loss(cycle_ihc, real_ihc)) / 2

        # Calculate adversarial losses for both generators
        valid_ihc = self.disc_ihc(fake_ihc)
        valid_hne = self.disc_hne(fake_hne)
        adv_loss = (self.adv_loss(valid_ihc, torch.ones_like(valid_ihc)) + self.adv_loss(valid_hne, torch.ones_like(valid_hne))) / 2

        id_loss_ihc = self.mae_loss(self.gen_hne2ihc(real_ihc), real_ihc)
        id_loss_hne = self.mae_loss(self.gen_ihc2hne(real_hne), real_hne)

        id_loss_ihc += self.ssim_loss(self.gen_hne2ihc(real_ihc), real_ihc)
        id_loss_hne += self.ssim_loss(self.gen_ihc2hne(real_hne), real_hne)

        id_loss = (id_loss_ihc + id_loss_hne) / 2

        # Total generator loss
        total_gen_loss = cycle_loss + adv_loss + id_loss

        # Perform backpropagation and optimization on generators
        self.manual_backward(total_gen_loss)
        opt_gen.step()
        opt_gen.zero_grad()

        # ---------------------
        #  Train Discriminator H&E
        # ---------------------
        # Calculate loss for real and fake images
        real_loss_hne = self.adv_loss(self.disc_hne(real_hne), torch.ones_like(self.disc_hne(real_hne)))
        fake_loss_hne = self.adv_loss(self.disc_hne(fake_hne.detach()), torch.zeros_like(self.disc_hne(fake_hne)))
        disc_hne_loss = (real_loss_hne + fake_loss_hne) / 2

        # Backprop and optimize discriminator H&E
        self.manual_backward(disc_hne_loss)
        opt_disc_hne.step()
        opt_disc_hne.zero_grad()

        # ---------------------
        #  Train Discriminator IHC
        # ---------------------
        real_loss_ihc = self.adv_loss(self.disc_ihc(real_ihc), torch.ones_like(self.disc_ihc(real_ihc)))
        fake_loss_ihc = self.adv_loss(self.disc_ihc(fake_ihc.detach()), torch.zeros_like(self.disc_ihc(fake_ihc)))
        disc_ihc_loss = (real_loss_ihc + fake_loss_ihc) / 2

        # Backprop and optimize discriminator IHC
        self.manual_backward(disc_ihc_loss)
        opt_disc_ihc.step()
        opt_disc_ihc.zero_grad()

        # Logging the losses
        self.log("train/cycle_loss", cycle_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/id_loss", id_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/adv_loss", adv_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/gen_loss", total_gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/disc_hne_loss", disc_hne_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/disc_ihc_loss", disc_ihc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx):
        pass

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code("train_gan")

    def configure_optimizers(self):
        opt_gen = Adam(list(self.gen_hne2ihc.parameters()) + list(self.gen_ihc2hne.parameters()), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_disc_hne = Adam(self.disc_hne.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_disc_ihc = Adam(self.disc_ihc.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_gen, opt_disc_hne, opt_disc_ihc], []
