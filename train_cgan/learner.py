import pytorch_lightning as pl
import torch
from torch import nn
from train_cgan.layers import UNet, Discriminator  # Assuming these are defined in your module
from pytorch_lightning.loggers import WandbLogger

class cGANLearner(pl.LightningModule):
    def __init__(self, n_channels=1, n_classes=3, bilinear=True, lambda_pixel=10):
        super().__init__()
        self.generator = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        self.discriminator = Discriminator(input_channels=n_channels + n_classes)
        self.loss_GAN = nn.BCELoss()
        self.loss_pixelwise = nn.L1Loss()
        self.lambda_pixel = lambda_pixel
        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return self.loss_GAN(y_hat, y)

    def pixelwise_loss(self, y_hat, y):
        return self.loss_pixelwise(y_hat, y)

    def training_step(self, batch, batch_idx):
        seg_maps, real_images = batch
        valid = torch.ones((real_images.size(0), 1), device=self.device, requires_grad=False)
        fake = torch.zeros((real_images.size(0), 1), device=self.device, requires_grad=False)

        # Get optimizers
        opt_G, opt_D = self.optimizers()

        # Train Generator
        opt_G.zero_grad()
        generated_images = self(seg_maps)
        pred_fake = self.discriminator(torch.cat((seg_maps, generated_images), 1))
        loss_GAN = self.adversarial_loss(pred_fake, valid)
        loss_pixel = self.pixelwise_loss(generated_images, real_images)
        loss_G = loss_GAN + self.lambda_pixel * loss_pixel
        self.manual_backward(loss_G)
        opt_G.step()

        # Log generator loss
        self.log('train/loss_G', loss_G, on_step=True, on_epoch=True, prog_bar=True)

        # Train Discriminator
        opt_D.zero_grad()
        pred_real = self.discriminator(torch.cat((seg_maps, real_images), 1))
        loss_real = self.adversarial_loss(pred_real, valid)
        pred_fake = self.discriminator(torch.cat((seg_maps, generated_images.detach()), 1))
        loss_fake = self.adversarial_loss(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        self.manual_backward(loss_D)
        opt_D.step()

        # Log discriminator loss
        self.log('train/loss_D', loss_D, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss_G': loss_G, 'loss_D': loss_D}

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [opt_G, opt_D], []

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code("train_seg")

