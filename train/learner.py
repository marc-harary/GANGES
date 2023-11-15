import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from train.data import GANDataModule
from train.layers import *
from typing import Iterable, Callable
from torch.optim import Optimizer


OptimizerCallable = Callable[[Iterable], Optimizer]


class GAN(pl.LightningModule):
    def __init__(
        self,
        optim_g: OptimizerCallable,
        optim_d: OptimizerCallable,
        im_shape=(3, 64, 64),
        latent_dim=100,
        hidden_dim=128,
        scale=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(
            im_shape=im_shape,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            scale=scale,
        )
        self.discriminator = Discriminator(
            im_shape=im_shape, hidden_dim=hidden_dim, scale=scale
        )
        self.optim_g = optim_g
        self.optim_d = optim_d

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        ims, _ = batch

        # Sample noise
        z = torch.randn(ims.shape[0], self.hparams.latent_dim)
        z = z.type_as(ims)

        # Train generator
        if optimizer_idx == 0:
            self.generated_ims = self(z)
            g_loss = self.adversarial_loss(
                self.discriminator(self.generated_ims), torch.ones(ims.size(0), 1)
            )
            self.log(
                f"train/g_loss",
                g_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return g_loss
        # Train discriminator
        elif optimizer_idx == 1:
            real_loss = self.adversarial_loss(
                self.discriminator(ims), torch.ones(ims.size(0), 1)
            )
            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_ims.detach()),
                torch.zeros(ims.size(0), 1),
            )
            d_loss = (real_loss + fake_loss) / 2
            self.log(
                f"train/d_loss",
                d_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return d_loss

    def _shared_step(self, batch, batch_idx, stage):
        ims, _ = batch

        # Sample noise
        z = torch.randn(ims.shape[0], self.hparams.latent_dim)
        z = z.type_as(ims)

        # Generate images
        generated_ims = self(z)

        # Calculate the generator loss
        g_loss = self.adversarial_loss(
            self.discriminator(generated_ims), torch.ones(ims.size(0), 1)
        )

        # Calculate the discriminator loss
        real_loss = self.adversarial_loss(
            self.discriminator(ims), torch.ones(ims.size(0), 1)
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(generated_ims.detach()), torch.zeros(ims.size(0), 1)
        )
        d_loss = (real_loss + fake_loss) / 2

        # Log the losses
        self.log(f"{stage}/g_loss", g_loss, on_epoch=True, prog_bar=False)
        self.log(f"{stage}/d_loss", d_loss, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optim_g = self.optim_g(self.parameters())
        optim_d = self.optim_d(self.parameters())
        return [optim_g, optim_d]
