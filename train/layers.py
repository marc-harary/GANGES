import torch
from torch import nn
import pytorch_lightning as pl
import wandb

class Generator(nn.Module):
    def __init__(self, im_shape, latent_dim=100, hidden_dim=128, scale=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(scale),
            nn.Linear(hidden_dim, int(torch.prod(torch.tensor(im_shape)))),
            nn.Tanh()
        )
        self.im_shape = im_shape

    def forward(self, z):
        im = self.model(z)
        im = im.view(im.size(0), *self.im_shape)
        return im

class Discriminator(nn.Module):
    def __init__(self, im_shape, hidden_dim=128, scale=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(im_shape))), hidden_dim),
            nn.LeakyReLU(scale),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, im):
        im_flat = im.view(im.size(0), -1)
        validity = self.model(im_flat)
        return validity
