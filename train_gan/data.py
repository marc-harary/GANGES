import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision.transforms import functional as TF
import openslide
import yaml


def create_paired_image_names(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    paired_names = [line.split(",") for line in lines]
    return paired_names


class SVSDataset(Dataset):
    def __init__(self, paired_image_paths, epoch_length, patch_size, transforms=None):
        self.paired_image_paths = paired_image_paths
        self.patch_size = patch_size
        self.transforms = transforms
        self.epoch_length = epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        he_path, ihc_path = random.choice(self.paired_image_paths)
        he_slide = openslide.OpenSlide(he_path)
        ihc_slide = openslide.OpenSlide(ihc_path)

        # Randomly choose a location to extract the patch
        width, height = he_slide.dimensions
        max_x, max_y = width - self.patch_size, height - self.patch_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        he_patch = he_slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
        ihc_patch = ihc_slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")

        if self.transforms:
            he_patch = self.transforms(he_patch)
            ihc_patch = self.transforms(ihc_patch)
        else:
            he_patch = TF.to_tensor(he_patch)
            ihc_patch = TF.to_tensor(ihc_patch)

        return he_patch, ihc_patch


class GANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path,
        thres,
        epoch_length=1000,
        train_batch_size=32,
        train_patch_size=64,
        val_batch_size=32,
        val_patch_size=64,
        split_ratios=[0.8, 0.10, 0.10],
        num_workers=1,
        transforms=None,
    ):
        super().__init__()
        self.file_path = file_path
        self.thres = thres
        self.train_batch_size = train_batch_size
        self.train_patch_size = train_patch_size
        self.val_batch_size = val_batch_size
        self.val_patch_size = val_patch_size
        self.split_ratios = split_ratios
        self.num_workers = num_workers
        self.transforms = transforms
        self.epoch_length = epoch_length

    def setup(self, stage=None):
        paired_image_names = create_paired_image_names(self.file_path)

        total_size = len(paired_image_names)
        train_size = int(total_size * self.split_ratios[0])
        val_size = int(total_size * self.split_ratios[1])
        test_size = total_size - train_size - val_size

        train_names, val_names, test_names = random_split(
            paired_image_names, [train_size, val_size, test_size]
        )

        self.train_dataset = SVSDataset(
            train_names, self.epoch_length, self.train_patch_size, self.transforms,
        )
        self.val_dataset = SVSDataset(
            val_names, self.epoch_length, self.val_patch_size, self.transforms,
        )
        self.test_dataset = SVSDataset(
            test_names, self.epoch_length, self.val_patch_size, self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)
