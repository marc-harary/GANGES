import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py

class Hdf5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __len__(self):
        with h5py.File(self.file_path, 'r') as file:
            return len(file['hne_patches'])

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            hne_patch = torch.from_numpy(file['hne_patches'][idx]).float()
            ihc_patch = torch.from_numpy(file['ihc_patches'][idx]).float()
        return hne_patch, ihc_patch

class GANDataModule(pl.LightningDataModule):
    def __init__(self, file_path, batch_size=32, train_val_test_split=[0.7, 0.15, 0.15]):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split

    def setup(self, stage=None):
        # Create dataset
        dataset = Hdf5Dataset(self.file_path)

        # Split dataset
        train_size = int(self.train_val_test_split[0] * len(dataset))
        val_size = int(self.train_val_test_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
