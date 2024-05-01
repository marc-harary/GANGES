import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.segmentation = self.hdf5_file["segmentation"]
        self.patch = self.hdf5_file["patch"]

    def __len__(self):
        return self.segmentation.shape[0]

    def __getitem__(self, index):
        # Load segmentation map
        seg = torch.tensor(self.segmentation[index]).float()
        # seg_bin = (seg > 0).float().unsqueeze(0)  # Adding channel dimension

        # Load corresponding RGB image
        rgb = torch.tensor(self.patch[index]).float().permute(2, 0, 1)  # Permute to CxHxW format
        rgb /= 255

        return seg.unsqueeze(0), rgb  # Return segmentation map and RGB image pair

    def close(self):
        self.hdf5_file.close()

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, file_path, batch_size=32, train_val_test_split=(0.7, 0.2, 0.1)):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split

    def setup(self, stage=None):
        full_dataset = HDF5Dataset(self.file_path)
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_val_test_split[0])
        val_size = int(total_size * self.train_val_test_split[1])
        test_size = total_size - train_size - val_size

        # Random split the dataset into training, validation, and test sets
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
