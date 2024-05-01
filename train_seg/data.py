import h5py
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import numpy as np

# class HDF5Dataset(Dataset):
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.hdf5_file = h5py.File(file_path, 'r')
#         self.dataset = self.hdf5_file["segmentation"]
# 
#     def __len__(self):
#         return self.dataset.shape[0]
# 
#     def __getitem__(self, index):
#         seg = torch.tensor(self.dataset[index]).long()
#         seg_bin = (seg > 0).float()
#         return seg_bin.unsqueeze(0), seg
# 
#     def close(self):
#         self.hdf5_file.close()
# 
# class SegmentationDataModule(pl.LightningDataModule):
#     def __init__(self, file_path, batch_size=32, train_val_test_split=(0.7, 0.2, 0.1)):
#         super().__init__()
#         self.file_path = file_path
#         self.batch_size = batch_size
#         self.train_val_test_split = train_val_test_split
# 
#     def setup(self, stage=None):
#         full_dataset = HDF5Dataset(self.file_path)
#         total_size = len(full_dataset)
#         train_size = int(total_size * self.train_val_test_split[0])
#         val_size = int(total_size * self.train_val_test_split[1])
#         test_size = total_size - train_size - val_size
# 
#         # Random split the dataset into training, validation, and test
#         self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
# 
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
# 
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size)
# 
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)

# class HDF5Dataset(Dataset):
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.hdf5_file = h5py.File(file_path, 'r')
#         self.dataset = self.hdf5_file
# 
#     def __len__(self):
#         return len(self.dataset.keys())
# 
#     def __getitem__(self, index):
#         seg = torch.tensor(self.dataset[index]).long()
#         seg_bin = (seg > 0).float()
#         return seg_bin.unsqueeze(0), seg
# 
#     def close(self):
#         self.hdf5_file.close()
# # class SegmentationDataModule(pl.LightningDataModule): #     def __init__(self, file_path, batch_size=32, train_val_test_split=(0.7, 0.2, 0.1)):
#         super().__init__()
#         self.file_path = file_path
#         self.batch_size = batch_size
#         self.train_val_test_split = train_val_test_split
# 
#     def setup(self, stage=None):
#         full_dataset = HDF5Dataset(self.file_path)
#         total_size = len(full_dataset)
#         train_size = int(total_size * self.train_val_test_split[0])
#         val_size = int(total_size * self.train_val_test_split[1])
#         test_size = total_size - train_size - val_size
# 
#         # Random split the dataset into training, validation, and test
#         self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
# 
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
# 
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size)
# 
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.keys = []
        self.index_map = []  # Maps a linear index to a (file_key, tile_index) tuple

        # Preprocess to map linear index to a specific tile in a specific whole-slide image
        with h5py.File(self.hdf5_path, 'r') as f:
            for key in f.keys():
                for tile_index in range(f[key].shape[0]):
                    self.keys.append(key)
                    self.index_map.append((key, tile_index))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        key, tile_index = self.index_map[idx]
        with h5py.File(self.hdf5_path, 'r') as f:
            tile = torch.tensor(f[key][tile_index])
        seg = tile[..., 0]
        seg = seg.long()
        seg_bin = (seg > 0).float().unsqueeze(0)
        return seg_bin, seg


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, hdf5_path, batch_size, train_val_test_split=(0.7, 0.2, 0.1)):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.dataset = HDF5Dataset(self.hdf5_path)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Calculate the sizes of each split
        train_size = int(self.train_val_test_split[0] * len(self.dataset))
        val_size = int(self.train_val_test_split[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

