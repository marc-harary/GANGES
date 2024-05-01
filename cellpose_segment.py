import os
import random
import sys
from pathlib import Path
import h5py
import openslide
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
# import squidpy as sq
from skimage import color, filters, measure, morphology
from tqdm import tqdm
from cellpose import models



def segment(im, model):
    # mask15, *_ = model.eval(im, diameter=5.0, invert=False, batch_size=32)
    mask30, *_ = model.eval(im, diameter=30.0, invert=False, batch_size=32)
    mask40, *_ = model.eval(im, diameter=40.0, invert=True, batch_size=32)

    mask = np.zeros_like(mask30)
    # mask30 += mask15.max()
    mask40 += mask30.max()
    # mask[mask15 > mask15.min()] = mask15[mask15 > mask15.min()]
    mask[mask30 > mask30.min()] = mask30[mask30 > mask30.min()]
    mask[mask40 > mask40.min()] = mask40[mask40 > mask40.min()]

    regions = measure.regionprops(mask)
    frac_thresh = .05
    dab_thresh = .25

    seg_both = np.zeros_like(mask)

    label = color.rgb2hed(im)
    dab = label[..., 2]
    dab_mask = dab > dab_thresh

    for region in regions:
        ii, jj = region.coords[:, 0], region.coords[:, 1]
        dab_reg = dab_mask[ii, jj]
        lab_val = 2 if dab_reg.mean() > frac_thresh else 1
        seg_both[ii, jj] = lab_val

    return seg_both


def process_slide(slide, mask, patch_size, model, output_path, num_samples=None):
    num_patches_y, num_patches_x = mask.shape

    # Determine the total number of patches to sample
    if num_samples is None or num_samples > mask.sum():
        num_samples = mask.sum()

    sampled_indices = []
    while len(sampled_indices) < num_samples:
        y = random.randint(0, num_patches_y - 1)
        x = random.randint(0, num_patches_x - 1)
        if mask[y, x] and (y, x) not in sampled_indices:
            sampled_indices.append((y, x))

    tot_patches = len(sampled_indices)

    # Create an HDF5 file for output
    with h5py.File(output_path, 'w') as h5f:
        dset_seg = h5f.create_dataset('segmentation', (tot_patches, patch_size, patch_size), dtype=np.int32)
        dset_patch = h5f.create_dataset('patch', (tot_patches, patch_size, patch_size, 3), dtype=np.int32)

        # Initialize the progress bar and offset for dataset indexing
        offset = 0
        with tqdm(total=tot_patches) as pbar:
            for (y, x) in sampled_indices:
                patch = slide.read_region((x*patch_size, y*patch_size), 0, (patch_size, patch_size))
                patch_rgb = np.array(patch.convert('RGB'))
                seg = segment(patch_rgb, model)
                dset_patch[offset, ...] = patch_rgb
                dset_seg[offset, ...] = seg
                pbar.update(1)
                offset += 1


def process_im(im, patch_size, model):
    height, width = im.shape[:2]
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    dset = np.zeros(im.shape[:2], dtype=int)

    # Initialize the progress bar
    with tqdm(total=num_patches_x * num_patches_y) as pbar:
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = im[y:y+patch_size, x:x+patch_size, :]
                segmentation = segment(patch, model)
                dset[y:y+patch_size, x:x+patch_size] = segmentation
                pbar.update(1)

    return dset


def find_file_starting_with(prefix, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                return os.path.join(root, file)
    return None


def main():
    df = pd.read_csv("ihc-hne.csv")
    model = models.Cellpose(gpu=True)
    thumbdir = "/mnt/disks/raw-data/thumbnails/thumbnails256/"

    files = df["vasa"]
    files = list(files[~files.isna()])
    random.shuffle(files)

    for slide_name in tqdm(files):
        slide_name = find_file_starting_with(slide_name, "/mnt/disks/raw-data")
        if not slide_name:
            continue

        ihc_thumbnail = slide_name.replace('/mnt/disks/raw-data/', thumbdir).replace("svs", "npy")
        try:
            ihc_thumbnail = 255 * np.load(ihc_thumbnail)
        except:
            continue

        ihc_gray = color.rgb2gray(ihc_thumbnail)
        ihc_mask = ihc_gray <= np.percentile(ihc_gray.ravel(), 50)

        slide = openslide.open_slide(slide_name)
        output_path = (Path("/mnt/disks/disk2/") / Path(slide_name).name).with_suffix(".h5")
        process_slide(slide=slide, mask=ihc_mask, patch_size=256, model=model, output_path=output_path, num_samples=1000)


if __name__ == "__main__":
    main()
