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
from train_seg.learner import *
from train_cgan.learner import *


def segment(im, model, hne=False):
    # mask15, *_ = model.eval(im, diameter=5.0, invert=False, batch_size=32)
    # mask30, *_ = model.eval(im, diameter=30.0, invert=False, batch_size=32)
    mask30, *_ = model.eval(im, diameter=30.0, invert=False, batch_size=32)
    mask40, *_ = model.eval(im, diameter=40.0, invert=hne, batch_size=32)

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


def process_slide(slide, mask, patch_size, model, output_path):
    num_patches_y, num_patches_x = mask.shape
    tot_patches = mask.sum()

    # Create an HDF5 file for output
    with h5py.File(output_path, 'w') as h5f:
        # Create a dataset to store the segmentation results
        dset_seg = h5f.create_dataset('segmentation', (tot_patches, patch_size, patch_size), dtype=np.int32)
        dset_patch = h5f.create_dataset('patch', (tot_patches, patch_size, patch_size, 3), dtype=np.int32)

        # Initialize the progress bar
        offset = 0
        with tqdm(total=tot_patches) as pbar:
            for y in range(num_patches_y):
                for x in range(num_patches_x):
                    if not mask[y, x]:
                        continue
                    patch = slide.read_region((x*patch_size, y*patch_size), 0, (patch_size, patch_size))
                    patch_rgb = np.array(patch.convert('RGB'))
                    seg = segment(patch_rgb, model)
                    dset_patch[offset, ...] = patch_rgb
                    dset_seg[offset, ...] = seg
                    pbar.update(1)
                    offset += 1



def process_im(im, patch_size, model, hne):
    height, width = im.shape[:2]
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    dset = np.zeros(im.shape[:2], dtype=int)

    # Initialize the progress bar
    with tqdm(total=num_patches_x * num_patches_y) as pbar:
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = im[y:y+patch_size, x:x+patch_size, :]
                segmentation = segment(patch, model, hne)
                dset[y:y+patch_size, x:x+patch_size] = segmentation
                pbar.update(1)

    return dset


def process_im_seg(im, patch_size, model, hne):
    height, width = im.shape[:2]
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    dset = np.zeros(im.shape[:2], dtype=int)

    # Initialize the progress bar
    with tqdm(total=num_patches_x * num_patches_y) as pbar:
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = im[y:y+patch_size, x:x+patch_size, :]
                segmentation = segment(patch, model, hne)
                dset[y:y+patch_size, x:x+patch_size] = segmentation
                pbar.update(1)

    return dset


def process_im_seg(seg, patch_size, model):
    """
    This function processes the segmentation result patch by patch using the provided model.
    """
    height, width = seg.shape
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    final_result = np.zeros(seg.shape, dtype=int)

    with tqdm(total=num_patches_x * num_patches_y) as pbar:
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = seg[y:y+patch_size, x:x+patch_size]
                patch_torch = torch.from_numpy(patch)[None, None, ...].float().cuda()
                with torch.no_grad():
                    output = model(patch_torch).squeeze().permute(1, 2, 0)
                result_patch = output.argmax(-1).cpu().numpy()
                final_result[y:y+patch_size, x:x+patch_size] = result_patch
                pbar.update(1)

    return final_result


def process_patches_with_cgan(input_array, patch_size, model):
    """
    This function processes the input array in patches, applying the cGAN model to each patch.
    """
    height, width = input_array.shape
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # Assuming the model output has the same size as input for simplicity
    final_result = np.zeros((height, width, 3), dtype=np.float32)

    with tqdm(total=num_patches_x * num_patches_y) as pbar:
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                # Extract the patch
                patch = input_array[y:y+patch_size, x:x+patch_size]
                patch_torch = torch.from_numpy(patch)[None, None, ...].float().cuda()

                # Process the patch with no_grad to prevent memory leak during inference
                with torch.no_grad():
                    output_patch = model(patch_torch).squeeze().permute(1, 2, 0).cpu().numpy()

                # Store the processed patch back into the result array
                final_result[y:y+patch_size, x:x+patch_size, :] = output_patch
                pbar.update(1)

    return final_result



def main():
    patch_size = 2 * 1024
    # im_size = 10 * 1024
    im = plt.imread("hne6.png")
    im = color.rgba2rgb(im)
    im = im[:-3000, :-3000, :]
    # # im = im[:im_size, :im_size, :]
    # # print(im.shape)

    # # # run cell pose
    # cell_pose_model = models.Cellpose(gpu=True)
    # seg = process_im(im, patch_size=patch_size, model=cell_pose_model, hne=True)

    # np.save("seg", seg)
    # exit()
    seg = np.load("seg.npy")
    seg = seg[:-3000, :-3000]
    # seg = seg[:1000, :1000]

    # run segmenter
    # path = Path("detect_pos/qz9qg97e/checkpoints/")
    path = Path("detect_pos/e3tyeod4/checkpoints/")
    file = list(path.iterdir())[0]
    segmenter = UNetLearner.load_from_checkpoint(file).eval()
    opt = process_im_seg(seg, patch_size, segmenter)
    # seg_torch = torch.from_numpy(seg)
    # seg_torch = seg_torch[None, None, ...].float().cuda()
    # with torch.no_grad():
    #     seg_lab = segmenter(seg_torch).squeeze().permute(1, 2, 0)
    # seg_lab = seg_lab.argmax(-1).float()

    # load cGAN
    path = Path("ihc_cgan/wgp6p79t/checkpoints/")
    file = list(path.iterdir())[0]
    learner = cGANLearner.load_from_checkpoint(file)
    cgan = learner.generator.eval()
    opt2 = process_patches_with_cgan(opt, patch_size, cgan)

    plt.imshow(opt2)
    plt.show()

    plt.imsave("cgan6_vasa.png", opt2)
    exit()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im)
    axs[1].imshow(opt2)
    plt.show()
    # with torch.no_grad():
    #     ihc = cgan(seg_lab[None, None, ...])

    # convert predu
    ihc = ihc.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    seg_lab = seg_lab.detach().cpu().numpy()

    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    axs[0].set_title("Input H&E")
    axs[0].imshow(im)
    axs[1].imshow(seg)
    axs[1].set_title("Nuclei segmentation (via CellPose)")
    axs[2].imshow(seg_lab)
    axs[2].set_title("Nuclei classification (OCT4+ vs. OCT4-)")
    axs[3].imshow(ihc)
    axs[3].set_title("Predicted IHC (via cGAN)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
