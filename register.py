import matplotlib.pyplot as plt
import openslide
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from skimage import transform, color, filters, feature
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac
from copy import deepcopy


def register(
    src_im,
    dst_im,
    n_keypoints=200,
    fast_threshold=0.05,
    min_samples=3,
    residual_threshold=2,
    max_trials=100,
):
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
    orb.detect_and_extract(dst_im)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    orb.detect_and_extract(src_im)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    # Match descriptors between images
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Select matched keypoints
    src = keypoints2[matches[:, 1]][:, ::-1]
    dst = keypoints1[matches[:, 0]][:, ::-1]

    # Perform robust affine transformation
    model_robust, _ = ransac(
        (src, dst),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
    )

    return model_robust


def warp_register(dst_im, model, scale=1):
    model_cpy = deepcopy(model)
    model_cpy.params[0, 2] *= scale
    model_cpy.params[1, 2] *= scale
    im_opt = warp(dst_im, model_cpy.inverse, mode="symmetric")
    return im_opt


def main():
    files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk("thumbnails256")
        for file in files
        if file.endswith(".npy")
    ]
    files.sort()

    hne = np.load(files[981])
    ihc = np.load(files[1034])
    ihc = transform.rotate(ihc, 180, resize=False)

    hne_gray = color.rgb2gray(hne)
    ihc_gray = color.rgb2gray(ihc)

    model = register(
        ihc_gray,
        hne_gray,
        n_keypoints=1000,
        fast_threshold=0.05,
        min_samples=3,
        residual_threshold=2,
        max_trials=1000,
    )
    ihc_image_registered = warp_register(ihc_gray, model)


    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(hne)
    axs[0].set_title("Destination H&E")
    axs[1].imshow(ihc_image_registered)
    axs[1].set_title("Registered IHC")
    axs[2].imshow(ihc)
    axs[2].set_title("Original IHC")
    plt.show()

    files32 = [
        os.path.join(root, file)
        for root, dirs, files in os.walk("thumbnails32")
        for file in files
        if file.endswith(".png")
    ]
    files32.sort()
    hne32 = plt.imread(files32[981])
    ihc32 = plt.imread(files32[1034])
    ihc32 = transform.rotate(ihc32, 180, resize=False)

    ihc_image_registered32 = warp_register(ihc32, model, scale=8)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(hne32)
    axs[0].set_title("Destination H&E")
    axs[1].imshow(ihc_image_registered32)
    axs[1].set_title("Registered IHC")
    axs[2].imshow(ihc32)
    axs[2].set_title("Original IHC")
    plt.show()


if __name__ == "__main__":
    main()
