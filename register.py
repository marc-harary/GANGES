# test

import matplotlib.pyplot as plt
import openslide
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from skimage import transform, color, filters, feature, exposure
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac
from copy import deepcopy
from skimage.feature import match_descriptors, plot_matches, SIFT, BRIEF
from skimage.exposure import match_histograms




def register(
    src_im,
    dst_im,
    n_keypoints=200,
    fast_threshold=0.05,
    min_samples=3,
    residual_threshold=2,
    max_trials=100,
):
    # orb = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
    # orb = SIFT(c_edge=10, c_dog=.02)
    orb = SIFT()
    orb.detect_and_extract(dst_im)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    orb.detect_and_extract(src_im)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    # Match descriptors between images
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)#, max_distance=10)#, max_ratio=1.0)

    # Select matched keypoints
    src = keypoints2[matches[:, 1]][:, ::-1]
    dst = keypoints1[matches[:, 0]][:, ::-1]

    fig, ax = plt.subplots()
    plot_matches(ax, src_im, dst_im, keypoints1, keypoints2, matches, keypoints_color='r')
    plt.show()
    # exit()

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
        for root, dirs, files in os.walk("../thumbnails/thumbnails256")
        for file in files
        if file.endswith(".npy")
    ]
    files.sort()
    files32 = [
        os.path.join(root, file)
        for root, dirs, files in os.walk("../thumbnails/thumbnails32")
        for file in files
        if file.endswith(".png")
    ]
    files32.sort()

    i = 927
    j = 1045
    rotate = 0

    hne = np.load(files[i])
    ihc = np.load(files[j])
    import pdb; pdb.set_trace()
    # hne = plt.imread(files32[981])
    # ihc = plt.imread(files32[1034])
    ihc = transform.rotate(ihc, rotate, resize=False)

    hne_gray = color.rgb2gray(hne)
    ihc_gray = color.rgb2gray(ihc)
    hne_gray = feature.corner_harris(hne_gray, sigma=1)
    ihc_gray = feature.corner_harris(ihc_gray, sigma=1)

    # hne_gray = exposure.equalize_adapthist(hne_gray)
    # ihc_gray = exposure.equalize_adapthist(ihc_gray)
    # hne_gray = match_histograms(hne_gray, ihc_gray)
    # hne_gray = filters.sobel(hne_gray)#, sigma=1)
    # ihc_gray = filters.sobel(ihc_gray)#, sigma=1)

    model = register(
        ihc_gray,
        hne_gray,
        n_keypoints=5000,
        fast_threshold=0.01,
        min_samples=3,
        residual_threshold=10,
        max_trials=1000,
    )
    ihc_image_registered = warp_register(ihc, model)


    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(hne)
    axs[0].set_title("Destination H&E")
    axs[1].imshow(ihc_image_registered)
    axs[1].set_title("Registered IHC")
    axs[2].imshow(ihc)
    axs[2].set_title("Original IHC")
    plt.show()
    # exit()

    hne32 = plt.imread(files32[i])
    ihc32 = plt.imread(files32[j])
    ihc32 = transform.rotate(ihc32, rotate, resize=False)

    ihc_image_registered32 = warp_register(ihc32, model, scale=8)

    np.save(f"ihc-{i}-{j}", ihc_image_registered32)
    np.save(f"hne-{j}-{i}", hne32)
    # exit()

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(hne32)
    # axs[0].set_title("Destination H&E")
    # axs[1].imshow(ihc_image_registered32)
    # axs[1].set_title("Registered IHC")
    # axs[2].imshow(ihc32)
    # axs[2].set_title("Original IHC")
    # plt.show()


if __name__ == "__main__":
    main()
