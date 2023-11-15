import h5py
import numpy as np
from skimage import io
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt


def embed(hne_paths, ihc_paths, patch_size, output_path):
    f = h5py.File(output_path, "w")

    tot_patches_all = 0
    for i, (he_path, ihc_path) in tqdm(list(enumerate(zip(hne_paths, ihc_paths)))):
        he_image = plt.imread(he_path)
        ihc_image = plt.imread(ihc_path)

        he_image = he_image[..., :3]
        ihc_image = ihc_image[..., :3]

        k_patches = min(he_image.shape[0], ihc_image.shape[0]) // patch_size
        l_patches = min(he_image.shape[1], ihc_image.shape[1]) // patch_size

        tot_patches_im = k_patches * l_patches
        tot_patches_all += k_patches * l_patches

    hne_patches = f.create_dataset(
        "hne_patches",
        (tot_patches_all, patch_size, patch_size, 3),
    )
    ihc_patches = f.create_dataset(
        "ihc_patches",
        (tot_patches_all, patch_size, patch_size, 3),
    )
    idxs = f.create_dataset("patch_idxs", (tot_patches_all,))

    idx_global = 0

    pbar = tqdm(total=tot_patches_all)
    for i, (hne_path, ihc_path) in enumerate(zip(hne_paths, ihc_paths)):
        hne_image = plt.imread(hne_path)
        ihc_image = plt.imread(ihc_path)

        hne_image = hne_image[..., :3]
        ihc_image = ihc_image[..., :3]

        k_patches = min(he_image.shape[0], ihc_image.shape[0]) // patch_size
        l_patches = min(he_image.shape[1], ihc_image.shape[1]) // patch_size

        for k in range(k_patches):
            for l in range(l_patches):
                # Extract patches
                hne_patch = hne_image[
                    k * patch_size : (k + 1) * patch_size,
                    l * patch_size : (l + 1) * patch_size,
                ]
                ihc_patch = ihc_image[
                    k * patch_size : (k + 1) * patch_size,
                    l * patch_size : (l + 1) * patch_size,
                ]

                # Store the patches
                hne_patches[idx_global, ...] = hne_patch
                ihc_patches[idx_global, ...] = ihc_patch
                idxs[idx_global] = i

                pbar.update(1)

    f.close()


def main():
    files32 = [
        os.path.join(root, file)
        for root, dirs, files in os.walk("../thumbnails/thumbnails32")
        for file in files
        if file.endswith(".png")
    ]
    files32.sort()
    idxs = [
        (927, 1045),
        (931, 1042),
        (980, 1035),
        (981, 1034),
        (982, 1033),
        (983, 1032),
        (984, 1030),
        (934, 1022),
        (935, 999),
        (882, 1005),
    ]
    idxs = sorted([pair[0] for pair in idxs])
    hne_paths = [files32[idx] for idx in idxs]

    ihc_paths = Path.cwd().glob("registered*.png")
    ihc_paths = sorted(ihc_paths)

    # for path in tqdm(ihc_paths):
    #     ihc = np.load(path)
    #     ihc = ihc.clip(0, 1)
    #     # ihc -= ihc.min()
    #     # ihc /= ihc.max()
    #     out_path = path.with_suffix(".png")
    #     plt.imsave(out_path, ihc)
    # exit()

    # hne = plt.imread(hne_paths[0])
    # ihc = plt.imread(ihc_paths[0])
    # plt.subplot(121).imshow(hne)
    # plt.subplot(122).imshow(ihc)
    # plt.show()

    # hne = plt.imread(hne_paths[-1])
    # ihc = plt.imread(ihc_paths[-1])
    # plt.subplot(121).imshow(hne)
    # plt.subplot(122).imshow(ihc)
    # plt.show()

    embed(hne_paths, ihc_paths, 256, "dataset-10-256.h5")


if __name__ == "__main__":
    main()
