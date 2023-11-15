import openslide
import itk
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from skimage.color import rgb2gray
from skimage.exposure import match_histograms
from skimage import transform
import os
from tqdm import tqdm


def main():
    files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk("thumbnails/thumbnails256")
        for file in files
        if file.endswith(".npy")
    ]
    files.sort()
    files32 = [
        os.path.join(root, file)
        for root, dirs, files in os.walk("thumbnails/thumbnails32")
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

    for i, j in tqdm(idxs):
        start = perf_counter()

        im_hne = plt.imread(files32[i])
        im_ihc = plt.imread(files32[j])

        if i in [981, 983]:
            im_ihc = transform.rotate(im_ihc, 180)

        height = min(im_hne.shape[0], im_ihc.shape[0])
        width = min(im_hne.shape[1], im_hne.shape[1])

        im_hne = im_hne[:height, :width, :]
        im_ihc = im_ihc[:height, :width, :]
        im_ihc = match_histograms(im_ihc, im_hne)

        im_hne_gray = rgb2gray(im_hne)
        im_ihc_gray = rgb2gray(im_ihc)

        registered_image, params = itk.elastix_registration_method(im_hne_gray, im_ihc_gray)

        red_channel, green_channel, blue_channel = (
            im_ihc[:, :, 0],
            im_ihc[:, :, 1],
            im_ihc[:, :, 2],
        )
        transformed_channels = []
        for channel in [red_channel, green_channel, blue_channel]:
            channel_image = itk.image_from_array(channel)
            transformed_channel = itk.transformix_filter(channel_image, params)
            transformed_channels.append(itk.array_from_image(transformed_channel))
        im_register = np.stack(transformed_channels, axis=-1)

        # print(perf_counter() - start)

        mask = im_register == np.zeros(3)
        mask = np.all(mask, -1)
        im_register[mask] = [1, 1, 1]

        np.save(f"registered-{i}-{j}", im_register)# registered_image)

        # plt.subplot(131).imshow(im_hne)
        # plt.subplot(132).imshow(im_register)
        # plt.subplot(133).imshow(im_ihc)

        # plt.show()


if __name__ == "__main__":
    main()
