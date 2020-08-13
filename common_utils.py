import numpy as np
import os
import cv2


def make_image_noisy(image, noise_typ):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 40
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape((row, col, ch))
        noisy_image = image + gauss
        return noisy_image.clip(0, 255)
    elif noise_typ == "zero":
        amount = 0.05  # percentage of zero pixels
        out = np.copy(image)
        num_zeros = np.ceil(amount * image.shape[0]*image.shape[1])
        coords = [np.random.randint(0, i - 1, int(num_zeros))
                  for i in image.shape[:2]]
        out[:, :, 0][coords] = 0
        out[:, :, 1][coords] = 0
        out[:, :, 2][coords] = 0
        return out.astype(np.uint8)
    elif noise_typ == "s&p":
        raise RuntimeError("Test it properly before using!")
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        raise RuntimeError("Test it properly before using!")
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image * vals) / float(vals)
        return noisy_image
    elif noise_typ == "speckle":
        raise RuntimeError("Test it properly before using!")
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape((row, col, ch))
        noisy_image = image + image * gauss
        return noisy_image
    else:
        raise RuntimeError(f"Unknown noisy_type: {noise_typ}")

