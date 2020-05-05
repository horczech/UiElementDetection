from utilities.general_utils import get_paths_of_files_with_suffix
import cv2
import numpy as np
from os import path


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
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
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0

        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy


    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * 0.1
        return noisy


def run(img_path, save_dir):
    img = cv2.imread(img_path)


    noisy_img = noisy('s&p',img)

    noisy_img = np.asarray(noisy_img,dtype=np.uint8)


    blured_img = cv2.GaussianBlur(noisy_img, (5, 5), 3)


    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(blured_img, -1, kernel)

    blured_img = cv2.medianBlur(sharpened,3)


    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blured_img, -1, kernel)

    # cv2.imshow('origo', img)
    # # cv2.imshow('noisy_img', noisy_img)
    # cv2.imshow('sharpened', sharpened)

    save_path = path.join(save_dir, path.basename(img_path))
    cv2.imwrite(save_path, sharpened)


if __name__ == '__main__':
    IMG_DIR_PATH = r'C:\Code\Dataset2\images\hidden_folders\android'
    save_path_dir = r'C:\Code\Dataset2\images\android'

    img_paths = get_paths_of_files_with_suffix(IMG_DIR_PATH, '.png')
    for idx, img_path in enumerate(img_paths):
        print(f'{idx}/{len(img_paths)}')
        run(img_path, save_path_dir)
