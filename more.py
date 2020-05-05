import cv2
import numpy as np
from utilities.general_utils import get_paths_of_files_with_suffix
import matplotlib.pyplot as plt


def crop_img_if_not_even(img):

    row, col = img.shape
    if row % 2 != 0:
        img = img[:-1, :]
    if col % 2 != 0:
        img = img[:, :-1]

    return img


def calculate_PSD(img, is_log=False):


    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # compute the PSD = sqrt(Re(DFT(I)) ^ 2 + Im(DFT(I)) ^ 2) ^ 2
    magnitude = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    psd = cv2.pow(magnitude,2)


    if is_log:
        psd = np.log(psd)

    return psd


def synthesizeFilterH(H, center, r):
    rows, cols = H.shape
    c2 = c3 = np.asarray(center)
    c2[1] = rows - center[1]
    c3[0] = cols - center[0]
    c4 = (c3[0], c2[1])

    H = cv2.circle(H, center, r, 0 ,-1,8)
    H = cv2.circle(H, tuple(c2), r, 0,-1,8)
    H = cv2.circle(H, tuple(c3), r, 0,-1,8)
    H = cv2.circle(H, tuple(c4), r, 0,-1,8)

    return H

def run(img_path):
    img_origo = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_origo = np.asarray(img_origo)
    
    img_origo = crop_img_if_not_even(img_origo)

    psd_img = calculate_PSD(img_origo.copy())

    psd_img = np.fft.fftshift(psd_img)

    psd_img = cv2.normalize(psd_img, psd_img, 0, 255, cv2.NORM_MINMAX)


    # H calculation (start)
    H = np.ones_like(img_origo)
    r = 21
    H = synthesizeFilterH(H, (705, 458), r)
    H =synthesizeFilterH(H, (850, 391), r)
    H =synthesizeFilterH(H, (993, 325), r)

    # // filtering (start)
    H = np.fft.fftshift(H)




    cv2.imshow("H", H*255)
    cv2.imshow("original image", img_origo)

    cv2.imshow("psd_normalized", psd_img)
    plt.imshow(psd_img, cmap='gray')
    plt.show()
    cv2.waitKey()


if __name__ == '__main__':
    # IMG_DIR_PATH = r'C:\Code\Dataset2\images\printer'
    #
    moire_img_path = r"C:\Code\Dataset2\images\printer\17.png"
    # android_img_path = r"C:\Code\Dataset2\images\android_original\438.png"

    # img_paths = get_paths_of_files_with_suffix(IMG_DIR_PATH, '.png')
    # img_path = r"C:\Users\horakm\Downloads\period_input.jpg"


    run(moire_img_path)