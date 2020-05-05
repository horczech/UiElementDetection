import cv2
import numpy as np
from utilities.general_utils import get_paths_of_files_with_suffix
import matplotlib.pyplot as plt

refPt = []
clone=None
filter_r = 10

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, clone
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        print(refPt)
        clone = cv2.circle(clone,(x,y), filter_r, 0, -1)



def get_points(image):
    # load the image, clone it, and setup the mouse callback function
    global clone, refPt
    refPt = []
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    return refPt


def draw_mask(img_shape, point_list, filter_r):
    mask_img = np.ones((img_shape[0], img_shape[1], 2))
    for center in point_list:
        mask_img = cv2.circle(mask_img, tuple(center), filter_r, 0, -1)
    return mask_img


def find_blobs(img):
    im = img.copy()


    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    return im_with_keypoints


def run(img_path):
    img_origo = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)




    img_color = cv2.imread(img_path)

    imgs = [img_color[:,:,0],img_color[:,:,1],img_color[:,:,2]]


    dft = cv2.dft(np.float32(img_origo), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.asarray(normalized, np.uint8)

    blobs = find_blobs(normalized)



    point_list = get_points(normalized)
    mask_img = draw_mask(img_origo.shape, point_list, filter_r)

    # apply mask and inverse DFT
    fshift = dft_shift * mask_img
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


    # plt.subplot(231), plt.imshow(img_origo, cmap='gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(233), plt.imshow(mask_img[:,:,0]*255, cmap='gray')
    # plt.title('Mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(234), plt.imshow(img_back, cmap='gray')
    # plt.title('Result'), plt.xticks([]), plt.yticks([])
    #
    cv2.imshow('origo', img_origo)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.asarray(img_back, np.uint8)
    cv2.imshow('result', img_back)

    cv2.imshow('blobs', blobs)



    plt.show()

    cv2.waitKey()



if __name__ == '__main__':
    IMG_DIR_PATH = r'C:\Code\Dataset2\images\printer'
    IMG_DIR_PATH = r'C:\Code\Dataset2\images\hidden_folders\printer'
    #
    moire_img_path = r"C:\Users\horakm\Desktop\New folder2\original.png"
    # android_img_path = r"C:\Code\Dataset2\images\android_original\438.png"

    img_paths = get_paths_of_files_with_suffix(IMG_DIR_PATH, '.png')
    # img_path = r"C:\Users\horakm\Downloads\period_input.jpg"

    # img_paths = get_paths_of_files_with_suffix(IMG_DIR_PATH, '.png')

    for moire_img_path in img_paths:
        run(moire_img_path)
