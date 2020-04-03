


from utilities.general_utils import get_paths_of_files_with_suffix
import cv2
import numpy as np
from os import path

def add_zero_padding(image_path):
    EXPECTED_INPUT_SHAPE = (1773, 1200,3)
    SHAPE = (1773, 3152, 3)
    TARGET_SHAPE = (720, 1280)
    img = cv2.imread(image_path)

    if img.shape != EXPECTED_INPUT_SHAPE:
        img = cv2.resize(img, EXPECTED_INPUT_SHAPE)

    result_img = np.zeros(SHAPE, dtype=np.uint8)
    result_img[:EXPECTED_INPUT_SHAPE[0], :EXPECTED_INPUT_SHAPE[1], :] = img
    result_img = cv2.resize(result_img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))

    return result_img


def run(image_dir, save_dir):
    image_paths = get_paths_of_files_with_suffix(image_dir, '.png')

    for idx, image_path in enumerate(image_paths):
        print(f'{idx}/{len(image_paths)}')

        full_save_path = path.join(save_dir, path.basename(image_path))

        padded_img = add_zero_padding(image_path)
        cv2.imwrite(full_save_path, padded_img)


if __name__ == '__main__':
    PATH_TO_IMAGE_DIR = r'C:\Code\Dataset2\images\android'
    PATH_TO_SAVE_DIR = r'C:\Code\Dataset2\images\android_with_printer_ratio'

    run(image_dir=PATH_TO_IMAGE_DIR,
        save_dir=PATH_TO_SAVE_DIR)