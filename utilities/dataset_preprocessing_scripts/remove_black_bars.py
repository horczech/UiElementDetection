"""
Every android imaege has bottom navigation bar and notification bar at the top. This script will remove it
"""

from utilities.general_utils import get_paths_of_files_with_suffix
import cv2
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET


def crop_black_bars(image_dir, ymin, ymax):
    image_paths = get_paths_of_files_with_suffix(image_dir, '.png')

    for idx, image_path in enumerate(image_paths):
        print(f'{idx}/{len(image_paths)}')
        img = cv2.imread(image_path)
        img_cropped = img[ymin:ymax, :, :]

        cv2.imwrite(image_path, img_cropped)


def change_bbox(xml_path, padding):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_list = root.findall('object')
    for xml_object in object_list:
        bbox = xml_object.find('bndbox')

        ymin = str(float(bbox.find('ymin').text) + padding)
        ymax = str(float(bbox.find('ymax').text) + padding)

        bbox.find('ymin').text = ymin
        bbox.find('ymax').text = ymax

    tree.write(xml_path)
    return False


def adapt_bboxes(annotation_dir, y_padding):
    annotation_paths = get_paths_of_files_with_suffix(annotation_dir, '.xml')

    for idx, annotation_path in enumerate(annotation_paths):
        print(f'{idx}/{len(annotation_paths)}')
        change_bbox(annotation_path, y_padding)


if __name__ == '__main__':
    # PATH_TO_IMAGE_DIR = r'C:\Code\Dataset2\images\android'
    # YMIN = 50
    # YMAX = 1823
    #
    # crop_black_bars(image_dir=PATH_TO_IMAGE_DIR,
    #                 ymin=YMIN,
    #                 ymax=YMAX)

    ANNOTATION_DIR_PATH = r'C:\Code\Dataset2\annotations\android\checked_dataset\all'
    Y_PADDING = -50
    adapt_bboxes(annotation_dir=ANNOTATION_DIR_PATH,
                 y_padding=Y_PADDING)
