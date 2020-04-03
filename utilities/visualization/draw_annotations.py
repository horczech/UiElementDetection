from utilities.general_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET
import cv2
from object_detection.utils import visualization_utils as vis_util
from utilities.visualization.bbox_drawer import BboxDrawer
from os import path
from utils import pil_img_to_numpy_array, load_img_to_np


def draw_bboxes(annotations_path, image_dir, save_dir, labelmap_path):
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    filename = root.find('filename').text.replace('/', '\\')
    source_img_path = path.join(image_dir, filename)
    save_image_path = path.join(save_dir, path.basename(source_img_path))

    object_list = root.findall('object')

    drawer = BboxDrawer(labelmap_path)

    bboxes = []
    class_indexes = []
    for xml_object in object_list:
        class_name = xml_object[0].text
        class_indexes.append(drawer.class_dict[class_name])

        bbox_element = xml_object.find('bndbox')
        xmin = int(float(bbox_element.find('xmin').text))
        xmax = int(float(bbox_element.find('xmax').text))
        ymin = int(float(bbox_element.find('ymin').text))
        ymax = int(float(bbox_element.find('ymax').text))
        bboxes.append([ymin, xmin, ymax, xmax])

    image = cv2.imread(source_img_path)
    drawn_image = drawer.draw_object_from_annotations(image, bboxes, class_indexes)

    cv2.imwrite(save_image_path, drawn_image)


def run(annotation_dir, image_dir, save_dir, labelmap_path):
    annotations_paths = get_paths_of_files_with_suffix(annotation_dir, '.xml')

    for img_idx, annotations_path in enumerate(annotations_paths):
        print(f'{img_idx}/{len(annotations_paths)}')
        draw_bboxes(annotations_path, image_dir, save_dir, labelmap_path)


if __name__ == '__main__':
    ANNOTATION_DIR_PATH = r'C:\Code\Dataset2\annotations\android\checked_dataset\all'
    IMAGE__DIR_PATH = r'C:\Code\Dataset2\images'
    PATH_TO_LABEL_MAP = r"C:\Code\Dataset2\label_maps\label_map_8_classes.pbtxt"

    SAVE_DIR_PATH = r'C:\Code\Dataset2\dummy\android_corrected'

    run(annotation_dir=ANNOTATION_DIR_PATH,
        image_dir=IMAGE__DIR_PATH,
        save_dir=SAVE_DIR_PATH,
        labelmap_path=PATH_TO_LABEL_MAP, )
