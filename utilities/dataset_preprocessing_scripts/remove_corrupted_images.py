"""
Some images are corrupted (black screen, only one image, web page, etc.). While annotating I was not able to delete
them directly so I marked them as NoneClass class that I use now to find these annotations and to remove them
"""

from os import remove
from utilities.general_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET


def xml_has_class(xml_path, class_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_list = root.findall('object')
    for xml_object in object_list:
        xml_class = xml_object[0].text
        if xml_class == class_name:
            return True
    return False


def clean_dataset(annotation_dir_path, class_name):
    annotation_path_list = get_paths_of_files_with_suffix(annotation_dir_path, r'.xml')

    removed_files_counter = 0
    for ann_path in annotation_path_list:
        if xml_has_class(ann_path, class_name):
            remove(ann_path)

            removed_files_counter += 1
            print(f'Removing annotation: {ann_path}')

    print(f'Cleaning of dataset finished successfully. Number of deleted files: {removed_files_counter} ')


if __name__ == '__main__':
    ANNOTATION_DIR = r'C:\Code\Dataset2\android\annotations\checked_dataset'
    CLASS_NAME = 'NoneClass'

    clean_dataset(annotation_dir_path=ANNOTATION_DIR,
                  class_name=CLASS_NAME)
