"""
Input: Path to directory with annotations in PASCAL VOC format and name of class to be removed from those annotations
Output: No output. It removes the selected class from the annotations. (It just removes the object not the whole annotation file)
"""

from utilities.general_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET
from os import path

def rename(xml_path, save_dir, original_class_name, new_class_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    save_path = path.join(save_dir, path.basename(xml_path))
    object_list = root.findall('object')
    for xml_object in object_list:
        xml_class = xml_object[0].text
        if xml_class == original_class_name:
            xml_object[0].text = new_class_name

    tree.write(save_path)
    return False


def rename_class(annotation_dir_path, save_dir,original_class_name, new_class_name):
    annotation_path_list = get_paths_of_files_with_suffix(annotation_dir_path, r'.xml')

    for file_idx, ann_path in enumerate(annotation_path_list):
        print(f'Working on file No.: {file_idx}')
        rename(ann_path, save_dir, original_class_name, new_class_name)

    print('DONE')


if __name__ == '__main__':
    ANNOTATION_DIR = r'C:\Code\Dataset2\annotations\android\checked_dataset\all'
    SAVE_ANNOTATION_DIR = r'C:\Code\Dataset2\annotations\android\checked_dataset\all_7classes'
    ORIGINAL_CLASS_NAME = 'ImageButton'
    NEW_CLASS_NAME = 'Button'

    rename_class(annotation_dir_path=ANNOTATION_DIR,
                 save_dir=SAVE_ANNOTATION_DIR,
                 original_class_name=ORIGINAL_CLASS_NAME,
                 new_class_name=NEW_CLASS_NAME)
