"""
Input: Path to directory with annotations in PASCAL VOC format and name of class to be removed from those annotations
Output: No output. It removes the selected class from the annotations. (It just removes the object not the whole annotation file)
"""

from utilities.general_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET


def filter_out_class(xml_path, class_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_list = root.findall('object')
    for xml_object in object_list:
        xml_class = xml_object[0].text
        if xml_class == class_name:
            root.remove(xml_object)

    tree.write(xml_path)
    return False


def remove_class(annotation_dir_path, class_name):
    annotation_path_list = get_paths_of_files_with_suffix(annotation_dir_path, r'.xml')

    for file_idx, ann_path in enumerate(annotation_path_list):
        print(f'Working on file No.: {file_idx}')
        filter_out_class(ann_path, class_name)

    print('DONE')


if __name__ == '__main__':
    ANNOTATION_DIR = r'C:\Code\Dataset2\android\annotations\checked_dataset'
    CLASS_NAME = 'ListItem'

    remove_class(annotation_dir_path=ANNOTATION_DIR,
                 class_name=CLASS_NAME)
