"""
Input: Path to directory with annotations in PASCAL VOC format.
Output: Dictionary with class names and its counts
"""
from utilities.general_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET


def get_distribution_in_file(xml_file, distribution_dic):
    try:
        tree = ET.parse(xml_file)
    except Exception as e:
        print(e)

    root = tree.getroot()

    object_list = root.findall('object')
    for xml_object in object_list:
        class_name = xml_object[0].text
        distribution_dic[class_name] = distribution_dic.get(class_name, 0) + 1

    return distribution_dic


def run(annotaiton_dir_path):
    distribution_dic = dict()
    file_paths_list = get_paths_of_files_with_suffix(annotaiton_dir_path, '.xml')

    for file_id, xml_file in enumerate(file_paths_list):
        print(f'Working on file No.:{xml_file}')
        distribution_dic = get_distribution_in_file(xml_file, distribution_dic)

    return distribution_dic


if __name__ == '__main__':
    PATH_TO_ANNOTATION_DIR = r"C:\Code\Dataset2\annotations\printer\phase_1\test"

    class_distribution_dic = run(annotaiton_dir_path=PATH_TO_ANNOTATION_DIR)

    print(f'Distribution of classes in dataset:\n{class_distribution_dic}')
