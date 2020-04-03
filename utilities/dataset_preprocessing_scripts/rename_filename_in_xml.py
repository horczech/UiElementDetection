"""
Input: Path to directory with annotations in PASCAL VOC format and name of class to be removed from those annotations
Output: No output. It removes the selected class from the annotations. (It just removes the object not the whole annotation file)
"""

from utilities.general_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET



def addpreffix_to_filename(xml_path, prefix):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findall('filename')[0].text
    root.findall('filename')[0].text = prefix + filename


    tree.write(xml_path)
    return False

def remove_class(annotation_dir_path, preffix_to_add):
    annotation_path_list = get_paths_of_files_with_suffix(annotation_dir_path, r'.xml')

    for file_idx, ann_path in enumerate(annotation_path_list):
        print(f'Working on file No.: {file_idx}')
        addpreffix_to_filename(ann_path, preffix_to_add)

    print('DONE')


if __name__ == '__main__':
    ANNOTATION_DIR = r'C:\Code\Dataset2\annotations\printer\entire_dataset'
    PREFFIX_TO_ADD = "printer/"

    remove_class(annotation_dir_path=ANNOTATION_DIR,
                 preffix_to_add=PREFFIX_TO_ADD)
