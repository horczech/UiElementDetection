"""
The goal of this utility is to copy unique annotation files. Exmaple: I hav directory with all annotations
and directory with checked annotations and I want to copy unchecked annotations from directory with all annoattions
"""

from utilities.general_utils import get_paths_of_files_with_suffix
from os import path
from shutil import copyfile


def run(all_annotations_dir, checked_annotation_dir, destination_dir):
    all_anns_list = list(map(path.basename, get_paths_of_files_with_suffix(all_annotations_dir, '.xml')))
    checked_anns_list = list(map(path.basename, get_paths_of_files_with_suffix(checked_annotation_dir, '.xml')))

    print(f'All annotations list count: {len(all_anns_list)}')
    print(f'Checked annotations list count: {len(checked_anns_list)}')

    all_anns_set = set(all_anns_list)
    checked_anns_set = set(checked_anns_list)

    print(f'All annotations set count: {len(all_anns_set)}')
    print(f'Checked annotations set count: {len(checked_anns_set)}')

    origo_vals = all_anns_set.difference(checked_anns_set)
    print(f'Original values: {len(origo_vals)}')

    for file_name in origo_vals:
        source_path = path.join(all_annotations_dir, file_name)
        destination_path = path.join(destination_dir, file_name)

        copyfile(source_path, destination_path)

    print("DONE")


if __name__ == '__main__':
    PATH_TO_ALL_ANNOTATIONS = r"C:\Code\Datasets\processed_datasets\big_unchecked_dataset\entire_dataset_anotations"
    PATH_TO_CHECKED_ANNOTATIONS_DIR = r'C:\Code\Datasets\processed_datasets\checked_android\dataset_without_bullshit_listitem_classes\all_annotations'
    PATH_TO_DESTINATION_DIR = r"C:\Code\Datasets\processed_datasets\big_unchecked_dataset\unchecked_unique_annotations"

    run(all_annotations_dir=PATH_TO_ALL_ANNOTATIONS,
        checked_annotation_dir=PATH_TO_CHECKED_ANNOTATIONS_DIR,
        destination_dir=PATH_TO_DESTINATION_DIR)
