from utilities.general_utils import get_paths_of_files_with_suffix
import random
from os import path
from shutil import copyfile

def copy_files(source_file_path, destination_dir_path):
    source_file_name = path.basename(source_file_path)
    destination_file_path = path.join(destination_dir_path, source_file_name)

    copyfile(source_file_path,destination_file_path)


def split(annotation_dir, output_train_dir, output_test_dir, split_ratio):
    annotation_file_list = get_paths_of_files_with_suffix(annotation_dir, '.xml')
    random.shuffle(annotation_file_list)

    split_abs = int(split_ratio*len(annotation_file_list))

    train_list = annotation_file_list[:split_abs]
    test_list = annotation_file_list[split_abs:]

    [copy_files(file_path, output_train_dir) for file_path in train_list]
    [copy_files(file_path, output_test_dir) for file_path in test_list]

    print('Successfully done')



if __name__ == '__main__':
    PATH_TO_ANNOTATION_DIR = r'C:\Code\Dataset2\annotations\android\checked_dataset\all'
    SPLIT_RATIO = 0.75
    OUTPUT_DIR_TRAIN = r'C:\Code\Dataset2\annotations\android\checked_dataset\splitted\train'
    OUTPUT_DIR_TEST = r'C:\Code\Dataset2\annotations\android\checked_dataset\splitted\test'

    split(annotation_dir=PATH_TO_ANNOTATION_DIR,
          output_train_dir=OUTPUT_DIR_TRAIN,
          output_test_dir=OUTPUT_DIR_TEST,
          split_ratio=SPLIT_RATIO)
