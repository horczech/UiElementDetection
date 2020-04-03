from glob import glob
from os import path


def get_paths_of_files_with_suffix(path_to_dir, file_suffix):
    '''
    :param file_suffix: must be with dot e.g. ".jpg"
    '''
    search_pattern = path.join(path_to_dir, f'*{file_suffix}')
    return glob(search_pattern)


def extract_image_name(annotation_path, img_suffix):
    ann_name = path.basename(annotation_path).split('.')[0]
    img_name = f'{ann_name}{img_suffix}'
    return img_name


def covert_annotation_path_to_image_path(annotation_path, image_dir_path, img_suffix):
    image_name = extract_image_name(annotation_path, img_suffix)
    return path.join(image_dir_path, image_name)
