from glob import glob
from os import path
from shutil import copyfile

from utilities.general_utils import extract_image_name


def copy_images_with_annotation(annotation_dir_path, source_dir_path, destination_dir_path, img_suffix='.png'):
    """
    Finds images with the same name as annotations and copies them to the save dir.
    E.g. 666.xml finds image 666.png and copies it to the save dir
    """

    annotation_search_path = path.join(annotation_dir_path, f'*.xml')
    annotation_paths = glob(annotation_search_path)

    img_names = [extract_image_name(ann_path, img_suffix) for ann_path in annotation_paths]
    source_img_paths = [path.join(source_dir_path, img_name) for img_name in img_names]

    save_img_paths = [path.join(destination_dir_path, img_name) for img_name in img_names]

    for img_idx, (source_path, destination_path) in enumerate(zip(source_img_paths, save_img_paths)):
        print(f'Copying img. No.{img_idx}')
        copyfile(source_path, destination_path)

    print('Successfully done')


if __name__ == '__main__':
    copy_images_with_annotation(annotation_dir_path=r'C:\Code\TFOD\assets\android_dataset\checked_dataset\annotations',
                                source_dir_path=r'C:\Code\TFOD\assets\android_dataset\allimgs\images',
                                destination_dir_path=r'C:\Code\TFOD\assets\android_dataset\checked_dataset\images',
                                img_suffix='.png')