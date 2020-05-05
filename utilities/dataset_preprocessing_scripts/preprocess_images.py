from utilities.general_utils import get_paths_of_files_with_suffix
import cv2
from os import path



def run(img_path, save_path_dir):
    img = cv2.imread(img_path)

    # img_median = cv2.GaussianBlur(img, (5,5),3)
    img_median = cv2.bilateralFilter(img, d=5,sigmaColor=11, sigmaSpace=11)
    # img_method1 = cv2.fastNlMeansDenoisingColored(img,None,6,6,7,21)


    cv2.imshow('origo',img)
    cv2.imshow('median', img_median)
    # cv2.imshow('median+bilatral',img_bilateral)
    # cv2.imshow('denoising',img_method1)
    # cv2.imshow('img_strong',img_strong)
    cv2.waitKey()

    # save_file_name = path.join(save_path_dir, path.basename(img_path))
    cv2.imwrite(save_path_dir, img_median)



if __name__ == '__main__':
    # IMG_DIR_PATH = r'C:\Code\Dataset2\images\printer'
    # save_path_dir = r'C:\Code\Dataset2\images\printer_denoised'
    #
    #
    # img_paths = get_paths_of_files_with_suffix(IMG_DIR_PATH, '.png')
    # for idx, img_path in enumerate(img_paths):
    #     print(f'{idx}/{len(img_paths)}')
    run(r"C:\Users\horakm\Desktop\New folder2\original.png", r"C:\Users\horakm\Desktop\New folder2\bilateral.png")