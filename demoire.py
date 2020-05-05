import cv2
from glob import  glob
from matplotlib import pyplot as plt
from utilities.general_utils import get_paths_of_files_with_suffix
import numpy as np
import random
from matplotlib import pyplot as plt

def add_moire(img_path):
  img = cv2.imread(img_path, )
  resize_factor = 0.5
  img = resize(img, resize_factor)

  b,g,r = cv2.split(img)
  shape = np.asarray(img.shape)
  shape[:2] = shape[:2]*3
  new_img = np.zeros(shape, dtype=np.uint8)

  for row_id in range(img.shape[0]):
    for col_id in range(img.shape[1]):

      b,g,r = img[row_id, col_id]

      subpixel = np.zeros((3, 3, 3))
      subpixel[0,0, 1] = 0
      subpixel[0,1, 1] = 0
      subpixel[0,2, 1] = 0

      subpixel[1,0, 2] = r
      subpixel[1,1, 1] = g
      subpixel[1,2, 0] = b

      subpixel[2,0, 2] = r
      subpixel[2,1, 1] = g
      subpixel[2,2, 0] = b


      row_min = row_id * 3
      row_max = row_min + 3
      col_min = col_id * 3
      col_max = col_min + 3

      new_img[row_min:row_max, col_min:col_max, :] = subpixel




  blured_img = cv2.GaussianBlur(new_img, (3,3), 0)
  cv2.imshow('blured_img', blured_img)

  # perspective TF
  max_x= 1200
  max_y = 1920

  rand_x = int(0.1*max_x)
  rand_y = int(0.1*max_y)

  pt1 = (random.randrange(0,rand_y), random.randrange(0,rand_x))
  pt2 = (random.randrange(0,rand_y), max_x-random.randrange(0,rand_x))
  pt3 = (max_y - random.randrange(0,rand_y), max_x-random.randrange(0,rand_x))
  pt4 = (max_y -random.randrange(0,rand_y), random.randrange(0,rand_x))


  pts1 = np.float32([[0, 0], [0, 1200], [1920, 1200], [1920, 0]])
  pts2 = np.float32([pt1, pt2, pt3, pt4])

  M = cv2.getPerspectiveTransform(pts1, pts2)

  dst = cv2.warpPerspective(blured_img.copy(), M, (max_x,max_y))

  increase =100
  improved_img = cv2.cvtColor(dst.copy(), cv2.COLOR_BGR2HSV)
  v = improved_img[:, :, 2]
  v = np.where(v <= 255 - increase, v + increase, 255)
  improved_img[:, :, 2] = v

  improved_img = cv2.cvtColor(improved_img, cv2.COLOR_HSV2BGR)


  plt.figure()
  plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
  plt.title('transformed')

  plt.figure()
  plt.imshow(cv2.cvtColor(improved_img, cv2.COLOR_BGR2RGB))
  plt.title('improved_img')


  cv2.imshow('tranformed', dst)
  cv2.imshow('reverse', improved_img)
  plt.show()
  # cv2.imshow('reverse', reverse)
  # cv2.imshow('dst', dst)
  # cv2.imshow('new_img', new_img)
  #
  # cv2.imshow('origo_img',img)
  # cv2.waitKey()


def improve_image(image, alpha, beta):
  new_image = np.zeros(image.shape, image.dtype)

  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      for c in range(image.shape[2]):
        new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

  return new_image

def resize(img, resize_factor):
  shape = np.asarray(img.shape)
  shape[:2] = shape[:2] * resize_factor
  shape = tuple(np.asarray(shape, dtype=int))
  return img


def run(img_dir):
  img_path_list = get_paths_of_files_with_suffix(img_dir, '.png')


  for img_path in img_path_list:
    print(img_path)
  img = cv2.resize(img, (shape[1], shape[0]))
    add_moire(img_path)



if __name__ == '__main__':
  IMG_DIR = r'C:\Code\Datasets\android_dataset\checked_android\dataset_without_bullshit_listitem_classes\all_images'

  run(img_dir = IMG_DIR)