import cv2
from glob import  glob
from matplotlib import pyplot as plt

img_path = r"C:\Code\Dataset2\android\images\765.png"
# img_path = r"C:\Code\Dataset2\printer\images\1.png"
paths = glob(r"C:\Code\Dataset2\android\images\*.png")


for img_path in paths:
  origo_img = cv2.imread(img_path)

  cropped_resized = cv2.imread(img_path)
  cropped_resized = cropped_resized[50:1823,:,:]
  cropped_resized = cv2.resize(cropped_resized, (1280,720))


  resized_img = cv2.imread(img_path)
  resized_img = cv2.resize(resized_img, (1280,720))


  cv2.imshow('origo', origo_img)
  cv2.imshow('resized_img', resized_img)
  cv2.imshow('cropped_resized', cropped_resized)
  cv2.waitKey()