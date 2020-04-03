import numpy as np
from PIL import Image

def pil_img_to_numpy_array(pil_image):
    (im_width, im_height) = pil_image.size

    # return np.array(pil_image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def load_img_to_np(img_path):
    image = Image.open(img_path)
    return pil_img_to_numpy_array(image)


