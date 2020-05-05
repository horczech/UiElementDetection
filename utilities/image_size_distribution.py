from utilities.general_utils import get_paths_of_files_with_suffix
import cv2
import seaborn as sns
from matplotlib import pyplot as plt


image_dir = r'C:\Code\Dataset2\images\printer'


image_paths = get_paths_of_files_with_suffix(image_dir, '.png')


heights = []
widths = []


for img_path in image_paths:
    img = cv2.imread(img_path)
    heights.append(img.shape[0])
    widths.append(img.shape[1])

plt.figure(1)
sns.distplot(heights)
plt.title('Heights')

plt.show()


plt.figure(2)
sns.distplot(widths)
plt.title('Widths')
plt.show()
a=5
