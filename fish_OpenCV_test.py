import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import os

# for i in range(2230):
#     image_path = 'C:/data/fish_dat/train1_fish_normal_2230/%d.jpg'%i
#     image = cv2.imread(image_path)
#     image2 = np.where((image <= 254) & (image != 0), 0, image)#254보다 작은건 모조리 0으로 처리
#     image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
#     image4 = cv2.medianBlur(src=image3, ksize= 5)
#     cv2.imwrite('C:/data/fish_data/train1_fish_normal_2230_noise/%0d.jpg'%i, image4)

img = cv2.imread("C:/data/fish_dat/train1_fish_normal_2230/1.jpg", 0)
edges = cv2.Canny(img, 125, 135)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()