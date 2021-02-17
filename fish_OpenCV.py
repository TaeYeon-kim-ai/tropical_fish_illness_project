import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
#opencv : [B, G, R]
#외곽선 검출
# for i in range(2229):
#     image_path = "C:/data/fish_data/train1_fish_normal_2230/" + str(i) + 'jpg'
#     #print(image_path)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     cv2.resize(image, (240, 160), interpolation = cv2.INTER_CUBIC)
#     blur = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0)
#     ret, thresh1 = cv2.threshold(blur, 100, 135, cv2.THRESH_BINARY)
#     edged = cv2.Canny(blur, 10, 90)
#     cv2.imwrite('C:/data/fish_data/train1_fish_normal_2230_noise' + str(i) + '.jpg', edged)

# 외곽선검출_정상
path = r'C:/data/fish_data/train1_fish_normal_2230' # Source Folder
dstpath = r'C:/data/fish_data/train1_fish_normal_2230_noise' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")

# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    image1 = cv2.resize(gray, (240, 160), interpolation = cv2.INTER_CUBIC)
    # image1 = cv2.GaussianBlur(image, ksize=(4,4), sigmaX=1)
    ret, thresh1 = cv2.threshold(image1, 100, 135, cv2.THRESH_BINARY)
    edged = cv2.Canny(image1, 10, 90)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    image1 = cv2.drawContours(edged, contours, -1, (0,255,0), 3)
    cv2.imwrite(os.path.join(dstpath,image),edged)



#외곽선 검출2_질병
path = r'C:/data/fish_data/train2_fish_illness' # Source Folder
dstpath = r'C:/data/fish_data/train2_fish_illness_noise' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")

# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    image1 = cv2.resize(gray, (240, 160), interpolation = cv2.INTER_CUBIC)
    # image1 = cv2.GaussianBlur(image, ksize=(4,4), sigmaX=1)
    ret, thresh1 = cv2.threshold(image1, 100, 135, cv2.THRESH_BINARY)
    edged = cv2.Canny(image1, 10, 90)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    image1 = cv2.drawContours(edged, contours, -1, (0,255,0), 3)
    cv2.imwrite(os.path.join(dstpath,image),edged)
