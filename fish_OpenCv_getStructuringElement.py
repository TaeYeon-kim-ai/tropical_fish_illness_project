
import matplotlib.pyplot as plt
import time 
import copy
import cv2
import numpy as np
import os
# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
# 팽창 연산 적용 ---②
#경계선 검출
# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
    # 열림 연산 적용 ---②
'''
# 외곽선검출 1
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
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dst = cv2.dilate(img, k)
    gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, k)
    size = cv2.resize(gradient, (240, 160), interpolation = cv2.INTER_CUBIC)
    # 결과 출력
    #merged = np.hstack((img, gradient))
    #cv2.imshow('gradient', merged)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)
'''
#=========================================================

# 외곽선검출 2
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
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dst = cv2.dilate(img, k)
    gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, k)
    size = cv2.resize(gradient, (240, 160), interpolation = cv2.INTER_CUBIC)
    결과 출력
    merged = np.hstack((img, gradient))
    cv2.imshow('gradient', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)
