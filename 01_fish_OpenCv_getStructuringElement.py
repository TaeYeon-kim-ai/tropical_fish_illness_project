
import matplotlib.pyplot as plt
import time 
import copy
import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import compare_ssim

'''
# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
# 팽창 연산 적용 ---②
#경계선 검출
# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
    # 열림 연산 적용 ---②

#이미지 중복제거 / 가로세로 픽셀일치여부
#os.listdir('path') : 경로에 있는 파일 목록 불러오기
#os.path.getsize('path/filename') : 파일 크기 구하기
#데이터 프레임 만들기
#정규표현식을 이용하여 ' 숫자'에 해당하는 값을 지워주었다.
#겹치는 파일명, 파일크기 확인
#이름 같지만 크기 다른것 찾기
# df.sort_values(['기준열']) : 정렬
# df.drop_duplicates(['기준열']) : 중복제거
# ' 숫자'인 파일을 제거하고 싶어서 내림차순으로 정렬해주었다. 이후 숫자가 제거된 파일명을 기준으로 중복제거하였다.

#0. resize
path = r'C:/data/fish_data/100.train1_fish_normal_2000_' # Source Folder
dstpath = r'C:/data/fish_data/100.train1_fish_normal_2000' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")
# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    size = cv2.resize(img, (360, 240), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(dstpath,image),size)

#1. 
photo_list = []

for f in os.listdir('C:/data/fish_data/100.train1_fish_normal_2000'):
    if 'jpg' in f:
        photo_list.append(f) #사진 목록의 ,jpg파일을 불러와 photo_list에 추가

#2. 사진 사이즈 
photo_size = list(map(lambda x: os.path.getsize('C:/data/fish_data/100.train1_fish_normal_2000' + '/' + x), photo_list))

#3. 데이터 프레임 만들기
# Find Same Photos
fsp = pd.DataFrame({'filename_raw':photo_list, 'size':photo_size})
print('사진의 갯수 :', len(fsp))

#4. 숫자 제거
import re   # 정규표현식
com = re.compile(' \d')
fsp['filename'] = list(map(lambda x: com.sub('', x), photo_list))

#5. 겹치는 파일명, 파일크기 확인
# Photo Value Counts
pvc = pd.DataFrame({'filename':fsp['filename'].value_counts().index, 'fn_counts':fsp['filename'].value_counts().values})   
psvc = pd.DataFrame({'size':fsp['size'].value_counts().index, 'size_counts':fsp['size'].value_counts().values})   

fsp = pd.merge(fsp, pvc, how = 'left', on = 'filename')
fsp = pd.merge(fsp, psvc, how = 'left', on = 'size')

fsp.sample(2)

#6. 이름 같지만 크기 다른것 찾기
for i in range(len(fsp)):
    if (fsp['fn_counts'][i] > 1) & (fsp['size_counts'][i] == 1):
        print(i)

#7. Find Same Phto_Not Same Name
fsp_nsn = fsp.sort_values(['filename_raw'], ascending = False).drop_duplicates(['filename'], keep = 'first')
print('남은 사진의 갯수 : {}\n지워진 사진의 갯수 : {}'.format(len(fsp_nsn), len(fsp)-len(fsp_nsn)))

#8
pvc_nsn = pd.DataFrame({'filename':fsp_nsn['filename'].value_counts().index, 'fn_counts_nsn':fsp_nsn['filename'].value_counts().values})   
psvc_nsn = pd.DataFrame({'size':fsp_nsn['size'].value_counts().index, 'size_counts_nsn':fsp_nsn['size'].value_counts().values})   

fsp_nsn = pd.merge(fsp_nsn, pvc_nsn, how = 'left', on = 'filename')
fsp_nsn = pd.merge(fsp_nsn, psvc_nsn, how = 'left', on = 'size')

#9. 이름 겹친것 찾기
fsp_nsn[fsp_nsn['fn_counts_nsn']!=1]

#10. 남은 사이즈 겹치는것 갯수
print('사이즈 겹치는 사진의 갯수 :', len(fsp_nsn[fsp_nsn['size_counts_nsn']!=1]))
print('중복 사이즈의 갯수 :', len(psvc_nsn[psvc_nsn['size_counts_nsn']>1]))

# 남은 것은 사이즈는 같지만 저장명은 완전히 다른 것들이다. 이미지 비교가 필요한 사진들이라 OpenCV와 skimage를 이용하였다.
#   1) 사이즈가 같은 두 이미지를 불러온다.
#   2) 이미지를 array로 변환했을 때 구조가 같다면 두 이미지에 차이가 있는지 확인한다.
#   3) 만약 두개의 이미지 구조가 같은데 차이가 존재한다면 그 사진은 직접 확인하기로 한다.1

#11. 삭제될 사진의 리스트

print(psvc_nsn)
print(fsp_nsn)

delete = []

for i in range(len(psvc_nsn)):
    
    # 중복된 크기(size)가 있는 경우
    if psvc_nsn['size_counts_nsn'][i] == 2:
        
        # 그 크기에 해당하는 사진을 불러온다. 
        temp = fsp_nsn[fsp_nsn['size']==psvc_nsn['size'][i]].reset_index(drop = True).sort_values(['filename'])
        
        # 사진 읽기
        imageA = cv2.imread('C:/data/fish_data/100.train1_fish_normal_2000/'+temp['filename_raw'][0])
        imageB = cv2.imread('C:/data/fish_data/100.train1_fish_normal_2000/'+temp['filename_raw'][1])
        
        # 이미지를 grayscale로 변환
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        
        # 이미지의 구조가 같다면 이미지 비교
        if len(grayA)==len(grayB):
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            
            # 차이가 없다면 하나는 delete에 넣어주기
            if score == 1:
                delete.append(temp['filename_raw'][1])
            
            # 구조가 같지만 차이가 존재한다면 직접 확인하기     
            else:
                print('확인해보시오! : ', temp['filename_raw'][0], '/', temp['filename_raw'][1], f'(score : {score})')


#12. 중복제거된 것들은 delete 리스트에 넣어주기
delete = delete + list(fsp[~fsp['filename_raw'].isin(fsp_nsn['filename_raw'])]['filename_raw'])
print('삭제할 목록 :', len(delete))

#13. 전체데이터 - delete = 남길데이터
# result : 처음(fsp)데이터에서 - delete를 제외한 것
#shutil.move('path/file', '이동할 경로') : 파일을 '이동할 경로'로 옮겨준다.
result = list(fsp[~fsp['filename_raw'].isin(delete)]['filename_raw'])

print('남길 목록 : ', len(result))

import shutil

for i in result:
    shutil.move('C:/data/fish_data/100.train1_fish_normal_2000/'+i, 'C:/data/fish_data/test_result') #사용할 이미지
    
for i in delete:
    shutil.move('C:/data/fish_data/100.train1_fish_normal_2000/'+i, 'C:/data/fish_data/test_del') #삭제할 이미지




#=====================================================
'''

# 외곽선검출 train_normal
path = r'C:/data/fish_data/fish_datasets/x_train/normal' # Source Folder
dstpath = r'C:/data/fish_data/fish_datasets/x1_train/normal' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")
# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dst = cv2.dilate(img, k)
    gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, k)
    size = cv2.resize(gradient, (128, 128), interpolation = cv2.INTER_CUBIC)
    #결과 출력
    # merged = np.hstack((img, gradient))
    # cv2.imshow('gradient', merged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)

#=========================================================

# 외곽선검출 train_illness
path = r'C:/data/fish_data/fish_datasets/x_train/illness' # Source Folder
dstpath = r'C:/data/fish_data/fish_datasets/x1_train/illness' # Destination Folder

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
    size = cv2.resize(gradient, (128, 128), interpolation = cv2.INTER_CUBIC)
    #결과 출력
    # merged = np.hstack((img, gradient))
    # cv2.imshow('gradient', merged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)


#=======================================================================


# 외곽선검출 test_normal
path = r'C:/data/fish_data/fish_datasets/x_test/normal' # Source Folder
dstpath = r'C:/data/fish_data/fish_datasets/x1_test/normal' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")
# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dst = cv2.dilate(img, k)
    gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, k)
    size = cv2.resize(gradient, (128, 128), interpolation = cv2.INTER_CUBIC)
    # 결과 출력
    #merged = np.hstack((img, gradient))
    #cv2.imshow('gradient', merged)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)

#=========================================================

# 외곽선검출 test_illness
path = r'C:/data/fish_data/fish_datasets/x_test/illness' # Source Folder
dstpath = r'C:/data/fish_data/fish_datasets/x1_test/illness' # Destination Folder

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
    size = cv2.resize(gradient, (128, 128), interpolation = cv2.INTER_CUBIC)
    #결과 출력
    # merged = np.hstack((img, gradient))
    # cv2.imshow('gradient', merged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)
