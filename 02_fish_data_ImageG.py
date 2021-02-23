import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random

#test illness fish Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.5, 1.0],
    horizontal_flip=True,
    width_shift_range=1.0,
    height_shift_range=1.0, 
    brightness_range=[0.2, 1.0]
)

xy_train = train_datagen.flow_from_directory(
    'C:/data/fish_data/fish_datasets/test',
    target_size = (240, 360),
    batch_size = 500,
    class_mode= 'binary' ,
    save_to_dir='C:/data/fish_data/fish_datasets/test_image' #정의해논걸 print로 한번 건드려줘야 작성함(건드려 준 만큼 이미지 생성됨)
)

gen  = int(6) #반복한 만큼 image수 *n 번 생성
    
for i in range(gen) :
    print(xy_train[0][1])


def keep_n_dir(directory, n):
    files = os.listdir(directory) 
    if len(files) > n: 
        diff = len(files) - n
        files_to_delete = random.sample(files, k=diff) 
        for file in files_to_delete: 
            os.remove(os.path.join(directory, file)) 

path_to_all_images_folder = 'C:/data/fish_data/fish_datasets/x_train/illness'
directories = os.listdir(path_to_all_images_folder)
directories = [os.path.join(path_to_all_images_folder, folder) for folder in directories]
for directory in directories:
    if os.path.isdir(directory):
        keep_n_dir(directory, n)

keep_n_dir('C:/data/fish_data/fish_datasets/x_train/illness', 1000)