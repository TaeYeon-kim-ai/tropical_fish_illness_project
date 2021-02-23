#이진분류로 할 경우 f1 score로 잡아야함.
#database 숫자 증폭 잘할 것


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cv2
import glob
import sys
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten, BatchNormalization, Input, AveragePooling2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


#선언
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.5, 1.0],
    horizontal_flip=True,
    width_shift_range=1.0,
    height_shift_range=1.0, 
    brightness_range=[0.2, 1.0]
)

test_datagen = ImageDataGenerator(rescale=1./255)

#xy_train
xy_train = train_datagen.flow_from_directory(
    'C:/data/fish_data/fish_datasets/x1_train',
    target_size = (128, 128), 
    batch_size = 128, 
    class_mode = 'binary',
    subset="training"
)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory(
    'C:/data/fish_data/fish_datasets/x1_test',
    target_size = (128, 128),
    batch_size = 128, 
    class_mode = 'binary'
)

#모델링
input1 = Input(shape=(128, 128 ,3))
x = Conv2D(64, 4, activation='relu')(input1)
x = BatchNormalization()(x)
x = Conv2D(64, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Dropout(0.1)(x)

x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

model.compile(
    loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
es = EarlyStopping(monitor='loss', patience=20, verbose=1, mode="auto",)
rl = ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto')
modelpath = 'C:/data/MC/best_fish_illness_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
hist= model.fit_generator(xy_train, 
    steps_per_epoch = 16, 
    epochs = 100, 
    validation_data = xy_test,
    callbacks=[es, rl, mc]
    )

model.save('C:/data/h5/fish_model2_4.h5')
model.save_weights('C:/data/h5/fish_weight_4.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#평가
loss, acc = model.evaluate(xy_test)
print('loss : ', loss)
print('acc : ', acc)

#검증
# predict image filter
path = r'C:/data/fish_data/fish_predicts_data/real' # Source Folder
dstpath = r'C:/data/fish_data/fish_predicts_data/image_filter' # Destination Folder
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
    size = cv2.resize(gradient, (360, 240), interpolation = cv2.INTER_CUBIC)
    #결과 출력
    # merged = np.hstack((img, gradient))
    # cv2.imshow('gradient', merged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)
    
def Dataization(img_path):
    image_w = 256
    image_h = 256
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)
 
src = []
name = []
test = []
image_dir = "C:/data/fish_data/fish_predicts_data/image_filter/"
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):      
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))

test = np.array(test)
y_pred = model.predict(test)
 
for i in range(len(test)):
    if y_pred[i] > 0.5 :
        print(name[i] + "열대어는" , y_pred[i]*100, "% 확률로 질병에 걸렸으므로 적절한 조치가 필요합니다.")
    else :
        print(name[i] + "열대어는" , (1-y_pred[i])*100, "% 확률로 정상입니다." )


# #시각화
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# #그래프 출력
# import matplotlib.pyplot as plt
# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

# loss :  0.0006684510735794902
# acc :  1.0