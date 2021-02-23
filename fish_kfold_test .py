#이진분류로 할 경우 f1 score로 잡아야함.
#database 숫자 증폭 잘할 것

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten, BatchNormalization, Input, AveragePooling2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import cv2
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
    batch_size = 2000, 
    class_mode = 'binary',
    subset="training"
)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory(
    'C:/data/fish_data/fish_datasets/x1_test',
    batch_size = 2000, 
    class_mode = 'binary'
)

np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_train_x.npy', arr=xy_train[0][0]) #x 1
np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_train_y.npy', arr=xy_train[0][1])
np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_test_x.npy', arr=xy_test[0][0]) #x 1
np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_test_y.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_train_x.npy') #x 1
y_train = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_train_y.npy')
x_test = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_test_x.npy') #x 1
y_test = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_test_y.npy')
print(x_train.shape, y_train.shape)


#모델링
input1 = Input(shape=(x_train.shape[1], x_train.shape[2] ,x_train.shape[3]))
x = Conv2D(64, 4, activation='relu')(input1)
x = BatchNormalization()(x)
x = Conv2D(128, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, activation='relu')(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Dropout(0.1)(x)
x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#훈련
model.compile(
    loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
es = EarlyStopping(monitor='loss', patience=70, verbose=1, mode="auto",)
rl = ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=20, verbose=1, mode='auto')
modelpath = 'C:/data/MC/best_fish_illness_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
hist= model.fit(x_train, y_train, epochs = 1000, batch_size = 32,validation_data = (x_test, y_test), callbacks=[es, rl, mc])

model.save('C:/data/h5/fish_model2_3.h5')
model.save_weights('C:/data/h5/fish_weight_3.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)


#검증
# predict image filter
path = r'C:/data/fish_data/fish_predicts_data/real' # Source Folder
dstpath = r'C:/data/fish_data/fish_predicts_data/opencv/image_filter' # Destination Folder
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
    merged = np.hstack((img, gradient))
    cv2.imshow('gradient', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),size)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_pred = test_datagen.flow_from_directory(
    'C:/data/fish_data/fish_predicts_data/opencv',
    batch_size = 128, 
    class_mode = 'binary'
)

# #true, fales
y_pred = model.predict(xy_test)
print(y_pred)

#true, fales
# results = model.predict(xy_pred)
# if results > 0.5 :
#     print("해당 열대어는" , results*100, "% 확률로 정상입니다.")
# else :
#     print("해당 열대어는" , (1-results)*100, "% 확률로 질병에 걸렸으므로 적절한 조치가 필요합니다." )


#시각화
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#그래프 출력
import matplotlib.pyplot as plt
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
