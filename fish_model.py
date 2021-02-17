#이진분류로 할 경우 f1 score로 잡아야함.
#database 숫자 증폭 잘할 것

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2

#선언
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=1.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split = 0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)


#xy_train
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128), 
    batch_size = 32, 
    class_mode = 'binary',
    subset="training"
)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory(
    'C:/data/image/gender',
    target_size = (128, 128),
    batch_size = 32, 
    class_mode = 'binary'
)

input1 = Input(shape=(128, 128 ,3))
x = Conv2D(64, 4, activation='relu')(input1)
x = BatchNormalization()(x)
x = Conv2D(128, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 2, activation='relu')(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Dropout(0.4)(x)
x = Flatten()(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(learning_rate=0.1), 
    metrics=['acc']
    
    )

hist= model.fit_generator(
    xy_train,
    steps_per_epoch=44,
    epochs=50,
    validation_data=xy_val,
    callbacks=[es, rl]
)
#평가
loss, acc = model.evaluate(xy_test)
print('loss : ', loss)
print('acc : ', acc)

#남/여
results = model.predict(xy1_test)
if results > 0.5 :
    print("정상일 확률 : " , results*100, "%")
else :
    print("질병에 길린 확률 : " , (1-results)*100, "%" )


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
