#이진분류로 할 경우 f1 score로 잡아야함.
#database 숫자 증폭 잘할 것
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten, BatchNormalization, Input, AveragePooling2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
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


# np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_train_x.npy', arr=xy_train[0][0]) #x 1
# np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_train_y.npy', arr=xy_train[0][1])
# np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_test_x.npy', arr=xy_test[0][0]) #x 1
# np.save('C:/data/fish_data/fish_datasets/fish_npy/fish_test_y.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_train_x.npy') #x 1
y_train = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_train_y.npy')
x_test = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_test_x.npy') #x 1
y_test = np.load('C:/data/fish_data/fish_datasets/fish_npy/fish_test_y.npy')
print(x_train.shape, y_train.shape)


#모델링
input1 = Input(shape=(x_train.shape[1], x_train.shape[2] ,x_train.shape[3]))
x = Conv2D(64, 4, activation='relu')(input1)
x = BatchNormalization()(x)
x = Conv2D(256, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, 3, activation='relu')(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
outputs = Dense(1, activation= 'softmax')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#훈련
model.compile(
    loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics='acc')
es = EarlyStopping(monitor='loss', patience=5, verbose=1, mode="auto",)
rl = ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto')
modelpath = 'C:/data/MC/best_fish_illness_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
hist= model.fit(x_train, y_train, epochs = 20, batch_size = 32,validation_data = (x_test, y_test), callbacks=[es, rl, mc])

model.save('C:/data/h5/fish_model2_3.h5')
model.save_weights('C:/data/h5/fish_weight_3.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#평가
loss, acc, f1_score, precision, recall = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('f1_score : ', f1_score)
print('precision : ', precision)
print('recall : ', recall)


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
    size = cv2.resize(gradient, (128, 128), interpolation = cv2.INTER_CUBIC)
    #결과 출력
    merged = np.hstack((img, gradient))
    cv2.imshow('gradient', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

x_test = np.array(test)
y_pred1 = model.predict(x_test)
y_pred = np.argmax(y_pred1, axis=1)

# Print f1, precision, and recall scores
print(precision_score(y_test, y_pred , average="macro"))
print(recall_score(y_test, y_pred , average="macro"))
print(f1_score(y_test, y_pred , average="macro"))


#시각화
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
