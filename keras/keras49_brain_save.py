from matplotlib.pyplot import axis
from keras.datasets import fashion_mnist,mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

#1.데이터
train_datagen0 = ImageDataGenerator(rescale=1./255,)  
test_datagen0 = ImageDataGenerator(rescale=1./255)   

xy_train = train_datagen0.flow_from_directory(
    'C:\\Users/asthe\Downloads/brain/train/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=500,
    class_mode='binary', 
    color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )#Found 160 images belonging to 2 classes

xy_test = test_datagen0.flow_from_directory(
    'C:\\Users/asthe\Downloads/brain/test/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,)

x_train=np.array(xy_train[0][0])
y_train=np.array(xy_train[0][1])
x_test=np.array(xy_test[0][0])
y_test=np.array(xy_test[0][1])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

augument_size = 64
randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)

x_augumented = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]

x_data_all = np.concatenate((x_train, x_augumented)) 

y_data_all = np.concatenate((y_train, y_augument))

# xy_train = test_datagen.flow(x_data_all, y_data_all, batch_size=augument_size, shuffle=False)

# xy_test = test_datagen.flow(x_test, y_test, batch_size=augument_size, shuffle=False)


np.save('c:/study_data/_save/keras49_05_train_x.npy', arr=x_data_all)
np.save('c:/study_data/_save/keras49_05_train_y.npy', arr=y_data_all)
np.save('c:/study_data/_save/keras49_05_test_x.npy', arr=x_test)
np.save('c:/study_data/_save/keras49_05_test_y.npy', arr=y_test)