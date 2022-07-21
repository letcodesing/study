#kaggle dog cat
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/training_set/training_set/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test_set/test_set/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary',
    # color_mode='grayscale',
    shuffle=True,
    )

# print(xy_train)
print(xy_test)

np.save('d:/study_data/_save/_npy/keras47_01_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras47_01_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras47_01_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras47_01_test_y.npy', arr=xy_test[0][1])

0000
print(x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size = argument_size)

print(np.min(randidx), np.max(randidx))
print(type(randidx))
x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

print(x_argumented.shape)
print(y_argumented.shape)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

x_argumented = x_argumented.reshape(x_argumented.shape[0], x_argumented.shape[1], x_argumented.shape[2],1)

x_argumented = train_datagen.flow(x_argumented, y_argumented,
                                  batch_size=argument_size,
                                  shuffle=False,).next()[0]
print(x_argumented)
print(x_argumented.shape)


x_train = np.concatenate((x_train, x_argumented))
y_train = np.concatenate((y_train, y_argumented))

import matplotlib.pyplot as plt
for i in range(20):
    if i <= 9:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_argumented[i], cmap='gray')
    else:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[randidx[i-10]], cmap='gray')
    
plt.show() 