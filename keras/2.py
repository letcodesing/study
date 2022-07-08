from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
# model.add(Dense(units=10, input_shape=(10,10,3)))
# model.add(Conv2D(filters=10, kernel_size=(3,3), input_shape=(10,10,3)))
# 8,8,10
# model.summary()
# model.add(Dense(units=10, input_shape=(3,)))
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))
# (None, 4, 4, 10)          50 출력
model.add(Conv2D(7, (2,2), activation='relu'))
# (None, 3, 3, 7)           287 출력
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 
5by5 이미지에서 10칼럼으로
model.summary()
