# lstm -> dnn 좋음
# dnn -> lstm 별로 dnn은 불연속적인 데이터에 많이 사용하기 때문
# cnn -> lstm -> dnn도 가능
 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
 
trainset = np.array(range(1,101))
testset = np.array(range(96,106))
print(len(trainset))

size = 5
def split_x(seq, size):  
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
train = split_x(trainset, size)

print('------------------')
print(train)
print(train.shape)
 
x = train[:, :-1]
y = train[:, -1] 
 
print(x.shape) 
print(y.shape) 




size = 4
test = split_x(testset, size)

print(test)
print(x.shape, y.shape, test.shape)



model = Sequential()
model.add(Dense(33, activation = 'relu', input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(1))
# LSTM을 DNN으로 구현 가능
 
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=1)



y_pred = model.predict(test)
print(y_pred)

