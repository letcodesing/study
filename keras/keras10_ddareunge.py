#1.데이터
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


path = './_data/ddareunge/'
train_set = pd.read_csv(path + 'train.csv')

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1892)

#dims 확인

print(x.shape)

#2.모델구성

model = Sequential()
model.add(Dense(100, input_dims=1))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=1000, batch_size=15, verbose=3)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)
print(y_predict)

r2_score = r2(y_test, y_predict)
print('r2', r2)