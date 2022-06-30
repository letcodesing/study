from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=False)

model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=20, validation_split=0.25)
submission. = pd.read_csv(path + 'submission.csv')
submission.to_csv(path + 'submission.csv')
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print(r2)

import numpy as np
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)
print(rmse)



