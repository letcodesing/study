#존나 많네
#데이터 불러오기 캐글 캘리포니아가 아니라서 다행이다
from sklearn.datasets import fetch_california_housing
data_set=fetch_california_housing()
print(data_set.feature_names)
print(data_set.DESCR)

x = data_set.data
y = data_set.target
    # :Number of Instances: 20640

    # :Number of Attributes: 8 numeric, predictive attributes and the target
#,xy나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.7, shuffle=False
)
#모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss = 'mae', optimizer='adam',
              metrics=['accuracy', 'mse'])
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor = 'val_loss', patience=20,
                   mode='min', restore_best_weights=True,
                   verbose=3)
hist = model.fit(x_train, y_train, epochs=100, 
                 validation_split=0.2,
                 verbose=2)
#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

#프린트 맨나중 그리기먼저
import matplotlib.pyplot as plt
plt.figure(figsize=(5,2))
plt.plot(hist.history['loss'], marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.',c='blue',label='val_loss')
plt.grid()
plt.xlabel('좋아')
plt.ylabel('모든것이좋아')
plt.legend()
plt.show()
print('loss', loss)
print('r2', r2)