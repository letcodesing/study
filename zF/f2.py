from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )


#2. 모델구성

model = Sequential()
model.add(Dense(200, input_dim=10))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(180))
model.add(Dense(170))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=15)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# r2 0.62 이상
# loss :  2452.336669921875
# r2스코어 :  0.6286149246878252

#https://github.com/thiefdirk/study-gkeol/blob/b63561b46a289ffa7dfe6a10661dc2c4ec34a2f0/keras/keras08_3_diabets.py