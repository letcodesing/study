#r2 0.62 이상


from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
#y는 어차피 결과값으로 나오는 것이기 때문에 여기서 전처리는 필요없다

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#2.모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(4, input_dim=10))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
history = model.fit(x_train, y_train, epochs=200, batch_size=100, validation_split=0.1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

print(history.history)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.figure(figsize=(2,3))
plt.grid()
plt.plot(history.history['loss'], marker='_', c='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', c='green', label='val_loss')
plt.title('나나미')
plt.ylabel('epochs')
plt.xlabel('loss')
plt.legend()
plt.show()


