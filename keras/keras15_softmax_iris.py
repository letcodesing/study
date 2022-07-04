import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#1.데이터
datasets = load_iris()
print(datasets.DESCR)
#4가지를 가르쳐줄테니 50개의 3종류 꽃으 맞춰라
# :Number of Instances: 150 (50 in each of three classes)
# # y=         - class:
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=False)

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
model = Sequential()
model.add(Dense(5, input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, verbose=1, 
          validation_split=0.2,
          callbacks=ES)

#4.평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score



pre2 = y_predict.flatten() # 차원 펴주기
pre3 = np.where(pre2 > 1, 2 , 0) #1보다크면 2, 작으면 0
acc = accuracy_score(y_test, pre3)

print(pre3)
print('loss', loss)
print('acc', acc)

# loss [0.3140193819999695, 0.0]
# r2 0.0

# loss [0.46477609872817993, 0.0]
# r2 0.0