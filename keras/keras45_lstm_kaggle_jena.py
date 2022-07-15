#split lstm conv1d rnn cnn 시계열 데이터
import pandas as pd
import numpy as np
data = pd.read_csv('c:/study/_data/kaggle_jena/jena_climate_2009_2016.csv')
print(data.shape) #(420551, 15) #하루 143행
gen = data.to_numpy()
print(gen)
def split(seq, size): #함수 split_x는 한 
    aaa = []
    for i in range(len(gen.loc[:]) -i*144):
        subset = seq.loc[i*575:i*575+i+574]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


# print(len(data.loc[:]))
size = 419976
gen2 = split(gen,size)
print(gen2.shape)
print(gen2)

# print(data)
# print(data.shape)
# print(np.unique(data, return_counts=True))
# print(data.value_counts())
# print(data.describe())
# print(data.isnull().sum())
# print(data.info())
# print(data[:5])
# print(data.shape)
# print(data_split.colu)
# data.reshape(6,4205510)
# x=data[:,:-1]
# y=data[:-1]
# print(x.shape, y.shape)
# print(x[:5],y[:5])
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=9812)

# from keras.models import Model
# from keras.layers import Conv2D,LSTM,Flatten,Reshape,Dense,Input
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# input1 = Input(shape=())
# c1 = Conv2D(10,(3,5))(input1)
# r1 = Reshape(4,60)(c1)
# l1 = LSTM(400, activation='relu',name='l1')(r1)
# d1 = Dense(20)(l1)
# out1 = Dense(1)(d1)
# model = Model(inputs=input1, outputs=out1)
# model.summary()

# model.compile(loss='mse', optimizer='adam',metrics='mad')
# model.fit(x,y,epochs=10, batch_size=2000)
# loss = model.evaluate(x_test,y_test)
# print(loss)


# def uni_data(dataset, start_index, end_index, history_size, target_size);
# 	data = []
# 	labels = []

# 	start_index = start_index + history_size
# 	if end_index is None:
# 		end_index = len(dataset) - target_size

# 	for i in range(start_index, end_index):
# 		indices = range(i-history_size, i)
# 		data.append(np.reshape(dataset[indices], (history_size, 1)))
# 		labels. append(dataset[i+target_size])
# 	return np.array(data), np.array(labels)

# data2 = uni_data(data,0,,10,1)