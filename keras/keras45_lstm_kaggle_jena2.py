#split lstm conv1d rnn cnn 시계열 데이터
import pandas as pd
import numpy as np
data = pd.read_csv('c:/study/_data/kaggle_jena/jena_climate_2009_2016.csv')
data = data.transpose()
print(data[:5])
print(data.shape)
def split(seq, size): 
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

data = data.iloc[:,:143]
print(data)