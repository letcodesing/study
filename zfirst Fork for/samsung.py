import pandas as pd
import numpy as np
df_price = pd.read_csv('c:/study/_data/test_amore_0718/삼성전자220718.csv', encoding='cp949',thousands=',' )
df_price.describe()


# pd.to_datetime(df_price['일자'], format='%y%m%d')
# # 0      2020-01-07
# # 1      2020-01-06
# # 2      2020-01-03
# # 3      2020-01-02
# # 4      2019-12-30

# df_price['일자'] = pd.to_datetime(df_price['일자'], format='%Y%m%d')
# df_price['연도'] =df_price['일자'].dt.year
# df_price['월'] =df_price['일자'].dt.month
# df_price['일'] =df_price['일자'].dt.day

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['시가', '고가', '저가', '종가', '거래량']
df_scaled = scaler.fit_transform(df_price[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

print(df_scaled)
train = df_scaled[:-200]
test = df_scaled[-200:]
def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
feature_cols = ['시가', '고가', '저가', '거래량']
label_cols = ['종가']

train_feature = train[feature_cols]
train_label = train[label_cols]

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)


# ((6086, 20, 4), (1522, 20, 4))

# test dataset (실제 예측 해볼 데이터)


# ((180, 20, 4), (180, 1))
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)
# filename = os.path.join(model_path, 'tmp_checkpoint.h5')
# checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=16,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop])
# model.load_weights(filename)

# 예측
pred = model.predict(test)
print(pred)