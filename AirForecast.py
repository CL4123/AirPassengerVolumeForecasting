import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

import warnings
warnings.filterwarnings('ignore')

# 解决中文乱码问题
#sans-serif就是无衬线字体，是一种通用字体族。
#常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等。
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#导入数据
df = pd.read_csv('DATA/AirPassengers.csv', sep=',')
df = df.set_index('time')

df.head()

#画图
df['passengers'].plot()
plt.ylabel("客运量（千人）")
plt.xlabel("时间")
plt.show()

#只用客运流量一列
df = pd.read_csv('DATA/AirPassengers.csv', sep=',', usecols=[1])
data_all = np.array(df).astype(float)

#数据归一化
scaler = MinMaxScaler()
data_all = scaler.fit_transform(data_all)

#时间序列
sequence_length=10
data = []
for i in range(len(data_all) - sequence_length - 1):
    data.append(data_all[i: i + sequence_length + 1])
reshaped_data = np.array(data).astype('float64')
reshaped_data

split = 0.8
np.random.shuffle(reshaped_data)
x = reshaped_data[:, :-1]
y = reshaped_data[:, -1]
split_boundary = int(reshaped_data.shape[0] * split)
train_x = x[: split_boundary]
test_x = x[split_boundary:]

train_y = y[: split_boundary]
test_y = y[split_boundary:]

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

#搭建LSTM模型
model = Sequential()
model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
print(model.layers)
model.add(LSTM(100, return_sequences=False))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(train_x, train_y, batch_size=512, nb_epoch=100, validation_split=0.1)
predict = model.predict(test_x)
predict = np.reshape(predict, (predict.size, ))

predict_y = scaler.inverse_transform([[i] for i in predict])
test = scaler.inverse_transform(test_y)

plt.plot(predict_y, 'g:', label='prediction')
plt.plot(test, 'r-', label='true')
plt.legend(['模型预测值', '真实值'])
plt.ylabel("客运量（千人）")
plt.xlabel("时间")
plt.show()
