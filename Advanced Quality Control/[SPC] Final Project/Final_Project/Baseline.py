import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import keras
import os, sys, math, copy
import scipy.io as sio
import tensorflow as tf
from keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from keras.layers import InputSpec
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from keras import initializers, regularizers, constraints
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras import backend as K
from keras.utils import np_utils

sys.setrecursionlimit(10000)

import matplotlib.pyplot as plt

train_sample = pd.read_csv("train.csv", header=0, encoding="utf-8")
path = r'C:\Users\leehy\Downloads\03. Dataset_CNC\dataset\1\CNC Virtual Data set _v2'
all_files = glob.glob(path + "\*.csv")

train_sample_np = np.array(train_sample.copy())

li_df = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li_df.append(df)

def tool_condition(input):
    for i in range(len(input)):
        if input[i, 4] == "unworn":
            input[i, 4] = 0
        else:
            input[i, 4] = 1
    return input

def item_inspection(input):
    for i in range(len(input)):
        if input[i, 5] == 'no':
            input[i, 6] = 2
        elif input[i, 5] == 'yes' and input[i, 6] == 'no':
            input[i, 6] = 1
        elif input[i, 5] == 'yes' and input [i, 6] == 'yes':
            input[i, 6] = 0
    return input

def machining_process(input):
    for i in range(len(input)):
        if input[i,47] == 'Prep':
            input[i,47] =0
        elif input[i,47] == 'Layer 1 Up':
            input[i,47] =1
        elif input[i,47] == 'Layer 1 Down':
            input[i,47] =2
        elif input[i,47] =='Layer 2 Up':
            input[i,47] = 3
        elif input[i,47] =='Layer 2 Down':
            input[i,47] =4
        elif input[i,47] =='Layer 3 Up':
            input[i,47] =5
        elif input[1,47] =='Layer 3 Down':
            input[i,47] =6
        elif input[i,47] =='Repositioning':
            input[i,47] =7
        elif input[i,47] =='End'or'end':
            input[i,47] =8
        elif input[i,47] =='Starting':
            input[i,47] =9
    return input

train_sample_info = np.array(train_sample_np.copy())
train_sample_info = tool_condition(train_sample_info)
train_sample_info = item_inspection(train_sample_info)

train_sample_info = np.delete(train_sample_info,5, 1)
train_sample_info = np.delete(train_sample_info,0, 1)
train_sample_info = np.delete(train_sample_info,0, 1)

k = 0
li_pass = []
li_pass_half = []
li_fail = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)

    if train_sample_info[k, 3] == 0:
        li_pass.append(df)
    elif train_sample_info[k, 3] == 1:
        li_pass_half.append(df)
    else :
        li_fail.append(df)

    k += 1

frame01 = pd.concat(li_pass,axis=0, ignore_index=True)
frame02 = pd.concat(li_pass_half,axis=0, ignore_index=True)
frame03 = pd.concat(li_fail,axis=0, ignore_index=True)
data_pass = np.array(frame01.copy())
data_pass_half = np.array(frame02.copy())
data_fail = np.array(frame03.copy())

data_pass = machining_process(data_pass)
data_pass_half = machining_process(data_pass_half)
data_fail = machining_process(data_fail)

data01 = data_pass[0:3228 + 6175, :]
data02 = data_pass_half[0:6175, :]
data03 = data_fail[0:3228, :]
data = np.concatenate((data01, data02), axis=0)
data = np.concatenate((data, data03), axis=0)
data_all = data_pass[3228 + 6175 : 22645, :]

sc = MinMaxScaler()
X_train = sc.fit_transform(data)
X_train = np.array(X_train)
X_test = sc.fit_transform(data_all)
X_test = np.array(X_test)

Y_train = np.zeros((len(X_train), 1), dtype='int')
Y_test = np.zeros((len(X_test), 1), dtype='int')
l = int(Y_train.shape[0]/2)
Y_train[0:l, :] = 0
Y_train[l:l*2, :] = 1

nb_classes = 2
batch_size = 1024
epochs = 300
lr = 1e-4

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

from tensorflow.keras.losses import Loss


class ContrastiveLoss(Loss):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = 48))
model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes, activation= 'sigmoid'))
model_checkpoint = ModelCheckpoint('weight_CNC_binary.mat', monitor='val_acc', save_best_only='True')
opt = Adam(lr)
model.summary()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])
history = History()

model.fit(X_train, Y_train, verbose=2, batch_size=batch_size, epochs=5, validation_split=0.1, shuffle=True, callbacks = [history])
model.save_weights('weight_CNC_binary.mat')

loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=32)
loss_and_metrics2= model.evaluate(X_test, Y_test, batch_size=32)

print(loss_and_metrics)
print(loss_and_metrics2)

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.title('Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Validation Accuracy', 'Training Accuracy'])
plt.show()

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Validation Loss', 'Training Loss'])
plt.show()