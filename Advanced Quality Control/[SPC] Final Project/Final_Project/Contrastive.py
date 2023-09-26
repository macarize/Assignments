import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm
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
from tensorflow.keras import layers
from keras.losses import binary_crossentropy
import tensorflow_addons as tfa
from keras.metrics import accuracy
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
epochs = 50
lr = 1e-4

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = np_utils.to_categorical(Y_train, nb_classes)

Y_test = np_utils.to_categorical(Y_test, nb_classes)
dataset_train = tf.data.Dataset.from_tensor_slices((X_train,Y_train[:, 0])).batch(1024).shuffle(buffer_size = 10 * batch_size)
dataset_test = tf.data.Dataset.from_tensor_slices((X_test,Y_test[:, 0])).batch(1024).shuffle(buffer_size = 10 * batch_size)
encoder = Sequential()

encoder.add(Dense(128, activation = 'relu', input_dim = 48))
encoder.add(Dropout(0.3))
encoder.add(Dense(256, activation= 'relu'))
encoder.add(Dropout(0.3))
encoder.add(Dense(512, activation= 'relu'))
encoder.add(Dropout(0.3))
encoder.add(Dense(512, activation= 'relu'))
encoder.add(Dropout(0.3))
encoder.add(Dense(256, activation= 'relu'))
encoder.add(Dropout(0.3))
encoder.add(Dense(128, activation= 'relu'))

opt = Adam(lr)
encoder.summary()
history = History()

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
            layer.trainable = trainable

    inputs = keras.Input(shape=48)
    features = encoder(inputs)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(1, activation="sigmoid")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

def supervised_nt_xent_loss(z, y, temperature=0.2, base_temperature=0.07):

    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max

    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask

    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss

classifier = create_classifier(encoder)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        r = encoder(images, training=True)
        logits = classifier(images, training=True)

        loss_contrast = supervised_nt_xent_loss(r, labels)
        loss_cross = binary_crossentropy(logits, labels)

        loss = loss_contrast * 0.0001 + loss_cross
    gradients = tape.gradient(loss, encoder.trainable_variables)
    opt.apply_gradients(zip(gradients, encoder.trainable_variables))

    return loss

train_loss_results = []

for epoch in tqdm(range(5)):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for images, labels in dataset_train:
        loss = train_step(images, labels)
        epoch_loss_avg.update_state(loss)
        train_loss_results.append(epoch_loss_avg.result())

plt.plot(train_loss_results)
plt.title("Supervised Contrastive Loss")
plt.show()

accuracy = classifier.evaluate(X_test, Y_test[:, 0])[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

