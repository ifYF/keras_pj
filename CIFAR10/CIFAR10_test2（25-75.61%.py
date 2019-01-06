# 初始化
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

np.random.seed(0)
print('Initialized!')

# 定义变量
batch_size = 32
nb_classes = 10
nb_epoch = 25
img_rows, img_cols = 32, 32
nb_filters = [32, 32, 64, 64]
pool_size = (2, 2)
kernel_size = (3, 3)

# 读取数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

input_shape = (img_rows, img_cols, 3)
# one_hot编码
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 图片生成器
datagen = ImageDataGenerator(
            featurewise_center=False,   # 数据集去中心化（均值为0）
            samplewise_center=False,    # 每个样本均值为0
            featurewise_std_normalization=False,    # 输入除以数据集的标准差以完成标准化
            samplewise_std_normalization=False, # 输入的每个样本除以其自身的标准差
            zca_whitening=False, # 对输入数据施加ZCA白化
            rotation_range=0,   # 数据提升时图片随机转动的角度
            width_shift_range=0.1,  # 数据提升时图片随机水平偏移的幅度
            height_shift_range=0.1, # 数据提升时图片随机竖直偏移的幅度
            horizontal_flip=True,   # 随机水平翻转
            vertical_flip=False)    # 随机竖直翻转
datagen.fit(X_train)

# 搭建结构
model = Sequential()
model.add(Conv2D(nb_filters[0], kernel_size, padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters[1], kernel_size))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters[2], kernel_size, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters[3], kernel_size))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 优化器
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 训练模型
# best_model = ModelCheckpoint('',)
t1 = time.time()
model.fit_generator(
                    datagen.flow(X_train, Y_train, batch_size=batch_size),
                    epochs=nb_epoch, verbose=1,
                    validation_data=(X_test, Y_test),
                    steps_per_epoch=X_train.shape[0] // batch_size)

# 模型评分
score = model.evaluate(X_test, Y_test, verbose=0)
t2 = time.time()
# 输出结果
print('Test score:', score[0])
print('Accuracy:%.2f%%' % (score[1]*100))
print('Time used: %.2f min' % ((t2 - t1)/60.))
print('Compiled!')




