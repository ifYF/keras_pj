from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'  # 只显示 Error

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train = X_train.reshape(-1, 1, 32, 32)/255.
#X_test = X_test.reshape(-1, 1, 32, 32)/255.
X_train = X_train.reshape(-1, 3, 32, 32)/255.
X_test = X_test.reshape(-1, 3, 32, 32)/255.


y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()

# model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
# Conv layer 1 output shape (filters32, 32, 32)
model.add(Convolution2D(
#    batch_input_shape=(None, 1, 32, 32),
    batch_input_shape=(None, 3, 32, 32),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))
# Pooling layer 1 (max pooling) output shape (32, 16, 16)
model.add(MaxPooling2D(
    pool_size=2,#池化核大小
    strides=2,#步长
    padding='same',    # Padding method
    data_format='channels_first',
))
#model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# Conv layer 2 output shape (64, 16, 16)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))
# Pooling layer 2 (max pooling) output shape (64, 8, 8)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
#model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected layer 1 input shape (64 * 8 * 8) = (4096), output shape (1024)
# model.add(Dense(512))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# model.summary()

# # initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
# Another way to define your optimizer
adam = Adam(lr=1e-4)

# # train the model using RMSprop
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# hist = model.fit(x_train, y_train, epochs=40, shuffle=True)
print('Training ------------')
# Another way to train the model
history = model.fit(X_train, y_train, epochs=20, verbose=1, batch_size=64)
# verbose=0(不显示),1(进度条显示),2(每一个epochs显示一个)
# history_dict = history.history
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']
# print(acc_values, val_acc_values)

# # evaluate
# loss, accuracy = model.evaluate(x_test, y_test)
# print loss, accuracy
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)