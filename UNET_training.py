import tensorflow as tf

from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, \
    BatchNormalization, UpSampling2D, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Concatenate

import pandas as pd
import numpy as np

train_data_dir = '../Data/train_roofs'
validation_data_dir = '../Data/validate_roofs'
labels = pd.read_csv('labels.csv')
# dimensions of our images.
img_width, img_height = 400, 400

nb_train_samples = 1101
nb_validation_samples = 363
epochs = 100
batch_size = 16

# UNET
# https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_buildings.py
concat_axis = 3

inputs = Input((400, 400, 3))

conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

up6 = Concatenate(axis=concat_axis)([UpSampling2D(size=(2, 2))(conv5), conv4])
conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

up7 = Concatenate(concat_axis)([UpSampling2D(size=(2, 2))(conv6), conv3])
conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

up8 = Concatenate(concat_axis)([UpSampling2D(size=(2, 2))(conv7), conv2])
conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

up9 =Concatenate(concat_axis)([UpSampling2D(size=(2, 2))(conv8), conv1])
conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

flatten = Flatten()(conv9)
Dense1 = Dense(512, activation='relu')(flatten)
BN = BatchNormalization()(Dense1)
Dense2 = Dense(1, activation='linear')(BN)

model = Model(input=inputs, output=Dense2)

opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=1e-6)

model.compile(loss='mse',
              optimizer=opt)


# https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model?noredirect=1#comment70692649_41749398
def regression_flow_from_directory(flow_from_directory_gen, list_of_values):
    for x, y in flow_from_directory_gen:
        yield x, list_of_values[y]


# prepare data augmentation configuration
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')

reg_gen_train = regression_flow_from_directory(train_generator, labels[nb_train_samples:]['total'].values)

reg_gen_validation = regression_flow_from_directory(validation_generator, labels[nb_validation_samples:]['total'].values)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0,
                          write_graph=True, write_images=False)

# train the model
history = model.fit_generator(
    reg_gen_train,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=reg_gen_validation,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tensorboard])

# plot training
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('training_history.png')
