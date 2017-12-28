# https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model?noredirect=1#comment70692649_41749398

import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop

import pandas as pd
import numpy as np

train_data_dir = 'Data/train_roofs'
validation_data_dir = 'Data/validate_roofs'
labels = pd.read_csv('labels.csv')
# dimensions of our images.
img_width, img_height = 400, 400

nb_train_samples = 1101
nb_validation_samples = 363
epochs = 20
batch_size = 16

# build the VGG16 network
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(400, 400, 3))

# to non-trainable (weights will not be updated)
for layer in base_model.layers:  # 18 leaves the last max pool out
    if ((layer.name == 'block5_conv1') |
            (layer.name == 'block5_conv2') |
            (layer.name == 'block5_conv3') |
            (layer.name == 'block5_pool')):
        layer.trainable = True
    else:
        layer.trainable = False

    print(layer.name, ' : ', layer.trainable)

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='linear'))

# add the model on top of the convolutional base
model = tf.keras.models.Model(inputs=base_model.input, outputs=top_model(base_model.output))

# compile model
opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=1e-6)

model.compile(loss='mse', optimizer=opt)

# GENERATORS
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
