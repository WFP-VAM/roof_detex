import json
import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model

# PARAMETERS ------------
img_rows, img_cols = 400, 400


# get images metadata -------------------------------------------
def load_training_data():
    # Load training data
    json_data = open('../GiveDirectlyData/data/train-data-2014-01-13.json')
    roof_data = [json.loads(json_line) for json_line in json_data]
    image_meta = pd.read_csv('../GiveDirectlyData/data/image_metadata.csv')
    roof_train = pd.DataFrame(roof_data)
    roof_train['image_tag'] = roof_train.image.map(lambda name: name.strip().split('-')[0])
    roof_train['image'] = roof_train.image.map(lambda name: name.strip())

    # Merge Training data
    all_train = pd.merge(image_meta, roof_train, left_on='GDID', right_on='image_tag')
    return all_train

training_data = load_training_data()


# create mask, 1 roof, 0 nothing ---------------------------
mask = []
for ix, row in training_data.iterrows():
    tmp = np.zeros([400,400])

    for roof in row['roofs']:
        tmp[int(roof['y']), int(roof['x'])] = 1

    mask.append(tmp)

# data loading routines ----------------------------------
# https://github.com/JamilGafur/Unet/blob/master/U-net%20Cell%20segment.ipynb

training_images = []
for file in os.listdir('../GiveDirectlyData/data/images'):
    if file.endswith(".png"):
        img = Image.open("../GiveDirectlyData/data/images/" + file)
        img.load()
        data = np.asarray(img, dtype="int32")
        training_images.append(data)


train_images = np.array(training_images).reshape(len(training_images), 400, 400, 1)
train_labels = np.array(mask).reshape(len(training_images), 400, 400, 1)


# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
inputs = Input((img_rows, img_cols, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels, batch_size=8, epochs=5, shuffle=True, validation_split=0.2)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('training_history.png')

# save model
model.save('UNET_model.h5')
