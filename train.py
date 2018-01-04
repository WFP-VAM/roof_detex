import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

# PARAMETERS ------------
img_rows, img_cols = 400, 400


# data loading routines ----------------------------------
# https://github.com/JamilGafur/Unet/blob/master/U-net%20Cell%20segment.ipynb
def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    #image = image.convert('RGB')
    image = np.array(image)
    return image


# get training images
training_images = []
for file in os.listdir('GiveDirectlyData/data/images'):
    if file.endswith(".png"):
        data = get_image('GiveDirectlyData/data/images/' + file)
        training_images.append(data)

# get masks
training_masks = []
for file in os.listdir('masks'):
    if file.endswith(".png"):
        data = get_image('masks/' + file)
        training_masks.append(data)


training_images = np.array(training_images) #.reshape(len(training_images), 400, 400, 3)
training_masks = np.array(training_masks)[:, :, :, 0]#.reshape(len(training_masks), 400, 400, 1)


# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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

model.compile(optimizer=Adam(lr=0.001, decay=0.0001), loss='binary_crossentropy', metrics=[dice_coef])

# normalize images
training_images = training_images.astype('float32')
mean = np.mean(training_images)  # mean for data centering
std = np.std(training_images)  # std for data normalization
training_images -= mean
training_images /= std
train_images = training_images/255.

# viz check
# plt.figure()
# plt.imshow(training_images[0])
# plt.imshow(training_masks[0], cmap='gray', alpha=0.5)
# plt.show()

tb = TensorBoard(log_dir='logs', histogram_freq=0,  write_graph=True, write_images=False)


history = model.fit(training_images.reshape(len(training_images),400,400,1),
                    training_masks.reshape(len(training_images),400,400,1),
                    batch_size=8, epochs=30, shuffle=True,
                    validation_split=0.3, callbacks=[tb])

# save training history plot
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
