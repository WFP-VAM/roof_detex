import numpy as np
import os
from utils import get_image, save_history_plot
from unet import unet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard

# PARAMETERS ------------
img_rows, img_cols = 400, 400
classes = 1
batch_size = 16
epochs = 20
# because of MaxPool layer width and height has to be divisible by 2^4
# use 320 on the rows to avoid the Google sign.

# get training images ----------------------
training_images = []
for file in os.listdir('GiveDirectlyData/data/images'):
    if file.endswith(".png"):
        data = get_image('GiveDirectlyData/data/images/' + file)
        training_images.append(data[:img_rows,:])


# get masks_1class -------------------------------
training_masks = []
for file in os.listdir('masks/1/masks'):
    if file.endswith(".png"):
        data = get_image('masks/1/masks/' + file)
        training_masks.append(data[:img_rows,:])


# reshape ---------------------------------
training_images = np.array(training_images)
training_masks = np.array(training_masks)[:, :, :, 0].reshape(len(training_masks), img_rows, img_cols, 1)

# generators for data augmentation -------
seed = 1
generator_x = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True)

generator_y = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    vertical_flip=True,
    horizontal_flip=True)

generator_x.fit(training_images, augment=True, seed=seed)
generator_y.fit(training_masks, augment=True, seed=seed)


image_generator = generator_x.flow_from_directory(
    'GiveDirectlyData/data',
    target_size=(img_rows, img_cols),
    class_mode=None,
    seed=seed)

mask_generator = generator_y.flow_from_directory(
    'masks/1',
    target_size=(img_rows, img_cols),
    class_mode=None,
    color_mode='grayscale',
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

# load model
model = unet(img_rows, img_cols, classes=classes)

# kick off training
tb = TensorBoard(log_dir='logs', histogram_freq=False,  write_graph=False, write_images=False)

model.fit_generator(train_generator, steps_per_epoch=int(len(training_images)/batch_size),
                    epochs=epochs, callbacks=[tb])

# save model
model.save('UNET_model_1class_aug.h5')