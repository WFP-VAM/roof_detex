import os
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from src.utils import get_image
from src.unet import unet

# PARAMETERS ------------
img_rows, img_cols = 400, 400
classes = 1
batch_size = 4
epochs = 20
inputDir = 'GiveDirectlyData/data'
maskDir = 'masks/1'
# because of MaxPool layer width and height has to be divisible by 2^4
# use 320 on the rows to avoid the Google sign.

# get training images ----------------------
training_images = []
for file in os.listdir(inputDir + '/images'):
    if file.endswith(".png"):
        data = get_image(inputDir + '/images/' + file)
        training_images.append(data[:img_rows,:])


# get masks_1class -------------------------------
training_masks = []
for file in os.listdir(maskDir + '/masks'):
    if file.endswith(".png"):
        data = get_image(maskDir + '/masks/' + file)
        training_masks.append(data[:img_rows, :])


# reshape ---------------------------------
training_images = np.array(training_images)
training_masks = np.array(training_masks)[:, :, :, 0].reshape(len(training_masks), img_rows, img_cols, 1)

# normalize RGB images
training_images = training_images.astype('float32')

# generators for data augmentation -------
generator_x = ImageDataGenerator(rescale=1./255., horizontal_flip=False)
generator_y = ImageDataGenerator(horizontal_flip=False)


# flow from directory (no need to fit without normalizaiton
dgdx = generator_x.flow_from_directory(inputDir, class_mode=None,
                                       target_size=(img_rows,img_cols), batch_size=2, seed=0, save_to_dir='augm/images')
dgdy = generator_y.flow_from_directory(maskDir,  class_mode=None,
                                       target_size=(img_rows,img_cols), batch_size=2, seed=0,color_mode='grayscale',save_to_dir='augm/masks')

# synchronize two generator and combine it into one
train_generator = zip(dgdx, dgdy)

# load model
model = unet(img_rows, img_cols, classes=classes)

# kick off training
tb = TensorBoard(log_dir='logs', histogram_freq=False,  write_graph=False, write_images=False)

model.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=epochs, callbacks=[tb])

# save model
model.save('UNET_model_1class_aug.h5')