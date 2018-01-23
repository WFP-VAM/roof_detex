import numpy as np
import os
from utils import get_image, save_history_plot
from unet import unet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard

# PARAMETERS ------------
img_rows, img_cols = 400, 400
classes = 1
batch_size = 4
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

# normalize RGB images
training_images = training_images.astype('float32')
mean = np.mean(training_images)  # mean for data centering
std = np.std(training_images)  # std for data normalization
training_images -= mean
training_images /= std
b=training_images[:,:,:,0]
g=training_images[:,:,:,1]
r=training_images[:,:,:,2]
sum=b+g+r
training_images[:,:,:,0]=b/sum*255.0
training_images[:,:,:,1]=g/sum*255.0
training_images[:,:,:,2]=r/sum*255.0



# generators for data augmentation -------
seed = 1
generator_x = ImageDataGenerator(
    vertical_flip=True)

generator_x.fit(training_images, augment=True, seed=seed)

# load model
model = unet(img_rows, img_cols, classes=classes)

# kick off training
tb = TensorBoard(log_dir='logs', histogram_freq=False,  write_graph=False, write_images=False)

model.fit_generator(generator_x.flow(training_images, training_masks, batch_size=batch_size), 
                    steps_per_epoch=int(len(training_images)/batch_size),
                    epochs=epochs, callbacks=[tb])

# save model
model.save('UNET_model_1class_aug.h5')
