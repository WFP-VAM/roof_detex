import os
from src.utils import get_image, load_training_metadata
from src.unet import unet
import numpy as np

# globals
IMAGES_DIR = 'GiveDirectlyData/data/images/'
MASKS_DIR = 'masks/1/masks/'
img_rows, img_cols = 400, 400
classes = 1
batch_size = 4
epochs = 10
split = 0.8

# list of files
data_list = load_training_metadata()['image']

training_size = int(len(data_list)*split)
validation_size = int(len(data_list) - training_size)

training_list = data_list[:training_size]
validation_list = data_list[training_size:]

print('training size: ', len(training_list), ' validation size: ', len(validation_list))


def load_images(files, directory):
    """ given a list of files it returns them in batch"""
    images = []
    for f in files:
        im = get_image(directory + f)
        images.append(im)
    return images

def image_preprocessing(img):
    """ scale images by ..."""
    img = img / np.amax(img)
    return img

def data_generator(files, batch_size, size):
    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < size:
            limit = min(batch_end, size)

            X = np.array(load_images(files[batch_start:limit], IMAGES_DIR))
            Y = np.array(load_images(files[batch_start:limit], MASKS_DIR))[:, :, :, 0]

            X = image_preprocessing(X)

            yield (X, Y.reshape(limit-batch_start,img_rows,img_cols,classes))

            batch_start += batch_size
            batch_end += batch_size


model = unet(None, None, classes=classes)

model.fit_generator(data_generator(training_list,
                                 batch_size,
                                 training_size),
                    steps_per_epoch = len(training_list)/batch_size,
                    validation_data=data_generator(validation_list,
                                                   batch_size,
                                                   validation_size),
                    validation_steps=len(validation_list)/batch_size,
                    epochs=epochs)

# save model
model.save('UNET_model_batch.h5')
