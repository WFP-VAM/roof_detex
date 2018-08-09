import os
from src.utils import get_image, save_history_plot
from src.unet import unet, unet_heavreg
import numpy as np
from random import shuffle
import click
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from time import time
from PIL import Image


@click.command()
@click.option('--image_dir', default="P:/VAM/spacenet/images/")
@click.option('--masks_dir', default="P:/VAM/spacenet/images/")
@click.option('--model_path_in', default='models/model_buildings.h5')
@click.option('--model_path_out', default=None)
def trainer(image_dir, masks_dir, model_path_in, model_path_out):
    # parameters (tbd) -------------------------
    IMAGES_DIR = image_dir
    MASKS_DIR = masks_dir
    img_rows, img_cols = 256, 256
    classes = 1
    batch_size = 4
    epochs = 10
    split = 0.8

    # list of files -----------------------------
    data_list = []
    for file in os.listdir(IMAGES_DIR):
        if file.endswith(".png"):
            data_list.append(file)

    shuffle(data_list)  # shuffle list

    training_size = int(len(data_list)*split)
    validation_size = int(len(data_list) - training_size)

    training_list = data_list[:training_size]
    validation_list = data_list[training_size:]

    print('training size: ', len(training_list), ' validation size: ', len(validation_list))

    def load_images(files, directory):
        """
        given a list of files it returns them from the directory
        - files: list
        - directory: str (path to directory)
        - returns: list containing the images
        """
        images = []
        for f in files:

            pg = Image.open(directory + f, 'r')
            pg = pg.convert('RGB')
            pg = pg.resize((img_rows, img_cols))
            im = np.array(pg)
            images.append(im)

        return images

    def image_preprocessing(img):
        """ scale images by ..."""
        img = img / np.amax(img)
        return img

    def data_generator(files, batch_size):
        """
        Python generator, returns batches of images in array format
        files: list
        batch_size: int
        returns: np arrays of (batch_size,rows,cols,channels)
        """

        size = len(files)

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

    model = unet(None, None)

    if model_path_in:
        model.load_weights(model_path_in)
        print('INFO: model loaded ...')

    # callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0, mode='auto')

    print('INFO: training ...')
    history = model.fit_generator(data_generator(training_list, batch_size),
                        validation_data=data_generator(validation_list, batch_size),
                        validation_steps=validation_size/batch_size, steps_per_epoch=training_size/batch_size,
                        epochs=epochs, callbacks=[tensorboard, stopper])

    # save training history plot
    save_history_plot(history, 'training_history.png')

    # save model
    model.save(model_path_out)

    print('INFO: et voila!')


if __name__ == '__main__':
    trainer()
    # rubbish collection
    #tf.keras.backend.clear_session()