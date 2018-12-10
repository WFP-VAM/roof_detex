import os
from src.unet import unet
import numpy as np
from random import shuffle
import click
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from PIL import Image
import datetime as dt
import random
random.seed(701)


@click.command()
@click.option('--image_dir', default="spacenet/images/")
@click.option('--masks_dir', default="spacenet/masks/")
@click.option('--model_path_in', default=None)
@click.option('--model_path_out', default='model_spacenet')
def trainer(image_dir, masks_dir, model_path_in, model_path_out):
    # parameters (tbd) -------------------------
    # spacenet: 256, 4, 30
    # GD: same
    IMAGES_DIR = image_dir
    MASKS_DIR = masks_dir
    IMG_SIZE = 256
    BATCH_SIZE = 4
    EPOCHS = 30
    split = 0.8

    # list of files -----------------------------
    data_list = []
    for file in os.listdir(IMAGES_DIR):
        if file.endswith(".png"):
            data_list.append(file)

    shuffle(data_list)  # shuffle list

    training_size = int(len(data_list)*split)

    training_list = data_list[:training_size]
    validation_list = data_list[training_size:]

    print('training size: ', len(training_list), ' validation size: ', len(validation_list))

    def load_images(files, directory, flip=False, scale=False, labels=False):
        """
        given a list of files it returns them from the directory
        - files: list
        - directory: str (path to directory)
        - returns: list containing the images
        """
        images = []
        if scale:
            m = 255.
        else:
            m = 1
        for f in files:
            pg = Image.open(directory + f, 'r')
            if (pg.mode != 'RGB') & (not labels): pg = pg.convert('RGB')
            if flip:
                pg = pg.rotate(180)
                im = np.array(pg) / m
            else:
                im = np.array(pg) / m

            images.append(im)

        return images

    def data_generator(files, batch_size, flip=False):
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

                X = load_images(files[batch_start:limit], IMAGES_DIR, scale=True)
                Y = load_images(files[batch_start:limit], MASKS_DIR, labels=True)
                if flip:  # also add the rotated images
                    X.extend(load_images(files[batch_start:limit], IMAGES_DIR, flip=True, scale=True))
                    Y.extend(load_images(files[batch_start:limit], MASKS_DIR, flip=True,labels=True))

                yield (np.array(X), np.array(Y).reshape(-1,IMG_SIZE,IMG_SIZE,1))

                batch_start += batch_size
                batch_end += batch_size

    model = unet(IMG_SIZE, IMG_SIZE)

    if model_path_in:
        model.load_weights(model_path_in)
        print('INFO: model loaded ...')

    # callbacks
    tboard = TensorBoard(log_dir="logs/{}-{}".format(
        model_path_out,
        dt.datetime.now().minute), write_graph=False)
    #stopper = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=4, verbose=1, mode='auto')

    print('INFO: training ...')
    history = model.fit_generator(
        generator=data_generator(training_list, BATCH_SIZE, flip=True),
        validation_data=data_generator(validation_list, BATCH_SIZE, flip=True),
        validation_steps=40,
        steps_per_epoch=80,
        epochs=EPOCHS, verbose=1, callbacks=[tboard])

    # save training history plot
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}.png'.format(model_path_out))

    # save model
    model.save('models/{}.h5'.format(model_path_out))

    print('INFO: et voila!')


if __name__ == '__main__':
    trainer()
    # rubbish collection
    #tf.keras.backend.clear_session()
