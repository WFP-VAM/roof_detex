import numpy as np
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
from scipy.misc import bytescale
from skimage.exposure import rescale_intensity
from PIL import Image


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


# losses -------------------------------------------------
def dice_coef(y_true, y_pred):
    """ref: https://arxiv.org/pdf/1606.04797v1.pdf"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def jaccard_distance(y_true, y_pred, smooth=100):
    """https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# save training history --------------------------------
def save_history_plot(history, path):
    plt.switch_backend('agg')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)


def uint16_to_uint8(mat):
    """ satellite images have often higher bit-depths! utility to convert them to uint8 for TF"""
    x, y, bands = mat.shape
    return_stack = np.zeros([x, y, bands], dtype=np.uint8)
    for b in range(bands):
        p1_pix, p2_pix = np.percentile(mat[:, :, b], (2, 98))
        return_stack[:, :, b] = bytescale(rescale_intensity(mat[:, :, b], out_range=(p1_pix, p2_pix)))

    return return_stack
