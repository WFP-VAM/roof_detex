from PIL import Image
import numpy as np
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import json
import pandas as pd


# data loading routines ----------------------------------
def get_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.convert('RGB')
    image = np.array(image)
    return image


# losses -------------------------------------------------
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


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


# load training metadata
def load_training_metadata():
    # Load training data
    json_data = open('GiveDirectlyData/data/train-data-2014-01-13.json')
    roof_data = [json.loads(json_line) for json_line in json_data]
    image_meta = pd.read_csv('GiveDirectlyData/data/image_metadata.csv')
    roof_train = pd.DataFrame(roof_data)
    roof_train['image_tag'] = roof_train.image.map(lambda name: name.strip().split('-')[0])
    roof_train['image'] = roof_train.image.map(lambda name: name.strip())

    # Merge Training data
    all_train = pd.merge(image_meta, roof_train, left_on='GDID', right_on='image_tag')
    return all_train