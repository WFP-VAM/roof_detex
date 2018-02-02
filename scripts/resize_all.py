import os
import numpy as np
from PIL import Image

# PARAMETERS ------------
img_rows, img_cols = 256, 256


def resizer(directory, rows, cols):
    for file in os.listdir(directory):
        if file.endswith(".png"):
            print(str(file))

            im = Image.open(directory + file, 'r')
            im = im.resize((rows, cols), Image.ANTIALIAS)
            im.save(directory + file)


resizer('GiveDirectlyData/data/images/', img_rows, img_cols)
resizer('GiveDirectlyData/masks/1/masks/', img_rows, img_cols)
