import os
from PIL import Image

def flip_90_in_dir(directory):
    for file in os.listdir(directory):
        if file.endswith(".png"):
            im = Image.open(directory + file, 'r')

            im = im.rotate(90)

            im.save(directory + file[4:] + '_90.png')



flip_90_in_dir('GiveDirectlyData/data/images/')
flip_90_in_dir('masks/2/')

