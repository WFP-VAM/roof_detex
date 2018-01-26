import os
from PIL import Image

bad_images = ['KE2013071222-grass.png', 'KE2013071223-grass.png', 'KE2013071224-grass.png', 'KE2013071225-grass.png',
              'KE2013071037-iron.png', 'KE2013071038-iron.png', 'KE2013071039-iron.png', 'KE2013071040-iron.png',
              'KE2013071041-iron.png', 'KE2013071121-iron.png', 'KE2013071223-grass.png', 'KE2013071204-grass.png'
              'KE2013071224-grass.png', 'KE2013071225-grass.png', 'KE2013071205-grass.png', 'KE2013071206-grass.png',
              'KE2013071940-iron.png', 'KE2013071941-iron.png', 'KE2013071942-iron.png', 'KE2013071943-iron.png',
              'KE2013072373-grass.png']


def flip_pm90_in_dir(directory):
    """ rotates all the images in a directory by + and - 90 degrees"""
    for file in os.listdir(directory):
        if file.endswith(".png") & (str(file) not in bad_images):
            print(str(file))
            im = Image.open(directory + file, 'r')

            imp = im.rotate(90)
            imp.save(directory + file[:-4] + '_p90.png')

            imm = im.rotate(-90)
            imm.save(directory + file[:-4] + '_m90.png')


flip_pm90_in_dir('GiveDirectlyData/data/images/')
flip_pm90_in_dir('masks/1/masks/')

