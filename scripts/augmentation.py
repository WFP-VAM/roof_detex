import os
from PIL import Image
from src.utils import flip_pm90_in_dir

bad_images = ['KE2013071222-grass.png', 'KE2013071223-grass.png', 'KE2013071224-grass.png', 'KE2013071225-grass.png',
              'KE2013071037-iron.png', 'KE2013071038-iron.png', 'KE2013071039-iron.png', 'KE2013071040-iron.png',
              'KE2013071041-iron.png', 'KE2013071121-iron.png', 'KE2013071223-grass.png', 'KE2013071204-grass.png'
              'KE2013071224-grass.png', 'KE2013071225-grass.png', 'KE2013071205-grass.png', 'KE2013071206-grass.png',
              'KE2013071940-iron.png', 'KE2013071941-iron.png', 'KE2013071942-iron.png', 'KE2013071943-iron.png',
              'KE2013072373-grass.png']

flip_pm90_in_dir('GiveDirectlyData/data/images/', bad_images)
flip_pm90_in_dir('masks/1/masks/', bad_images)

