import numpy as np
from PIL import Image
import pandas as pd
import json
import os
import random

BUFFER = 6
IMG_SIZE = 400

# get images metadata -------------------------------------------
json_data = open('GiveDirectlyData/train-data-2014-01-13.json')
roof_data = [json.loads(json_line) for json_line in json_data]
image_meta = pd.read_csv('GiveDirectlyData/image_metadata.csv')
roof_train = pd.DataFrame(roof_data)
roof_train['image_tag'] = roof_train.image.map(lambda name: name.strip().split('-')[0])
roof_train['image'] = roof_train.image.map(lambda name: name.strip())

# Merge Training data
training_data = pd.merge(image_meta, roof_train, left_on='GDID', right_on='image_tag')

# create mask --------
mask = []
# loop over images
for ix, row in training_data.iterrows():
    tmp = np.zeros([IMG_SIZE,IMG_SIZE])
    file = row['image']
    # loop over all roofs labelled
    for roof in row['roofs']:
        y = int(roof['y'])
        x = int(roof['x'])
        for i in range(-BUFFER, BUFFER+1):
            for j in range(-BUFFER, BUFFER+1):
                try:
                    if roof['type'] == 'thatched':
                        tmp[y+i, x+j] = 1
                    elif roof['type'] == 'iron':
                        tmp[y+i, x+j] = 1
                except IndexError:
                    pass

    mask.append(tmp)

    # write to folder
    im = Image.fromarray(tmp.astype(np.uint8), mode='L')
    im.save("GiveDirectlyData/data/masks/"+file)


# # Wanna see?
image = Image.open('GiveDirectlyData/data/masks/KE2013071000-iron.png', 'r')
im = np.array(image)*100
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(np.array(Image.open('GiveDirectlyData/data/images/KE2013071000-iron.png', 'r')))
plt.imshow(Image.fromarray(im), alpha=0.6)
plt.show()

# lets go from 400 -> 256 taking 2 random crops
dx = dy = 256
for file in os.listdir('GiveDirectlyData/data/images/'):

    image = Image.open('GiveDirectlyData/data/images/{}'.format(file))
    mask = Image.open('GiveDirectlyData/data/masks/{}'.format(file))

    for i in [1, 2]:
        newname = '{:03d}.{}'.format(i, file)
        w, h = image.size
        x = random.randint(0, w-dx-1)
        y = random.randint(0, h-dy-1)
        print("Cropping {}: {},{} -> {},{}".format(str(file), x,y, x+dx, y+dy))
        img_crop = image.crop((x, y, x + dx, y + dy)).save(os.path.join('GiveDirectlyData/data/images', newname))
        mask_crop = mask.crop((x, y, x + dx, y + dy)).save(os.path.join('GiveDirectlyData/data/masks', newname))

    os.remove('GiveDirectlyData/data/images/{}'.format(file))
    os.remove('GiveDirectlyData/data/masks/{}'.format(file))