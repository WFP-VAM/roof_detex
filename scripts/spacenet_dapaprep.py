"""
Script to prepare the images and masks from the Karthoum dataset on Spacenet for training.
https://spacenetchallenge.github.io/
"""

import rasterio
import numpy as np
import geojson
from rasterio import features
from PIL import Image
from src.utils import uint16_to_uint8

# parameters
img_dir = 'spacenet/AOI_5_Khartoum_Train/RGB-PanSharpen/'
masks_dir = 'spacenet/AOI_5_Khartoum_Train/geojson/buildings/'

for i in range(1, 1687):

    # load raster --------------------------------------------
    try:
        rs = rasterio.open(img_dir+'RGB-PanSharpen_AOI_5_Khartoum_img{}.tif'.format(i))
        img = uint16_to_uint8(np.rollaxis(rs.read(), 0, 3))
        print('raster shape: ', img.shape)

        # get mask -----------------------------------
        with open(masks_dir+'buildings_AOI_5_Khartoum_img{}.geojson'.format(i)) as f:
            gjs = geojson.load(f)

        geoms = [feature["geometry"] for feature in gjs['features']]

        try:
            msk = features.rasterize(geoms, (rs.shape[0], rs.shape[1]), transform=rs.transform)
        except ValueError:
            print('no buildings here!')
            msk = np.zeros((rs.shape[0], rs.shape[1])).astype('uint8')

        # write
        Image.fromarray(img).save('spacenet/images/{}.png'.format(i))
        Image.fromarray(msk).save('spacenet/masks/{}.png'.format(i))

    except rasterio.errors.RasterioIOError:
        print('No image: ', 'RGB-PanSharpen_AOI_5_Khartoum_img{}.tif'.format(i))


# make 2 random crops 256x256 for each image.
import random
import os
dx = dy = 256

for n in range(1, 1687):
    try:
        image = Image.open('spacenet/images/{}.png'.format(n))
        mask = Image.open('spacenet/masks/{}.png'.format(n))

        for i in [1, 2]:
            newname = '{}._{:03d}.png'.format(n, i)
            w, h = image.size
            x = random.randint(0, w-dx-1)
            y = random.randint(0, h-dy-1)
            print("Cropping {}: {},{} -> {},{}".format(str(n), x,y, x+dx, y+dy))
            img_crop = image.crop((x, y, x + dx, y + dy))
            mask_crop = mask.crop((x, y, x + dx, y + dy))
            # fill black with mean
            img_arr = np.array(img_crop)
            mask_arr = np.array(mask_crop)
            img_arr[img_arr == 0] = np.mean(img_arr)
            mask_arr[mask_arr == 0] = np.mean(mask_arr)
            Image.fromarray(img_arr).save(os.path.join('spacenet/images', newname))
            Image.fromarray(mask_arr).save(os.path.join('spacenet/masks', newname))

        os.remove('spacenet/images/{}.png'.format(n))
        os.remove('spacenet/masks/{}.png'.format(n))

    except FileNotFoundError:
        pass

# removed by hand some images I didnt like, prune the masks too
for image in os.listdir('spacenet/masks'):
    try:
        Image.open('spacenet/images/{}'.format(image))
    except FileNotFoundError:
        os.remove('spacenet/masks/{}'.format(image))