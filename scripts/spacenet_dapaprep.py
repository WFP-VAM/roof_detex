import rasterio
import numpy as np
import geojson
from rasterio import features
from PIL import Image
from src.utils import uint16_to_uint8

# parameters
img_dir = 'P:/VAM/spacenet/AOI_5_Khartoum_Train/RGB-PanSharpen/'
masks_dir = 'P:/VAM/spacenet/AOI_5_Khartoum_Train/geojson/buildings/'

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
        Image.fromarray(img).save('P:/VAM/spacenet/images/{}.png'.format(i))
        Image.fromarray(msk).save('P:/VAM/spacenet/masks/{}.png'.format(i))

    except rasterio.errors.RasterioIOError:
        print('No image: ', 'RGB-PanSharpen_AOI_5_Khartoum_img{}.tif'.format(i))

from src.utils import flip_pm90_in_dir
flip_pm90_in_dir('P:/VAM/spacenet/images/')
flip_pm90_in_dir('P:/VAM/spacenet/masks/')

# checks
# pg = Image.open('P:/VAM/spacenet/images/' + '58.png', 'r')
# mk = Image.open('P:/VAM/spacenet/masks/' + '58.png', 'r')
#
# import matplotlib.pyplot as plt
# plt.imshow(np.array(pg))
# plt.imshow(np.array(mk)*100, alpha=0.6)
