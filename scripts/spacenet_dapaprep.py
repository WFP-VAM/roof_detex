import rasterio
import numpy as np
import geojson
from rasterio import features
from PIL import Image

# parameters
img_dir = 'P:/VAM/spacenet/AOI_5_Khartoum_Train/RGB-PanSharpen/'
masks_dir = 'P:/VAM/spacenet/AOI_5_Khartoum_Train/geojson/buildings/'

for i in range(1, 1687):

    # load raster --------------------------------------------
    try:
        rs = rasterio.open(img_dir+'RGB-PanSharpen_AOI_5_Khartoum_img{}.tif'.format(i))
        img = np.array(rs.read()).astype('uint8')
        print('raster shape: ', img.shape)
        img = np.moveaxis(img[:3,:,:], 0, 2)

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

