import numpy as np
import pandas as pd
import cv2
from shapely.wkt import loads as wkt_loads
from os import path
from PIL import Image
import gdal


def _get_image_names(base_path, imageId):
    """Get the names of the tiff files"""
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
    img_mask = np.zeros(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,1)
    return mask

# ----------------------
inDir = 'dstl/three_band'

# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv')

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


for i in df['ImageId'].unique():
    rs = gdal.Open('dstl/three_band/images/{}.tif'.format(i))
    _, rows, cols = np.array(rs.ReadAsArray()).astype('uint8').shape

    mask = generate_mask_for_image_and_class((rows, cols), i, 1, gs, df)
    cv2.imwrite("dstl/three_band/masks/{}.png".format(i), mask*255)


# check
import gdal
from PIL import Image

rs = gdal.Open('dstl/three_band/images/6060_2_3.tif')
rs = np.array(rs.ReadAsArray()).astype('uint16')
rs = np.moveaxis(rs, 0, -1).astype('uint8')
rs = Image.fromarray(rs).convert('RGB')
mk = Image.open('dstl/three_band/masks/6060_2_3.png')
mk = np.asarray(mk)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(rs, vmin=0,vmax=255)
plt.imshow(mk, cmap='gray', alpha=0.6)
plt.show()


# crop images ------------------------------------------
def crop(image, mask, img_dir, msk_dir, height ,width):
    """
    given tif (for images) and png (for masks) it crops them to height-width
    and writes them to the imag_dir and png_dir.
    """
    img = gdal.Open('dstl/three_band/images/{}.tif'.format(image))
    mk = Image.open('dstl/three_band/masks/{}.png'.format(mask))

    # dstl has channel first
    img = np.asarray(img.ReadAsArray()).swapaxes(0,2).swapaxes(0,1)

    im = Image.fromarray(img, mode='RGB')
    imgwidth, imgheight = im.size

    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            # print (i,j)
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            im_crop = im.crop(box)
            mk_crop = mk.crop(box)

            arr = np.array(mk_crop)
            # only save the cropped picture if there is at least one hut
            if arr.any() >= 1:
                print(img_dir + image + '_' + str(i) + '_' + str(j) + '.png', msk_dir + mask + '_' + str(i) + '_' + str(j) + '.png')
                im_crop.save(img_dir + image + '_' + str(i) + '_' + str(j) + '.png')
                mk_crop.save(msk_dir + mask + '_' + str(i) + '_' + str(j) + '.png')

img_size = 256
img_dir = 'dstl/images/'
masks_dir = 'dstl/masks/'

for i in df['ImageId'].unique():
    print('dstl/three_band/images/{}.tif'.format(i), 'dstl/three_band/masks/{}.png'.format(i))
    crop(i, i, img_dir, masks_dir, img_size, img_size)