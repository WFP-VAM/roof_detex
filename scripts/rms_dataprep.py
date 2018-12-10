from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# parameters
img_size = 256
img_dir = 'VAMdata/images/'
masks_dir = 'VAMdata/masks/'
raster_image = 'stack_palorinya_22Jan2018.tif'

# load raster --------------------------------------------
rs = gdal.Open('VAMdata/'+raster_image)
rs = np.array(rs.ReadAsArray()).astype('uint8')
print('raster shape: ', rs.shape)

# get image and mask -----------------------------------
img = np.moveaxis(rs[:3,:,:], 0, 2)
mask = rs[3,:,:]

# add buffer to mask ----------------------------------
buffer = 3
tmp = np.zeros((mask.shape))
for y in range(mask.shape[0]):
    for x in range(mask.shape[1]):
        if mask[y,x] == 1.0:
            for i in range(-buffer, buffer + 1):
                for j in range(-buffer, buffer + 1):
                    try:
                        tmp[y + i, x + j] = 1.
                    except IndexError:
                         pass
mask = tmp

# mask[mask > 0.].sum()
plt.figure()
plt.imshow(img)
plt.imshow(mask, cmap='gray', alpha=0.6)
plt.show()


# crop image ------------------------------------------
def crop(image, mask, img_dir, msk_dir, height ,width):
    im = Image.fromarray(image.astype('uint8'), mode='RGB')
    mk = Image.fromarray(mask.astype('uint8'))
    imgwidth, imgheight = im.size

    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            # print (i,j)
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            im_crop = im.crop(box)
            mk_crop = mk.crop(box)

            arr = np.array(mk_crop)
            # only save the cropped picture if there is at least one hut
            if arr.any() == 1:
                im_crop.save(img_dir + raster_image + '_' + str(i) + '_' + str(j) + '.png')
                mk_crop.save(msk_dir + raster_image + '_' + str(i) + '_' + str(j) + '.png')

crop(img, mask, img_dir, masks_dir, img_size, img_size)

# checks
im = np.asarray(Image.open('VAMdata/images/stack_palorinya_22Jan2018.tif_33_21.png'))
mk = np.asarray(Image.open('VAMdata/masks/stack_palorinya_22Jan2018.tif_33_21.png'))
plt.imshow(im)
plt.imshow(mk, cmap='gray', alpha=0.2)

