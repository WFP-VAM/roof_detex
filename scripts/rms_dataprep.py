from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.utils import flip_pm90_in_dir

# load raster --------------------------------------------
rs = gdal.Open('VAMdata/MLstack_ari_koukouri_c.tif')
rs = np.array(rs.ReadAsArray())

# get image and mask -----------------------------------
img = np.moveaxis(rs[:3,:,:], 0, 2) # TODO: I think order of colours is wrong? plt.imshow(img)
mask = rs[3,:,:]


# add buffer to mask ----------------------------------
buffer = 4
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
# plt.figure()
# plt.imshow(img)
# plt.imshow(mask, cmap='gray', alpha=0.6)
# plt.show()


# crop image ------------------------------------------
def crop_center(img, cropx, cropy, centrex, centrey, masks=False):
    startx = centrex -(cropx//2)
    starty = centrey -(cropy//2)
    # print(startx, starty)
    if masks:
        return img[starty:starty + cropy, startx:startx + cropx]
    else:
        return img[starty:starty + cropy, startx:startx + cropx, :]


y, x = img.shape[0], img.shape[1]
for offx, offy in zip([-500, -400, -300,-200, -100, 0, 100, 200, 300, 400, 500],[-400, -300, -200, -100, 0, 100, 200, 300, 400]):

    im_crop = crop_center(img, 256, 256, centrex=x // 2 + offx, centrey=y // 2 + offy)
    mk_crop = crop_center(mask, 256, 256, centrex=x // 2 + offx, centrey=y // 2 + offy, masks=True)

    im = Image.fromarray(im_crop.astype(np.uint8))
    mk = Image.fromarray(mk_crop.astype(np.uint8))

    im.save("VAMdata/images/MLstack_ari_koukouri_c_" + str(offx) + str(offy) + '.png')
    mk.save("VAMdata/masks/MLstack_ari_koukouri_c_" + str(offx) + str(offy) + '.png')


# augmentation ---------------------------------------
# plt.imshow(np.asarray(Image.open('VAMdata/images/MLstack_ari_koukouri_c_-1000.png')))
# plt.imshow(np.asarray(Image.open('VAMdata/masks/MLstack_ari_koukouri_c_-1000.png')), alpha=0.6)
flip_pm90_in_dir('VAMdata/images/')
flip_pm90_in_dir('VAMdata/masks/')
