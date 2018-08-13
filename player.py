
from PIL import Image
import numpy as np
from tensorflow.python.keras.models import load_model
from src.utils import dice_coef, dice_coef_loss
import rasterio
import matplotlib.pyplot as plt

model = load_model('models/model_buildings.h5', custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
directory = 'P:/VAM/spacenet/Wau_split_rasters/'
pg = np.array(Image.open('P:/VAM/spacenet/images/' + '234.png', 'r').resize((512,512), Image.ANTIALIAS))
msk = np.array(Image.open('P:/VAM/spacenet/masks/' + '234.png', 'r').resize((512,512), Image.ANTIALIAS))
res_im = model.predict(pg.reshape(1,512,512,3)/255.)
#plt.hist(res_im.ravel())
plt.imshow(pg)
plt.imshow(msk, alpha=0.3)
plt.imshow(res_im.reshape(512,512), alpha=0.6)

im = np.rollaxis(np.array(rasterio.open(directory+'SSD_Wau_23Jun2018_R3C211.tif').read())[:, 2000:2512,2000:2512], 0, 3)
res = model.predict(im.reshape(1,512,512,3)/255.)
plt.imshow(im)
plt.imshow(res.reshape(512,512), alpha=0.6)