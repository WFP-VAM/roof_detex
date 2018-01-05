import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from utils import get_image, dice_coef, dice_coef_loss
from scipy import ndimage, signal

# PARAMETERS ------------
img_rows, img_cols = 320, 400

# load image ------------
img = np.array(get_image("GiveDirectlyData/data/images/" + 'KE2013071406-grass.png').astype('float32')) #KE2013071529-iron.png

# load model ------------
model = load_model("models/UNET_model.h5", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

# crop only to relevant pixels
img = img[:img_rows, :img_cols]

# score single image -----------
res = model.predict(img.reshape(1,img_rows, img_cols,1))
#im = signal.convolve2d(res.reshape(img_rows, img_cols), np.ones((30,30)), mode="same")
blobs, number_of_blobs = ndimage.label(res)
print('Number of blobls:', number_of_blobs)


# show
plt.figure()
plt.imshow(res.reshape(img_rows, img_cols), alpha=0.6)
#plt.imshow(blobs, cmap='gray', alpha=0.6)
#plt.imshow(get_image("masks/" + 'KE2013071296-iron.png').astype(np.uint8)*255, cmap='gray', alpha=0.8)
plt.show()


# score 20 images -----------
from annotations import load_training_metadata
import pandas as pd

metadata = load_training_metadata()

results = pd.DataFrame(columns=['img', 'roofs', 'blobs'])
count = 0
for ix, row in metadata.iterrows():

    img = np.array(get_image("GiveDirectlyData/data/images/" + row['image']).astype('float32'))
    img = img[:img_rows, :img_cols]
    res = model.predict(img.reshape(1, img_rows, img_cols, 1))
    #im = ndimage.filters.convolve(res.reshape(img_rows, img_cols), np.ones((30, 30)))
    blobs, number_of_blobs = ndimage.label(res)

    print('Image: ', row['GDID'], ' -total Roofs: ', row['total'], ' -Number of blobls:', number_of_blobs)

    results.loc[ix,'img'] = row['image']
    results.loc[ix, 'roofs'] = row['total']
    results.loc[ix, 'blobs'] = number_of_blobs

    count = count + 1

    if count == 10: break

results.to_clipboard()