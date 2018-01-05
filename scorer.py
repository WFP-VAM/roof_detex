import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from utils import get_image, dice_coef, dice_coef_loss
from scipy import ndimage

# PARAMETERS ------------
img_rows, img_cols = 256, 256

# load image ------------
img = np.array(get_image("GiveDirectlyData/data/images/" + 'KE2013071296-iron.png').astype('float32'))

# load model ------------
model = load_model("models/UNET_model.h5", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

# crop only to relevant pixels
img = img[:img_rows, 72:328]

# score single image -----------
res = model.predict(img.reshape(1,img_rows, img_cols,1))
blobs, number_of_blobs = ndimage.label(res.reshape(img_rows, img_cols))
print('Number of blobls:', number_of_blobs)

# show
plt.figure()
plt.imshow(img.reshape(img_rows, img_cols))
plt.imshow(res.reshape(img_rows, img_cols), cmap='gray', alpha=0.8)
#plt.imshow(get_image("masks/" + 'KE2013071296-iron.png').astype(np.uint8)*255, cmap='gray', alpha=0.8)
plt.show()


# score all images -----------
from annotations import load_training_metadata
import pandas as pd

metadata = load_training_metadata()

results = pd.DataFrame(columns=['img', 'roofs', 'blobs'])
for ix, row in metadata.iterrows():

    img = np.array(get_image("GiveDirectlyData/data/images/" + row['image']).astype('float32'))
    img = img[:img_rows, 72:328]
    res = model.predict(img.reshape(1, img_rows, img_cols, 1))
    blobs, number_of_blobs = ndimage.label(res.reshape(img_rows, img_cols))
    print('Image: ', row['GDID'], ' -total Roofs: ', row['total'], ' -Number of blobls:', number_of_blobs)

    results.append({'img': row['image'], 'roofs': row['total'], 'blobs': number_of_blobs}, ignore_index=True)