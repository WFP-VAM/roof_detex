import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from utils import get_image, dice_coef, dice_coef_loss

# PARAMETERS ------------
img_rows, img_cols = 320, 400

# load image ------------
img = np.array(get_image("GiveDirectlyData/data/images/" + 'KE2013071019-iron.png').astype('float32'))

# load model ------------
model = load_model("models/UNET_model.h5", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

# take only 256 by 256
img = img[:img_rows, :]

# score image -----------
res = model.predict(img.reshape(1,img_rows, img_cols,1))

# show
plt.figure()
plt.imshow(img.reshape(img_rows, img_cols))
plt.imshow(res.reshape(img_rows, img_cols), cmap='gray', alpha=0.5)
plt.show()

