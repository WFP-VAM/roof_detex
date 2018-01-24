import os

import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from utils import get_image, save_history_plot

from src.unet import unet

# PARAMETERS ------------
img_rows, img_cols = 400, 400
# because of MaxPool layer width and height has to be divisible by 2^4
# use 320 on the rows to avoid the Google sign.

# get training images ----------------------
training_images = []
for file in os.listdir('GiveDirectlyData/data/images'):
    if file.endswith(".png"):
        data = get_image('GiveDirectlyData/data/images/' + file)
        training_images.append(data[:img_rows,:])

# get masks_1class -------------------------------
training_masks = []
for file in os.listdir('masks/1/masks'):
    if file.endswith(".png"):
        data = get_image('masks/1/masks/' + file)
        training_masks.append(data[:img_rows,:])


# reshape ---------------------------------
training_images = np.array(training_images) #.reshape(len(training_images), 400, 400, 3)
training_masks = np.array(training_masks)[:, :, :, 0].reshape(len(training_masks), 400, 400, 1)

# instantiate model ----------------------
model = unet(img_rows, img_cols)  # with dropout
#model.load_weights('models/unet_10_jk0.7878') # pre trained weights from this guy https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras

# normalize RGB images
training_images = training_images.astype('float32')
mean = np.mean(training_images)  # mean for data centering
std = np.std(training_images)  # std for data normalization
training_images -= mean
training_images /= std
b=training_images[:,:,:,0]
g=training_images[:,:,:,1]
r=training_images[:,:,:,2]
sum=b+g+r
training_images[:,:,:,0]=b/sum*255.0
training_images[:,:,:,1]=g/sum*255.0
training_images[:,:,:,2]=r/sum*255.0


# viz check
# plt.figure()
# plt.imshow(training_images[508])
# plt.imshow(training_masks[508], cmap='gray', alpha=0.5)
# plt.show()

tb = TensorBoard(log_dir='logs', histogram_freq=False,  write_graph=False, write_images=False)

history = model.fit(training_images,
                    training_masks,
                    batch_size=4, epochs=10, shuffle=True,
                    validation_split=0.3, callbacks=[tb])

# save training history plot
save_history_plot(history, 'training_history.png')

# save model
model.save('UNET_model_2class.h5')
