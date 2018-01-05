import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from utils import get_image, save_history_plot
from unet import unet

# PARAMETERS ------------
img_rows, img_cols = 256, 256


# get training images ----------------------
training_images = []
for file in os.listdir('GiveDirectlyData/data/images'):
    if file.endswith(".png"):
        data = get_image('GiveDirectlyData/data/images/' + file)

        # remove Google sign at the bottom and take central part
        training_images.append(data[:256,72:328])

# get masks -------------------------------
training_masks = []
for file in os.listdir('masks'):
    if file.endswith(".png"):
        data = get_image('masks/' + file)
        training_masks.append(data[:256,72:328])


# reshape ---------------------------------
training_images = np.array(training_images) #.reshape(len(training_images), 400, 400, 3)
training_masks = np.array(training_masks)[:, :, :, 0]#.reshape(len(training_masks), 400, 400, 1)

# instantiate model ----------------------
model = unet(img_rows, img_cols)


# normalize images
training_images = training_images.astype('float32')
mean = np.mean(training_images)  # mean for data centering
std = np.std(training_images)  # std for data normalization
training_images -= mean
training_images /= std
train_images = training_images/255.

# viz check
# plt.figure()
# plt.imshow(training_images[0])
# plt.imshow(training_masks[0], cmap='gray', alpha=0.5)
# plt.show()

tb = TensorBoard(log_dir='logs', histogram_freq=2,  write_graph=False, write_images=False)


history = model.fit(training_images.reshape(len(training_images),256,256,1),
                    training_masks.reshape(len(training_images),256,256,1),
                    batch_size=8, epochs=30, shuffle=True,
                    validation_split=0.3, callbacks=[tb])

# save training history plot
save_history_plot(history, 'training_history.png')

# save model
model.save('UNET_model.h5')
