import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.number_of_islands import Graph
from tensorflow.python.keras.models import load_model

from src.utils import get_image, dice_coef, dice_coef_loss


@click.command()
@click.option('--image', default="KE2013071000-iron.png")
@click.option('--img_rows', default=400)
@click.option('--img_cols', default=400)
def main(image, img_rows, img_cols):

    # load model ------------
    model = load_model("models/UNET_model_1class_aug.h5", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

    # SINGLE IMAGE SCORE ------------
    img = np.array(get_image("GiveDirectlyData/data/images/" + image).astype('float32')) #KE2013071529-iron.png

    # crop only to relevant pixels
    #img = img[:img_rows, :img_cols]

    # score single image
    res = model.predict(img.reshape(1,img_rows, img_cols,3))
    g = Graph(img_rows, img_cols, res.reshape(img_rows, img_cols))
    g.countIslands()
    print('Number of roofs:', g.countIslands())

    # show
    plt.figure()
    plt.imshow(img)
    plt.imshow(res.reshape(img_rows, img_cols), cmap='gray', alpha=0.6)
    #plt.imshow(get_image("masks_1class/" + image).astype(np.uint8)*255, cmap='gray', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    main()
    # rubbish collection
    tf.keras.backend.clear_session()


# manual -----------------------------
model = load_model("models/UNET_model_1class_aug.h5", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
img = np.array(get_image("GiveDirectlyData/data/images/" + 'KE2013071529-iron.png').astype('float32'))
mask = np.array(get_image("masks/1/masks/" + 'KE2013071529-iron.png').astype('float32'))[:,:,0]
model.evaluate(img.reshape(1,400,400,3)/255., mask.reshape(1,400,400,1))
res = model.predict(img.reshape(1,400,400,3)/255.).reshape(400,400)
res[res > 0.4] = 255.
res[res <= 0.4] = 0

plt.imshow(res, cmap='gray', alpha=0.6)

from PIL import Image
img = np.array(Image.open("augm/masks/_0_3695.png"))
plt.imshow(img*100., cmap='gray', alpha=0.6)
img[img > 0.]