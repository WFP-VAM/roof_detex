import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.number_of_islands import Graph
from tensorflow.python.keras.models import load_model
from src.utils import dice_coef, dice_coef_loss, load_images


@click.command()
@click.option('--image', default="KE2013071000-iron.png")
@click.option('--img_rows', default=400)
@click.option('--img_cols', default=400)
def main(image, img_rows, img_cols):

    # load model ------------
    model = load_model("models/model_gd.h5", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

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
IMG_SIZE = 256
IMG = 'stack_palorinya_22Jan2018.tif_14_4.png'
model = load_model("models/model_spacenet_vam.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
img = load_images([IMG], "VAMdata/images/", scale=True)[0]
mask = load_images([IMG], "VAMdata/masks/", labels=True)[0]
model.evaluate(img.reshape(1,IMG_SIZE,IMG_SIZE,3), mask.reshape(1,IMG_SIZE,IMG_SIZE,1))
res = model.predict(img.reshape(1,IMG_SIZE,IMG_SIZE,3)).reshape(IMG_SIZE,IMG_SIZE)
np.histogram(res)


plt.imshow((img*255.).astype(int))
plt.imshow(res*255, cmap='RdGy', alpha=0.6)
