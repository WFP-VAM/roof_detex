import numpy as np
from PIL import Image
from utils import load_training_metadata
buffer = 4

# get images metadata -------------------------------------------
training_data = load_training_metadata()


# create mask, 2 iron, 1 thatched, 0 nothing on channel 0 --------
mask = []
# loop over images
for ix, row in training_data.iterrows():
    tmp = np.zeros([400,400,3])
    file = row['image']
    # loop over all roofs labelled
    for roof in row['roofs']:
        y = int(roof['y'])
        x = int(roof['x'])
        for i in range(-buffer, buffer+1):
            for j in range(-buffer, buffer+1):
                try:
                    if roof['type'] == 'thatched':
                        tmp[y+i, x+j, 0] = 1
                    elif roof['type'] == 'iron':
                        tmp[y+i, x+j, 0] = 2
                except IndexError:
                    pass

    mask.append(tmp)

    # write to folder
    im = Image.fromarray(tmp.astype(np.uint8))
    im.save("masks/2class/"+file)


# # Wanna see?
# image = Image.open('masks/2class/KE2013071025-iron.png', 'r')
# im = np.array(image)
# im = Image.fromarray(im*100)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(np.array(Image.open('GiveDirectlyData/data/images/KE2013071025-iron.png', 'r')))
# plt.imshow(im, alpha=0.6)
# plt.show()
