import numpy as np
from PIL import Image
from utils import load_training_metadata
buffer = 4

# get images metadata -------------------------------------------
training_data = load_training_metadata()


# create mask, 1 roof, 0 nothing on channel 0 ---------------------------
mask = []
for ix, row in training_data.iterrows():
    tmp = np.zeros([400,400,3])
    file = row['image']
    for roof in row['roofs']:
        y = int(roof['y'])
        x = int(roof['x'])
        for i in range(-buffer, buffer+1):
            for j in range(-buffer, buffer+1):
                try:
                    tmp[y+i, x+j, 0] = 1
                except IndexError:
                    pass

    mask.append(tmp)

    # write to masks folder
    print('roofs in image ', file, ' = ', (tmp != 0).sum()/81.)
    im = Image.fromarray(tmp.astype(np.uint8))
    im.save("masks/"+file)


