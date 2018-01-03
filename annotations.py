import json
import pandas as pd
import numpy as np
from PIL import Image

buffer = 4

# get images metadata -------------------------------------------
def load_training_metadata():
    # Load training data
    json_data = open('GiveDirectlyData/data/train-data-2014-01-13.json')
    roof_data = [json.loads(json_line) for json_line in json_data]
    image_meta = pd.read_csv('GiveDirectlyData/data/image_metadata.csv')
    roof_train = pd.DataFrame(roof_data)
    roof_train['image_tag'] = roof_train.image.map(lambda name: name.strip().split('-')[0])
    roof_train['image'] = roof_train.image.map(lambda name: name.strip())

    # Merge Training data
    all_train = pd.merge(image_meta, roof_train, left_on='GDID', right_on='image_tag')
    return all_train

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

# Viz checks
import matplotlib.pyplot as plt
plt.imshow(mask[1]) #Needs to be in row,col order
# plt.imshow(Image.open("GiveDirectlyData/data/images/" + training_data.loc[68,'image']))
