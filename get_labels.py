import json
import pandas as pd
import shutil
import os

# CREATE LABELS ---------------------------------------
meta = []
with open('data/train-data-2014-01-13.json') as file:
    for l in file:
        meta.append(json.loads(l))

df = pd.DataFrame(columns=['image', 'number_iron', 'number_thatched', 'total'], index=range(len(meta)))
cnt=0
for line in meta:
    print(line['image'].lstrip('\n'), line['number_iron'], line['number_thatched'], line['total'])
    df.loc[cnt] = pd.Series({
        'image': line['image'].replace('\n', '').replace(' ', ''),
        'number_iron': line['number_iron'],
        'number_thatched': line['number_thatched'],
        'total': line['total']})
    cnt = cnt+1

df.to_csv('labels.csv', index=False)

# COPY TO CUSTOM FOLDERS -------------------------------
cnt=0
for ix, row in df.iterrows():
    org_dir = 'data/images/'
    if cnt <= 1100: dest_dir = 'data/train_roofs/'+str(row['total'])+'/'
    if cnt > 1100: dest_dir = 'data/validate_roofs/'+str(row['total'])+'/'

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print('writing  ', dest_dir + str(row['image']))
    shutil.copy2(org_dir + row['image'], dest_dir + str(row['image']))

    cnt = cnt+1

