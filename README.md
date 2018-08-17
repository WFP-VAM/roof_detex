! WORK IN PROGRESS !

Codebase for training models to detecting structures from very high resolution satellite images (~0.4m). 

## preprocessing
scripts to preprocess the data into images and lables are in `scripts/`. Each set of images and labels used for training come in different format so it requires custom preprocessing. 4 scripts are developed to handle 4 different data streams:
- [giveDirectly_dataprep.py](https://github.com/WFP-VAM/roof_detex/blob/master/scripts/dstl_dataprep.py), to prepare images and labels provided by the people behind this [paper](http://ssg.mit.edu/~krv/pubs/AbelsonVS_kdd2014.pdf). 1468 google maps images, 400 by 400, zoom level = 16. Between 0 and 24 huts per image, with an average between 5 and 6.
- [rms_dataprep.py](https://github.com/WFP-VAM/roof_detex/blob/master/scripts/rms_dataprep.py) to prepare proprietary satellite and labels genrated from VAM's geospatial team. These are very high resolution RGB images labelled from the remote monitoring team at WFP. 1 channel containing the mask (1 for roof, 0 for no roof)
- [dstl_dataprep.py](https://github.com/WFP-VAM/roof_detex/blob/master/scripts/dstl_dataprep.py) to prepare images and labels from the [DSTL Kaggle competition](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection). 
- [spacenet_dataprep.py](https://github.com/WFP-VAM/roof_detex/blob/master/scripts/spacenet_dapaprep.py) to prepare images and labels fof the RGB Khartoum image set from [Spacenet](https://spacenetchallenge.github.io).

The labels for each dataset change slightly: for VAM data huts are identified of one pixel for where the roof is, i.e. 1 pixel per roof. To reduce class imbalance between roofs and not roofs, a buffer is added around the single pixels, between 3x3 to 9x9 depending on image. So the final mask is a square that idially overlaps with the roof. One of the big improvements to do here.

## Training

`batch_training.py` trains the netowrk in batches using a generator to load the images in memory at run-time. use `python batch_training.py --help` to get avaialble parameters, including directory of the training data, what weights to use and what images' names to use.

## src/
**number_of_islands.py** class to count islands in boolean 2D matrix.<br>
**unet.py** the network architecture, [UNet](https://arxiv.org/abs/1505.04597).
**utils** all the handy stuff, inclusing cv and IO routines. Loss funciton, Dice Coefficient, is defined here too.

## Infrastructure
AWS g2.2xlarge
