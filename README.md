**WHY:** Automate the monitoring of idp and refugee camps.<br>
**WHAT:** Detecting and counting huts/houses in camps in areas of interest of the World Food Programme.<br>
**HOW:** ConvNets trained on internally and externally labelled satellite imaginery.<br>

# Data
## GMaps
1468 google maps images, 400 by 400, zoom level = 16. Between 0 and 24 huts per image, with an average between 5 and 6. Source: GiveDirectly (more details [here](http://ssg.mit.edu/~krv/pubs/AbelsonVS_kdd2014.pdf)).<br>
## VAM
very high resolution RGB images labelled from the remote monitoring team at WFP. 1 channel containing the mask (1 for roof, 0 for no roof)<br> 
## DSTL
the dataset from Kaggle DSTL's [competition](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection). Apparently shots are from Nigeria.

# How to use
### batch_training.py
trains the netowrk in batches using a generator to load the images in memory at run-time. use `python batch_training.py --help` to get avaialble parameters, including directory of the training data, what weights to use and what images' names to use.
### scripts/
handy scripts used for data prep, like generating masks, cropping and augmenting the original images. Most importantly **rms_dataprep.py**, used to prepare, crop, rotate the VHS rasters into images and masks to be used for training. 
### src/
**number_of_islands.py** program to count islands in boolean 2D matrix.<br>
**unet.py** the network architecture
**utils** all the handy stuff, inclusing cv and IO routines. Alos loss is defined here too.

 
# Approach
## Data Pipeline
### masks
the labels for each hut are made of the pixel location where the roof is, i.e. 1 pixel per roof. To reduce class imbalance between roofs and not roofs, a buffer is added around the single pixels, between 3x3 to 9x9 depending on image. So the final mask is a square that idially overlaps with the roof. One of the big improvements to do here.
### data augmentation
positive and negative rotations for (almost) every image.

## Architecture
[UNet](https://arxiv.org/abs/1505.04597)
## Evaluation Metric
Dice Coefficient.
## Infrastructure
AWS g2.2xlarge
