1468 images, 400 by 400, zoom level = 16. Between 0 and 24 huts per image, with an average between 5 and 6.

# Files Index
  ...

# Evaluation
  Dice Coefficient
  
# Approache
- UNet
  best config: 
        `Adam(lr=0.00001, decay=0.0000001), loss=dice_coef_loss
        training_images.reshape(len(training_images),256,256,1),
                    training_masks.reshape(len(training_images),256,256,1),
                    batch_size=8, epochs=30, shuffle=True,
                    validation_split=0.3`
  score: `loss: 0.5424 - dice_coef: 0.4576 - val_loss: 0.5816 - val_dice_coef: 0.4184`
