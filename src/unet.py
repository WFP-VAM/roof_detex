from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from src.utils import dice_coef_loss, dice_coef


def unet(img_rows, img_cols, start_neurons=8):
    # Build U-Net modelz
    input_img = Input((img_rows, img_cols, 3), name='img')

    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_img)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.25)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.25)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.25)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.25)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs=input_img, outputs=output_layer)

    #model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.00001), metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=0.001, decay=0.00001), loss=dice_coef_loss, metrics=['binary_accuracy'])

    return model


def unet_heavreg(img_rows, img_cols, classes=1, conv_size=3):

    inputs = Input((img_rows, img_cols, 3))

    conv1 = Conv2D(32, (conv_size, conv_size), padding='same', name='conv1')(inputs)
    conv1 = BatchNormalization(axis=3, name='bn0')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (conv_size, conv_size), activation='relu', padding='same', name='conv1.1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (conv_size, conv_size), activation='relu', padding='same', name='conv2')(pool1)
    conv2 = Conv2D(64, (conv_size, conv_size), activation='relu', padding='same', name='conv2.1')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (conv_size, conv_size), activation='relu', padding='same', name='conv3')(pool2)
    conv3 = Conv2D(128, (conv_size, conv_size), activation='relu', padding='same', name='conv3.1')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (conv_size, conv_size), activation='relu', padding='same', name='conv4')(pool3)
    conv4 = Conv2D(256, (conv_size, conv_size), activation='relu', padding='same', name='conv4.1')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (conv_size, conv_size), activation='relu', padding='same', name='conv5')(pool4)
    conv5 = Conv2D(512, (conv_size, conv_size), activation='relu', padding='same', name='conv5.1')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (conv_size, conv_size), activation='relu', padding='same', name='conv6')(up6)
    conv6 = Conv2D(256, (conv_size, conv_size), activation='relu', padding='same', name='conv6.1')(conv6)
    conv6 = Dropout(0.5)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (conv_size, conv_size), activation='relu', padding='same', name='conv7')(up7)
    conv7 = Conv2D(128, (conv_size, conv_size), activation='relu', padding='same', name='conv7.1')(conv7)
    conv7 = Dropout(0.5)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (conv_size, conv_size), activation='relu', padding='same', name='conv8')(up8)
    conv8 = Conv2D(64, (conv_size, conv_size), activation='relu', padding='same', name='conv8.1')(conv8)
    conv8 = Dropout(0.5)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (conv_size, conv_size), activation='relu', padding='same', name='conv9')(up9)
    conv9 = Conv2D(32, (conv_size, conv_size), activation='relu', padding='same', name='conv9.1')(conv9)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(classes, (1, 1), activation='sigmoid', name='out')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.0001, decay=0.000001), loss=dice_coef_loss, metrics=[dice_coef])

    return model
