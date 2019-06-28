import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

LATENT_SPACE_SIZE = 100


class Generator:
    """
    build and save detector model
    """
    def __init__(self, image_size, deconv_layers, kernel_sizes, pooling_sizes, drop_outs):
        image_resize = image_size
        for pool_size in pooling_sizes:
            image_resize //= pool_size

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(image_resize * image_resize * deconv_layers[0] * 2, input_dim=LATENT_SPACE_SIZE))
        model.add(keras.layers.BatchNormalization(momentum=0.9))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Reshape((image_resize, image_resize, deconv_layers[0] * 2)))
        model.add(keras.layers.Dropout(0.5))

        for i, (layer, kernel, pool_size, drop_out) in enumerate(zip(deconv_layers, kernel_sizes, pooling_sizes, drop_outs)):
#           not toroughly written:
            if pool_size > 1:
              model.add(keras.layers.UpSampling2D())
            model.add(keras.layers.Conv2DTranspose(layer, kernel, padding='same'))
#             model.add(keras.layers.Dropout(drop_out))
            if i != len(deconv_layers) -1 :
                model.add(keras.layers.BatchNormalization(momentum=0.9))
                model.add(keras.layers.Activation('relu'))
            else:
                model.add(keras.layers.Activation('sigmoid'))
        model.summary()
        self.model = model
