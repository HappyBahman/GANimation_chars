import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Detector:
    """
    build and save detector model
    """
    def __init__(self, image_size, conv_layers, kernel_sizes, pooling_sizes, drop_outs, dense_layer):
        model = keras.models.Sequential()
        first_itr = True
        for layer, kernel, pool_size, drop_out in zip(conv_layers, kernel_sizes, pooling_sizes, drop_outs):
            if first_itr:
                model.add(keras.layers.Conv2D(layer, kernel, strides=pool_size, input_shape=(image_size[0], image_size[1], 1), padding='same'))
                model.add(keras.layers.LeakyReLU(alpha=0.2))
                model.add(keras.layers.Dropout(drop_out))
                first_itr = False
            else:
                model.add(keras.layers.Conv2D(layer, kernel, strides=pool_size, padding='same'))
                model.add(keras.layers.LeakyReLU(alpha=0.2))
                model.add(keras.layers.Dropout(drop_out))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation('sigmoid'))
        model.summary()
        self.model = model
