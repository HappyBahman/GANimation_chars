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
                model.add(keras.layers.Conv2D(layer, kernel, input_shape=(image_size[0], image_size[1], 1), padding='same'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Activation('relu'))
                model.add(keras.layers.MaxPool2D(pool_size=pool_size))
                model.add(keras.layers.Dropout(drop_out))
            else:
                model.add(keras.layers.Conv2D(layer, kernel, padding='same'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Activation('relu'))
                model.add(keras.layers.MaxPool2D(pool_size=pool_size))
                model.add(keras.layers.Dropout(drop_out))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(dense_layer))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation('sigmoid'))
        self.model = model

    def train(self, x_train, y_train, batch_size, epochs, x_test, y_test):
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True)


