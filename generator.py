import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Generator:
    """
    build and save detector model
    """
    def __init__(self, input_size, image_size, deconv_layers, kernel_sizes, pooling_sizes, drop_outs, batch_size=100):
        image_resize = image_size
        for pool_size in pooling_sizes:
            image_resize //= pool_size

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(image_resize * image_resize * deconv_layers[0]))
        model.add(keras.layers.Reshape((image_resize, image_resize, deconv_layers[0])))

        for i, (layer, kernel, pool_size, drop_out) in enumerate(zip(deconv_layers, kernel_sizes, pooling_sizes, drop_outs)):
            model.add(keras.layers.Conv2DTranspose(layer, kernel, padding='same', strides=pool_size))
            model.add(keras.layers.BatchNormalization())
            if i != len(deconv_layers) -1 :
                model.add(keras.layers.Activation('relu'))
            else:
                model.add(keras.layers.Activation('sigmoid'))
            model.add(keras.layers.Dropout(drop_out))

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

