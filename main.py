from tensorflow import keras
# from tensorflow.keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train[np.logical_or(y_train==9, y_train==5)]
# x_train = x_train[y_train==5]

# x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255

BATCH_SIZE = 100
IMAGE_SIZE = [28, 28]

# DETECTOR_KERNELS = [64, 8, 1]
# DETECTOR_FILTERS = [10, 5, 1]
# DETECTOR_STRIDES = [2, 1, 1]
# DETECTOR_DROPOUTS = [0.5, 0.5, 0.5]
# DETECTOR_DENSE = 10
DETECTOR_KERNELS = [5, 5, 5, 5]
DETECTOR_FILTERS = [64, 128, 256, 512]
DETECTOR_STRIDES = [2, 2, 2, 1]
DETECTOR_DROPOUTS = [0.5, 0.5, 0.5, 0.5]
DETECTOR_DENSE = 10

GENERATOR_KERNELS = [5, 5, 5, 5]
GENERATOR_FILTERS = [128, 64, 32, 1]
GENERATOR_STRIDES = [2, 2, 1, 1]
GENERATOR_DROPOUTS = [0.5, 0.5, 0.5, 0.5]
GENERATOR_DENSE = 5
# GENERATOR_KERNELS = DETECTOR_KERNELS
# GENERATOR_FILTERS = [128, 64, 32, 16, 1]
# GENERATOR_STRIDES = DETECTOR_STRIDES
# GENERATOR_DROPOUTS = DETECTOR_DROPOUTS
# GENERATOR_DENSE = 5

TRAIN_STEPS = 10000

SAVE_INTERVAL = 100
MODEL_SAVE_INTERVAL = 1000
SHOW_ACC_INTERVAL = 10

LATENT_SPACE_SIZE = 100


class GANimation:
    def __init__(self):
        self.det = Detector(IMAGE_SIZE, DETECTOR_FILTERS, DETECTOR_KERNELS,
                            DETECTOR_STRIDES, DETECTOR_DROPOUTS, DETECTOR_DENSE)
        self.gen = Generator(IMAGE_SIZE[0], GENERATOR_FILTERS, GENERATOR_KERNELS,
                             GENERATOR_STRIDES, GENERATOR_DROPOUTS)
        optimizer = keras.optimizers.RMSprop(lr=0.0002, decay=6e-8)
        #         self.det.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        #         self.dm = self.det.model
        self.dm = keras.models.Sequential()
        self.dm.add(self.det.model)
        self.dm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.adv = keras.models.Sequential()
        self.adv.add(self.gen.model)
        self.adv.add(self.det.model)
        optimizer = keras.optimizers.RMSprop(lr=0.0001, decay=3e-8)
        self.adv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = '/content/drive/My Drive/ANN_models/GAN/mnist.png'
        if fake:
            if noise is None:
                noise = gen_noise(16)
            else:
                filename = "/content/drive/My Drive/ANN_models/GAN/mnist_unl_%d.png" % step
            images = self.gen.model.predict(noise)
        else:
            i = np.random.randint(0, x_train.shape[0], samples)
            images = x_train[i]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


def gen_noise(size):
    #   return np.random.uniform(-1, 1, size=[size, LATENT_SPACE_SIZE])
    return np.random.normal(0, 1, size=[size, LATENT_SPACE_SIZE])


def main():
    gan = GANimation()
    noise_input = gen_noise(16)
    for i in range(TRAIN_STEPS):
        gen2det_ratio = 1
        #         gen2det_ratio = int(np.log(np.log(i + 2) + 2) * 5) + 1
        #         gen2det_ratio = int(np.log(i + 2)) + 1
        #         gen2det_ratio = int(i/np.log(i + 2)) + 1
        images_train = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE)]
        #         images_train = images_train.reshape([BATCH_SIZE, 28, 28, 1])
        noise = gen_noise(BATCH_SIZE)
        images_fake = gan.gen.model.predict(noise, batch_size=100)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2 * BATCH_SIZE, 1])
        y[BATCH_SIZE:, :] = 0

        #         idx = np.arange(BATCH_SIZE)
        #         np.random.shuffle(idx)
        #         x = x[idx]
        #         y = y[idx]

        gan.dm.trainable = True
        gan.det.trainable = True
        d_loss = gan.dm.train_on_batch(x, y)
        gan.dm.trainable = False
        gan.det.trainable = False

        for j in range(gen2det_ratio):
            y = np.ones([BATCH_SIZE, 1])
            noise = gen_noise(BATCH_SIZE)
        a_loss = gan.adv.train_on_batch(noise, y)
        if i % SHOW_ACC_INTERVAL == 0:
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
        if i % SAVE_INTERVAL == 0:
            gan.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))
        if (i + 1) % MODEL_SAVE_INTERVAL == 0:
            gan.gen.model.save('/content/drive/My Drive/ANN_models/GAN/models/gen1_' + str(i))
            gan.adv.save('/content/drive/My Drive/ANN_models/GAN/models/adv1_' + str(i))
            gan.dm.save('/content/drive/My Drive/ANN_models/GAN/models/dm1_' + str(i))


main()
