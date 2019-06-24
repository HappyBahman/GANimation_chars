from detector import Detector
from generator import Generator
from tensorflow import keras
# from tensorflow.keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)

BATCH_SIZE = 100
IMAGE_SIZE = [28, 28]

DETECTOR_KERNELS = [2, 2]
DETECTOR_FILTERS = [5, 1]
DETECTOR_STRIDES = [2, 2]
DETECTOR_DROPOUTS = [0.5, 0.5]
DETECTOR_DENSE = 5

GENERATOR_KERNELS = [2, 2]
GENERATOR_FILTERS = [5, 1]
GENERATOR_STRIDES = [2, 2]
GENERATOR_DROPOUTS = [0.5, 0.5]
GENERATOR_DENSE = 5
SAVE_INTERVAL = 500
TRAIN_STEPS = 5000


class GANimation:
    def __init__(self):
        self.det = Detector(IMAGE_SIZE, DETECTOR_FILTERS, DETECTOR_KERNELS,
                       DETECTOR_STRIDES, DETECTOR_DROPOUTS, DETECTOR_DENSE)
        self.gen = Generator(IMAGE_SIZE[0], GENERATOR_FILTERS, GENERATOR_KERNELS,
                        GENERATOR_STRIDES, GENERATOR_DROPOUTS)

        optimizer = keras.optimizers.RMSprop(lr=0.0002, decay=6e-8)
        self.dm = keras.models.Sequential()
        self.dm.add(self.det.model)
        self.dm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.adv = keras.models.Sequential()
        self.adv.add(self.gen.model)
        self.adv.add(self.det.model)
        optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.adv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'content/gdrive/My Drive/ANN_models/GAN/mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "content/gdrive/My Drive/ANN_models/GAN/mnist_%d.png" % step
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


def main():
    gan = GANimation()
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

    for i in range(TRAIN_STEPS):
        images_train = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE)]
        noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100])
        images_fake = gan.gen.model.predict(noise)
        x = np.concatenate((images_train, images_fake.reshape(images_train.shape)))
        y = np.ones([2 * BATCH_SIZE, 1])
        y[BATCH_SIZE:, :] = 0
        d_loss = gan.dm.train_on_batch(x, y)

        y = np.ones([BATCH_SIZE, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100])
        a_loss = gan.adv.train_on_batch(noise, y)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)
        if (i+1)%SAVE_INTERVAL==0:
            gan.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))


main()
