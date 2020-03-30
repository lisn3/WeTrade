import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.datasets import cifar10

def load_data(name):
    if name=='mnist':
        img_size = 28
        print('\nLoading MNIST')

        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = np.reshape(X_train, [-1, img_size * img_size])
        X_train = X_train.astype(np.float32) / 255
        X_test = np.reshape(X_test, [-1, img_size * img_size])
        X_test = X_test.astype(np.float32) / 255

        to_categorical = tf.keras.utils.to_categorical
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    else:
        if name=='cifar10':
            nb_classes=10
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')

            # Convert class vectors to binary class matrices.
            y_train = keras.utils.to_categorical(y_train, nb_classes)
            y_test = keras.utils.to_categorical(y_test, nb_classes)

            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train /= 255
            X_test /= 255

    return X_train, X_test, y_train, y_test