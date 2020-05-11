"""Keras based implementation of Autoencoding beyond pixels using a learned similarity metric

Updated 05/11/2020 for Python 3 and Tensorflow 2 by Eric Bassett

References:

Autoencoding beyond pixels using a learned similarity metric
by: Anders Boesen Lindbo Larsen, Soren Kaae Sonderby, Hugo Larochelle, Ole Winther
https://arxiv.org/abs/1512.09300

Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Alec Radford, Luke Metz, Soumith Chintala
https://arxiv.org/abs/1511.06434
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Convolution2D, LeakyReLU, Flatten, BatchNormalization as BN

learning_rate = .0002
beta1 = .5
z_dim = 512

def generator(batch_size, gf_dim, ch, rows, cols):

    model = tf.keras.Sequential()

    model.add(Dense(gf_dim*8*rows[0]*cols[0], input_shape=(batch_size, z_dim), name="g_h0_lin", init=normal))
    model.add(Reshape((rows[0], cols[0], gf_dim*8)))
    model.add(BN(mode=2, axis=3, name="g_bn0", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim*4, 5, 5, subsample=(2, 2), name="g_h1", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn1", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim*2, 5, 5, subsample=(2, 2), name="g_h2", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn2", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim, 5, 5, subsample=(2, 2), name="g_h3", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn3", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(ch, 5, 5, subsample=(2, 2), name="g_h4", init=normal))
    model.add(Activation("tanh"))

    return model