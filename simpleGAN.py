#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:17:37 2020

@author: oscar
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
tf.keras.backend.clear_session()


"""


Discriminator Data Generation

"""
from data_utils import gaussian_samples as generate_sp_samples
"""

Discriminator Methods

"""

from tqdm import tqdm


def discriminator(n_inputs=2):
    model = Sequential()
    # model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    # model.add(Dense(1, activation='sigmoid'))

    model.add(Dense(512, input_dim=n_inputs, activation='relu', kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


"""

Generator Methods

"""


def z_gen(noise_dim=5, n=100):
    x_input = np.random.randn(noise_dim * n)
    x_input = x_input.reshape(n, noise_dim)
    return x_input


def generator(noise_dim, n_outputs=2):
    model = Sequential()
    # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=noise_dim))
    # model.add(Dense(n_outputs, activation='linear'))

    model.add(Dense(256, kernel_initializer='he_uniform', input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_outputs, activation='linear'))
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def generate_from_noise(generator, noise_dim, n):
    x_input = z_gen(noise_dim, n)
    X = generator.predict(x_input)
    y = np.zeros((n, 1))
    return X, y


"""

GAN Methods

"""


def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


"""

Evaluation

"""


# evaluate the discriminator and oit_not real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_sp_samples(n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_from_noise(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    plt.plot(x_real[:, 0], x_real[:, 1], color='red')
    # plt.plot(x_real, label='real')
    # plt.plot(x_fake, label='generated')
    # plt.legend(['real', 'generated'])
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    plt.show()


"""

GAN Training

"""


# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=20000, n_batch=8000, n_eval=200):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    d_loss1, d_acc, g_loss1 = [], [], []

    # manually enumerate epochs
    for i in tqdm(range(n_epochs)):
        # prepare real samples
        x_real, y_real = generate_sp_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = generate_from_noise(g_model, latent_dim, half_batch)
        # update discriminator
        d_loss_real = d_model.train_on_batch(x_real, y_real)
        d_loss_fake = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # prepare points in latent space as input for the generator
        x_gan = z_gen(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(x_gan, y_gan)
        d_loss1.append(d_loss[0])
        d_acc.append(d_loss[1])
        g_loss1.append(g_loss)
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100 * d_loss[1], g_loss))
        # evaluate the model every n_eval epochs
        if (i + 1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim, n=half_batch)
    # g_model.save('generator.h5')


# size of the latent space
noise_dim = 5
# define the discriminator model
generator = generator(noise_dim)
discriminator = discriminator()
gan_model = define_gan(generator, discriminator)
# summarize gan model
train(generator, discriminator, gan_model, noise_dim, n_epochs=60, n_batch=256, n_eval=30)
