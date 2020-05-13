#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:34:45 2020

@author: oscar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:17:37 2020

@author: oscar
"""

import matplotlib.pyplot as plt
# example of generating random samples from X^2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import tensorflow as tf
tf.keras.backend.clear_session()

"""


Discriminator Data Generation

"""
def generate_sp_samples(n):
    data = pd.read_csv('./SPY_daily.csv')[['date', '4. close']].set_index('date').sort_index()
    X1 = np.arange(len(data[0:n]))
    # X2 = np.log(data) - np.log(data.shift(1))
    X2 = data[0:n].values
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y
"""
Discriminator Methods

"""
n_critic = 5
clip_value = 0.01
optimizer = RMSprop(lr=0.00005)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)



from tqdm import tqdm


def discriminator(n_inputs=2):
    model = Sequential()

    model.add(Convolution1D(64, (1), activation='relu', batch_input_shape=(None, 1, 2)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling1D(pool_size=(3), strides=(2), padding="same"))
    model.add(Convolution1D(64, (1), activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Convolution1D(64, (1), activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling1D(pool_size=(3), strides=(2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss=wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
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
    model.add(Reshape((1,2)))
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
    model.compile(loss=wasserstein_loss, optimizer=optimizer)
    return model


"""

Evaluation

"""


# evaluate the discriminator and oit_not real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_sp_samples(n)
    # evaluate discriminator on real examples
    #_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_from_noise(generator, latent_dim, n)
    # evaluate discriminator on fake examples
   # _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch)
    # scatter plot real and fake data points
    plt.plot(x_real[:, 0], x_real[:, 1], color='red')
    #plt.plot(x_real, color='red', label='real')
    #plt.plot(x_fake, color='red')
    x_fake = x_fake.reshape(n,2)
    plt.scatter(x_fake[:, 0], x_fake[:, 1], label='fake',color='blue')
    plt.legend(['real','fake'])
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
        x_fake = x_fake.reshape(half_batch, 1, 2)
        x_real = x_real.reshape(half_batch, 1,2)
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
        for l in d_model.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)
        if (i + 1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim, n=half_batch)
    # g_model.save('generator.h5')
    return d_loss1, d_acc, g_loss1


# size of the latent space
noise_dim = 5
# define the discriminator model
generator = generator(noise_dim)
discriminator = discriminator()
gan_model = define_gan(generator, discriminator)
# summarize gan model
history = train(generator, discriminator, gan_model, noise_dim, n_epochs=30000, n_batch=5000, n_eval=50)
