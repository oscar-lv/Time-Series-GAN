#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:17:37 2020

@author: oscar
"""

# example of generating random samples from X^2
import numpy as np
from numpy.random import rand
from numpy import hstack
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential

"""

Discriminator Data Generation

"""

# generate n real samples with class labels
def generate_real_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = np.arange(n)
    X1 = X1 * np.pi/180
    # generate outputs X^2
    X2 = np.cos(X1)
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = np.ones((n, 1))
    return X, y

import pandas as pd


def generate_sp_samples(n):
   data = pd.read_csv('./SPY_daily.csv')[['date', '4. close']].set_index('date').sort_index()
   X1 = np.arange(len(data[0:n]))
   X2 = data[0:n].values
   X1 = X1.reshape(n, 1)
   X2 = X2.reshape(n, 1)
   X = hstack((X1, X2))
   y = np.ones((n, 1))
   return X, y

    

def plot_n(n):
    X,y = generate_real_samples(n)
    plt.scatter(X[:,0], X[:,1])
    

# generate n fake samples with class labels
def generate_fake_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = rand(n) - 0.5
    # generate outputs X^2
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = np.zeros((n, 1))
    return X, y

"""

Discriminator Methods

"""

from tqdm import tqdm

def discriminator(n_inputs = 2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 

def train_discriminator(model, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch/2)
    for e in tqdm(range(n_epochs)):
        X_real, y_real = generate_real_samples(half_batch)
        model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(half_batch)
        model.train_on_batch(X_fake, y_fake)
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
        print('Iteration: '+ str(e) + '/' + str(n_epochs-1) + " - Real Acc: " + str(acc_real) + " - Fake Acc: " + str(acc_fake))

# model = discriminator()
# train_discriminator(model)

"""

Generator Methods

"""

def z_gen(noise_dim=5, n=100):
    x_input = np.random.randn(noise_dim * n)
    x_input = x_input.reshape(n, noise_dim)
    return x_input
    
def generator(noise_dim, n_outputs = 2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=noise_dim))
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
# evaluate the discriminator and plot real and fake points
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
    plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    plt.show()
    
"""

GAN Training

"""

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=20000, n_batch=512, n_eval=1000):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in tqdm(range(n_epochs)):
        # prepare real samples
        x_real, y_real = generate_sp_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = generate_from_noise(g_model, latent_dim, half_batch)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = z_gen(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim, n=half_batch)
    #g_model.save('generator.h5')


# size of the latent space
noise_dim = 5
# define the discriminator model
generator = generator(noise_dim)
discriminator = discriminator()
gan_model = define_gan(generator, discriminator)
# summarize gan model
train(generator, discriminator, gan_model, noise_dim)



