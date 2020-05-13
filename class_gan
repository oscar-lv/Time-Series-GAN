#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:47:40 2020

@author: oscar
"""

# -*- coding: utf-8 -*-

from __future__ import print_function, division



from tensorflow.keras.layers import Input, Reshape, Convolution2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from numpy import expand_dims

# import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
import pandas as pd

np.random.seed(1129)

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

class GAN():
    def __init__(self):
        self.data_rows = 1
        self.data_cols = 1
        self.data_shape = (self.data_rows, self.data_cols)
        # self.data_shape = shape
        self.latent_dim = 5
        self.d_loss1 = []
        self.d_acc = []
        self.g_loss1 = []

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.generator()

        z = Input(shape=(self.latent_dim,))
        data = self.generator(z)
        self.discriminator.trainable = False

        validity = self.discriminator(data)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def generator(self, n_outputs=2):
        model = Sequential()


        model.add(Dense(256, kernel_initializer='he_uniform', input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(n_outputs, activation='linear'))
        noise = Input(shape=(self.latent_dim,))
        data = model(noise)
        return Model(noise, data)

    def discriminator(self, n_inputs=2):
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

        # data = Input(shape=(None,2))
        # validity = model(data)

        return model


    def plot_gan_result(self, n):
        noise = np.random.normal(0, 1, (n, self.latent_dim))
        gen_data = self.generator.predict(noise)
        b = gen_data.reshape(n, 2)
        b = gen_data
        fig, axs = plt.subplots()
        print("noise shape")
        print(noise.shape)
        print(noise[0])
        axs.scatter(b[:, 0], b[:, 1], color='red', label='generated')
        # axs.plot(b, color = "red", label = 'generated')

    def train(self, epochs, batch_size=128, sample_interval=50):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in tqdm(range(epochs)):

            data_s, _ = generate_sp_samples(batch_size)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_data = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(data_s, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            self.d_loss1.append(d_loss[0])
            self.d_acc.append(d_loss[1])
            self.g_loss1.append(g_loss)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.plot_gan_result(batch_size)
                c = data_s.reshape(data_s.shape[0], 2)
                c = data_s
                fig, axs = plt.subplots()
                axs.scatter(c[:, 0], c[:, 1], color='blue')
                # axs.plot(c, color = "blue", label = 'true')
                plt.savefig('fig'+str(epoch), dpi=500)
                plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10000, batch_size=4000, sample_interval=500)
