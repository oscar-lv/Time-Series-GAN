#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:02:24 2020

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
import keras.backend as K


from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm



import pandas as pd
df = pd.read_csv("SPY_daily.csv").fillna(0).set_index('date').sort_index()
ts = df[["4. close"]]
X_train = ts.dropna().values

X_train = np.cos(np.arange(100))
# # X_train= np.reshape(X_train,(1,X_train.shape[0], X_train.shape[1]))

# X_train = np.expand_dims(X_train, axis=2)
# shape=(X_train.shape)


class GAN():
    def __init__(self):
        self.data_rows = 1
        self.data_cols = 2
        self.data_shape = (500, 2,1)#(self.data_rows, self.data_cols)
        # self.data_shape = shape
        self.latent_dim = 48
        self.d_loss1 = []
        self.d_acc = []
        self.g_loss1 = []

        optimizer = Adam(0.0002, 0.5)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.generator()

        z = Input(shape=(self.latent_dim,))
        data = self.generator(z)
        self.discriminator.trainable = False

        validity = self.discriminator(data)
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.data_shape), activation='linear'))
        model.add(Reshape(self.data_shape))

        model.summary()

        # model = Sequential()
        # model.add(Dense(48,W_regularizer=regularizers.l2(l=0.01), input_dim=self.latent_dim))
        # model.add(Bidirectional(LSTM(64, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(64, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
        # model.add(Dropout(0.2))
        # model.add(Flatten())
        # model.add(Dense(np.prod(self.data_shape), activation='linear'))
        # model.add(Reshape(self.data_shape))

        noise = Input(shape=(self.latent_dim,))
        data = model(noise)

        return Model(noise, data)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def generator(self, n_outputs=1):
        model = Sequential()
        # model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=noise_dim))
        # model.add(Dense(n_outputs, activation='linear'))

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


        # model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def discriminator(self, n_inputs=1):
        model = Sequential()
        # model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
        # model.add(Dense(1, activation='sigmoid'))

        model.add(Dense(512, input_dim=n_inputs, activation='relu', kernel_initializer='he_uniform'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='linear'))
        
        # model.add(Convolution2D(64, (1), activation='relu', input_shape=self.data_shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(MaxPool2D(pool_size=(3), strides=(2), padding="same"))
        # model.add(Convolution2D(64, (1), activation='relu', input_shape=self.data_shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Convolution2D(64, (1), activation='relu', input_shape=self.data_shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(MaxPool2D(pool_size=(3), strides=(2), padding="same"))
        # model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.data_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        # model.add(Convolution1D(64, (1), activation='relu', input_shape=shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))
        # model.add(Convolution1D(64, (1), activation='relu', input_shape=shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Convolution1D(64, (1), activation='relu', input_shape=shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))
        # model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))

        data = Input(shape=self.data_shape)
        validity = model(data)

        return Model(data, validity)

    def plot_gan_result(self, n):
        noise = np.random.normal(0, 1, (n, self.latent_dim))
        gen_data = self.generator.predict(noise)
        b = gen_data.reshape(n, 1)
        b = gen_data
        fig, axs = plt.subplots()
        print("noise shape")
        print(noise.shape)
        print(noise[0])
        # axs.scatter(b[:, 0], b[:, 1], color='red', label='generated')
        axs.plot(b, color = "red", label = 'generated')

    def train(self, epochs, batch_size=128, sample_interval=50):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in tqdm(range(epochs)):

            data_s = X_train
            # data_s=np.expand_dims((data_s), axis=2)
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
            for l in self.discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                l.set_weights(weights)

            if epoch % sample_interval == 0:
                self.plot_gan_result(batch_size)
                c = data_s.reshape(data_s.shape[0], 1)
                c = data_s
                fig, axs = plt.subplots()
                # axs.scatter(c[:, 0], c[:, 1], color='blue')
                axs.plot(c, color = "blue", label = 'true')
                plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=5000, batch_size=100, sample_interval=200)
