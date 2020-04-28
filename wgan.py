
from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from tensorflow.keras.layers import LSTM,Bidirectional, GRU 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers,regularizers
from numpy import expand_dims
import keras.backend as K
#import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

import pandas as pd

import sys

import numpy as np
from tqdm import tqdm

# -*- coding: utf-8 -*-

df = pd.read_csv("SPY_daily.csv").fillna(0).set_index('date').sort_index()
ts = df[["4. close"]]
# df = pd.read_csv("AirPassengers.csv")
# ts = df[["#Passengers"]]
X_train = ts.values
shape=(X_train.shape[1],1)

# Rescale -1 to 1
#X_train = X_train / 127.5 - 1.

X_train = np.expand_dims(X_train, axis=2)

print("X_train")
print(X_train.shape)


class GAN():
    def __init__(self):
        self.data_rows = 1
        self.data_cols = 1
        self.data_shape = shape
        self.latent_dim = 48
        self.d_loss1 = []
        self.d_acc = []
        self.g_loss1 = []

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

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

    def build_discriminator(self):
        
        
        model = Sequential()

        # model.add(Flatten(input_shape=self.data_shape))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        model.add(Convolution1D(64, (1), activation='relu', input_shape=shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))
        model.add(Convolution1D(64, (1), activation='relu', input_shape=shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Convolution1D(64, (1), activation='relu', input_shape=shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))


        data = Input(shape=self.data_shape)
        validity = model(data)

        return Model(data, validity)

    def plot_gan_result(self, n):
        noise = np.random.normal(0, 1, (n, self.latent_dim))
        gen_data = self.generator.predict(noise)

        b = gen_data.reshape(n, 1)
        fig, axs = plt.subplots()
        print("noise shape")
        print(noise.shape)
        print(noise[0])
        axs.plot(b, color = "red", label = 'generated')
        
    def train(self, epochs, batch_size=128, sample_interval=50):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in tqdm(range(epochs)):

            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # data_s = X_train[idx]
            data_s = X_train[0:batch_size]


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
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
            # Clip critic weights
            for l in self.discriminator.layers:
              weights = l.get_weights()
              weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
              l.set_weights(weights)

            if epoch % sample_interval == 0:
                self.plot_gan_result(batch_size)
                c = X_train.reshape(X_train.shape[0], 1)
                fig, axs = plt.subplots()
                axs.plot(c, color = "blue", label = 'true')
                plt.show()
        print('End of traning')



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=200, sample_interval=100)
