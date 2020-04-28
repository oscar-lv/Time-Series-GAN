# -*- coding: utf-8 -*-

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import LSTM,Bidirectional, GRU 
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import optimizers,regularizers
from numpy import expand_dims
import numpy as np

#import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

import pandas as pd

import sys

import numpy as np
from tqdm import tqdm

# -*- coding: utf-8 -*-

df = pd.read_csv("SPY_daily.csv").fillna(0).set_index('date').sort_index()
ts = df[["4. close"]]
# ts = np.log(ts) - np.log(ts.shift(1))
df = pd.read_csv("AirPassengers.csv")
ts = df[["#Passengers"]]
X_train = ts.dropna().values

# X_train= np.reshape(X_train,(1,X_train.shape[0], X_train.shape[1]))
X_train = np.expand_dims(X_train, axis=2)
shape=(X_train.shape)


print("X_train")
print(X_train.shape)

def generate_sp_samples(n):
   data = pd.read_csv('./SPY_daily.csv')[['date', '4. close']].set_index('date').sort_index()
   X1 = np.arange(len(data[0:n]))
   X2 = np.log(data) - np.log(data.shift(1))
   X2 = X2[1:n].values
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
        self.latent_dim = 48
        self.d_loss1 = []
        self.d_acc = []
        self.g_loss1 = []

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        data = self.generator(z)
        self.discriminator.trainable = False

        validity = self.discriminator(data)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


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
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


            if epoch % sample_interval == 0:
                self.plot_gan_result(batch_size)
                c = X_train.reshape(X_train.shape[0], 1)
                fig, axs = plt.subplots()
                axs.plot(c, color = "blue", label = 'true')
                plt.show()



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=1000, batch_size=120, sample_interval=20)
