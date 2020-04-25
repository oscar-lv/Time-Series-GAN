#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:01:42 2020

@author: oscar
"""

# Imports
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()
from pandas_datareader import data
from time import time
from matplotlib.colors import hsv_to_rgb
from scipy.stats import ks_2samp
import yfinance as yf

""" Environement Constants """
learning_rate = 0.001
lr = learning_rate
batch_size = 28
latent_dim = 20
num_epochs = 50
vis_freq = 2
labels = None
D_rounds = 3
G_rounds = 1
hidden_units_g = 150
hidden_units_d = 150
num_generated_features = 1

# download Apple price data into DataFrame

# STEP 1 SIMULATE LOG RETURNS VIA MONTE CARLO SIMULATIONS ( GENERATE SOME THE DATA USED TO LEARN THE GAN)
# ESTIMATION OF THE MEAN AND VOLATILITY
T = 9  # Number of trading days
# download Apple price data into DataFrame
spy = data.get_data_yahoo('SPY', start='1/1/2000 ')
# calculate the compound annual growth rate ( CAGR ) which
# will give us our mean return input (mu)
days = (spy.index[-1] - spy.index[0]).days

cagr = ((((spy['Adj Close'][-1]) / spy['Adj Close'][1])) **
        (365.0 / days)) - 1
print('CAGR =', str(round(cagr, 4) * 100) + "%")
mu = cagr
# create a series of percentage returns and calculate the annual volatility of returns
spy['Returns'] = spy['Adj Close'].pct_change()
vol = spy['Returns'].std() * math.sqrt(T)
print(" Annual Volatility =", str(round(vol, 4) * 100) + "%")
# GENERATION OF THE DATA USED TO TRAIN THE GAN
# Define Variables
S = spy['Adj Close'][-1]  # starting stock price (i.e. last available real stock price )

n_samples = 50000
seq_length = 10  # change line 202 and 227 as well


# log returns
def sample_data(n=n_samples):
    vectors = []
    for i in range(n):
        # create list of daily returns using random normal
        d_returns = np.random.normal(mu / T, vol / math.sqrt(T),
                                     seq_length)  # T+1 since we work with the vectors = [] and
        vectors.append(np.log(d_returns + 1))
    dataset = np.array(vectors)
    # dataset = np. expand_dims ( dataset , axis =0)
    # dataset = dataset . reshape (38 , 253 , 1)
    dataset.reshape(-1, seq_length, 1)
    return dataset


CG = tf.placeholder(tf.float32, [batch_size, seq_length])  # Placeholder 0(shape: (50, 253) ) is it 0 or seq_length
CD = tf.placeholder(tf.float32, [batch_size, seq_length])  # Placerholder 1
Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim
                                ])  # Placeholder 2
W_out_G = tf.Variable(tf.truncated_normal([hidden_units_g,
                                           num_generated_features]))
b_out_G = tf.Variable(tf.truncated_normal([num_generated_features])
                      )
X = tf.placeholder(tf.float32, [batch_size, seq_length,
                                num_generated_features])
W_out_D = tf.Variable(tf.truncated_normal([hidden_units_d, 1]))
b_out_D = tf.Variable(tf.truncated_normal([1]))


def sample_Z(batch_size, seq_length, latent_dim):
    sample = np.float32(np.random.normal(size=[batch_size,
                                               seq_length, latent_dim]))
    return sample


def generator(z, c):
    with tf.variable_scope("generator") as scope:
        # each step of the generator takes a random seed + the conditional embedding
        repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
        repeated_encoding = tf.reshape(repeated_encoding, [tf.shape
                                                           (z)[0], tf.shape(z)[1], 10])
        generator_input = tf.concat([repeated_encoding, z], 2)
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_g,
                                       state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length] * batch_size,
            inputs=generator_input)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1,
                                                  hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length,
                                           num_generated_features])
    return output_3d


def discriminator(x, c, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        # each step of the generator takes one time step of the signal to evaluate its conditional embedding
        repeated_encoding = tf.tile(c, [1, tf.shape(x)[1]])
        repeated_encoding = tf.reshape(repeated_encoding, [tf.shape
                                                           (x)[0], tf.shape(x)[1], 10])
        decoder_input = tf.concat([repeated_encoding, x], 2)
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d,
                                       state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=decoder_input)
        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1,
                                                    hidden_units_g])
        logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D
        output = tf.nn.sigmoid(logits)
    return output, logits


G_sample = generator(Z, CG)
D_real, D_logit_real = discriminator(X, CD)
D_fake, D_logit_fake = discriminator(G_sample, CG, reuse=True)
generator_vars = [v for v in tf.trainable_variables() if v.name.
    startswith('generator')]
discriminator_vars = [v for v in tf.trainable_variables() if v.name
    .startswith('discriminator')]

D_loss_real = tf.reduce_mean(tf.nn.
                             sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                               labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.
                             sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                               labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake,
    labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(D_loss, var_list=discriminator_vars)
# D_solver = tf. train . AdamOptimizer (). minimize (D_loss , var_list = generator_vars )
# G_solver = tf. train . GradientDescentOptimizer ( learning_rate =lr).minimize(G_loss, var_list=discriminator_vars)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator_vars)

# Train the GAN on the Monte Carlo simulations

# Starting a tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
vis_Z = sample_Z(batch_size, seq_length, latent_dim)
t0 = time()


def get_batch(samples, batch_idx, batch_size):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos]


def train_generator(batch_idx, offset):
    # update the generator
    for g in range(G_rounds):
        # X_mb = get_batch ( samples , batch_size , batch_idx + g + offset )
        Y_mb = get_batch(samples, batch_size, batch_idx + g +
                         offset)
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={CG: Y_mb,
                                             Z: sample_Z(batch_size
                                                         , seq_length,
                                                         latent_dim)})
    return G_loss_curr


def train_discriminator(batch_idx, offset):
    # update the discriminator
    for d in range(D_rounds):
        # using same input sequence for both the synthetic data and the real one, probably it is not a good idea ...
        X_mb = get_batch(samples, batch_size, batch_idx + d +
                         offset)
        X_mb = X_mb.reshape(batch_size, seq_length, 1)
        Y_mb = get_batch(samples, batch_size, batch_idx + d +
                         offset)
        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={CD: Y_mb, CG: Y_mb, X:
                                      X_mb,
                                             Z: sample_Z(batch_size
                                                         , seq_length,
                                                         latent_dim)})
    return D_loss_curr


print('num_epoch \t D_loss_curr \t G_loss_curr \t time elapsed ')
samples = sample_data(n_samples) 
d_loss = []
g_loss = []
for num_epoch in range(num_epochs):
    for batch_idx in range(0, int(n_samples / batch_size) - (
            D_rounds + G_rounds), D_rounds + G_rounds):
        if num_epoch % 2 == 0:
            G_loss_curr = train_generator(batch_idx, 0)
            D_loss_curr = train_discriminator(batch_idx, G_rounds)
        else:
            D_loss_curr = train_discriminator(batch_idx, 0)
            G_loss_curr = train_generator(batch_idx, D_rounds)
            d_loss.append(D_loss_curr)
            g_loss.append(G_loss_curr)
            t = time() - t0
            print(num_epoch, '\t', D_loss_curr, '\t', G_loss_curr, '\t', t)
    # save synthetic data
    if num_epoch % 5 == 0:
        # generate synthetic dataset
        gen_samples = []
        for batch_idx in range(int(len(samples) / batch_size)):
            X_mb = get_batch(samples, batch_size, batch_idx)
            Y_mb = get_batch(samples, batch_size, batch_idx)
            z_ = sample_Z(batch_size, seq_length, latent_dim)
            gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_,
                                                           CG: Y_mb})
            gen_samples.append(gen_samples_mb)
            print(batch_idx)

        #gen_samples = np.vstack(gen_samples)







##### Results¨
            
ax = pd.DataFrame({'Generative Loss ': g_loss, ' Discriminative Loss ': d_loss, }).plot(title='Training loss ',
                                                                                        logy=True)
ax.set_xlabel(" Training iteration ")
ax.set_ylabel(" Loss ")

generated_data = np.transpose(gen_samples)
# plot the log - returns
gen_ind = 1  # change in function price as well
pd.DataFrame(generated_data[0, :, gen_ind]).plot()  # 1 is the index of the plotted sample out of the 1000 generated


# get the prices from the log returns
def price_gen(ind_gen_sample=1):
    daily_log_returns = generated_data[0, :, ind_gen_sample]
    price_list = [S]  # initial price ( today )
    for x in np.transpose(daily_log_returns):
        price_list.append(price_list[-1] * np.exp(x))
    # price_list = np. asarray ( price_list )
    return price_list


def price_real():
    daily_returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1

    price_list = [S]
    for x in daily_returns:
        price_list.append(price_list[-1] * x)
    return price_list


def prices_gen_data_frame(num_samples=1):
    df = pd.DataFrame([])
    for i in range(num_samples):
        df[i] = price_real()
    return df


def prices_real_data_frame(num_samples=1):
    df = pd.DataFrame()


    for i in range(num_samples):
        df[i] = price_real()
    return df

# Plot the price evolution of the generated sample
def plot_price_gen ( num_gen_sample = 1) :
    for i in range ( num_gen_sample ) :
       daily_log_returns = generated_data [0 ,: , i ]
       price_list = [ S ]
       for x in daily_log_returns:
           price_list.append(price_list[-1] * np.exp(x))
       plt.plot(price_list)
    return plt.show()


# Plot one real evolution
def plot_price_real(num_gen_samples=1):
    for i in range(num_gen_samples):

        # create list of daily returns using random normaldistribution
        daily_returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
        # set starting price and create price series generated by above randomdailyreturns
        price_list = [S]
        for x in daily_returns:
            price_list.append(price_list[-1] * x)
        # plot data from each individual run which we will plot at theend
        plt.plot(price_list)
        # pd. DataFrame ( price_list ). plot () #( if we want plots on separatefigures )
    return plt.show()  # return None (if we want plots on separate figures )


plot_price_gen(20000)
plot_price_real(20000)
