#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:41:53 2020

@author: oscar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plot the training for a model
def plot_gan_training(d_loss,g_loss):
    return 

# Plot the distribution of generated samples
def plot_dist(prices : pd.DataFrame, title:str='Distribution Plot',save:bool=False, fname:str='distplot'):
    lrets = np.log(prices) - np.log(prices.shift(1))
    sns.distplot(lrets, bins=30, color='skyblue', axlabel='Log Returns', label='Log returns')
    plt.title(title)
    if save==True:
        plt.savefig(fname, dpi=500)
    plt.show()

# Plot the training history and save
def plot_gan_history(history, title:str='Distribution Plot',save:bool=False, fname:str='distplot'):
    plt.plot(history[0].values, label='discriminator')
    plt.plot(history[2].values,label='generator')
    plt.legend(['Discriminator Loss', 'Generator Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    if save==True:
        plt.savefig(fname, dpi=500)
    plt.show()

# Plot the resulting generated samples
def plot_gan_result(generator, n, latent_dim, fname, save, title, flip):
    gen_data = generate_samples(generator, n, latent_dim)
    b = gen_data.reshape(n, 2)
    b = gen_data
    fig, axs = plt.subplots()
    plt.ylabel('Generated Price (USD)')
    plt.xlabel('Periods')
    axs.scatter(b[:, 0], b[:, 1], color='blue', label='generated')
    plt.title(title)
    if flip==True:
        plt.gca().invert_xaxis()
    if save==True:
        plt.savefig(fname, dpi=500)
    plt.show()
    # axs.plot(b, color = "red", label = 'generated')

# Generate samples from noise
def generate_samples(generator, n, noise_dim):
    noise = np.random.normal(0, 1, (n,noise_dim))
    generated_samples = generator.predict(noise)
    return generated_samples